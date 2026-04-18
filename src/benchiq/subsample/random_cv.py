"""Cross-validated random subsampling to fixed preselection sizes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from benchiq.io.load import Bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.logging import update_manifest
from benchiq.preprocess.filters import PreprocessResult
from benchiq.preprocess.scores import SCORE_FULL, ScoreResult
from benchiq.preprocess.stats import build_benchmark_matrix
from benchiq.reconstruct.gam import DEFAULT_LAM_GRID, cross_validate_gam
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID, SPLIT
from benchiq.split.splitters import SplitResult

REDUCED_SCORE = "reduced_score"
REDUCED_ITEM_COVERAGE = "reduced_item_coverage"
ITERATION_ID = "iteration_id"
SUBSET_SEED = "subset_seed"
STATUS = "status"
VAL_RMSE_MAX = "max_rmse_val"
VAL_RMSE_MEAN = "mean_rmse_val"
TEST_RMSE_MEAN = "mean_rmse_test"
BEST_LAM = "best_lam"
SAMPLED_ITEM_IDS = "sampled_item_ids"
FAILED_REASON = "failed_reason"
METHOD = "method"
METHOD_RANDOM_CV = "random_cv"
METHOD_DETERMINISTIC_INFO = "deterministic_info"
INFO_PROXY_SCORE = "information_proxy_score"


@dataclass(slots=True)
class BenchmarkSubsampleResult:
    """Per-benchmark random-CV subsampling outputs."""

    benchmark_id: str
    preselect_items: pd.DataFrame
    cv_results: pd.DataFrame
    subsample_report: dict[str, Any]
    ranking_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifact_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(slots=True)
class SubsampleResult:
    """Stage-04 subsampling outputs."""

    benchmarks: dict[str, BenchmarkSubsampleResult]
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None


def subsample_bundle(
    bundle: Bundle,
    preprocess_result: PreprocessResult,
    score_result: ScoreResult,
    split_result: SplitResult,
    *,
    method: str = METHOD_RANDOM_CV,
    k_preselect: int | None = None,
    n_iter: int = 2000,
    cv_folds: int = 5,
    checkpoint_interval: int = 25,
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> SubsampleResult:
    """Run random-CV subsampling for each benchmark and optionally write artifacts."""

    run_root, manifest_path = _resolve_run_root(bundle, out_dir=out_dir, run_id=run_id)
    resolved_method = _resolve_method(method)
    benchmark_results: dict[str, BenchmarkSubsampleResult] = {}
    for benchmark_id in sorted(preprocess_result.benchmarks):
        checkpoint_dir = None
        if run_root is not None:
            checkpoint_dir = (
                run_root / "artifacts" / "04_subsample" / "per_benchmark" / benchmark_id
            )
        benchmark_results[benchmark_id] = subsample_benchmark(
            bundle,
            preprocess_result,
            score_result,
            split_result,
            benchmark_id=benchmark_id,
            method=resolved_method,
            k_preselect=k_preselect,
            n_iter=n_iter,
            cv_folds=cv_folds,
            checkpoint_interval=checkpoint_interval,
            lam_grid=lam_grid,
            checkpoint_dir=checkpoint_dir,
        )

    result = SubsampleResult(benchmarks=benchmark_results)
    if run_root is not None:
        artifact_paths = _write_subsample_artifacts(result, run_root=run_root)
        result.artifact_paths = artifact_paths
        result.manifest_path = manifest_path
        if manifest_path is not None:
            update_manifest(
                manifest_path,
                {
                    "artifacts": {
                        "04_subsample": {
                            "per_benchmark": {
                                benchmark_id: {
                                    name: str(path) for name, path in sorted(paths.items())
                                }
                                for benchmark_id, paths in sorted(
                                    artifact_paths["per_benchmark"].items()
                                )
                            },
                        },
                    },
                },
            )
    return result


def subsample_benchmark(
    bundle: Bundle,
    preprocess_result: PreprocessResult,
    score_result: ScoreResult,
    split_result: SplitResult,
    *,
    benchmark_id: str,
    method: str = METHOD_RANDOM_CV,
    k_preselect: int | None = None,
    n_iter: int = 2000,
    cv_folds: int = 5,
    checkpoint_interval: int = 25,
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
    checkpoint_dir: str | Path | None = None,
) -> BenchmarkSubsampleResult:
    """Run one supported preselection method for a benchmark."""

    resolved_method = _resolve_method(method)
    if resolved_method == METHOD_RANDOM_CV:
        return _subsample_benchmark_random_cv(
            bundle,
            preprocess_result,
            score_result,
            split_result,
            benchmark_id=benchmark_id,
            k_preselect=k_preselect,
            n_iter=n_iter,
            cv_folds=cv_folds,
            checkpoint_interval=checkpoint_interval,
            lam_grid=lam_grid,
            checkpoint_dir=checkpoint_dir,
        )
    if resolved_method == METHOD_DETERMINISTIC_INFO:
        return _subsample_benchmark_deterministic_info(
            bundle,
            preprocess_result,
            score_result,
            split_result,
            benchmark_id=benchmark_id,
            k_preselect=k_preselect,
            cv_folds=cv_folds,
            lam_grid=lam_grid,
            checkpoint_dir=checkpoint_dir,
        )
    raise ValueError(f"unsupported subsample method: {method}")


def _subsample_benchmark_random_cv(
    bundle: Bundle,
    preprocess_result: PreprocessResult,
    score_result: ScoreResult,
    split_result: SplitResult,
    *,
    benchmark_id: str,
    k_preselect: int | None = None,
    n_iter: int = 2000,
    cv_folds: int = 5,
    checkpoint_interval: int = 25,
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
    checkpoint_dir: str | Path | None = None,
) -> BenchmarkSubsampleResult:
    """Run random-CV subsampling for one benchmark."""

    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be at least 1")

    benchmark_preprocess = preprocess_result.benchmarks[benchmark_id]
    if benchmark_preprocess.refused:
        return _skipped_benchmark_result(
            benchmark_id,
            method=METHOD_RANDOM_CV,
            selection_rule="minimax_validation_rmse",
            skipped_reason="benchmark_refused_in_preprocess",
            candidate_item_count=0,
            requested_k_preselect=k_preselect,
            effective_k_preselect=0,
            n_iter=n_iter,
            cv_folds=cv_folds,
        )

    candidate_item_ids = (
        benchmark_preprocess.filtered_items[ITEM_ID]
        .dropna()
        .astype("string")
        .sort_values()
        .reset_index(drop=True)
        .tolist()
    )
    split_frame = split_result.per_benchmark_splits.get(benchmark_id)
    if split_frame is None or split_frame.empty:
        return _skipped_benchmark_result(
            benchmark_id,
            method=METHOD_RANDOM_CV,
            selection_rule="minimax_validation_rmse",
            skipped_reason="no_split_models_available",
            candidate_item_count=len(candidate_item_ids),
            requested_k_preselect=k_preselect,
            effective_k_preselect=0,
            n_iter=n_iter,
            cv_folds=cv_folds,
        )

    score_lookup = (
        score_result.scores_full.loc[
            score_result.scores_full[BENCHMARK_ID] == benchmark_id,
            [MODEL_ID, SCORE_FULL],
        ]
        .dropna(subset=[SCORE_FULL])
        .astype({MODEL_ID: "string"})
        .set_index(MODEL_ID)[SCORE_FULL]
        .astype("Float64")
    )
    split_lookup = split_frame.set_index(MODEL_ID)[SPLIT]
    pool_model_ids = [
        model_id
        for model_id in split_lookup.index.astype("string").tolist()
        if split_lookup.loc[model_id] != "test" and model_id in score_lookup.index
    ]
    test_model_ids = [
        model_id
        for model_id in split_lookup.index.astype("string").tolist()
        if split_lookup.loc[model_id] == "test" and model_id in score_lookup.index
    ]
    if not pool_model_ids:
        return _skipped_benchmark_result(
            benchmark_id,
            method=METHOD_RANDOM_CV,
            selection_rule="minimax_validation_rmse",
            skipped_reason="no_train_val_models_available",
            candidate_item_count=len(candidate_item_ids),
            requested_k_preselect=k_preselect,
            effective_k_preselect=0,
            n_iter=n_iter,
            cv_folds=cv_folds,
        )

    effective_k = _resolve_k_preselect(
        requested_k_preselect=k_preselect,
        candidate_item_count=len(candidate_item_ids),
        pool_model_count=len(pool_model_ids),
    )
    if effective_k is None:
        return _skipped_benchmark_result(
            benchmark_id,
            method=METHOD_RANDOM_CV,
            selection_rule="minimax_validation_rmse",
            skipped_reason="k_preselect_exceeds_candidate_items",
            candidate_item_count=len(candidate_item_ids),
            requested_k_preselect=k_preselect,
            effective_k_preselect=0,
            n_iter=n_iter,
            cv_folds=cv_folds,
        )
    if len(pool_model_ids) < cv_folds:
        return _skipped_benchmark_result(
            benchmark_id,
            method=METHOD_RANDOM_CV,
            selection_rule="minimax_validation_rmse",
            skipped_reason="cv_folds_exceed_train_val_models",
            candidate_item_count=len(candidate_item_ids),
            requested_k_preselect=k_preselect,
            effective_k_preselect=effective_k,
            n_iter=n_iter,
            cv_folds=cv_folds,
        )

    benchmark_matrix = build_benchmark_matrix(
        bundle.responses_long, benchmark_id=benchmark_id
    ).reindex(
        index=pd.Index(pool_model_ids + test_model_ids, dtype="string"),
        columns=candidate_item_ids,
    )

    iteration_records: list[dict[str, Any]] = []
    valid_iterations: list[dict[str, Any]] = []
    warnings: list[str] = []
    benchmark_offset = _benchmark_seed_offset(benchmark_id)
    progress_payload = _progress_payload(
        benchmark_id=benchmark_id,
        candidate_item_count=len(candidate_item_ids),
        requested_k_preselect=k_preselect,
        effective_k_preselect=effective_k,
        n_iter=n_iter,
        completed_iterations=0,
        valid_iterations=0,
        failed_iterations=0,
        best_iteration=None,
    )

    for iteration_id in range(n_iter):
        subset_seed = bundle.config.random_seed + benchmark_offset + iteration_id
        sampled_item_ids = _sample_item_subset(
            candidate_item_ids,
            subset_size=effective_k,
            seed=subset_seed,
        )
        reduced_scores = _build_reduced_score_frame(
            benchmark_matrix,
            selected_item_ids=sampled_item_ids,
        )
        pool_frame = _eligible_model_frame(
            reduced_scores=reduced_scores,
            model_ids=pool_model_ids,
            score_lookup=score_lookup,
            min_item_coverage=bundle.config.min_item_coverage,
        )
        test_frame = _eligible_model_frame(
            reduced_scores=reduced_scores,
            model_ids=test_model_ids,
            score_lookup=score_lookup,
            min_item_coverage=bundle.config.min_item_coverage,
        )

        failed_reason = _iteration_failed_reason(pool_frame, cv_folds=cv_folds)
        if failed_reason is not None:
            iteration_records.append(
                _failed_iteration_row(
                    benchmark_id=benchmark_id,
                    iteration_id=iteration_id,
                    subset_seed=subset_seed,
                    sampled_item_ids=sampled_item_ids,
                    candidate_item_count=len(candidate_item_ids),
                    effective_k_preselect=effective_k,
                    valid_pool_count=len(pool_frame.index),
                    valid_test_count=len(test_frame.index),
                    failed_reason=failed_reason,
                )
            )
        else:
            gam_result = cross_validate_gam(
                pool_frame[[REDUCED_SCORE]],
                pool_frame[SCORE_FULL],
                lam_grid=lam_grid,
                cv_folds=cv_folds,
                random_seed=subset_seed,
                feature_names=[REDUCED_SCORE],
                target_name=SCORE_FULL,
                X_test=test_frame[[REDUCED_SCORE]] if not test_frame.empty else None,
                y_test=test_frame[SCORE_FULL] if not test_frame.empty else None,
            )
            best_cv_rows = gam_result.cv_results.loc[
                np.isclose(
                    gam_result.cv_results["lam"].astype(float).to_numpy(),
                    gam_result.best_lam,
                )
            ].copy()
            max_rmse_val = float(best_cv_rows["val_rmse"].max())
            mean_rmse_val = float(best_cv_rows["val_rmse"].mean())
            mean_rmse_test = (
                float(best_cv_rows["test_rmse"].dropna().mean())
                if best_cv_rows["test_rmse"].notna().any()
                else None
            )
            for fold_row in best_cv_rows.to_dict(orient="records"):
                iteration_records.append(
                    {
                        BENCHMARK_ID: benchmark_id,
                        ITERATION_ID: iteration_id,
                        SUBSET_SEED: subset_seed,
                        STATUS: "ok",
                        FAILED_REASON: pd.NA,
                        SAMPLED_ITEM_IDS: json.dumps(sampled_item_ids),
                        "candidate_item_count": len(candidate_item_ids),
                        "k_preselect": effective_k,
                        "valid_pool_model_count": len(pool_frame.index),
                        "valid_test_model_count": len(test_frame.index),
                        BEST_LAM: gam_result.best_lam,
                        VAL_RMSE_MAX: max_rmse_val,
                        VAL_RMSE_MEAN: mean_rmse_val,
                        TEST_RMSE_MEAN: mean_rmse_test,
                        "fold_id": fold_row["fold_id"],
                        "train_rows": fold_row["train_rows"],
                        "val_rows": fold_row["val_rows"],
                        "val_rmse": fold_row["val_rmse"],
                        "baseline_val_rmse": fold_row["baseline_val_rmse"],
                        "test_rmse": fold_row["test_rmse"],
                    }
                )
            valid_iterations.append(
                {
                    ITERATION_ID: iteration_id,
                    SUBSET_SEED: subset_seed,
                    "sampled_item_ids": sampled_item_ids,
                    "best_lam": gam_result.best_lam,
                    "max_rmse_val": max_rmse_val,
                    "mean_rmse_val": mean_rmse_val,
                    "mean_rmse_test": mean_rmse_test,
                    "cv_report": gam_result.cv_report,
                    "valid_pool_model_count": len(pool_frame.index),
                    "valid_test_model_count": len(test_frame.index),
                }
            )

        completed_iterations = iteration_id + 1
        progress_payload = _progress_payload(
            benchmark_id=benchmark_id,
            candidate_item_count=len(candidate_item_ids),
            requested_k_preselect=k_preselect,
            effective_k_preselect=effective_k,
            n_iter=n_iter,
            completed_iterations=completed_iterations,
            valid_iterations=len(valid_iterations),
            failed_iterations=completed_iterations - len(valid_iterations),
            best_iteration=_best_iteration_summary(valid_iterations),
        )
        if completed_iterations % checkpoint_interval == 0 or completed_iterations == n_iter:
            warnings_payload = progress_payload.setdefault("warnings", [])
            if completed_iterations < n_iter:
                warnings_payload.append("checkpoint written before all iterations completed")
            _write_checkpoint(
                checkpoint_dir,
                benchmark_id=benchmark_id,
                cv_results=_build_cv_results_frame(iteration_records),
                best_iteration=_select_best_iteration(valid_iterations),
                progress_payload=progress_payload,
                effective_k_preselect=effective_k,
            )

    cv_results = _build_cv_results_frame(iteration_records)
    best_iteration = _select_best_iteration(valid_iterations)
    if best_iteration is None:
        warnings.append(f"{benchmark_id}: no valid iterations completed")
        preselect_items = _empty_preselect_items_frame()
        skipped_reason = "no_valid_iterations"
    else:
        preselect_items = _build_preselect_items_frame(
            benchmark_id=benchmark_id,
            selected_item_ids=best_iteration["sampled_item_ids"],
            selected_iteration=best_iteration[ITERATION_ID],
            subset_seed=best_iteration[SUBSET_SEED],
            k_preselect=effective_k,
            max_rmse_val=best_iteration["max_rmse_val"],
            mean_rmse_val=best_iteration["mean_rmse_val"],
        )
        skipped_reason = None

    subsample_report = {
        "benchmark_id": benchmark_id,
        "method": METHOD_RANDOM_CV,
        "selection_rule": "minimax_validation_rmse",
        "skipped": skipped_reason is not None,
        "skipped_reason": skipped_reason,
        "warnings": warnings,
        "parameters": {
            "requested_k_preselect": k_preselect,
            "effective_k_preselect": effective_k,
            "n_iter": n_iter,
            "cv_folds": cv_folds,
            "checkpoint_interval": checkpoint_interval,
            "lam_grid": list(lam_grid),
            "min_item_coverage": bundle.config.min_item_coverage,
        },
        "counts": {
            "candidate_item_count": len(candidate_item_ids),
            "pool_model_count": len(pool_model_ids),
            "test_model_count": len(test_model_ids),
            "completed_iterations": n_iter,
            "valid_iterations": len(valid_iterations),
            "failed_iterations": n_iter - len(valid_iterations),
            "cv_rows": int(len(cv_results.index)),
        },
        "best_iteration": _best_iteration_summary(valid_iterations),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return BenchmarkSubsampleResult(
        benchmark_id=benchmark_id,
        preselect_items=preselect_items,
        cv_results=cv_results,
        subsample_report=subsample_report,
        ranking_table=_empty_ranking_table_frame(),
        artifact_paths={"progress_payload": progress_payload},
    )


def _build_reduced_score_frame(
    matrix: pd.DataFrame,
    *,
    selected_item_ids: Sequence[str],
) -> pd.DataFrame:
    selected_matrix = matrix.loc[:, list(selected_item_ids)].copy()
    answered_items = selected_matrix.notna().sum(axis=1)
    reduced_coverage = answered_items / max(1, len(selected_item_ids))
    reduced_score = selected_matrix.mean(axis=1, skipna=True) * 100.0
    return pd.DataFrame(
        {
            MODEL_ID: pd.Series(selected_matrix.index.astype("string"), dtype="string"),
            REDUCED_SCORE: pd.Series(reduced_score.to_numpy(), dtype="Float64"),
            REDUCED_ITEM_COVERAGE: pd.Series(reduced_coverage.to_numpy(), dtype="Float64"),
        },
    ).set_index(MODEL_ID)


def _eligible_model_frame(
    *,
    reduced_scores: pd.DataFrame,
    model_ids: Sequence[str],
    score_lookup: pd.Series,
    min_item_coverage: float,
) -> pd.DataFrame:
    if not model_ids:
        return pd.DataFrame(
            {
                MODEL_ID: pd.Series(dtype="string"),
                REDUCED_SCORE: pd.Series(dtype="Float64"),
                REDUCED_ITEM_COVERAGE: pd.Series(dtype="Float64"),
                SCORE_FULL: pd.Series(dtype="Float64"),
            },
        )

    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        reduced_row = reduced_scores.loc[model_id]
        coverage = reduced_row[REDUCED_ITEM_COVERAGE]
        reduced_score = reduced_row[REDUCED_SCORE]
        full_score = score_lookup.loc[model_id]
        if pd.isna(coverage) or float(coverage) < min_item_coverage:
            continue
        if pd.isna(reduced_score) or pd.isna(full_score):
            continue
        rows.append(
            {
                MODEL_ID: model_id,
                REDUCED_SCORE: float(reduced_score),
                REDUCED_ITEM_COVERAGE: float(coverage),
                SCORE_FULL: float(full_score),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            {
                MODEL_ID: pd.Series(dtype="string"),
                REDUCED_SCORE: pd.Series(dtype="Float64"),
                REDUCED_ITEM_COVERAGE: pd.Series(dtype="Float64"),
                SCORE_FULL: pd.Series(dtype="Float64"),
            },
        )
    frame[MODEL_ID] = frame[MODEL_ID].astype("string")
    frame[REDUCED_SCORE] = pd.Series(frame[REDUCED_SCORE], dtype="Float64")
    frame[REDUCED_ITEM_COVERAGE] = pd.Series(frame[REDUCED_ITEM_COVERAGE], dtype="Float64")
    frame[SCORE_FULL] = pd.Series(frame[SCORE_FULL], dtype="Float64")
    return frame.sort_values(MODEL_ID).reset_index(drop=True)


def _iteration_failed_reason(pool_frame: pd.DataFrame, *, cv_folds: int) -> str | None:
    if pool_frame.empty:
        return "no_valid_models_after_reduced_coverage_filter"
    if len(pool_frame.index) < cv_folds:
        return "too_few_valid_models_for_cv"
    if pool_frame[REDUCED_SCORE].nunique(dropna=True) < 2:
        return "reduced_score_not_variable"
    return None


def _subsample_benchmark_deterministic_info(
    bundle: Bundle,
    preprocess_result: PreprocessResult,
    score_result: ScoreResult,
    split_result: SplitResult,
    *,
    benchmark_id: str,
    k_preselect: int | None = None,
    cv_folds: int = 5,
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
    checkpoint_dir: str | Path | None = None,
) -> BenchmarkSubsampleResult:
    """Run a deterministic information-proxy rank-and-select baseline."""

    benchmark_preprocess = preprocess_result.benchmarks[benchmark_id]
    if benchmark_preprocess.refused:
        return _skipped_benchmark_result(
            benchmark_id,
            method=METHOD_DETERMINISTIC_INFO,
            selection_rule="top_k_information_proxy",
            skipped_reason="benchmark_refused_in_preprocess",
            candidate_item_count=0,
            requested_k_preselect=k_preselect,
            effective_k_preselect=0,
            n_iter=1,
            cv_folds=cv_folds,
        )

    ranking_table = _build_information_proxy_ranking_table(
        benchmark_id=benchmark_id,
        filtered_items=benchmark_preprocess.filtered_items,
    )
    candidate_item_ids = ranking_table[ITEM_ID].astype("string").tolist()
    split_frame = split_result.per_benchmark_splits.get(benchmark_id)
    if split_frame is None or split_frame.empty:
        return _skipped_benchmark_result(
            benchmark_id,
            method=METHOD_DETERMINISTIC_INFO,
            selection_rule="top_k_information_proxy",
            skipped_reason="no_split_models_available",
            candidate_item_count=len(candidate_item_ids),
            requested_k_preselect=k_preselect,
            effective_k_preselect=0,
            n_iter=1,
            cv_folds=cv_folds,
        )

    score_lookup = (
        score_result.scores_full.loc[
            score_result.scores_full[BENCHMARK_ID] == benchmark_id,
            [MODEL_ID, SCORE_FULL],
        ]
        .dropna(subset=[SCORE_FULL])
        .astype({MODEL_ID: "string"})
        .set_index(MODEL_ID)[SCORE_FULL]
        .astype("Float64")
    )
    split_lookup = split_frame.set_index(MODEL_ID)[SPLIT]
    pool_model_ids = [
        model_id
        for model_id in split_lookup.index.astype("string").tolist()
        if split_lookup.loc[model_id] != "test" and model_id in score_lookup.index
    ]
    test_model_ids = [
        model_id
        for model_id in split_lookup.index.astype("string").tolist()
        if split_lookup.loc[model_id] == "test" and model_id in score_lookup.index
    ]
    if not pool_model_ids:
        return _skipped_benchmark_result(
            benchmark_id,
            method=METHOD_DETERMINISTIC_INFO,
            selection_rule="top_k_information_proxy",
            skipped_reason="no_train_val_models_available",
            candidate_item_count=len(candidate_item_ids),
            requested_k_preselect=k_preselect,
            effective_k_preselect=0,
            n_iter=1,
            cv_folds=cv_folds,
        )

    effective_k = _resolve_k_preselect(
        requested_k_preselect=k_preselect,
        candidate_item_count=len(candidate_item_ids),
        pool_model_count=len(pool_model_ids),
    )
    if effective_k is None:
        return _skipped_benchmark_result(
            benchmark_id,
            method=METHOD_DETERMINISTIC_INFO,
            selection_rule="top_k_information_proxy",
            skipped_reason="k_preselect_exceeds_candidate_items",
            candidate_item_count=len(candidate_item_ids),
            requested_k_preselect=k_preselect,
            effective_k_preselect=0,
            n_iter=1,
            cv_folds=cv_folds,
        )
    if len(pool_model_ids) < cv_folds:
        return _skipped_benchmark_result(
            benchmark_id,
            method=METHOD_DETERMINISTIC_INFO,
            selection_rule="top_k_information_proxy",
            skipped_reason="cv_folds_exceed_train_val_models",
            candidate_item_count=len(candidate_item_ids),
            requested_k_preselect=k_preselect,
            effective_k_preselect=effective_k,
            n_iter=1,
            cv_folds=cv_folds,
        )

    selected_item_ids = (
        ranking_table.loc[:, ITEM_ID].astype("string").head(effective_k).sort_values().tolist()
    )
    ranking_table["selected"] = ranking_table[ITEM_ID].isin(selected_item_ids)
    rank_lookup = {item_id: rank for rank, item_id in enumerate(selected_item_ids, start=1)}
    ranking_table["selection_rank"] = (
        ranking_table[ITEM_ID].astype("string").map(rank_lookup).astype("Float64").astype("Int64")
    )

    benchmark_matrix = build_benchmark_matrix(
        bundle.responses_long, benchmark_id=benchmark_id
    ).reindex(
        index=pd.Index(pool_model_ids + test_model_ids, dtype="string"),
        columns=candidate_item_ids,
    )
    reduced_scores = _build_reduced_score_frame(
        benchmark_matrix,
        selected_item_ids=selected_item_ids,
    )
    pool_frame = _eligible_model_frame(
        reduced_scores=reduced_scores,
        model_ids=pool_model_ids,
        score_lookup=score_lookup,
        min_item_coverage=bundle.config.min_item_coverage,
    )
    test_frame = _eligible_model_frame(
        reduced_scores=reduced_scores,
        model_ids=test_model_ids,
        score_lookup=score_lookup,
        min_item_coverage=bundle.config.min_item_coverage,
    )
    failed_reason = _iteration_failed_reason(pool_frame, cv_folds=cv_folds)
    if failed_reason is not None:
        progress_payload = _progress_payload(
            benchmark_id=benchmark_id,
            candidate_item_count=len(candidate_item_ids),
            requested_k_preselect=k_preselect,
            effective_k_preselect=effective_k,
            n_iter=1,
            completed_iterations=1,
            valid_iterations=0,
            failed_iterations=1,
            best_iteration=None,
        )
        _write_checkpoint(
            checkpoint_dir,
            benchmark_id=benchmark_id,
            cv_results=_build_cv_results_frame(
                [
                    _failed_iteration_row(
                        benchmark_id=benchmark_id,
                        iteration_id=0,
                        subset_seed=bundle.config.random_seed,
                        sampled_item_ids=selected_item_ids,
                        candidate_item_count=len(candidate_item_ids),
                        effective_k_preselect=effective_k,
                        valid_pool_count=len(pool_frame.index),
                        valid_test_count=len(test_frame.index),
                        failed_reason=failed_reason,
                    )
                ]
            ),
            best_iteration=None,
            progress_payload=progress_payload,
            effective_k_preselect=effective_k,
        )
        result = _skipped_benchmark_result(
            benchmark_id,
            method=METHOD_DETERMINISTIC_INFO,
            selection_rule="top_k_information_proxy",
            skipped_reason=failed_reason,
            candidate_item_count=len(candidate_item_ids),
            requested_k_preselect=k_preselect,
            effective_k_preselect=effective_k,
            n_iter=1,
            cv_folds=cv_folds,
        )
        result.ranking_table = ranking_table
        result.artifact_paths["progress_payload"] = progress_payload
        return result

    gam_result = cross_validate_gam(
        pool_frame[[REDUCED_SCORE]],
        pool_frame[SCORE_FULL],
        lam_grid=lam_grid,
        cv_folds=cv_folds,
        random_seed=bundle.config.random_seed,
        feature_names=[REDUCED_SCORE],
        target_name=SCORE_FULL,
        X_test=test_frame[[REDUCED_SCORE]] if not test_frame.empty else None,
        y_test=test_frame[SCORE_FULL] if not test_frame.empty else None,
    )
    best_cv_rows = gam_result.cv_results.loc[
        np.isclose(
            gam_result.cv_results["lam"].astype(float).to_numpy(),
            gam_result.best_lam,
        )
    ].copy()
    max_rmse_val = float(best_cv_rows["val_rmse"].max())
    mean_rmse_val = float(best_cv_rows["val_rmse"].mean())
    mean_rmse_test = (
        float(best_cv_rows["test_rmse"].dropna().mean())
        if best_cv_rows["test_rmse"].notna().any()
        else None
    )
    cv_results = _build_cv_results_frame(
        _single_iteration_cv_records(
            benchmark_id=benchmark_id,
            subset_seed=bundle.config.random_seed,
            selected_item_ids=selected_item_ids,
            candidate_item_count=len(candidate_item_ids),
            effective_k_preselect=effective_k,
            valid_pool_count=len(pool_frame.index),
            valid_test_count=len(test_frame.index),
            best_lam=float(gam_result.best_lam),
            max_rmse_val=max_rmse_val,
            mean_rmse_val=mean_rmse_val,
            mean_rmse_test=mean_rmse_test,
            cv_rows=best_cv_rows.to_dict(orient="records"),
        )
    )
    best_iteration = {
        ITERATION_ID: 0,
        SUBSET_SEED: bundle.config.random_seed,
        "sampled_item_ids": selected_item_ids,
        "best_lam": float(gam_result.best_lam),
        "max_rmse_val": max_rmse_val,
        "mean_rmse_val": mean_rmse_val,
        "mean_rmse_test": mean_rmse_test,
        "valid_pool_model_count": len(pool_frame.index),
        "valid_test_model_count": len(test_frame.index),
    }
    progress_payload = _progress_payload(
        benchmark_id=benchmark_id,
        candidate_item_count=len(candidate_item_ids),
        requested_k_preselect=k_preselect,
        effective_k_preselect=effective_k,
        n_iter=1,
        completed_iterations=1,
        valid_iterations=1,
        failed_iterations=0,
        best_iteration=_best_iteration_summary([best_iteration]),
    )
    _write_checkpoint(
        checkpoint_dir,
        benchmark_id=benchmark_id,
        cv_results=cv_results,
        best_iteration=best_iteration,
        progress_payload=progress_payload,
        effective_k_preselect=effective_k,
    )
    subsample_report = {
        "benchmark_id": benchmark_id,
        "method": METHOD_DETERMINISTIC_INFO,
        "selection_rule": "top_k_information_proxy",
        "skipped": False,
        "skipped_reason": None,
        "warnings": [],
        "parameters": {
            "requested_k_preselect": k_preselect,
            "effective_k_preselect": effective_k,
            "n_iter": 1,
            "cv_folds": cv_folds,
            "checkpoint_interval": 1,
            "lam_grid": list(lam_grid),
            "min_item_coverage": bundle.config.min_item_coverage,
        },
        "counts": {
            "candidate_item_count": len(candidate_item_ids),
            "pool_model_count": len(pool_model_ids),
            "test_model_count": len(test_model_ids),
            "completed_iterations": 1,
            "valid_iterations": 1,
            "failed_iterations": 0,
            "cv_rows": int(len(cv_results.index)),
        },
        "best_iteration": _best_iteration_summary([best_iteration]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return BenchmarkSubsampleResult(
        benchmark_id=benchmark_id,
        preselect_items=_build_preselect_items_frame(
            benchmark_id=benchmark_id,
            selected_item_ids=selected_item_ids,
            selected_iteration=0,
            subset_seed=bundle.config.random_seed,
            k_preselect=effective_k,
            max_rmse_val=max_rmse_val,
            mean_rmse_val=mean_rmse_val,
        ),
        cv_results=cv_results,
        subsample_report=subsample_report,
        ranking_table=ranking_table,
        artifact_paths={"progress_payload": progress_payload},
    )


def _resolve_method(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_")
    if normalized in {METHOD_RANDOM_CV, "random"}:
        return METHOD_RANDOM_CV
    if normalized in {METHOD_DETERMINISTIC_INFO, "deterministic", "info_rank"}:
        return METHOD_DETERMINISTIC_INFO
    raise ValueError(f"unsupported subsample method: {method}")


def _build_information_proxy_ranking_table(
    *,
    benchmark_id: str,
    filtered_items: pd.DataFrame,
) -> pd.DataFrame:
    if filtered_items.empty:
        return _empty_ranking_table_frame()
    ranking = filtered_items.loc[
        :,
        [ITEM_ID, "point_biserial", "mean", "sd", "item_coverage"],
    ].copy()
    point_biserial = ranking["point_biserial"].astype("Float64").fillna(0.0).abs()
    mean_score = ranking["mean"].astype("Float64").fillna(0.0)
    coverage = ranking["item_coverage"].astype("Float64").fillna(0.0)
    ranking[INFO_PROXY_SCORE] = (
        (point_biserial.astype(float) ** 2)
        * mean_score.astype(float)
        * (1.0 - mean_score.astype(float))
        * coverage.astype(float)
    )
    ranking.insert(0, BENCHMARK_ID, pd.Series([benchmark_id] * len(ranking.index), dtype="string"))
    ranking["selected"] = False
    ranking["selection_rank"] = pd.Series([pd.NA] * len(ranking.index), dtype="Int64")
    return (
        ranking.sort_values([INFO_PROXY_SCORE, ITEM_ID], ascending=[False, True])
        .reset_index(drop=True)
        .astype(
            {
                BENCHMARK_ID: "string",
                ITEM_ID: "string",
                "point_biserial": "Float64",
                "mean": "Float64",
                "sd": "Float64",
                "item_coverage": "Float64",
                INFO_PROXY_SCORE: "Float64",
                "selected": "boolean",
                "selection_rank": "Int64",
            }
        )
    )


def _single_iteration_cv_records(
    *,
    benchmark_id: str,
    subset_seed: int,
    selected_item_ids: Sequence[str],
    candidate_item_count: int,
    effective_k_preselect: int,
    valid_pool_count: int,
    valid_test_count: int,
    best_lam: float,
    max_rmse_val: float,
    mean_rmse_val: float,
    mean_rmse_test: float | None,
    cv_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fold_row in cv_rows:
        records.append(
            {
                BENCHMARK_ID: benchmark_id,
                ITERATION_ID: 0,
                SUBSET_SEED: subset_seed,
                STATUS: "ok",
                FAILED_REASON: pd.NA,
                SAMPLED_ITEM_IDS: json.dumps(list(selected_item_ids)),
                "candidate_item_count": candidate_item_count,
                "k_preselect": effective_k_preselect,
                "valid_pool_model_count": valid_pool_count,
                "valid_test_model_count": valid_test_count,
                BEST_LAM: best_lam,
                VAL_RMSE_MAX: max_rmse_val,
                VAL_RMSE_MEAN: mean_rmse_val,
                TEST_RMSE_MEAN: mean_rmse_test,
                "fold_id": fold_row["fold_id"],
                "train_rows": fold_row["train_rows"],
                "val_rows": fold_row["val_rows"],
                "val_rmse": fold_row["val_rmse"],
                "baseline_val_rmse": fold_row["baseline_val_rmse"],
                "test_rmse": fold_row["test_rmse"],
            }
        )
    return records


def _failed_iteration_row(
    *,
    benchmark_id: str,
    iteration_id: int,
    subset_seed: int,
    sampled_item_ids: Sequence[str],
    candidate_item_count: int,
    effective_k_preselect: int,
    valid_pool_count: int,
    valid_test_count: int,
    failed_reason: str,
) -> dict[str, Any]:
    return {
        BENCHMARK_ID: benchmark_id,
        ITERATION_ID: iteration_id,
        SUBSET_SEED: subset_seed,
        STATUS: "failed",
        FAILED_REASON: failed_reason,
        SAMPLED_ITEM_IDS: json.dumps(list(sampled_item_ids)),
        "candidate_item_count": candidate_item_count,
        "k_preselect": effective_k_preselect,
        "valid_pool_model_count": valid_pool_count,
        "valid_test_model_count": valid_test_count,
        BEST_LAM: pd.NA,
        VAL_RMSE_MAX: pd.NA,
        VAL_RMSE_MEAN: pd.NA,
        TEST_RMSE_MEAN: pd.NA,
        "fold_id": pd.NA,
        "train_rows": pd.NA,
        "val_rows": pd.NA,
        "val_rmse": pd.NA,
        "baseline_val_rmse": pd.NA,
        "test_rmse": pd.NA,
    }


def _build_cv_results_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            {
                BENCHMARK_ID: pd.Series(dtype="string"),
                ITERATION_ID: pd.Series(dtype="Int64"),
                SUBSET_SEED: pd.Series(dtype="Int64"),
                STATUS: pd.Series(dtype="string"),
                FAILED_REASON: pd.Series(dtype="string"),
                SAMPLED_ITEM_IDS: pd.Series(dtype="string"),
                "candidate_item_count": pd.Series(dtype="Int64"),
                "k_preselect": pd.Series(dtype="Int64"),
                "valid_pool_model_count": pd.Series(dtype="Int64"),
                "valid_test_model_count": pd.Series(dtype="Int64"),
                BEST_LAM: pd.Series(dtype="Float64"),
                VAL_RMSE_MAX: pd.Series(dtype="Float64"),
                VAL_RMSE_MEAN: pd.Series(dtype="Float64"),
                TEST_RMSE_MEAN: pd.Series(dtype="Float64"),
                "fold_id": pd.Series(dtype="Int64"),
                "train_rows": pd.Series(dtype="Int64"),
                "val_rows": pd.Series(dtype="Int64"),
                "val_rmse": pd.Series(dtype="Float64"),
                "baseline_val_rmse": pd.Series(dtype="Float64"),
                "test_rmse": pd.Series(dtype="Float64"),
            },
        )
    frame = pd.DataFrame(records)
    return (
        frame.astype(
            {
                BENCHMARK_ID: "string",
                ITERATION_ID: "Int64",
                SUBSET_SEED: "Int64",
                STATUS: "string",
                FAILED_REASON: "string",
                SAMPLED_ITEM_IDS: "string",
                "candidate_item_count": "Int64",
                "k_preselect": "Int64",
                "valid_pool_model_count": "Int64",
                "valid_test_model_count": "Int64",
                BEST_LAM: "Float64",
                VAL_RMSE_MAX: "Float64",
                VAL_RMSE_MEAN: "Float64",
                TEST_RMSE_MEAN: "Float64",
                "fold_id": "Int64",
                "train_rows": "Int64",
                "val_rows": "Int64",
                "val_rmse": "Float64",
                "baseline_val_rmse": "Float64",
                "test_rmse": "Float64",
            }
        )
        .sort_values([ITERATION_ID, "fold_id"], na_position="last")
        .reset_index(drop=True)
    )


def _select_best_iteration(valid_iterations: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not valid_iterations:
        return None
    return min(
        valid_iterations,
        key=lambda row: (row["max_rmse_val"], row["mean_rmse_val"], row[ITERATION_ID]),
    )


def _best_iteration_summary(valid_iterations: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    best_iteration = _select_best_iteration(valid_iterations)
    if best_iteration is None:
        return None
    return {
        ITERATION_ID: int(best_iteration[ITERATION_ID]),
        SUBSET_SEED: int(best_iteration[SUBSET_SEED]),
        BEST_LAM: float(best_iteration["best_lam"]),
        VAL_RMSE_MAX: float(best_iteration["max_rmse_val"]),
        VAL_RMSE_MEAN: float(best_iteration["mean_rmse_val"]),
        TEST_RMSE_MEAN: (
            float(best_iteration["mean_rmse_test"])
            if best_iteration["mean_rmse_test"] is not None
            else None
        ),
        "valid_pool_model_count": int(best_iteration["valid_pool_model_count"]),
        "valid_test_model_count": int(best_iteration["valid_test_model_count"]),
        "sampled_item_ids": list(best_iteration["sampled_item_ids"]),
    }


def _build_preselect_items_frame(
    *,
    benchmark_id: str,
    selected_item_ids: Sequence[str],
    selected_iteration: int,
    subset_seed: int,
    k_preselect: int,
    max_rmse_val: float,
    mean_rmse_val: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series([benchmark_id] * len(selected_item_ids), dtype="string"),
            ITEM_ID: pd.Series(list(selected_item_ids), dtype="string"),
            "selection_rank": pd.Series(range(1, len(selected_item_ids) + 1), dtype="Int64"),
            ITERATION_ID: pd.Series([selected_iteration] * len(selected_item_ids), dtype="Int64"),
            SUBSET_SEED: pd.Series([subset_seed] * len(selected_item_ids), dtype="Int64"),
            "k_preselect": pd.Series([k_preselect] * len(selected_item_ids), dtype="Int64"),
            VAL_RMSE_MAX: pd.Series([max_rmse_val] * len(selected_item_ids), dtype="Float64"),
            VAL_RMSE_MEAN: pd.Series([mean_rmse_val] * len(selected_item_ids), dtype="Float64"),
        }
    )


def _empty_preselect_items_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            ITEM_ID: pd.Series(dtype="string"),
            "selection_rank": pd.Series(dtype="Int64"),
            ITERATION_ID: pd.Series(dtype="Int64"),
            SUBSET_SEED: pd.Series(dtype="Int64"),
            "k_preselect": pd.Series(dtype="Int64"),
            VAL_RMSE_MAX: pd.Series(dtype="Float64"),
            VAL_RMSE_MEAN: pd.Series(dtype="Float64"),
        }
    )


def _empty_ranking_table_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            ITEM_ID: pd.Series(dtype="string"),
            "point_biserial": pd.Series(dtype="Float64"),
            "mean": pd.Series(dtype="Float64"),
            "sd": pd.Series(dtype="Float64"),
            "item_coverage": pd.Series(dtype="Float64"),
            INFO_PROXY_SCORE: pd.Series(dtype="Float64"),
            "selected": pd.Series(dtype="boolean"),
            "selection_rank": pd.Series(dtype="Int64"),
        }
    )


def _progress_payload(
    *,
    benchmark_id: str,
    candidate_item_count: int,
    requested_k_preselect: int | None,
    effective_k_preselect: int,
    n_iter: int,
    completed_iterations: int,
    valid_iterations: int,
    failed_iterations: int,
    best_iteration: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "candidate_item_count": candidate_item_count,
        "requested_k_preselect": requested_k_preselect,
        "effective_k_preselect": effective_k_preselect,
        "n_iter": n_iter,
        "completed_iterations": completed_iterations,
        "valid_iterations": valid_iterations,
        "failed_iterations": failed_iterations,
        "best_iteration": best_iteration,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _resolve_k_preselect(
    *,
    requested_k_preselect: int | None,
    candidate_item_count: int,
    pool_model_count: int,
) -> int | None:
    if candidate_item_count < 1:
        return None
    if requested_k_preselect is not None:
        if requested_k_preselect < 1:
            raise ValueError("k_preselect must be at least 1 when provided")
        if requested_k_preselect > candidate_item_count:
            return None
        return requested_k_preselect
    default_cap = max(1, pool_model_count // 4)
    return min(candidate_item_count, default_cap)


def _sample_item_subset(
    candidate_item_ids: Sequence[str],
    *,
    subset_size: int,
    seed: int,
) -> list[str]:
    rng = np.random.default_rng(seed)
    selection = rng.choice(
        np.asarray(candidate_item_ids, dtype=object),
        size=subset_size,
        replace=False,
    )
    return sorted(str(item_id) for item_id in selection.tolist())


def _skipped_benchmark_result(
    benchmark_id: str,
    *,
    method: str,
    selection_rule: str,
    skipped_reason: str,
    candidate_item_count: int,
    requested_k_preselect: int | None,
    effective_k_preselect: int,
    n_iter: int,
    cv_folds: int,
) -> BenchmarkSubsampleResult:
    return BenchmarkSubsampleResult(
        benchmark_id=benchmark_id,
        preselect_items=_empty_preselect_items_frame(),
        cv_results=_build_cv_results_frame([]),
        ranking_table=_empty_ranking_table_frame(),
        subsample_report={
            "benchmark_id": benchmark_id,
            "method": method,
            "selection_rule": selection_rule,
            "skipped": True,
            "skipped_reason": skipped_reason,
            "warnings": [],
            "parameters": {
                "requested_k_preselect": requested_k_preselect,
                "effective_k_preselect": effective_k_preselect,
                "n_iter": n_iter,
                "cv_folds": cv_folds,
            },
            "counts": {
                "candidate_item_count": candidate_item_count,
                "pool_model_count": 0,
                "test_model_count": 0,
                "completed_iterations": 0,
                "valid_iterations": 0,
                "failed_iterations": 0,
                "cv_rows": 0,
            },
            "best_iteration": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        artifact_paths={
            "progress_payload": _progress_payload(
                benchmark_id=benchmark_id,
                candidate_item_count=candidate_item_count,
                requested_k_preselect=requested_k_preselect,
                effective_k_preselect=effective_k_preselect,
                n_iter=n_iter,
                completed_iterations=0,
                valid_iterations=0,
                failed_iterations=0,
                best_iteration=None,
            )
        },
    )


def _write_subsample_artifacts(
    result: SubsampleResult,
    *,
    run_root: Path,
) -> dict[str, Any]:
    stage_dir = run_root / "artifacts" / "04_subsample"
    per_benchmark_paths: dict[str, dict[str, Path]] = {}
    for benchmark_id, benchmark_result in sorted(result.benchmarks.items()):
        benchmark_dir = stage_dir / "per_benchmark" / benchmark_id
        progress_path = write_json(
            benchmark_result.artifact_paths.get("progress_payload", {}),
            benchmark_dir / "progress.json",
        )
        benchmark_paths = {
            "preselect_items": write_parquet(
                benchmark_result.preselect_items,
                benchmark_dir / "preselect_items.parquet",
            ),
            "cv_results": write_parquet(
                benchmark_result.cv_results,
                benchmark_dir / "cv_results.parquet",
            ),
            "subsample_report": write_json(
                benchmark_result.subsample_report,
                benchmark_dir / "subsample_report.json",
            ),
            "progress": progress_path,
        }
        if not benchmark_result.ranking_table.empty:
            benchmark_paths["ranking_table"] = write_parquet(
                benchmark_result.ranking_table,
                benchmark_dir / "ranking_table.parquet",
            )
        per_benchmark_paths[benchmark_id] = benchmark_paths
    return {"per_benchmark": per_benchmark_paths}


def _write_checkpoint(
    checkpoint_dir: str | Path | None,
    *,
    benchmark_id: str,
    cv_results: pd.DataFrame,
    best_iteration: dict[str, Any] | None,
    progress_payload: dict[str, Any],
    effective_k_preselect: int,
) -> None:
    if checkpoint_dir is None:
        return
    resolved_dir = Path(checkpoint_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(cv_results, resolved_dir / "cv_results.parquet")
    if best_iteration is None:
        preselect_items = _empty_preselect_items_frame()
    else:
        preselect_items = _build_preselect_items_frame(
            benchmark_id=benchmark_id,
            selected_item_ids=best_iteration["sampled_item_ids"],
            selected_iteration=best_iteration[ITERATION_ID],
            subset_seed=best_iteration[SUBSET_SEED],
            k_preselect=effective_k_preselect,
            max_rmse_val=best_iteration["max_rmse_val"],
            mean_rmse_val=best_iteration["mean_rmse_val"],
        )
    write_parquet(preselect_items, resolved_dir / "preselect_items.parquet")
    write_json(progress_payload, resolved_dir / "progress.json")


def _benchmark_seed_offset(benchmark_id: str) -> int:
    return sum(ord(character) for character in benchmark_id)


def _resolve_run_root(
    bundle: Bundle,
    *,
    out_dir: str | Path | None,
    run_id: str | None,
) -> tuple[Path | None, Path | None]:
    if out_dir is not None:
        resolved_run_id = run_id or bundle.run_id or _default_run_id()
        run_root = Path(out_dir) / resolved_run_id
        return run_root, run_root / "manifest.json"
    if bundle.manifest_path is not None:
        return bundle.manifest_path.parent, bundle.manifest_path
    return None, None


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
