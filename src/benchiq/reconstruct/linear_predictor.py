"""Stage-08 benchmark-specific linear predictors and reduced subscores."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

from benchiq.io.load import Bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.logging import update_manifest
from benchiq.preprocess.scores import SCORE_FULL, ScoreResult
from benchiq.preprocess.stats import build_benchmark_matrix
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID, SPLIT
from benchiq.split.splitters import SplitResult

if TYPE_CHECKING:
    from benchiq.select.information_filter import SelectResult

REDUCED_SUBSCORE = "sub_b"
REDUCED_ITEM_COVERAGE = "reduced_item_coverage"
OBSERVED_ITEM_COUNT = "observed_item_count"
SELECTED_ITEM_COUNT = "selected_item_count"
LINEAR_PREDICTION = "lin_b"
LINEAR_PREDICTION_MISSING_REASON = "linear_prediction_missing_reason"
TRAINING_ELIGIBLE = "training_eligible"
COEFFICIENT = "coefficient"
MODEL_KIND = "model_kind"

DEFAULT_RIDGE_ALPHA_GRID = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
CONDITION_NUMBER_THRESHOLD = 1e8


@dataclass(slots=True)
class BenchmarkLinearPredictorResult:
    """Per-benchmark stage-08 linear-predictor outputs."""

    benchmark_id: str
    model_outputs: pd.DataFrame
    coefficients: pd.DataFrame
    linear_predictor_report: dict[str, Any]
    artifact_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(slots=True)
class LinearPredictorResult:
    """Stage-08 linear-predictor outputs."""

    benchmarks: dict[str, BenchmarkLinearPredictorResult]
    feature_report: dict[str, Any]
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None


def fit_linear_predictor_bundle(
    bundle: Bundle,
    score_result: ScoreResult,
    split_result: SplitResult,
    select_result: SelectResult,
    *,
    ridge_alpha_grid: Sequence[float] = DEFAULT_RIDGE_ALPHA_GRID,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> LinearPredictorResult:
    """Fit benchmark-specific no-intercept linear predictors and write stage-08 artifacts."""

    benchmark_results: dict[str, BenchmarkLinearPredictorResult] = {}
    warnings: list[dict[str, Any]] = []
    for benchmark_id in sorted(select_result.benchmarks):
        benchmark_result = fit_linear_predictor_benchmark(
            bundle,
            score_result,
            split_result,
            select_result,
            benchmark_id=benchmark_id,
            ridge_alpha_grid=ridge_alpha_grid,
        )
        benchmark_results[benchmark_id] = benchmark_result
        for warning in benchmark_result.linear_predictor_report["warnings"]:
            warnings.append({"benchmark_id": benchmark_id, **warning})

    feature_report = _build_feature_report(
        benchmark_results=benchmark_results,
        warnings=warnings,
        ridge_alpha_grid=ridge_alpha_grid,
    )
    result = LinearPredictorResult(
        benchmarks=benchmark_results,
        feature_report=feature_report,
    )

    run_root, manifest_path = _resolve_run_root(bundle, out_dir=out_dir, run_id=run_id)
    if run_root is not None:
        artifact_paths = _write_linear_predictor_artifacts(result, run_root=run_root)
        result.artifact_paths = artifact_paths
        result.manifest_path = manifest_path
        if manifest_path is not None:
            update_manifest(
                manifest_path,
                {
                    "artifacts": {
                        "08_features": {
                            "feature_report": str(artifact_paths["feature_report"]),
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


def fit_linear_predictor_benchmark(
    bundle: Bundle,
    score_result: ScoreResult,
    split_result: SplitResult,
    select_result: SelectResult,
    *,
    benchmark_id: str,
    ridge_alpha_grid: Sequence[float] = DEFAULT_RIDGE_ALPHA_GRID,
) -> BenchmarkLinearPredictorResult:
    """Fit one benchmark-specific linear predictor from reduced item responses."""

    selection = select_result.benchmarks[benchmark_id]
    if selection.selection_report["skipped"] or selection.subset_final.empty:
        return _skipped_benchmark_result(
            benchmark_id,
            skipped_reason=selection.selection_report["skipped_reason"]
            or "no_selected_items_available",
            ridge_alpha_grid=ridge_alpha_grid,
        )

    split_frame = split_result.per_benchmark_splits.get(benchmark_id)
    if split_frame is None or split_frame.empty:
        return _skipped_benchmark_result(
            benchmark_id,
            skipped_reason="no_split_models_available",
            ridge_alpha_grid=ridge_alpha_grid,
        )

    selected_item_ids = (
        selection.subset_final[ITEM_ID]
        .dropna()
        .astype("string")
        .sort_values()
        .reset_index(drop=True)
        .tolist()
    )
    if not selected_item_ids:
        return _skipped_benchmark_result(
            benchmark_id,
            skipped_reason="no_selected_items_available",
            ridge_alpha_grid=ridge_alpha_grid,
        )

    benchmark_matrix = build_benchmark_matrix(
        bundle.responses_long,
        benchmark_id=benchmark_id,
    ).reindex(
        index=split_frame[MODEL_ID].dropna().astype("string").tolist(),
        columns=selected_item_ids,
    )
    score_lookup = (
        score_result.scores_full.loc[
            score_result.scores_full[BENCHMARK_ID] == benchmark_id,
            [MODEL_ID, SCORE_FULL],
        ]
        .dropna(subset=[SCORE_FULL])
        .astype({MODEL_ID: "string"})
        .set_index(MODEL_ID)[SCORE_FULL]
        .astype(float)
    )

    observed_item_count = benchmark_matrix.notna().sum(axis=1).astype("Int64")
    selected_item_count = len(selected_item_ids)
    reduced_item_coverage = (
        observed_item_count.astype(float) / float(selected_item_count)
        if selected_item_count > 0
        else pd.Series(0.0, index=benchmark_matrix.index)
    )
    reduced_subscore = (benchmark_matrix.mean(axis=1, skipna=True) * 100.0).astype("Float64")

    train_model_ids = (
        split_frame.loc[split_frame[SPLIT] == "train", MODEL_ID].dropna().astype("string").tolist()
    )
    training_eligible_ids = [
        model_id
        for model_id in train_model_ids
        if model_id in score_lookup.index
        and int(observed_item_count.loc[model_id]) == selected_item_count
    ]
    if not training_eligible_ids:
        return _skipped_benchmark_result(
            benchmark_id,
            skipped_reason="no_complete_train_rows_for_linear_predictor",
            selected_item_count=selected_item_count,
            ridge_alpha_grid=ridge_alpha_grid,
            model_outputs=_build_model_outputs_frame(
                benchmark_id=benchmark_id,
                split_frame=split_frame,
                score_lookup=score_lookup,
                reduced_subscore=reduced_subscore,
                reduced_item_coverage=reduced_item_coverage,
                observed_item_count=observed_item_count,
                selected_item_count=selected_item_count,
                linear_prediction=pd.Series(pd.NA, index=benchmark_matrix.index, dtype="Float64"),
                missing_reason_by_model={
                    model_id: "insufficient_reduced_item_coverage"
                    for model_id in benchmark_matrix.index.astype("string")
                },
                training_eligible_ids=set(),
            ),
        )

    X_train = benchmark_matrix.loc[training_eligible_ids].to_numpy(dtype=float)
    y_train = score_lookup.loc[training_eligible_ids].to_numpy(dtype=float)
    fit_result = fit_no_intercept_linear_predictor(
        X_train,
        y_train,
        feature_names=selected_item_ids,
        ridge_alpha_grid=ridge_alpha_grid,
    )

    complete_coverage_ids = [
        model_id
        for model_id in benchmark_matrix.index.astype("string").tolist()
        if int(observed_item_count.loc[model_id]) == selected_item_count
    ]
    prediction_series = pd.Series(pd.NA, index=benchmark_matrix.index, dtype="Float64")
    if complete_coverage_ids:
        prediction_series.loc[complete_coverage_ids] = (
            benchmark_matrix.loc[complete_coverage_ids].to_numpy(dtype=float)
            @ fit_result["coefficients"]
        )

    missing_reason_by_model: dict[str, str | None] = {}
    for model_id in benchmark_matrix.index.astype("string").tolist():
        if model_id in complete_coverage_ids:
            missing_reason_by_model[model_id] = None
        else:
            missing_reason_by_model[model_id] = "insufficient_reduced_item_coverage"

    model_outputs = _build_model_outputs_frame(
        benchmark_id=benchmark_id,
        split_frame=split_frame,
        score_lookup=score_lookup,
        reduced_subscore=reduced_subscore,
        reduced_item_coverage=reduced_item_coverage,
        observed_item_count=observed_item_count,
        selected_item_count=selected_item_count,
        linear_prediction=prediction_series,
        missing_reason_by_model=missing_reason_by_model,
        training_eligible_ids=set(training_eligible_ids),
    )
    coefficients = pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series([benchmark_id] * len(selected_item_ids), dtype="string"),
            ITEM_ID: pd.Series(selected_item_ids, dtype="string"),
            COEFFICIENT: pd.Series(fit_result["coefficients"], dtype="Float64"),
        }
    )
    linear_predictor_report = _build_benchmark_report(
        benchmark_id=benchmark_id,
        model_outputs=model_outputs,
        coefficients=coefficients,
        fit_result=fit_result,
        selected_item_count=selected_item_count,
        training_eligible_count=len(training_eligible_ids),
        ridge_alpha_grid=ridge_alpha_grid,
    )
    return BenchmarkLinearPredictorResult(
        benchmark_id=benchmark_id,
        model_outputs=model_outputs,
        coefficients=coefficients,
        linear_predictor_report=linear_predictor_report,
    )


def fit_no_intercept_linear_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_names: Sequence[str],
    ridge_alpha_grid: Sequence[float] = DEFAULT_RIDGE_ALPHA_GRID,
) -> dict[str, Any]:
    """Fit OLS without intercept and fall back deterministically to ridge when needed."""

    if X_train.ndim != 2:
        raise ValueError("X_train must be a two-dimensional matrix")
    if len(y_train.shape) != 1:
        y_train = y_train.reshape(-1)
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same row count")
    if X_train.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the number of columns in X_train")
    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        raise ValueError("training matrix must be non-empty")

    rank = int(np.linalg.matrix_rank(X_train))
    xtx = X_train.T @ X_train
    condition_number = float(np.linalg.cond(xtx)) if xtx.size > 0 else float("inf")
    use_ridge = (
        rank < X_train.shape[1]
        or not np.isfinite(condition_number)
        or (condition_number > CONDITION_NUMBER_THRESHOLD)
    )
    fallback_reason: str | None = None
    if rank < X_train.shape[1]:
        fallback_reason = "rank_deficient_design_matrix"
        use_ridge = True
    elif not np.isfinite(condition_number) or condition_number > CONDITION_NUMBER_THRESHOLD:
        fallback_reason = "ill_conditioned_design_matrix"
        use_ridge = True

    ridge_candidates: list[dict[str, float]] = []
    if use_ridge:
        candidate_results: list[tuple[float, np.ndarray, float]] = []
        for alpha in ridge_alpha_grid:
            coefficients = _fit_ridge_no_intercept(X_train, y_train, alpha=float(alpha))
            predictions = X_train @ coefficients
            rmse = float(np.sqrt(np.mean((predictions - y_train) ** 2)))
            ridge_candidates.append({"alpha": float(alpha), "train_rmse": rmse})
            candidate_results.append((float(alpha), coefficients, rmse))
        chosen_alpha, coefficients, train_rmse = min(
            candidate_results,
            key=lambda item: (item[2], item[0]),
        )
        model_kind = "ridge_no_intercept"
    else:
        coefficients, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        predictions = X_train @ coefficients
        train_rmse = float(np.sqrt(np.mean((predictions - y_train) ** 2)))
        chosen_alpha = 0.0
        model_kind = "ols_no_intercept"

    return {
        "coefficients": np.asarray(coefficients, dtype=float).reshape(-1),
        "model_kind": model_kind,
        "train_rmse": train_rmse,
        "rank": rank,
        "condition_number": condition_number,
        "fallback_used": use_ridge,
        "fallback_reason": fallback_reason,
        "chosen_alpha": float(chosen_alpha),
        "ridge_candidates": ridge_candidates,
        "train_row_count": int(X_train.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "feature_names": list(feature_names),
    }


def _fit_ridge_no_intercept(
    X_train: np.ndarray, y_train: np.ndarray, *, alpha: float
) -> np.ndarray:
    n_features = X_train.shape[1]
    ridge_matrix = X_train.T @ X_train + float(alpha) * np.eye(n_features)
    ridge_target = X_train.T @ y_train
    return np.linalg.solve(ridge_matrix, ridge_target)


def _build_model_outputs_frame(
    *,
    benchmark_id: str,
    split_frame: pd.DataFrame,
    score_lookup: pd.Series,
    reduced_subscore: pd.Series,
    reduced_item_coverage: pd.Series,
    observed_item_count: pd.Series,
    selected_item_count: int,
    linear_prediction: pd.Series,
    missing_reason_by_model: dict[str, str | None],
    training_eligible_ids: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_lookup = split_frame.set_index(MODEL_ID)[SPLIT].astype("string")
    for model_id in split_frame[MODEL_ID].dropna().astype("string").tolist():
        rows.append(
            {
                BENCHMARK_ID: benchmark_id,
                MODEL_ID: model_id,
                SPLIT: split_lookup.loc[model_id],
                SCORE_FULL: score_lookup.get(model_id, pd.NA),
                REDUCED_SUBSCORE: reduced_subscore.get(model_id, pd.NA),
                REDUCED_ITEM_COVERAGE: reduced_item_coverage.get(model_id, pd.NA),
                OBSERVED_ITEM_COUNT: observed_item_count.get(model_id, pd.NA),
                SELECTED_ITEM_COUNT: selected_item_count,
                LINEAR_PREDICTION: linear_prediction.get(model_id, pd.NA),
                LINEAR_PREDICTION_MISSING_REASON: missing_reason_by_model.get(model_id),
                TRAINING_ELIGIBLE: model_id in training_eligible_ids,
            }
        )

    frame = pd.DataFrame.from_records(rows)
    return frame.astype(
        {
            BENCHMARK_ID: "string",
            MODEL_ID: "string",
            SPLIT: "string",
            SCORE_FULL: "Float64",
            REDUCED_SUBSCORE: "Float64",
            REDUCED_ITEM_COVERAGE: "Float64",
            OBSERVED_ITEM_COUNT: "Int64",
            SELECTED_ITEM_COUNT: "Int64",
            LINEAR_PREDICTION: "Float64",
            LINEAR_PREDICTION_MISSING_REASON: "string",
            TRAINING_ELIGIBLE: bool,
        }
    )


def _build_benchmark_report(
    *,
    benchmark_id: str,
    model_outputs: pd.DataFrame,
    coefficients: pd.DataFrame,
    fit_result: dict[str, Any],
    selected_item_count: int,
    training_eligible_count: int,
    ridge_alpha_grid: Sequence[float],
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    if fit_result["fallback_used"]:
        warnings.append(
            {
                "code": "ridge_fallback_used",
                "message": (
                    f"{fit_result['fallback_reason']} triggered deterministic ridge fallback "
                    f"with alpha={fit_result['chosen_alpha']}."
                ),
                "severity": "warning",
            }
        )
    incomplete_count = int(model_outputs[LINEAR_PREDICTION].isna().sum())
    if incomplete_count > 0:
        warnings.append(
            {
                "code": "incomplete_reduced_item_coverage",
                "message": (
                    f"{incomplete_count} models were not scored by the linear predictor because "
                    "their reduced-item coverage was incomplete."
                ),
                "severity": "warning",
            }
        )

    return {
        "benchmark_id": benchmark_id,
        "skipped": False,
        "skipped_reason": None,
        "warnings": warnings,
        "parameters": {
            "selected_item_count": selected_item_count,
            "ridge_alpha_grid": list(map(float, ridge_alpha_grid)),
        },
        "training_diagnostics": {
            MODEL_KIND: fit_result["model_kind"],
            "train_rmse": fit_result["train_rmse"],
            "rank": fit_result["rank"],
            "condition_number": fit_result["condition_number"],
            "fallback_used": fit_result["fallback_used"],
            "fallback_reason": fit_result["fallback_reason"],
            "chosen_alpha": fit_result["chosen_alpha"],
            "ridge_candidates": fit_result["ridge_candidates"],
            "train_row_count": fit_result["train_row_count"],
            "feature_count": fit_result["feature_count"],
        },
        "counts": {
            "model_count": int(len(model_outputs.index)),
            "training_eligible_count": training_eligible_count,
            "predicted_model_count": int(model_outputs[LINEAR_PREDICTION].notna().sum()),
            "incomplete_coverage_count": int(model_outputs[LINEAR_PREDICTION].isna().sum()),
            "coefficient_count": int(len(coefficients.index)),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_feature_report(
    *,
    benchmark_results: dict[str, BenchmarkLinearPredictorResult],
    warnings: list[dict[str, Any]],
    ridge_alpha_grid: Sequence[float],
) -> dict[str, Any]:
    return {
        "warnings": warnings,
        "parameters": {
            "ridge_alpha_grid": list(map(float, ridge_alpha_grid)),
        },
        "counts": {
            "benchmark_count": len(benchmark_results),
            "skipped_benchmark_count": int(
                sum(
                    result.linear_predictor_report["skipped"]
                    for result in benchmark_results.values()
                )
            ),
        },
        "benchmarks": {
            benchmark_id: result.linear_predictor_report
            for benchmark_id, result in sorted(benchmark_results.items())
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _skipped_benchmark_result(
    benchmark_id: str,
    *,
    skipped_reason: str,
    selected_item_count: int = 0,
    ridge_alpha_grid: Sequence[float] = DEFAULT_RIDGE_ALPHA_GRID,
    model_outputs: pd.DataFrame | None = None,
) -> BenchmarkLinearPredictorResult:
    if model_outputs is None:
        model_outputs = _empty_model_outputs_frame()
    return BenchmarkLinearPredictorResult(
        benchmark_id=benchmark_id,
        model_outputs=model_outputs,
        coefficients=_empty_coefficients_frame(),
        linear_predictor_report={
            "benchmark_id": benchmark_id,
            "skipped": True,
            "skipped_reason": skipped_reason,
            "warnings": [],
            "parameters": {
                "selected_item_count": selected_item_count,
                "ridge_alpha_grid": list(map(float, ridge_alpha_grid)),
            },
            "training_diagnostics": {
                MODEL_KIND: None,
                "train_rmse": None,
                "rank": None,
                "condition_number": None,
                "fallback_used": False,
                "fallback_reason": None,
                "chosen_alpha": None,
                "ridge_candidates": [],
                "train_row_count": 0,
                "feature_count": 0,
            },
            "counts": {
                "model_count": int(len(model_outputs.index)),
                "training_eligible_count": 0,
                "predicted_model_count": int(model_outputs[LINEAR_PREDICTION].notna().sum())
                if not model_outputs.empty
                else 0,
                "incomplete_coverage_count": int(model_outputs[LINEAR_PREDICTION].isna().sum())
                if not model_outputs.empty
                else 0,
                "coefficient_count": 0,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _write_linear_predictor_artifacts(
    result: LinearPredictorResult,
    *,
    run_root: Path,
) -> dict[str, Any]:
    stage_dir = run_root / "artifacts" / "08_features"
    per_benchmark_paths: dict[str, dict[str, Path]] = {}
    for benchmark_id, benchmark_result in sorted(result.benchmarks.items()):
        benchmark_dir = stage_dir / "per_benchmark" / benchmark_id
        per_benchmark_paths[benchmark_id] = {
            "model_outputs": write_parquet(
                benchmark_result.model_outputs,
                benchmark_dir / "model_outputs.parquet",
            ),
            "coefficients": write_parquet(
                benchmark_result.coefficients,
                benchmark_dir / "coefficients.parquet",
            ),
            "linear_predictor_report": write_json(
                benchmark_result.linear_predictor_report,
                benchmark_dir / "linear_predictor_report.json",
            ),
        }
    feature_report_path = write_json(result.feature_report, stage_dir / "feature_report.json")
    return {
        "feature_report": feature_report_path,
        "per_benchmark": per_benchmark_paths,
    }


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


def _empty_model_outputs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            SPLIT: pd.Series(dtype="string"),
            SCORE_FULL: pd.Series(dtype="Float64"),
            REDUCED_SUBSCORE: pd.Series(dtype="Float64"),
            REDUCED_ITEM_COVERAGE: pd.Series(dtype="Float64"),
            OBSERVED_ITEM_COUNT: pd.Series(dtype="Int64"),
            SELECTED_ITEM_COUNT: pd.Series(dtype="Int64"),
            LINEAR_PREDICTION: pd.Series(dtype="Float64"),
            LINEAR_PREDICTION_MISSING_REASON: pd.Series(dtype="string"),
            TRAINING_ELIGIBLE: pd.Series(dtype=bool),
        }
    )


def _empty_coefficients_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            ITEM_ID: pd.Series(dtype="string"),
            COEFFICIENT: pd.Series(dtype="Float64"),
        }
    )
