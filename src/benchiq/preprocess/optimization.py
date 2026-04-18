"""Preprocessing optimization experiment helpers for BenchIQ."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from benchiq.config import BenchIQConfig
from benchiq.io.load import Bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.runner import run as run_pipeline
from benchiq.schema.tables import BENCHMARK_ID, SPLIT

THRESHOLD_COLUMNS = (
    "drop_low_tail_models_quantile",
    "min_item_sd",
    "max_item_mean",
    "min_abs_point_biserial",
    "min_models_per_item",
    "min_item_coverage",
)
MARGINAL_MODEL = "marginal"
JOINT_MODEL = "joint"


@dataclass(slots=True, frozen=True)
class PreprocessingOptimizationProfile:
    """One named preprocessing rule family."""

    profile_id: str
    family: str
    description: str
    config_overrides: Mapping[str, Any]
    is_baseline: bool = False


@dataclass(slots=True, frozen=True)
class PreprocessingOptimizationDataset:
    """A reusable dataset/setup pair for preprocessing experiments."""

    dataset_id: str
    label: str
    source_path: str
    base_config: Mapping[str, Any]
    base_stage_options: Mapping[str, Mapping[str, Any]]
    notes: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class PreprocessingOptimizationPlan:
    """One stage of a preprocessing experiment sweep."""

    search_stage: str
    dataset_id: str
    profile_ids: tuple[str, ...]
    preselection_methods: tuple[str, ...]
    seeds: tuple[int, ...]


@dataclass(slots=True)
class PreprocessingExperimentRawResult:
    """Raw run outputs before aggregation."""

    experiment_matrix: pd.DataFrame
    run_index: pd.DataFrame
    per_run_metrics: pd.DataFrame
    selection_sets: pd.DataFrame


@dataclass(slots=True)
class PreprocessingOptimizationResult:
    """Aggregated optimization outputs."""

    experiment_matrix: pd.DataFrame
    run_index: pd.DataFrame
    per_run_metrics: pd.DataFrame
    selection_sets: pd.DataFrame
    selection_stability: pd.DataFrame
    summary: pd.DataFrame
    report: dict[str, Any]
    artifact_paths: dict[str, Path] = field(default_factory=dict)


def execute_preprocessing_experiment_plans(
    *,
    bundles: Mapping[str, Bundle],
    datasets: Sequence[PreprocessingOptimizationDataset],
    profiles: Sequence[PreprocessingOptimizationProfile],
    plans: Sequence[PreprocessingOptimizationPlan],
    out_dir: str | Path,
) -> PreprocessingExperimentRawResult:
    """Run one or more preprocessing experiment plans and return raw tables."""

    dataset_map = {dataset.dataset_id: dataset for dataset in datasets}
    profile_map = {profile.profile_id: profile for profile in profiles}
    workdir = Path(out_dir) / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for plan in plans:
        dataset = dataset_map[plan.dataset_id]
        bundle = bundles[plan.dataset_id]
        for profile_id in plan.profile_ids:
            profile = profile_map[profile_id]
            for method in plan.preselection_methods:
                for seed in plan.seeds:
                    matrix_rows.append(
                        _experiment_matrix_row(
                            dataset=dataset,
                            profile=profile,
                            plan=plan,
                            preselection_method=method,
                            seed=seed,
                        )
                    )
                    resolved_config = resolve_experiment_config(
                        dataset=dataset,
                        profile=profile,
                        seed=seed,
                    )
                    stage_options = resolve_experiment_stage_options(
                        dataset=dataset,
                        preselection_method=method,
                    )
                    run_id = (
                        f"{plan.search_stage}__{dataset.dataset_id}__{profile.profile_id}"
                        f"__{method}__seed-{seed}"
                    )
                    run_result = run_pipeline(
                        bundle,
                        config=resolved_config,
                        out_dir=workdir,
                        run_id=run_id,
                        stage_options=stage_options,
                        stop_after="09_reconstruct",
                    )
                    run_rows.append(
                        _run_index_row(
                            run_result=run_result,
                            dataset=dataset,
                            profile=profile,
                            plan=plan,
                            preselection_method=method,
                            seed=seed,
                            config=resolved_config,
                        )
                    )
                    metric_rows.extend(
                        _benchmark_metric_rows(
                            run_result=run_result,
                            dataset=dataset,
                            profile=profile,
                            plan=plan,
                            preselection_method=method,
                            seed=seed,
                        )
                    )
                    selection_rows.extend(
                        _selection_rows(
                            run_result=run_result,
                            dataset=dataset,
                            profile=profile,
                            plan=plan,
                            preselection_method=method,
                            seed=seed,
                        )
                    )

    experiment_matrix = _build_experiment_matrix_frame(matrix_rows)
    run_index = _build_run_index_frame(run_rows)
    per_run_metrics = _build_per_run_metrics_frame(metric_rows)
    selection_sets = _build_selection_sets_frame(selection_rows)
    return PreprocessingExperimentRawResult(
        experiment_matrix=experiment_matrix,
        run_index=run_index,
        per_run_metrics=per_run_metrics,
        selection_sets=selection_sets,
    )


def combine_preprocessing_experiment_raw_results(
    results: Sequence[PreprocessingExperimentRawResult],
) -> PreprocessingExperimentRawResult:
    """Concatenate multiple raw-result batches."""

    if not results:
        return PreprocessingExperimentRawResult(
            experiment_matrix=_build_experiment_matrix_frame([]),
            run_index=_build_run_index_frame([]),
            per_run_metrics=_build_per_run_metrics_frame([]),
            selection_sets=_build_selection_sets_frame([]),
        )
    return PreprocessingExperimentRawResult(
        experiment_matrix=pd.concat(
            [result.experiment_matrix for result in results],
            ignore_index=True,
        ).drop_duplicates(),
        run_index=pd.concat(
            [result.run_index for result in results],
            ignore_index=True,
        ).drop_duplicates(),
        per_run_metrics=pd.concat(
            [result.per_run_metrics for result in results],
            ignore_index=True,
        ),
        selection_sets=pd.concat(
            [result.selection_sets for result in results],
            ignore_index=True,
        ).drop_duplicates(),
    )


def summarize_preprocessing_experiments(
    raw_result: PreprocessingExperimentRawResult,
    *,
    out_dir: str | Path | None = None,
) -> PreprocessingOptimizationResult:
    """Aggregate raw experiment tables into summary artifacts."""

    selection_stability = _selection_stability_frame(raw_result.selection_sets)
    summary = _summary_frame(
        per_run_metrics=raw_result.per_run_metrics,
        selection_stability=selection_stability,
    )
    report = _build_report(
        experiment_matrix=raw_result.experiment_matrix,
        run_index=raw_result.run_index,
        per_run_metrics=raw_result.per_run_metrics,
        selection_stability=selection_stability,
        summary=summary,
    )
    result = PreprocessingOptimizationResult(
        experiment_matrix=raw_result.experiment_matrix,
        run_index=raw_result.run_index,
        per_run_metrics=raw_result.per_run_metrics,
        selection_sets=raw_result.selection_sets,
        selection_stability=selection_stability,
        summary=summary,
        report=report,
    )
    if out_dir is not None:
        result.artifact_paths = _write_summary_artifacts(result, out_dir=Path(out_dir))
    return result


def resolve_experiment_config(
    *,
    dataset: PreprocessingOptimizationDataset,
    profile: PreprocessingOptimizationProfile,
    seed: int,
) -> BenchIQConfig:
    """Merge a dataset base config, profile overrides, and the run seed."""

    payload = dict(dataset.base_config)
    payload.update(profile.config_overrides)
    payload["random_seed"] = int(seed)
    return BenchIQConfig.model_validate(payload)


def resolve_experiment_stage_options(
    *,
    dataset: PreprocessingOptimizationDataset,
    preselection_method: str,
) -> dict[str, dict[str, Any]]:
    """Deep-copy dataset stage options and inject the stage-04 method."""

    stage_options = deepcopy(dict(dataset.base_stage_options))
    stage_options.setdefault("04_subsample", {})
    stage_options["04_subsample"]["method"] = preselection_method
    return stage_options


def top_summary_rows(
    summary: pd.DataFrame,
    *,
    dataset_id: str,
    search_stage: str | None = None,
    preselection_method: str | None = None,
    limit: int = 5,
) -> pd.DataFrame:
    """Return the best rows under the default winner sort."""

    frame = summary.copy()
    if search_stage is not None:
        frame = frame.loc[frame["search_stage"] == search_stage].copy()
    if preselection_method is not None:
        frame = frame.loc[frame["preselection_method"] == preselection_method].copy()
    frame = frame.loc[frame["dataset_id"] == dataset_id].copy()
    if frame.empty:
        return frame
    return frame.sort_values(
        [
            "best_available_test_rmse_mean",
            "seed_rmse_std",
            "final_selection_instability",
            "run_runtime_mean_seconds",
            "retained_items_mean",
        ],
        ascending=[True, True, True, True, True],
    ).head(limit)


def best_summary_row(
    summary: pd.DataFrame,
    *,
    dataset_id: str,
    search_stage: str | None = None,
    preselection_method: str | None = None,
) -> dict[str, Any] | None:
    """Return the top-ranked summary row as a plain dict."""

    top_rows = top_summary_rows(
        summary,
        dataset_id=dataset_id,
        search_stage=search_stage,
        preselection_method=preselection_method,
        limit=1,
    )
    if top_rows.empty:
        return None
    return _json_safe(top_rows.iloc[0].to_dict())


def _experiment_matrix_row(
    *,
    dataset: PreprocessingOptimizationDataset,
    profile: PreprocessingOptimizationProfile,
    plan: PreprocessingOptimizationPlan,
    preselection_method: str,
    seed: int,
) -> dict[str, Any]:
    row = {
        "search_stage": plan.search_stage,
        "dataset_id": dataset.dataset_id,
        "dataset_label": dataset.label,
        "source_path": dataset.source_path,
        "profile_id": profile.profile_id,
        "family": profile.family,
        "description": profile.description,
        "preselection_method": preselection_method,
        "seed": int(seed),
        "is_baseline": bool(profile.is_baseline),
    }
    for threshold_name in THRESHOLD_COLUMNS:
        row[f"requested_{threshold_name}"] = profile.config_overrides.get(
            threshold_name,
            dataset.base_config.get(threshold_name),
        )
    return row


def _run_index_row(
    *,
    run_result,
    dataset: PreprocessingOptimizationDataset,
    profile: PreprocessingOptimizationProfile,
    plan: PreprocessingOptimizationPlan,
    preselection_method: str,
    seed: int,
    config: BenchIQConfig,
) -> dict[str, Any]:
    stage_records = run_result.summary()["stage_records"]
    return {
        "run_signature": _run_signature(
            search_stage=plan.search_stage,
            dataset_id=dataset.dataset_id,
            profile_id=profile.profile_id,
            preselection_method=preselection_method,
            seed=seed,
        ),
        "run_id": run_result.summary()["run_id"],
        "search_stage": plan.search_stage,
        "dataset_id": dataset.dataset_id,
        "dataset_label": dataset.label,
        "profile_id": profile.profile_id,
        "family": profile.family,
        "description": profile.description,
        "preselection_method": preselection_method,
        "seed": int(seed),
        "is_baseline": bool(profile.is_baseline),
        "run_root": str(run_result.run_root),
        "config_path": str(run_result.run_root / "config_resolved.json"),
        "metrics_path": str(run_result.run_root / "reports" / "metrics.json"),
        "run_runtime_seconds": float(
            sum(float(record["duration_seconds"]) for record in stage_records.values())
        ),
        "stage01_runtime_seconds": float(stage_records["01_preprocess"]["duration_seconds"]),
        "stage04_runtime_seconds": float(stage_records["04_subsample"]["duration_seconds"]),
        "stage09_runtime_seconds": float(stage_records["09_reconstruct"]["duration_seconds"]),
        "warning_count": int(run_result.summary()["warning_count"]),
        **{threshold_name: getattr(config, threshold_name) for threshold_name in THRESHOLD_COLUMNS},
    }


def _benchmark_metric_rows(
    *,
    run_result,
    dataset: PreprocessingOptimizationDataset,
    profile: PreprocessingOptimizationProfile,
    plan: PreprocessingOptimizationPlan,
    preselection_method: str,
    seed: int,
) -> list[dict[str, Any]]:
    run_summary = run_result.summary()
    stage_records = run_summary["stage_records"]
    preprocess_result = run_result.stage_results["01_preprocess"]
    select_result = run_result.stage_results["06_select"]
    reconstruction_summary = run_result.stage_results["09_reconstruct"].reconstruction_summary
    rows: list[dict[str, Any]] = []
    for benchmark_id, preprocess_benchmark in sorted(preprocess_result.benchmarks.items()):
        preprocess_counts = preprocess_benchmark.preprocess_report["counts"]
        preprocess_thresholds = preprocess_benchmark.preprocess_report["thresholds"]
        selection_benchmark = select_result.benchmarks.get(benchmark_id)
        marginal_metrics = _split_metrics(
            reconstruction_summary,
            benchmark_id=benchmark_id,
            model_type=MARGINAL_MODEL,
            split_name="test",
        )
        joint_metrics = _split_metrics(
            reconstruction_summary,
            benchmark_id=benchmark_id,
            model_type=JOINT_MODEL,
            split_name="test",
        )
        best_available = joint_metrics if joint_metrics["rmse"] is not None else marginal_metrics
        row = {
            "run_signature": _run_signature(
                search_stage=plan.search_stage,
                dataset_id=dataset.dataset_id,
                profile_id=profile.profile_id,
                preselection_method=preselection_method,
                seed=seed,
            ),
            "run_id": run_summary["run_id"],
            "search_stage": plan.search_stage,
            "dataset_id": dataset.dataset_id,
            "dataset_label": dataset.label,
            "profile_id": profile.profile_id,
            "family": profile.family,
            "description": profile.description,
            "preselection_method": preselection_method,
            "seed": int(seed),
            "is_baseline": bool(profile.is_baseline),
            BENCHMARK_ID: benchmark_id,
            "refused": bool(preprocess_benchmark.refused),
            "refusal_reasons": ";".join(preprocess_benchmark.preprocess_report["refusal_reasons"])
            or None,
            "benchmark_warning_count": int(
                len(preprocess_benchmark.preprocess_report["warnings"])
                + len(selection_benchmark.selection_report["warnings"])
                if selection_benchmark is not None
                else len(preprocess_benchmark.preprocess_report["warnings"])
            ),
            "run_warning_count": int(run_summary["warning_count"]),
            "retained_models": int(preprocess_counts["retained_models"]),
            "retained_items": int(preprocess_counts["retained_items"]),
            "selected_items_final": (
                int(selection_benchmark.selection_report["counts"]["selected_item_count"])
                if selection_benchmark is not None
                else 0
            ),
            "selected_items_preselect": int(
                len(
                    run_result.stage_results["04_subsample"]
                    .benchmarks[benchmark_id]
                    .preselect_items.index
                )
            ),
            "joint_available": bool(joint_metrics["rmse"] is not None),
            "best_available_model_type": best_available["model_type"],
            "best_available_test_rmse": best_available["rmse"],
            "best_available_test_mae": best_available["mae"],
            "best_available_test_pearson": best_available["pearson_r"],
            "best_available_test_spearman": best_available["spearman_r"],
            "marginal_test_rmse": marginal_metrics["rmse"],
            "marginal_test_mae": marginal_metrics["mae"],
            "marginal_test_pearson": marginal_metrics["pearson_r"],
            "marginal_test_spearman": marginal_metrics["spearman_r"],
            "joint_test_rmse": joint_metrics["rmse"],
            "joint_test_mae": joint_metrics["mae"],
            "joint_test_pearson": joint_metrics["pearson_r"],
            "joint_test_spearman": joint_metrics["spearman_r"],
            "run_runtime_seconds": float(
                sum(float(record["duration_seconds"]) for record in stage_records.values())
            ),
            "stage01_runtime_seconds": float(stage_records["01_preprocess"]["duration_seconds"]),
            "stage04_runtime_seconds": float(stage_records["04_subsample"]["duration_seconds"]),
            "stage09_runtime_seconds": float(stage_records["09_reconstruct"]["duration_seconds"]),
        }
        for threshold_name in THRESHOLD_COLUMNS:
            row[f"requested_{threshold_name}"] = getattr(
                run_result.stage_results["00_bundle"].config,
                threshold_name,
            )
            row[f"effective_{threshold_name}"] = preprocess_thresholds.get(
                threshold_name,
                getattr(run_result.stage_results["00_bundle"].config, threshold_name),
            )
        rows.append(row)
    return rows


def _selection_rows(
    *,
    run_result,
    dataset: PreprocessingOptimizationDataset,
    profile: PreprocessingOptimizationProfile,
    plan: PreprocessingOptimizationPlan,
    preselection_method: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_signature = _run_signature(
        search_stage=plan.search_stage,
        dataset_id=dataset.dataset_id,
        profile_id=profile.profile_id,
        preselection_method=preselection_method,
        seed=seed,
    )
    for benchmark_id, benchmark_result in sorted(
        run_result.stage_results["04_subsample"].benchmarks.items()
    ):
        preselect_items = (
            benchmark_result.preselect_items["item_id"]
            .dropna()
            .astype("string")
            .sort_values()
            .tolist()
        )
        rows.append(
            {
                "run_signature": run_signature,
                "search_stage": plan.search_stage,
                "dataset_id": dataset.dataset_id,
                "dataset_label": dataset.label,
                "profile_id": profile.profile_id,
                "family": profile.family,
                "description": profile.description,
                "preselection_method": preselection_method,
                "seed": int(seed),
                "is_baseline": bool(profile.is_baseline),
                BENCHMARK_ID: benchmark_id,
                "selection_stage": "preselect",
                "selected_items": json.dumps(preselect_items),
                "selected_item_count": len(preselect_items),
            }
        )
    for benchmark_id, benchmark_result in sorted(
        run_result.stage_results["06_select"].benchmarks.items()
    ):
        final_items = (
            benchmark_result.subset_final["item_id"]
            .dropna()
            .astype("string")
            .sort_values()
            .tolist()
        )
        rows.append(
            {
                "run_signature": run_signature,
                "search_stage": plan.search_stage,
                "dataset_id": dataset.dataset_id,
                "dataset_label": dataset.label,
                "profile_id": profile.profile_id,
                "family": profile.family,
                "description": profile.description,
                "preselection_method": preselection_method,
                "seed": int(seed),
                "is_baseline": bool(profile.is_baseline),
                BENCHMARK_ID: benchmark_id,
                "selection_stage": "final",
                "selected_items": json.dumps(final_items),
                "selected_item_count": len(final_items),
            }
        )
    return rows


def _split_metrics(
    reconstruction_summary: pd.DataFrame,
    *,
    benchmark_id: str,
    model_type: str,
    split_name: str,
) -> dict[str, Any]:
    rows = reconstruction_summary.loc[
        (reconstruction_summary[BENCHMARK_ID] == benchmark_id)
        & (reconstruction_summary["model_type"] == model_type)
        & (reconstruction_summary[SPLIT] == split_name)
    ].copy()
    if rows.empty:
        return {
            "model_type": model_type,
            "rmse": None,
            "mae": None,
            "pearson_r": None,
            "spearman_r": None,
        }
    row = rows.iloc[0]
    return {
        "model_type": model_type,
        "rmse": _float_or_none(row["rmse"]),
        "mae": _float_or_none(row["mae"]),
        "pearson_r": _float_or_none(row["pearson_r"]),
        "spearman_r": _float_or_none(row["spearman_r"]),
    }


def _selection_stability_frame(selection_sets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "search_stage",
        "dataset_id",
        "dataset_label",
        "profile_id",
        "family",
        "description",
        "preselection_method",
        "is_baseline",
        BENCHMARK_ID,
        "selection_stage",
    ]
    if selection_sets.empty:
        return _build_selection_stability_frame(rows)
    for group_key, frame in selection_sets.groupby(group_columns, dropna=False):
        item_sets = [set(json.loads(payload)) for payload in frame["selected_items"].astype(str)]
        pairwise = [
            _jaccard(left, right)
            for index, left in enumerate(item_sets)
            for right in item_sets[index + 1 :]
        ]
        (
            search_stage,
            dataset_id,
            dataset_label,
            profile_id,
            family,
            description,
            preselection_method,
            is_baseline,
            benchmark_id,
            selection_stage,
        ) = group_key
        rows.append(
            {
                "search_stage": search_stage,
                "dataset_id": dataset_id,
                "dataset_label": dataset_label,
                "profile_id": profile_id,
                "family": family,
                "description": description,
                "preselection_method": preselection_method,
                "is_baseline": is_baseline,
                BENCHMARK_ID: benchmark_id,
                "selection_stage": selection_stage,
                "pairwise_jaccard_mean": (
                    1.0 if not pairwise else float(sum(pairwise) / len(pairwise))
                ),
                "pairwise_jaccard_min": 1.0 if not pairwise else float(min(pairwise)),
                "seed_count": int(len(item_sets)),
            }
        )
    return _build_selection_stability_frame(rows)


def _summary_frame(
    *,
    per_run_metrics: pd.DataFrame,
    selection_stability: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "search_stage",
        "dataset_id",
        "dataset_label",
        "profile_id",
        "family",
        "description",
        "preselection_method",
        "is_baseline",
    ]
    if per_run_metrics.empty:
        return _build_summary_frame([])
    grouped = (
        per_run_metrics.groupby(group_columns, dropna=False)
        .agg(
            run_count=("run_signature", "nunique"),
            benchmark_rows=(BENCHMARK_ID, "count"),
            benchmark_count=(BENCHMARK_ID, "nunique"),
            best_available_test_rmse_mean=("best_available_test_rmse", "mean"),
            best_available_test_rmse_std=("best_available_test_rmse", "std"),
            best_available_test_mae_mean=("best_available_test_mae", "mean"),
            best_available_test_mae_std=("best_available_test_mae", "std"),
            best_available_test_pearson_mean=("best_available_test_pearson", "mean"),
            best_available_test_spearman_mean=("best_available_test_spearman", "mean"),
            retained_items_mean=("retained_items", "mean"),
            retained_items_std=("retained_items", "std"),
            selected_items_final_mean=("selected_items_final", "mean"),
            selected_items_preselect_mean=("selected_items_preselect", "mean"),
            retained_models_mean=("retained_models", "mean"),
            run_runtime_mean_seconds=("run_runtime_seconds", "mean"),
            stage01_runtime_mean_seconds=("stage01_runtime_seconds", "mean"),
            stage04_runtime_mean_seconds=("stage04_runtime_seconds", "mean"),
            stage09_runtime_mean_seconds=("stage09_runtime_seconds", "mean"),
            run_warning_count_mean=("run_warning_count", "mean"),
            benchmark_warning_count_mean=("benchmark_warning_count", "mean"),
            refused_benchmark_count=("refused", "sum"),
            joint_available_rate=("joint_available", "mean"),
        )
        .reset_index()
    )
    seed_spread = (
        per_run_metrics.groupby(group_columns + ["seed"], dropna=False)["best_available_test_rmse"]
        .mean()
        .groupby(level=list(range(len(group_columns))))
        .std()
        .rename("seed_rmse_std")
        .reset_index()
    )
    grouped = grouped.merge(seed_spread, on=group_columns, how="left")
    grouped["seed_rmse_std"] = grouped["seed_rmse_std"].fillna(0.0)

    if selection_stability.empty:
        grouped["preselect_selection_stability_mean"] = pd.NA
        grouped["preselect_selection_stability_min"] = pd.NA
        grouped["final_selection_stability_mean"] = pd.NA
        grouped["final_selection_stability_min"] = pd.NA
    else:
        stability_wide = (
            selection_stability.groupby(group_columns + ["selection_stage"], dropna=False)
            .agg(
                stability_mean=("pairwise_jaccard_mean", "mean"),
                stability_min=("pairwise_jaccard_min", "min"),
            )
            .reset_index()
            .pivot_table(
                index=group_columns,
                columns="selection_stage",
                values=["stability_mean", "stability_min"],
                aggfunc="first",
            )
        )
        stability_wide.columns = [
            f"{selection_stage}_selection_{metric}"
            for metric, selection_stage in stability_wide.columns.to_flat_index()
        ]
        stability_wide = stability_wide.reset_index()
        grouped = grouped.merge(stability_wide, on=group_columns, how="left")
        grouped = grouped.rename(
            columns={
                "preselect_selection_stability_mean": "preselect_selection_stability_mean",
                "preselect_selection_stability_min": "preselect_selection_stability_min",
                "final_selection_stability_mean": "final_selection_stability_mean",
                "final_selection_stability_min": "final_selection_stability_min",
            }
        )
    grouped["final_selection_instability"] = 1.0 - grouped["final_selection_stability_mean"].fillna(
        1.0
    )
    grouped["rmse_rank_within_dataset_stage"] = (
        grouped.groupby(["search_stage", "dataset_id", "preselection_method"], dropna=False)[
            "best_available_test_rmse_mean"
        ]
        .rank(method="dense")
        .astype("Int64")
    )
    return _build_summary_frame(grouped.to_dict(orient="records"))


def _build_report(
    *,
    experiment_matrix: pd.DataFrame,
    run_index: pd.DataFrame,
    per_run_metrics: pd.DataFrame,
    selection_stability: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    winners_by_group: list[dict[str, Any]] = []
    if not summary.empty:
        for (search_stage, dataset_id, preselection_method), frame in summary.groupby(
            ["search_stage", "dataset_id", "preselection_method"],
            dropna=False,
        ):
            ordered = frame.sort_values(
                [
                    "best_available_test_rmse_mean",
                    "seed_rmse_std",
                    "final_selection_instability",
                    "run_runtime_mean_seconds",
                    "retained_items_mean",
                ],
                ascending=[True, True, True, True, True],
            )
            winners_by_group.append(
                {
                    "search_stage": search_stage,
                    "dataset_id": dataset_id,
                    "preselection_method": preselection_method,
                    "winner": _json_safe(ordered.iloc[0].to_dict()),
                }
            )
    return {
        "counts": {
            "planned_runs": int(len(experiment_matrix.index)),
            "completed_runs": (
                int(run_index["run_signature"].nunique()) if not run_index.empty else 0
            ),
            "benchmark_metric_rows": int(len(per_run_metrics.index)),
            "selection_rows": int(selection_stability.index.size),
            "summary_rows": int(len(summary.index)),
        },
        "winners_by_group": winners_by_group,
        "baseline_rows": _json_safe(
            summary.loc[summary["is_baseline"].fillna(False)].to_dict(orient="records")
            if not summary.empty
            else []
        ),
    }


def _write_summary_artifacts(
    result: PreprocessingOptimizationResult,
    *,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "experiment_matrix_parquet": write_parquet(
            result.experiment_matrix,
            out_dir / "experiment_matrix.parquet",
        ),
        "experiment_matrix_csv": _write_csv(
            result.experiment_matrix,
            out_dir / "experiment_matrix.csv",
        ),
        "run_index_parquet": write_parquet(result.run_index, out_dir / "run_index.parquet"),
        "run_index_csv": _write_csv(result.run_index, out_dir / "run_index.csv"),
        "per_run_metrics_parquet": write_parquet(
            result.per_run_metrics,
            out_dir / "per_run_metrics.parquet",
        ),
        "per_run_metrics_csv": _write_csv(result.per_run_metrics, out_dir / "per_run_metrics.csv"),
        "selection_sets_parquet": write_parquet(
            result.selection_sets,
            out_dir / "selection_sets.parquet",
        ),
        "selection_sets_csv": _write_csv(result.selection_sets, out_dir / "selection_sets.csv"),
        "selection_stability_parquet": write_parquet(
            result.selection_stability,
            out_dir / "selection_stability.parquet",
        ),
        "selection_stability_csv": _write_csv(
            result.selection_stability,
            out_dir / "selection_stability.csv",
        ),
        "summary_parquet": write_parquet(result.summary, out_dir / "summary.parquet"),
        "summary_csv": _write_csv(result.summary, out_dir / "summary.csv"),
        "report_json": write_json(result.report, out_dir / "report.json"),
        "summary_md": _write_text(_summary_markdown(result), out_dir / "summary.md"),
    }
    artifact_paths["rmse_vs_family_plot"] = _plot_rmse_vs_family(
        result.summary,
        out_path=plots_dir / "rmse_vs_preprocessing_family.png",
    )
    artifact_paths["seed_spread_plot"] = _plot_seed_spread(
        result.summary,
        out_path=plots_dir / "seed_spread_vs_stability.png",
    )
    artifact_paths["runtime_vs_rmse_plot"] = _plot_runtime_vs_rmse(
        result.summary,
        out_path=plots_dir / "runtime_vs_rmse.png",
    )
    artifact_paths["retained_items_vs_rmse_plot"] = _plot_retained_items_vs_rmse(
        result.summary,
        out_path=plots_dir / "retained_items_vs_rmse.png",
    )
    artifact_paths["default_vs_winner_plot"] = _plot_default_vs_winner(
        result.summary,
        out_path=plots_dir / "default_vs_winner.png",
    )
    return artifact_paths


def _summary_markdown(result: PreprocessingOptimizationResult) -> str:
    lines = [
        "# preprocessing optimization",
        "",
        "## winners",
        "",
    ]
    if not result.report["winners_by_group"]:
        lines.append("- none")
    else:
        for winner_payload in result.report["winners_by_group"]:
            winner = winner_payload["winner"]
            lines.append(
                "- "
                f"{winner_payload['search_stage']} / {winner_payload['dataset_id']} / "
                f"{winner_payload['preselection_method']}: "
                f"`{winner['profile_id']}` "
                f"(rmse_mean={winner['best_available_test_rmse_mean']:.4f}, "
                f"seed_rmse_std={winner['seed_rmse_std']:.4f}, "
                f"final_stability={winner.get('final_selection_stability_mean')})"
            )
    lines.extend(
        [
            "",
            "## summary",
            "",
        ]
    )
    if result.summary.empty:
        lines.append("- none")
    else:
        for row in result.summary.sort_values(
            [
                "search_stage",
                "dataset_id",
                "preselection_method",
                "best_available_test_rmse_mean",
            ]
        ).to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['search_stage']} / {row['dataset_id']} / {row['preselection_method']} / "
                f"`{row['profile_id']}`: "
                f"rmse_mean={row['best_available_test_rmse_mean']:.4f}, "
                f"mae_mean={row['best_available_test_mae_mean']:.4f}, "
                f"pearson_mean={row['best_available_test_pearson_mean']:.4f}, "
                f"spearman_mean={row['best_available_test_spearman_mean']:.4f}, "
                f"seed_rmse_std={row['seed_rmse_std']:.4f}, "
                f"runtime_mean_seconds={row['run_runtime_mean_seconds']:.4f}, "
                f"retained_items_mean={row['retained_items_mean']:.2f}, "
                f"selected_items_final_mean={row['selected_items_final_mean']:.2f}, "
                f"final_selection_stability_mean={row.get('final_selection_stability_mean')}"
            )
    lines.append("")
    return "\n".join(lines)


def _plot_rmse_vs_family(summary: pd.DataFrame, *, out_path: Path) -> Path:
    figure, axes = _dataset_subplots(summary)
    for axis, dataset_id in axes:
        frame = summary.loc[summary["dataset_id"] == dataset_id].copy()
        if frame.empty:
            axis.set_visible(False)
            continue
        families = frame["family"].astype(str).dropna().unique().tolist()
        family_positions = {family: index for index, family in enumerate(families)}
        for method, marker in (("deterministic_info", "o"), ("random_cv", "s")):
            method_frame = frame.loc[frame["preselection_method"] == method].copy()
            if method_frame.empty:
                continue
            x_values = [family_positions[family] for family in method_frame["family"].astype(str)]
            axis.scatter(
                x_values,
                method_frame["best_available_test_rmse_mean"].astype(float),
                label=method,
                marker=marker,
                alpha=0.85,
            )
        axis.set_xticks(
            list(family_positions.values()),
            list(family_positions.keys()),
            rotation=30,
            ha="right",
        )
        axis.set_ylabel("mean held-out rmse")
        axis.set_title(str(frame["dataset_label"].iloc[0]))
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def _plot_seed_spread(summary: pd.DataFrame, *, out_path: Path) -> Path:
    figure, axes = _dataset_subplots(summary)
    for axis, dataset_id in axes:
        frame = summary.loc[summary["dataset_id"] == dataset_id].copy()
        if frame.empty:
            axis.set_visible(False)
            continue
        axis.scatter(
            frame["seed_rmse_std"].astype(float),
            frame["final_selection_stability_mean"].fillna(1.0).astype(float),
            c=frame["run_runtime_mean_seconds"].astype(float),
            cmap="viridis",
            alpha=0.85,
        )
        axis.set_xlabel("seed rmse std")
        axis.set_ylabel("final selection stability")
        axis.set_title(str(frame["dataset_label"].iloc[0]))
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def _plot_runtime_vs_rmse(summary: pd.DataFrame, *, out_path: Path) -> Path:
    figure, axes = _dataset_subplots(summary)
    for axis, dataset_id in axes:
        frame = summary.loc[summary["dataset_id"] == dataset_id].copy()
        if frame.empty:
            axis.set_visible(False)
            continue
        for method, marker in (("deterministic_info", "o"), ("random_cv", "s")):
            method_frame = frame.loc[frame["preselection_method"] == method].copy()
            if method_frame.empty:
                continue
            axis.scatter(
                method_frame["run_runtime_mean_seconds"].astype(float),
                method_frame["best_available_test_rmse_mean"].astype(float),
                label=method,
                marker=marker,
                alpha=0.85,
            )
        axis.set_xlabel("runtime (s)")
        axis.set_ylabel("mean held-out rmse")
        axis.set_title(str(frame["dataset_label"].iloc[0]))
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def _plot_retained_items_vs_rmse(summary: pd.DataFrame, *, out_path: Path) -> Path:
    figure, axes = _dataset_subplots(summary)
    for axis, dataset_id in axes:
        frame = summary.loc[summary["dataset_id"] == dataset_id].copy()
        if frame.empty:
            axis.set_visible(False)
            continue
        axis.scatter(
            frame["retained_items_mean"].astype(float),
            frame["best_available_test_rmse_mean"].astype(float),
            alpha=0.85,
        )
        axis.set_xlabel("retained items after preprocessing")
        axis.set_ylabel("mean held-out rmse")
        axis.set_title(str(frame["dataset_label"].iloc[0]))
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def _plot_default_vs_winner(summary: pd.DataFrame, *, out_path: Path) -> Path:
    figure, axes = _dataset_subplots(summary)
    for axis, dataset_id in axes:
        frame = summary.loc[summary["dataset_id"] == dataset_id].copy()
        if frame.empty:
            axis.set_visible(False)
            continue
        baseline_frame = frame.loc[frame["is_baseline"].fillna(False)].sort_values(
            "best_available_test_rmse_mean"
        )
        winner_frame = top_summary_rows(
            frame,
            dataset_id=dataset_id,
            search_stage=None,
            preselection_method="deterministic_info"
            if (frame["preselection_method"] == "deterministic_info").any()
            else None,
            limit=1,
        )
        if baseline_frame.empty or winner_frame.empty:
            axis.set_visible(False)
            continue
        baseline = baseline_frame.iloc[0]
        winner = winner_frame.iloc[0]
        labels = ["rmse", "seed std", "runtime", "final instability"]
        baseline_values = [
            float(baseline["best_available_test_rmse_mean"]),
            float(baseline["seed_rmse_std"]),
            float(baseline["run_runtime_mean_seconds"]),
            float(1.0 - baseline["final_selection_stability_mean"])
            if pd.notna(baseline["final_selection_stability_mean"])
            else 0.0,
        ]
        winner_values = [
            float(winner["best_available_test_rmse_mean"]),
            float(winner["seed_rmse_std"]),
            float(winner["run_runtime_mean_seconds"]),
            float(1.0 - winner["final_selection_stability_mean"])
            if pd.notna(winner["final_selection_stability_mean"])
            else 0.0,
        ]
        positions = list(range(len(labels)))
        axis.bar(
            [position - 0.2 for position in positions],
            baseline_values,
            width=0.4,
            label=str(baseline["profile_id"]),
        )
        axis.bar(
            [position + 0.2 for position in positions],
            winner_values,
            width=0.4,
            label=str(winner["profile_id"]),
        )
        axis.set_xticks(positions, labels)
        axis.set_title(str(frame["dataset_label"].iloc[0]))
        axis.legend()
        axis.grid(True, alpha=0.25, axis="y")
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def _dataset_subplots(summary: pd.DataFrame) -> tuple[plt.Figure, list[tuple[Any, str]]]:
    dataset_ids = summary["dataset_id"].astype(str).dropna().unique().tolist()
    if not dataset_ids:
        figure, axis = plt.subplots(figsize=(6, 4))
        return figure, [(axis, "empty")]
    figure, raw_axes = plt.subplots(1, len(dataset_ids), figsize=(7 * len(dataset_ids), 4))
    axes = list(raw_axes.flatten()) if hasattr(raw_axes, "flatten") else [raw_axes]
    return figure, list(zip(axes, dataset_ids, strict=True))


def _build_experiment_matrix_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "search_stage",
                "dataset_id",
                "dataset_label",
                "source_path",
                "profile_id",
                "family",
                "description",
                "preselection_method",
                "seed",
                "is_baseline",
                *[f"requested_{name}" for name in THRESHOLD_COLUMNS],
            ]
        )
    return pd.DataFrame.from_records(rows)


def _build_run_index_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_signature",
                "run_id",
                "search_stage",
                "dataset_id",
                "dataset_label",
                "profile_id",
                "family",
                "description",
                "preselection_method",
                "seed",
                "is_baseline",
                "run_root",
                "config_path",
                "metrics_path",
                "run_runtime_seconds",
                "stage01_runtime_seconds",
                "stage04_runtime_seconds",
                "stage09_runtime_seconds",
                "warning_count",
                *THRESHOLD_COLUMNS,
            ]
        )
    return pd.DataFrame.from_records(rows)


def _build_per_run_metrics_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_signature",
                "run_id",
                "search_stage",
                "dataset_id",
                "dataset_label",
                "profile_id",
                "family",
                "description",
                "preselection_method",
                "seed",
                "is_baseline",
                BENCHMARK_ID,
                "refused",
                "refusal_reasons",
                "benchmark_warning_count",
                "run_warning_count",
                "retained_models",
                "retained_items",
                "selected_items_final",
                "selected_items_preselect",
                "joint_available",
                "best_available_model_type",
                "best_available_test_rmse",
                "best_available_test_mae",
                "best_available_test_pearson",
                "best_available_test_spearman",
                "marginal_test_rmse",
                "marginal_test_mae",
                "marginal_test_pearson",
                "marginal_test_spearman",
                "joint_test_rmse",
                "joint_test_mae",
                "joint_test_pearson",
                "joint_test_spearman",
                "run_runtime_seconds",
                "stage01_runtime_seconds",
                "stage04_runtime_seconds",
                "stage09_runtime_seconds",
                *[
                    column_name
                    for threshold_name in THRESHOLD_COLUMNS
                    for column_name in (
                        f"requested_{threshold_name}",
                        f"effective_{threshold_name}",
                    )
                ],
            ]
        )
    return pd.DataFrame.from_records(rows)


def _build_selection_sets_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_signature",
                "search_stage",
                "dataset_id",
                "dataset_label",
                "profile_id",
                "family",
                "description",
                "preselection_method",
                "seed",
                "is_baseline",
                BENCHMARK_ID,
                "selection_stage",
                "selected_items",
                "selected_item_count",
            ]
        )
    return pd.DataFrame.from_records(rows)


def _build_selection_stability_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "search_stage",
                "dataset_id",
                "dataset_label",
                "profile_id",
                "family",
                "description",
                "preselection_method",
                "is_baseline",
                BENCHMARK_ID,
                "selection_stage",
                "pairwise_jaccard_mean",
                "pairwise_jaccard_min",
                "seed_count",
            ]
        )
    return pd.DataFrame.from_records(rows)


def _build_summary_frame(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    else:
        frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "search_stage",
                "dataset_id",
                "dataset_label",
                "profile_id",
                "family",
                "description",
                "preselection_method",
                "is_baseline",
                "run_count",
                "benchmark_rows",
                "benchmark_count",
                "best_available_test_rmse_mean",
                "best_available_test_rmse_std",
                "best_available_test_mae_mean",
                "best_available_test_mae_std",
                "best_available_test_pearson_mean",
                "best_available_test_spearman_mean",
                "retained_items_mean",
                "retained_items_std",
                "selected_items_final_mean",
                "selected_items_preselect_mean",
                "retained_models_mean",
                "run_runtime_mean_seconds",
                "stage01_runtime_mean_seconds",
                "stage04_runtime_mean_seconds",
                "stage09_runtime_mean_seconds",
                "run_warning_count_mean",
                "benchmark_warning_count_mean",
                "refused_benchmark_count",
                "joint_available_rate",
                "seed_rmse_std",
                "preselect_selection_stability_mean",
                "preselect_selection_stability_min",
                "final_selection_stability_mean",
                "final_selection_stability_min",
                "final_selection_instability",
                "rmse_rank_within_dataset_stage",
            ]
        )
    return frame


def _run_signature(
    *,
    search_stage: str,
    dataset_id: str,
    profile_id: str,
    preselection_method: str,
    seed: int,
) -> str:
    return f"{search_stage}::{dataset_id}::{profile_id}::{preselection_method}::seed-{int(seed)}"


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value
