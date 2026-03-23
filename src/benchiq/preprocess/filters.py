"""Benchmark-wise preprocessing filters and stage-01 artifact writing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from benchiq.io.load import Bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.logging import update_manifest
from benchiq.preprocess.stats import (
    build_benchmark_matrix,
    compute_item_stats,
    compute_model_coverage,
    compute_model_scores,
    effective_min_models_per_item,
    select_low_tail_model_ids,
)
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID


@dataclass(slots=True)
class BenchmarkPreprocessResult:
    """Per-benchmark preprocessing outputs."""

    benchmark_id: str
    item_stats: pd.DataFrame
    filtered_items: pd.DataFrame
    filtered_models: pd.DataFrame
    preprocess_report: dict[str, Any]
    refused: bool


@dataclass(slots=True)
class PreprocessResult:
    """Stage-01 preprocessing outputs."""

    benchmarks: dict[str, BenchmarkPreprocessResult]
    summary: pd.DataFrame
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    manifest_path: Path | None = None


def preprocess_bundle(
    bundle: Bundle,
    *,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> PreprocessResult:
    """Compute per-benchmark stats/filters and optionally write stage-01 artifacts."""

    benchmark_results: dict[str, BenchmarkPreprocessResult] = {}
    summary_rows: list[dict[str, Any]] = []
    benchmark_ids = (
        bundle.responses_long[BENCHMARK_ID]
        .dropna()
        .astype("string")
        .sort_values()
        .unique()
        .tolist()
    )
    for benchmark_id in benchmark_ids:
        result = preprocess_benchmark(bundle, benchmark_id=benchmark_id)
        benchmark_results[benchmark_id] = result
        summary_rows.append(_summary_row_from_result(result))

    summary = (
        pd.DataFrame.from_records(summary_rows).sort_values("benchmark_id").reset_index(drop=True)
    )
    preprocess_result = PreprocessResult(benchmarks=benchmark_results, summary=summary)
    run_root, manifest_path = _resolve_run_root(bundle, out_dir=out_dir, run_id=run_id)
    if run_root is not None:
        artifact_paths = _write_preprocess_artifacts(preprocess_result, run_root=run_root)
        preprocess_result.artifact_paths = artifact_paths
        preprocess_result.manifest_path = manifest_path
        if manifest_path is not None:
            update_manifest(
                manifest_path,
                {
                    "artifacts": {
                        "01_preprocess": {
                            "summary": str(artifact_paths["summary"]),
                            "per_benchmark": {
                                benchmark_id: {
                                    "filtered_items": str(paths["filtered_items"]),
                                    "filtered_models": str(paths["filtered_models"]),
                                    "item_stats": str(paths["item_stats"]),
                                    "preprocess_report": str(paths["preprocess_report"]),
                                }
                                for benchmark_id, paths in sorted(
                                    artifact_paths["per_benchmark"].items(),
                                )
                            },
                        },
                    },
                },
            )
    return preprocess_result


def preprocess_benchmark(bundle: Bundle, *, benchmark_id: str) -> BenchmarkPreprocessResult:
    """Preprocess one benchmark and return the filtered views/report."""

    config = bundle.config
    matrix = build_benchmark_matrix(bundle.responses_long, benchmark_id=benchmark_id)
    initial_models = len(matrix.index)
    initial_items = len(matrix.columns)
    warnings: list[str] = []
    refusal_reasons: list[str] = []

    model_scores = compute_model_scores(matrix)
    low_tail_model_ids = select_low_tail_model_ids(
        model_scores,
        quantile=config.drop_low_tail_models_quantile,
    )
    if low_tail_model_ids:
        matrix = matrix.drop(index=low_tail_model_ids).copy()

    n_models_for_coverage = len(matrix.index)
    min_models_per_item = effective_min_models_per_item(
        n_models_benchmark=n_models_for_coverage,
        min_models_per_item=config.min_models_per_item,
    )
    item_stats = compute_item_stats(matrix, benchmark_id=benchmark_id)
    item_stats = apply_item_filter_flags(
        item_stats,
        min_item_sd=config.min_item_sd,
        max_item_mean=config.max_item_mean,
        min_abs_point_biserial=config.min_abs_point_biserial,
        min_models_per_item=min_models_per_item,
    )

    kept_items = item_stats.loc[item_stats["retained_after_item_filters"], "item_id"].tolist()
    final_matrix = matrix.loc[:, kept_items].copy()
    low_coverage_models: list[str] = []
    dropped_item_coverage_ids: set[str] = set(
        item_stats.loc[item_stats["drop_item_coverage"], "item_id"].tolist(),
    )
    while kept_items and not final_matrix.empty:
        model_coverage = compute_model_coverage(final_matrix, benchmark_id=benchmark_id)
        keep_model_mask = model_coverage["model_coverage"] >= config.min_item_coverage
        dropped_models = model_coverage.loc[~keep_model_mask, "model_id"].astype("string").tolist()
        if dropped_models:
            low_coverage_models.extend(dropped_models)
            final_matrix = final_matrix.loc[keep_model_mask.to_numpy()].copy()

        item_response_counts = final_matrix.notna().sum(axis=0)
        coverage_keep_mask = item_response_counts >= min_models_per_item
        dropped_items = item_response_counts.index[~coverage_keep_mask].astype("string").tolist()
        if dropped_items:
            dropped_item_coverage_ids.update(dropped_items)
            final_matrix = final_matrix.loc[:, coverage_keep_mask.to_numpy()].copy()

        if not dropped_models and not dropped_items:
            break
        kept_items = final_matrix.columns.astype("string").tolist()

    item_stats["retained"] = item_stats["item_id"].isin(final_matrix.columns.astype("string"))
    item_stats["drop_reasons"] = item_stats.apply(
        lambda row: _item_drop_reasons(
            row=row,
            dropped_item_coverage_ids=dropped_item_coverage_ids,
        ),
        axis=1,
    )

    final_model_coverage = compute_model_coverage(final_matrix, benchmark_id=benchmark_id)
    final_model_ids = final_model_coverage["model_id"].astype("string").tolist()
    refused = False

    if len(final_model_ids) < config.min_models_per_benchmark:
        refusal_reasons.append("too_few_models_after_filtering")
        refused = True
    elif len(final_model_ids) < config.warn_models_per_benchmark:
        if config.allow_low_n:
            warnings.append("low_n_warning")
        else:
            refusal_reasons.append("allow_low_n_required")
            refused = True

    if len(final_matrix.columns) < config.min_items_after_filtering:
        refusal_reasons.append("too_few_items_after_filtering")
        refused = True

    filtered_items = _build_filtered_items(bundle, benchmark_id=benchmark_id, item_stats=item_stats)
    filtered_models = _build_filtered_models(
        bundle,
        benchmark_id=benchmark_id,
        model_coverage=final_model_coverage,
    )
    if refused:
        filtered_items = filtered_items.iloc[0:0].copy()
        filtered_models = filtered_models.iloc[0:0].copy()

    preprocess_report = {
        "benchmark_id": benchmark_id,
        "refused": refused,
        "refusal_reasons": refusal_reasons,
        "warnings": warnings,
        "thresholds": {
            "drop_low_tail_models_quantile": config.drop_low_tail_models_quantile,
            "min_item_sd": config.min_item_sd,
            "max_item_mean": config.max_item_mean,
            "min_abs_point_biserial": config.min_abs_point_biserial,
            "min_models_per_item": min_models_per_item,
            "min_item_coverage": config.min_item_coverage,
            "min_models_per_benchmark": config.min_models_per_benchmark,
            "warn_models_per_benchmark": config.warn_models_per_benchmark,
            "min_items_after_filtering": config.min_items_after_filtering,
        },
        "counts": {
            "initial_models": initial_models,
            "initial_items": initial_items,
            "low_tail_models_dropped": len(low_tail_model_ids),
            "coverage_models_dropped": len(set(low_coverage_models)),
            "retained_models": len(final_model_ids),
            "retained_items": len(final_matrix.columns),
            "dropped_low_variance_items": int(item_stats["drop_low_variance"].sum()),
            "dropped_near_ceiling_items": int(item_stats["drop_near_ceiling"].sum()),
            "dropped_low_discrimination_items": int(item_stats["drop_low_discrimination"].sum()),
            "dropped_low_coverage_items": len(dropped_item_coverage_ids),
        },
        "dropped_model_ids": {
            "low_tail": sorted(low_tail_model_ids),
            "coverage": sorted(set(low_coverage_models)),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return BenchmarkPreprocessResult(
        benchmark_id=benchmark_id,
        item_stats=item_stats,
        filtered_items=filtered_items,
        filtered_models=filtered_models,
        preprocess_report=preprocess_report,
        refused=refused,
    )


def apply_item_filter_flags(
    item_stats: pd.DataFrame,
    *,
    min_item_sd: float,
    max_item_mean: float,
    min_abs_point_biserial: float,
    min_models_per_item: int,
) -> pd.DataFrame:
    """Apply preprocessing thresholds to per-item statistics."""

    flagged = item_stats.copy()
    flagged["point_biserial"] = pd.to_numeric(
        flagged["point_biserial"],
        errors="coerce",
    ).astype("Float64")
    flagged["drop_low_variance"] = flagged["sd"].fillna(0.0) <= min_item_sd
    flagged["drop_near_ceiling"] = flagged["mean"].fillna(0.0) >= max_item_mean
    flagged["drop_low_discrimination"] = flagged["point_biserial"].isna() | (
        flagged["point_biserial"].abs() < min_abs_point_biserial
    )
    flagged["drop_item_coverage"] = flagged["n_responses"] < min_models_per_item
    flagged["retained_after_item_filters"] = ~flagged[
        [
            "drop_low_variance",
            "drop_near_ceiling",
            "drop_low_discrimination",
            "drop_item_coverage",
        ]
    ].any(axis=1)
    return flagged


def _item_drop_reasons(
    *,
    row: pd.Series,
    dropped_item_coverage_ids: set[str],
) -> str | None:
    reasons: list[str] = []
    if bool(row["drop_low_variance"]):
        reasons.append("low_variance")
    if bool(row["drop_near_ceiling"]):
        reasons.append("near_ceiling")
    if bool(row["drop_low_discrimination"]):
        reasons.append("low_discrimination")
    if row["item_id"] in dropped_item_coverage_ids:
        reasons.append("insufficient_item_coverage")
    if not reasons:
        return None
    return ";".join(sorted(set(reasons)))


def _build_filtered_items(
    bundle: Bundle,
    *,
    benchmark_id: str,
    item_stats: pd.DataFrame,
) -> pd.DataFrame:
    metadata = bundle.items.loc[bundle.items[BENCHMARK_ID] == benchmark_id].copy()
    filtered = metadata.merge(item_stats, on=[BENCHMARK_ID, ITEM_ID], how="inner")
    filtered = filtered.loc[filtered["retained"]].copy()
    return filtered.sort_values(ITEM_ID).reset_index(drop=True)


def _build_filtered_models(
    bundle: Bundle,
    *,
    benchmark_id: str,
    model_coverage: pd.DataFrame,
) -> pd.DataFrame:
    metadata = bundle.models.copy()
    filtered = metadata.merge(model_coverage, on=MODEL_ID, how="inner")
    if BENCHMARK_ID not in filtered.columns:
        filtered.insert(0, BENCHMARK_ID, benchmark_id)
    return filtered.sort_values(MODEL_ID).reset_index(drop=True)


def _summary_row_from_result(result: BenchmarkPreprocessResult) -> dict[str, Any]:
    counts = result.preprocess_report["counts"]
    return {
        "benchmark_id": result.benchmark_id,
        "refused": result.refused,
        "refusal_reasons": ";".join(result.preprocess_report["refusal_reasons"]) or None,
        "warning_count": len(result.preprocess_report["warnings"]),
        "initial_models": counts["initial_models"],
        "initial_items": counts["initial_items"],
        "retained_models": counts["retained_models"],
        "retained_items": counts["retained_items"],
        "low_tail_models_dropped": counts["low_tail_models_dropped"],
        "coverage_models_dropped": counts["coverage_models_dropped"],
        "dropped_low_variance_items": counts["dropped_low_variance_items"],
        "dropped_near_ceiling_items": counts["dropped_near_ceiling_items"],
        "dropped_low_discrimination_items": counts["dropped_low_discrimination_items"],
        "dropped_low_coverage_items": counts["dropped_low_coverage_items"],
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


def _write_preprocess_artifacts(
    result: PreprocessResult,
    *,
    run_root: Path,
) -> dict[str, Any]:
    stage_dir = run_root / "artifacts" / "01_preprocess"
    per_benchmark_paths: dict[str, dict[str, Path]] = {}
    for benchmark_id, benchmark_result in sorted(result.benchmarks.items()):
        benchmark_dir = stage_dir / "per_benchmark" / benchmark_id
        per_benchmark_paths[benchmark_id] = {
            "filtered_items": write_parquet(
                benchmark_result.filtered_items,
                benchmark_dir / "filtered_items.parquet",
            ),
            "filtered_models": write_parquet(
                benchmark_result.filtered_models,
                benchmark_dir / "filtered_models.parquet",
            ),
            "item_stats": write_parquet(
                benchmark_result.item_stats,
                benchmark_dir / "item_stats.parquet",
            ),
            "preprocess_report": write_json(
                benchmark_result.preprocess_report,
                benchmark_dir / "preprocess_report.json",
            ),
        }

    summary_path = write_parquet(result.summary, stage_dir / "summary.parquet")
    return {"summary": summary_path, "per_benchmark": per_benchmark_paths}


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
