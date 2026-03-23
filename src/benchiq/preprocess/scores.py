"""Score computation and stage-02 artifact writing for BenchIQ."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from benchiq.io.load import Bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.logging import update_manifest
from benchiq.preprocess.filters import PreprocessResult
from benchiq.preprocess.stats import build_benchmark_matrix
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID

SCORE_FULL = "score_full"
ITEM_COVERAGE = "item_coverage"
ANSWERED_ITEMS = "answered_items"
RETAINED_ITEM_COUNT = "retained_item_count"
SCORE_MISSING_REASON = "score_missing_reason"
GRAND_MEAN_SCORE = "grand_mean_score"


@dataclass(slots=True)
class ScoreResult:
    """Stage-02 score outputs."""

    scores_full: pd.DataFrame
    scores_grand: pd.DataFrame
    score_report: dict[str, Any]
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    manifest_path: Path | None = None


def compute_scores(
    bundle: Bundle,
    preprocess_result: PreprocessResult,
    *,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> ScoreResult:
    """Compute stage-02 full and grand score tables."""

    all_model_ids = (
        bundle.models[MODEL_ID]
        .dropna()
        .astype("string")
        .sort_values()
        .reset_index(drop=True)
        .tolist()
    )
    benchmark_ids = sorted(preprocess_result.benchmarks)

    benchmark_reports: dict[str, dict[str, Any]] = {}
    score_frames: list[pd.DataFrame] = []
    warnings: list[str] = []

    for benchmark_id in benchmark_ids:
        benchmark_result = preprocess_result.benchmarks[benchmark_id]
        benchmark_scores, benchmark_report = _compute_benchmark_scores(
            bundle,
            benchmark_result=benchmark_result,
            all_model_ids=all_model_ids,
        )
        score_frames.append(benchmark_scores)
        benchmark_reports[benchmark_id] = benchmark_report
        if benchmark_result.refused:
            warnings.append(f"{benchmark_id}: benchmark refused during preprocessing")
        elif benchmark_report["missing_low_coverage_count"] > 0:
            missing_low_coverage = benchmark_report["missing_low_coverage_count"]
            warnings.append(
                f"{benchmark_id}: {missing_low_coverage} low-coverage models missing full scores",
            )

    scores_full = (
        pd.concat(score_frames, ignore_index=True) if score_frames else _empty_scores_full_frame()
    )
    scores_grand, grand_report, grand_warnings = _compute_scores_grand(
        scores_full,
        benchmark_ids=benchmark_ids,
        min_overlap_models_for_joint=bundle.config.min_overlap_models_for_joint,
    )
    warnings.extend(grand_warnings)
    score_report = _build_score_report(
        scores_full,
        scores_grand,
        benchmark_reports=benchmark_reports,
        benchmark_ids=benchmark_ids,
        grand_report=grand_report,
        warnings=warnings,
    )

    score_result = ScoreResult(
        scores_full=scores_full,
        scores_grand=scores_grand,
        score_report=score_report,
    )
    run_root, manifest_path = _resolve_run_root(bundle, out_dir=out_dir, run_id=run_id)
    if run_root is not None:
        artifact_paths = _write_score_artifacts(score_result, run_root=run_root)
        score_result.artifact_paths = artifact_paths
        score_result.manifest_path = manifest_path
        if manifest_path is not None:
            update_manifest(
                manifest_path,
                {
                    "artifacts": {
                        "02_scores": {
                            "scores_full": str(artifact_paths["scores_full"]),
                            "scores_grand": str(artifact_paths["scores_grand"]),
                            "score_report": str(artifact_paths["score_report"]),
                        },
                    },
                },
            )

    return score_result


def _compute_benchmark_scores(
    bundle: Bundle,
    *,
    benchmark_result: Any,
    all_model_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    benchmark_id = benchmark_result.benchmark_id
    retained_item_count = int(benchmark_result.preprocess_report["counts"]["retained_items"])
    retained_model_count = int(benchmark_result.preprocess_report["counts"]["retained_models"])

    if benchmark_result.refused:
        scores_full = pd.DataFrame(
            {
                BENCHMARK_ID: pd.Series([benchmark_id] * len(all_model_ids), dtype="string"),
                MODEL_ID: pd.Series(all_model_ids, dtype="string"),
                RETAINED_ITEM_COUNT: pd.Series(
                    [retained_item_count] * len(all_model_ids), dtype="Int64"
                ),
                ANSWERED_ITEMS: pd.Series([0] * len(all_model_ids), dtype="Int64"),
                ITEM_COVERAGE: pd.Series([pd.NA] * len(all_model_ids), dtype="Float64"),
                SCORE_FULL: pd.Series([pd.NA] * len(all_model_ids), dtype="Float64"),
                SCORE_MISSING_REASON: pd.Series(
                    ["benchmark_refused"] * len(all_model_ids),
                    dtype="string",
                ),
            },
        )
        return scores_full, {
            "refused": True,
            "refusal_reasons": benchmark_result.preprocess_report["refusal_reasons"],
            "retained_item_count": retained_item_count,
            "preprocess_retained_model_count": retained_model_count,
            "valid_score_count": 0,
            "missing_score_count": len(all_model_ids),
            "missing_low_coverage_count": 0,
            "missing_preprocess_filtered_out_count": 0,
            "missing_benchmark_refused_count": len(all_model_ids),
        }

    retained_item_ids = (
        benchmark_result.filtered_items[ITEM_ID]
        .dropna()
        .astype("string")
        .sort_values()
        .reset_index(drop=True)
        .tolist()
    )
    all_model_index = pd.Index(all_model_ids, dtype="string", name=MODEL_ID)
    matrix = build_benchmark_matrix(bundle.responses_long, benchmark_id=benchmark_id).reindex(
        index=all_model_index,
        columns=retained_item_ids,
    )
    answered_items = matrix.notna().sum(axis=1).astype("Int64")
    item_coverage = _coverage_from_answered_items(
        answered_items,
        retained_item_count=len(retained_item_ids),
    )
    raw_scores = (matrix.mean(axis=1, skipna=True) * 100).astype("Float64")
    retained_model_ids = set(
        benchmark_result.filtered_models[MODEL_ID].dropna().astype("string").tolist(),
    )

    score_values: list[float | Any] = []
    missing_reasons: list[str | None] = []
    for model_id in all_model_ids:
        coverage = item_coverage.loc[model_id]
        if pd.isna(coverage) or float(coverage) < bundle.config.min_item_coverage:
            score_values.append(pd.NA)
            missing_reasons.append("insufficient_item_coverage")
        elif model_id not in retained_model_ids:
            score_values.append(pd.NA)
            missing_reasons.append("preprocess_filtered_out")
        else:
            score_values.append(raw_scores.loc[model_id])
            missing_reasons.append(None)

    score_frame = (
        pd.DataFrame(
            {
                BENCHMARK_ID: pd.Series([benchmark_id] * len(all_model_ids), dtype="string"),
                MODEL_ID: pd.Series(all_model_ids, dtype="string"),
                RETAINED_ITEM_COUNT: pd.Series(
                    [len(retained_item_ids)] * len(all_model_ids),
                    dtype="Int64",
                ),
                ANSWERED_ITEMS: answered_items.to_numpy(),
                ITEM_COVERAGE: item_coverage.to_numpy(),
                SCORE_FULL: pd.Series(score_values, dtype="Float64"),
                SCORE_MISSING_REASON: pd.Series(missing_reasons, dtype="string"),
            },
        )
        .sort_values([BENCHMARK_ID, MODEL_ID])
        .reset_index(drop=True)
    )

    return score_frame, {
        "refused": False,
        "refusal_reasons": [],
        "retained_item_count": len(retained_item_ids),
        "preprocess_retained_model_count": retained_model_count,
        "valid_score_count": int(score_frame[SCORE_FULL].notna().sum()),
        "missing_score_count": int(score_frame[SCORE_FULL].isna().sum()),
        "missing_low_coverage_count": int(
            (score_frame[SCORE_MISSING_REASON] == "insufficient_item_coverage").sum(),
        ),
        "missing_preprocess_filtered_out_count": int(
            (score_frame[SCORE_MISSING_REASON] == "preprocess_filtered_out").sum(),
        ),
        "missing_benchmark_refused_count": 0,
    }


def _compute_scores_grand(
    scores_full: pd.DataFrame,
    *,
    benchmark_ids: list[str],
    min_overlap_models_for_joint: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    if not benchmark_ids:
        return (
            _empty_scores_grand_frame(),
            {
                "complete_overlap_model_count": 0,
                "required_benchmark_count": 0,
                "skipped": True,
                "skip_reason": "no_benchmarks_available",
            },
            ["grand scores skipped: no benchmarks available"],
        )

    valid_scores = scores_full.loc[
        scores_full[SCORE_FULL].notna(),
        [MODEL_ID, BENCHMARK_ID, SCORE_FULL],
    ].copy()
    benchmark_counts = valid_scores.groupby(MODEL_ID)[BENCHMARK_ID].nunique()
    overlap_model_ids = (
        benchmark_counts.loc[benchmark_counts == len(benchmark_ids)]
        .sort_index()
        .index.astype("string")
        .tolist()
    )
    overlap_count = len(overlap_model_ids)
    grand_report = {
        "complete_overlap_model_count": overlap_count,
        "required_benchmark_count": len(benchmark_ids),
        "required_benchmark_ids": benchmark_ids,
        "min_overlap_models_for_joint": min_overlap_models_for_joint,
        "overlap_model_ids": overlap_model_ids,
        "skipped": False,
        "skip_reason": None,
    }
    warnings: list[str] = []

    if overlap_count == 0:
        grand_report["skipped"] = True
        grand_report["skip_reason"] = "no_complete_bundle_overlap"
        warnings.append("grand scores skipped: no model has valid scores on every benchmark")
        return _empty_scores_grand_frame(), grand_report, warnings

    if overlap_count < min_overlap_models_for_joint:
        grand_report["skipped"] = True
        grand_report["skip_reason"] = "overlap_below_joint_threshold"
        warnings.append(
            "grand scores skipped: complete-overlap models are below min_overlap_models_for_joint",
        )
        return _empty_scores_grand_frame(), grand_report, warnings

    grand_scores = (
        valid_scores.loc[valid_scores[MODEL_ID].isin(overlap_model_ids)]
        .groupby(MODEL_ID, as_index=False)[SCORE_FULL]
        .mean()
        .rename(columns={SCORE_FULL: GRAND_MEAN_SCORE})
        .sort_values(MODEL_ID)
        .reset_index(drop=True)
    )
    grand_scores["benchmark_count"] = pd.Series(
        [len(benchmark_ids)] * len(grand_scores.index),
        dtype="Int64",
    )
    grand_scores[MODEL_ID] = grand_scores[MODEL_ID].astype("string")
    grand_scores[GRAND_MEAN_SCORE] = grand_scores[GRAND_MEAN_SCORE].astype("Float64")
    return grand_scores, grand_report, warnings


def _build_score_report(
    scores_full: pd.DataFrame,
    scores_grand: pd.DataFrame,
    *,
    benchmark_reports: dict[str, dict[str, Any]],
    benchmark_ids: list[str],
    grand_report: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "score_scale": "percent",
        "benchmark_ids": benchmark_ids,
        "warnings": warnings,
        "counts": {
            "benchmark_count": len(benchmark_ids),
            "scores_full_rows": int(len(scores_full.index)),
            "scores_full_valid": int(scores_full[SCORE_FULL].notna().sum()),
            "scores_full_missing": int(scores_full[SCORE_FULL].isna().sum()),
            "scores_grand_rows": int(len(scores_grand.index)),
        },
        "benchmarks": benchmark_reports,
        "grand_scores": grand_report,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _coverage_from_answered_items(
    answered_items: pd.Series,
    *,
    retained_item_count: int,
) -> pd.Series:
    if retained_item_count <= 0:
        return pd.Series(pd.NA, index=answered_items.index, dtype="Float64")
    return (answered_items.astype("Float64") / retained_item_count).astype("Float64")


def _empty_scores_full_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            RETAINED_ITEM_COUNT: pd.Series(dtype="Int64"),
            ANSWERED_ITEMS: pd.Series(dtype="Int64"),
            ITEM_COVERAGE: pd.Series(dtype="Float64"),
            SCORE_FULL: pd.Series(dtype="Float64"),
            SCORE_MISSING_REASON: pd.Series(dtype="string"),
        },
    )


def _empty_scores_grand_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            MODEL_ID: pd.Series(dtype="string"),
            GRAND_MEAN_SCORE: pd.Series(dtype="Float64"),
            "benchmark_count": pd.Series(dtype="Int64"),
        },
    )


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


def _write_score_artifacts(
    result: ScoreResult,
    *,
    run_root: Path,
) -> dict[str, Path]:
    stage_dir = run_root / "artifacts" / "02_scores"
    return {
        "scores_full": write_parquet(result.scores_full, stage_dir / "scores_full.parquet"),
        "scores_grand": write_parquet(result.scores_grand, stage_dir / "scores_grand.parquet"),
        "score_report": write_json(result.score_report, stage_dir / "score_report.json"),
    }


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
