"""Stage-08 marginal and joint feature-table assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from benchiq.io.load import Bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.logging import update_manifest
from benchiq.preprocess.scores import SCORE_FULL, ScoreResult
from benchiq.reconstruct.linear_predictor import (
    LINEAR_PREDICTION,
    REDUCED_SUBSCORE,
    LinearPredictorResult,
)
from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID, SPLIT
from benchiq.split.splitters import SplitResult

if TYPE_CHECKING:
    from benchiq.irt.theta import ThetaResult

MARGINAL_TARGET = "score_full_b"
MARGINAL_THETA = "theta_b"
MARGINAL_THETA_SE = "theta_se_b"
GRAND_SUB = "grand_sub"
GRAND_LIN = "grand_lin"
THETA_HAT = "theta_hat"
THETA_METHOD = "theta_method"
THETA_SE = "theta_se"


@dataclass(slots=True)
class FeatureTableResult:
    """Stage-08 feature-table outputs."""

    features_marginal: pd.DataFrame
    features_joint: pd.DataFrame
    feature_report: dict[str, Any]
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None


def build_feature_tables(
    bundle: Bundle,
    score_result: ScoreResult,
    split_result: SplitResult,
    theta_result: ThetaResult,
    linear_result: LinearPredictorResult,
    *,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> FeatureTableResult:
    """Assemble stage-08 marginal and optional joint feature tables."""

    benchmark_ids = sorted(split_result.per_benchmark_splits)
    features_marginal = _build_features_marginal(
        benchmark_ids=benchmark_ids,
        score_result=score_result,
        split_result=split_result,
        theta_result=theta_result,
        linear_result=linear_result,
    )
    features_joint, joint_report = _build_features_joint(
        benchmark_ids=benchmark_ids,
        features_marginal=features_marginal,
        score_result=score_result,
        split_result=split_result,
        min_overlap_models_for_joint=bundle.config.min_overlap_models_for_joint,
    )
    feature_report = _build_feature_report(
        features_marginal=features_marginal,
        features_joint=features_joint,
        benchmark_ids=benchmark_ids,
        split_result=split_result,
        joint_report=joint_report,
        theta_report=theta_result.theta_report,
        linear_feature_report=linear_result.feature_report,
        score_report=score_result.score_report,
    )
    result = FeatureTableResult(
        features_marginal=features_marginal,
        features_joint=features_joint,
        feature_report=feature_report,
    )

    run_root, manifest_path = _resolve_run_root(bundle, out_dir=out_dir, run_id=run_id)
    if run_root is not None:
        artifact_paths = _write_feature_artifacts(result, run_root=run_root)
        result.artifact_paths = artifact_paths
        result.manifest_path = manifest_path
        if manifest_path is not None:
            manifest_payload: dict[str, Any] = {
                "artifacts": {
                    "08_features": {
                        "features_marginal": str(artifact_paths["features_marginal"]),
                        "feature_report": str(artifact_paths["feature_report"]),
                    }
                }
            }
            if "features_joint" in artifact_paths:
                manifest_payload["artifacts"]["08_features"]["features_joint"] = str(
                    artifact_paths["features_joint"]
                )
            update_manifest(manifest_path, manifest_payload)
    return result


def _build_features_marginal(
    *,
    benchmark_ids: list[str],
    score_result: ScoreResult,
    split_result: SplitResult,
    theta_result: ThetaResult,
    linear_result: LinearPredictorResult,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for benchmark_id in benchmark_ids:
        split_frame = split_result.per_benchmark_splits[benchmark_id].copy()
        split_frame = split_frame[[BENCHMARK_ID, MODEL_ID, SPLIT]].copy()
        split_frame[BENCHMARK_ID] = split_frame[BENCHMARK_ID].astype("string")
        split_frame[MODEL_ID] = split_frame[MODEL_ID].astype("string")
        split_frame[SPLIT] = split_frame[SPLIT].astype("string")

        score_frame = (
            score_result.scores_full.loc[
                score_result.scores_full[BENCHMARK_ID] == benchmark_id,
                [MODEL_ID, SCORE_FULL],
            ]
            .rename(columns={SCORE_FULL: MARGINAL_TARGET})
            .astype({MODEL_ID: "string"})
        )
        theta_frame = (
            theta_result.theta_estimates.loc[
                theta_result.theta_estimates[BENCHMARK_ID] == benchmark_id,
                [MODEL_ID, THETA_HAT, THETA_SE, THETA_METHOD],
            ]
            .rename(columns={THETA_HAT: MARGINAL_THETA, THETA_SE: MARGINAL_THETA_SE})
            .astype({MODEL_ID: "string"})
        )
        linear_frame = (
            linear_result.benchmarks[benchmark_id]
            .model_outputs.loc[
                :,
                [MODEL_ID, REDUCED_SUBSCORE, LINEAR_PREDICTION],
            ]
            .astype({MODEL_ID: "string"})
        )

        benchmark_rows = split_frame.merge(score_frame, on=MODEL_ID, how="left")
        benchmark_rows = benchmark_rows.merge(theta_frame, on=MODEL_ID, how="left")
        benchmark_rows = benchmark_rows.merge(linear_frame, on=MODEL_ID, how="left")
        rows.append(benchmark_rows)

    if not rows:
        return _empty_features_marginal_frame()

    combined = pd.concat(rows, ignore_index=True)
    return combined.astype(_empty_features_marginal_frame().dtypes.to_dict())


def _build_features_joint(
    *,
    benchmark_ids: list[str],
    features_marginal: pd.DataFrame,
    score_result: ScoreResult,
    split_result: SplitResult,
    min_overlap_models_for_joint: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    grand_report = score_result.score_report["grand_scores"]
    overlap_model_ids = [str(model_id) for model_id in grand_report.get("overlap_model_ids", [])]
    if not overlap_model_ids:
        return _empty_features_joint_frame(benchmark_ids), {
            "written": False,
            "skipped": True,
            "skip_reason": grand_report.get("skip_reason") or "no_complete_bundle_overlap",
            "overlap_model_count_from_scores": 0,
            "complete_feature_model_count": 0,
            "benchmark_row_counts": {benchmark_id: 0 for benchmark_id in benchmark_ids},
            "required_overlap_count": min_overlap_models_for_joint,
            "warnings": [],
        }

    theta_wide = (
        features_marginal.pivot(index=MODEL_ID, columns=BENCHMARK_ID, values=MARGINAL_THETA)
        .rename(columns=lambda benchmark_id: f"theta_{benchmark_id}")
        .astype("Float64")
    )
    sub_wide = (
        features_marginal.pivot(index=MODEL_ID, columns=BENCHMARK_ID, values=REDUCED_SUBSCORE)
        .rename(columns=lambda benchmark_id: f"sub_{benchmark_id}")
        .astype("Float64")
    )
    lin_wide = (
        features_marginal.pivot(index=MODEL_ID, columns=BENCHMARK_ID, values=LINEAR_PREDICTION)
        .rename(columns=lambda benchmark_id: f"lin_{benchmark_id}")
        .astype("Float64")
    )

    required_theta_columns = [f"theta_{benchmark_id}" for benchmark_id in benchmark_ids]
    required_sub_columns = [f"sub_{benchmark_id}" for benchmark_id in benchmark_ids]
    required_lin_columns = [f"lin_{benchmark_id}" for benchmark_id in benchmark_ids]

    aligned_index = pd.Index(overlap_model_ids, dtype="string", name=MODEL_ID)
    theta_wide = theta_wide.reindex(aligned_index)
    sub_wide = sub_wide.reindex(aligned_index)
    lin_wide = lin_wide.reindex(aligned_index)

    complete_mask = (
        theta_wide[required_theta_columns].notna().all(axis=1)
        & sub_wide[required_sub_columns].notna().all(axis=1)
        & lin_wide[required_lin_columns].notna().all(axis=1)
    )
    complete_model_ids = theta_wide.index[complete_mask].astype("string").tolist()

    if len(complete_model_ids) < min_overlap_models_for_joint:
        skip_reason = (
            "joint_feature_overlap_below_threshold"
            if len(complete_model_ids) > 0
            else "joint_feature_values_missing"
        )
        return _empty_features_joint_frame(benchmark_ids), {
            "written": False,
            "skipped": True,
            "skip_reason": skip_reason,
            "overlap_model_count_from_scores": len(overlap_model_ids),
            "complete_feature_model_count": len(complete_model_ids),
            "benchmark_row_counts": {benchmark_id: 0 for benchmark_id in benchmark_ids},
            "required_overlap_count": min_overlap_models_for_joint,
            "warnings": [
                {
                    "code": "joint_features_incomplete",
                    "message": (
                        f"Only {len(complete_model_ids)} complete-overlap models had all benchmark "
                        "theta/sub/lin features available."
                    ),
                    "severity": "warning",
                }
            ],
        }

    sub_complete = sub_wide.loc[complete_model_ids, required_sub_columns]
    lin_complete = lin_wide.loc[complete_model_ids, required_lin_columns]
    theta_complete = theta_wide.loc[complete_model_ids, required_theta_columns]
    grand_summary = pd.DataFrame(
        {
            MODEL_ID: pd.Series(complete_model_ids, dtype="string"),
            GRAND_SUB: sub_complete.mean(axis=1).astype("Float64").to_numpy(),
            GRAND_LIN: lin_complete.mean(axis=1).astype("Float64").to_numpy(),
        }
    )

    joint_rows: list[pd.DataFrame] = []
    benchmark_row_counts: dict[str, int] = {}
    for benchmark_id in benchmark_ids:
        target_rows = features_marginal.loc[
            (features_marginal[BENCHMARK_ID] == benchmark_id)
            & (features_marginal[MODEL_ID].isin(complete_model_ids)),
            [
                BENCHMARK_ID,
                MODEL_ID,
                SPLIT,
                MARGINAL_TARGET,
                MARGINAL_THETA_SE,
                REDUCED_SUBSCORE,
                LINEAR_PREDICTION,
            ],
        ].copy()
        target_rows = target_rows.merge(grand_summary, on=MODEL_ID, how="left")
        target_rows = target_rows.merge(
            theta_complete.reset_index(),
            on=MODEL_ID,
            how="left",
        )
        benchmark_row_counts[benchmark_id] = int(len(target_rows.index))
        joint_rows.append(target_rows)

    features_joint = pd.concat(joint_rows, ignore_index=True)
    features_joint[BENCHMARK_ID] = features_joint[BENCHMARK_ID].astype("string")
    features_joint[MODEL_ID] = features_joint[MODEL_ID].astype("string")
    features_joint[SPLIT] = features_joint[SPLIT].astype("string")
    features_joint[MARGINAL_TARGET] = pd.Series(features_joint[MARGINAL_TARGET], dtype="Float64")
    features_joint[MARGINAL_THETA_SE] = pd.Series(
        features_joint[MARGINAL_THETA_SE], dtype="Float64"
    )
    features_joint[REDUCED_SUBSCORE] = pd.Series(features_joint[REDUCED_SUBSCORE], dtype="Float64")
    features_joint[LINEAR_PREDICTION] = pd.Series(
        features_joint[LINEAR_PREDICTION], dtype="Float64"
    )
    features_joint[GRAND_SUB] = pd.Series(features_joint[GRAND_SUB], dtype="Float64")
    features_joint[GRAND_LIN] = pd.Series(features_joint[GRAND_LIN], dtype="Float64")
    for column_name in required_theta_columns:
        features_joint[column_name] = pd.Series(features_joint[column_name], dtype="Float64")

    ordered_columns = [
        BENCHMARK_ID,
        MODEL_ID,
        SPLIT,
        MARGINAL_TARGET,
        MARGINAL_THETA_SE,
        REDUCED_SUBSCORE,
        LINEAR_PREDICTION,
        GRAND_SUB,
        GRAND_LIN,
        *required_theta_columns,
    ]
    features_joint = features_joint.loc[:, ordered_columns]

    return features_joint, {
        "written": True,
        "skipped": False,
        "skip_reason": None,
        "overlap_model_count_from_scores": len(overlap_model_ids),
        "complete_feature_model_count": len(complete_model_ids),
        "benchmark_row_counts": benchmark_row_counts,
        "required_overlap_count": min_overlap_models_for_joint,
        "warnings": [],
    }


def _build_feature_report(
    *,
    features_marginal: pd.DataFrame,
    features_joint: pd.DataFrame,
    benchmark_ids: list[str],
    split_result: SplitResult,
    joint_report: dict[str, Any],
    theta_report: dict[str, Any],
    linear_feature_report: dict[str, Any],
    score_report: dict[str, Any],
) -> dict[str, Any]:
    marginal_row_counts = {
        benchmark_id: int((features_marginal[BENCHMARK_ID] == benchmark_id).sum())
        for benchmark_id in benchmark_ids
    }
    split_counts = (
        features_marginal[SPLIT].value_counts(dropna=False).sort_index().astype(int).to_dict()
        if not features_marginal.empty
        else {}
    )
    warnings: list[dict[str, Any]] = list(joint_report.get("warnings", []))
    if joint_report["skipped"]:
        warnings.append(
            {
                "code": "joint_features_skipped",
                "message": f"joint feature table skipped: {joint_report['skip_reason']}",
                "severity": "warning",
            }
        )

    return {
        "warnings": warnings,
        "parameters": {
            "benchmark_ids": benchmark_ids,
            "min_overlap_models_for_joint": score_report["grand_scores"][
                "min_overlap_models_for_joint"
            ],
        },
        "counts": {
            "benchmark_count": len(benchmark_ids),
            "features_marginal_rows": int(len(features_marginal.index)),
            "features_joint_rows": int(len(features_joint.index)),
            "split_counts": split_counts,
        },
        "marginal": {
            "written": True,
            "row_counts": marginal_row_counts,
        },
        "joint": joint_report,
        "upstream": {
            "scores": score_report,
            "theta": theta_report,
            "linear_predictors": linear_feature_report,
            "splits": split_result.split_report,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_feature_artifacts(
    result: FeatureTableResult,
    *,
    run_root: Path,
) -> dict[str, Path]:
    stage_dir = run_root / "artifacts" / "08_features"
    artifact_paths: dict[str, Path] = {
        "features_marginal": write_parquet(
            result.features_marginal,
            stage_dir / "features_marginal.parquet",
        ),
        "feature_report": write_json(
            result.feature_report,
            stage_dir / "feature_report.json",
        ),
    }
    if not result.feature_report["joint"]["skipped"]:
        artifact_paths["features_joint"] = write_parquet(
            result.features_joint,
            stage_dir / "features_joint.parquet",
        )
    return artifact_paths


def _empty_features_marginal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            SPLIT: pd.Series(dtype="string"),
            MARGINAL_TARGET: pd.Series(dtype="Float64"),
            MARGINAL_THETA: pd.Series(dtype="Float64"),
            MARGINAL_THETA_SE: pd.Series(dtype="Float64"),
            REDUCED_SUBSCORE: pd.Series(dtype="Float64"),
            LINEAR_PREDICTION: pd.Series(dtype="Float64"),
            THETA_METHOD: pd.Series(dtype="string"),
        }
    )


def _empty_features_joint_frame(benchmark_ids: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            SPLIT: pd.Series(dtype="string"),
            MARGINAL_TARGET: pd.Series(dtype="Float64"),
            MARGINAL_THETA_SE: pd.Series(dtype="Float64"),
            REDUCED_SUBSCORE: pd.Series(dtype="Float64"),
            LINEAR_PREDICTION: pd.Series(dtype="Float64"),
            GRAND_SUB: pd.Series(dtype="Float64"),
            GRAND_LIN: pd.Series(dtype="Float64"),
        }
    )
    for benchmark_id in benchmark_ids:
        frame[f"theta_{benchmark_id}"] = pd.Series(dtype="Float64")
    return frame


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
