"""Deployment-time prediction from a saved BenchIQ calibration bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchiq.calibration import CALIBRATION_BUNDLE_DIRNAME, CALIBRATION_BUNDLE_KIND
from benchiq.config import BenchIQConfig
from benchiq.io.load import Bundle, load_bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.irt.theta import (
    MISSING_HEAVY,
    OBSERVED_ITEM_COUNT,
    REDUCED_SCORE,
    RESPONSE_PATTERN,
    SATURATED,
    SATURATION_SIDE,
    SELECTED_ITEM_COUNT,
    THETA_HAT,
    THETA_METHOD,
    THETA_SE,
    estimate_theta_responses,
)
from benchiq.logging import update_manifest
from benchiq.preprocess.stats import build_benchmark_matrix
from benchiq.reconstruct.features import GRAND_LIN, GRAND_SUB, MARGINAL_THETA, MARGINAL_THETA_SE
from benchiq.reconstruct.gam import FittedGAM, load_gam
from benchiq.reconstruct.linear_predictor import (
    COEFFICIENT,
    LINEAR_PREDICTION,
    LINEAR_PREDICTION_MISSING_REASON,
    REDUCED_SUBSCORE,
)
from benchiq.reconstruct.reconstruction import JOINT_MODEL, MARGINAL_MODEL
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID

PREDICTION_AVAILABLE = "prediction_available"
PREDICTION_MISSING_REASON = "prediction_missing_reason"
PREDICTED_SCORE = "predicted_score"
MODEL_TYPE = "model_type"
SELECTED_MODEL_TYPE = "selected_model_type"


@dataclass(slots=True, frozen=True)
class CalibrationBenchmarkSpec:
    """Loaded per-benchmark calibration bundle metadata."""

    benchmark_id: str
    selected_item_ids: tuple[str, ...]
    subset_final_path: Path
    irt_item_params_path: Path
    theta_scoring_metadata_path: Path
    linear_predictor_coefficients_path: Path
    linear_predictor_report_path: Path
    reconstruction_report_path: Path
    reconstruction: dict[str, dict[str, Any]]


@dataclass(slots=True, frozen=True)
class LoadedCalibrationBundle:
    """Validated calibration bundle handle."""

    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    config: BenchIQConfig
    benchmarks: dict[str, CalibrationBenchmarkSpec]

    def resolve_path(self, relative_path: str | None) -> Path | None:
        if relative_path is None:
            return None
        candidate = Path(relative_path)
        if candidate.is_absolute():
            return candidate
        return self.root / candidate


@dataclass(slots=True)
class PredictionResult:
    """Deployment prediction outputs."""

    run_root: Path
    manifest_path: Path
    calibration_bundle_path: Path
    theta_estimates: pd.DataFrame
    features_marginal: pd.DataFrame
    features_joint: pd.DataFrame
    predictions: pd.DataFrame
    predictions_best_available: pd.DataFrame
    prediction_report: dict[str, Any]
    artifact_paths: dict[str, Any] = field(default_factory=dict)


def load_calibration_bundle(path: str | Path) -> LoadedCalibrationBundle:
    """Load and validate a calibration bundle manifest."""

    manifest_path = _resolve_calibration_manifest_path(path)
    root = manifest_path.parent
    manifest = _read_json(manifest_path)
    if manifest.get("kind") != CALIBRATION_BUNDLE_KIND:
        raise ValueError(f"{manifest_path} is not a BenchIQ calibration bundle manifest")
    config = BenchIQConfig.model_validate(manifest["config"])
    benchmarks: dict[str, CalibrationBenchmarkSpec] = {}
    for benchmark_id, payload in sorted(manifest["benchmarks"].items()):
        spec = CalibrationBenchmarkSpec(
            benchmark_id=benchmark_id,
            selected_item_ids=tuple(payload["selected_item_ids"]),
            subset_final_path=_require_existing(root / payload["subset_final_path"]),
            irt_item_params_path=_require_existing(root / payload["irt_item_params_path"]),
            theta_scoring_metadata_path=_require_existing(
                root / payload["theta_scoring_metadata_path"]
            ),
            linear_predictor_coefficients_path=_require_existing(
                root / payload["linear_predictor_coefficients_path"]
            ),
            linear_predictor_report_path=_require_existing(
                root / payload["linear_predictor_report_path"]
            ),
            reconstruction_report_path=_require_existing(
                root / payload["reconstruction_report_path"]
            ),
            reconstruction=_validate_reconstruction_payload(
                root=root,
                payload=payload["reconstruction"],
            ),
        )
        benchmarks[benchmark_id] = spec
    return LoadedCalibrationBundle(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        config=config,
        benchmarks=benchmarks,
    )


def predict(
    calibration_bundle: str | Path,
    responses_path: str | Path,
    *,
    out_dir: str | Path,
    run_id: str | None = None,
    items_path: str | Path | None = None,
    models_path: str | Path | None = None,
) -> PredictionResult:
    """Score new reduced responses from a previously fitted calibration bundle."""

    loaded_bundle = load_calibration_bundle(calibration_bundle)
    resolved_run_id = run_id or _default_run_id()
    input_bundle = load_bundle(
        responses_path,
        items_path,
        models_path,
        config=loaded_bundle.config,
        out_dir=out_dir,
        run_id=resolved_run_id,
    )
    model_ids = input_bundle.models[MODEL_ID].astype("string").tolist()
    if not model_ids:
        raise ValueError("prediction input did not contain any models")

    input_benchmark_ids = sorted(
        input_bundle.responses_long[BENCHMARK_ID].dropna().astype("string").unique().tolist()
    )
    calibrated_benchmark_ids = sorted(loaded_bundle.benchmarks)
    if not any(benchmark_id in input_benchmark_ids for benchmark_id in calibrated_benchmark_ids):
        raise ValueError(
            "prediction input does not contain any benchmarks that were calibrated in the bundle"
        )

    warnings: list[dict[str, Any]] = []
    extra_input_benchmarks = sorted(set(input_benchmark_ids) - set(calibrated_benchmark_ids))
    if extra_input_benchmarks:
        warnings.append(
            {
                "code": "input_benchmarks_ignored",
                "message": (
                    "prediction input included benchmarks that are not present in the calibration "
                    f"bundle and were ignored: {extra_input_benchmarks}"
                ),
                "severity": "warning",
            }
        )

    theta_frames: list[pd.DataFrame] = []
    marginal_feature_frames: list[pd.DataFrame] = []
    benchmark_reports: dict[str, Any] = {}
    for benchmark_id in calibrated_benchmark_ids:
        spec = loaded_bundle.benchmarks[benchmark_id]
        benchmark_theta, benchmark_features, benchmark_report = (
            _build_benchmark_deployment_features(
                input_bundle=input_bundle,
                model_ids=model_ids,
                spec=spec,
            )
        )
        theta_frames.append(benchmark_theta)
        marginal_feature_frames.append(benchmark_features)
        benchmark_reports[benchmark_id] = benchmark_report
        warnings.extend(benchmark_report["warnings"])

    theta_estimates = (
        pd.concat(theta_frames, ignore_index=True)
        if theta_frames
        else _empty_theta_estimates_frame()
    )
    features_marginal = (
        pd.concat(marginal_feature_frames, ignore_index=True)
        if marginal_feature_frames
        else _empty_features_marginal_frame()
    ).astype(_empty_features_marginal_frame().dtypes.to_dict())
    features_joint = _build_deployment_joint_features(
        model_ids=model_ids,
        benchmark_ids=calibrated_benchmark_ids,
        features_marginal=features_marginal,
    )

    prediction_frames: list[pd.DataFrame] = []
    for benchmark_id in calibrated_benchmark_ids:
        spec = loaded_bundle.benchmarks[benchmark_id]
        benchmark_marginal = features_marginal.loc[
            features_marginal[BENCHMARK_ID] == benchmark_id
        ].copy()
        prediction_frames.append(
            _predict_model_type(
                benchmark_id=benchmark_id,
                model_ids=model_ids,
                feature_frame=benchmark_marginal,
                model_type=MARGINAL_MODEL,
                model_payload=spec.reconstruction[MARGINAL_MODEL],
                bundle=loaded_bundle,
                missing_reason="missing_required_marginal_features",
            )
        )
        benchmark_joint = features_joint.loc[features_joint[BENCHMARK_ID] == benchmark_id].copy()
        prediction_frames.append(
            _predict_model_type(
                benchmark_id=benchmark_id,
                model_ids=model_ids,
                feature_frame=benchmark_joint,
                model_type=JOINT_MODEL,
                model_payload=spec.reconstruction[JOINT_MODEL],
                bundle=loaded_bundle,
                missing_reason="missing_required_joint_features",
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True).astype(
        _empty_predictions_frame().dtypes.to_dict()
    )
    predictions_best_available = _select_best_available_predictions(
        predictions=predictions,
        benchmark_ids=calibrated_benchmark_ids,
        model_ids=model_ids,
    )

    prediction_report = _build_prediction_report(
        calibration_bundle_path=loaded_bundle.manifest_path,
        predictions=predictions,
        predictions_best_available=predictions_best_available,
        theta_estimates=theta_estimates,
        benchmark_reports=benchmark_reports,
        warnings=warnings,
        input_benchmark_ids=input_benchmark_ids,
        calibrated_benchmark_ids=calibrated_benchmark_ids,
    )
    run_root = (
        input_bundle.manifest_path.parent
        if input_bundle.manifest_path is not None
        else Path(out_dir) / resolved_run_id
    )
    artifact_paths = _write_prediction_artifacts(
        run_root=run_root,
        theta_estimates=theta_estimates,
        features_marginal=features_marginal,
        features_joint=features_joint,
        predictions=predictions,
        predictions_best_available=predictions_best_available,
        prediction_report=prediction_report,
    )
    if input_bundle.manifest_path is None:
        raise ValueError("prediction run did not create a manifest")
    update_manifest(
        input_bundle.manifest_path,
        {
            "prediction": {
                "calibration_bundle_manifest": str(loaded_bundle.manifest_path),
                "generated_at": prediction_report["generated_at"],
            },
            "artifacts": {
                "01_predict": {
                    "theta_estimates": str(artifact_paths["theta_estimates"]),
                    "features_marginal": str(artifact_paths["features_marginal"]),
                    "features_joint": str(artifact_paths["features_joint"]),
                    "predictions": str(artifact_paths["predictions"]),
                    "predictions_best_available": str(artifact_paths["predictions_best_available"]),
                    "prediction_report": str(artifact_paths["prediction_report"]),
                }
            },
        },
    )
    return PredictionResult(
        run_root=run_root,
        manifest_path=input_bundle.manifest_path,
        calibration_bundle_path=loaded_bundle.manifest_path,
        theta_estimates=theta_estimates,
        features_marginal=features_marginal,
        features_joint=features_joint,
        predictions=predictions,
        predictions_best_available=predictions_best_available,
        prediction_report=prediction_report,
        artifact_paths=artifact_paths,
    )


def _build_benchmark_deployment_features(
    *,
    input_bundle: Bundle,
    model_ids: list[str],
    spec: CalibrationBenchmarkSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    benchmark_id = spec.benchmark_id
    theta_scoring = _read_json(spec.theta_scoring_metadata_path)
    item_params = pd.read_parquet(spec.irt_item_params_path)
    coefficients = pd.read_parquet(spec.linear_predictor_coefficients_path)
    linear_predictor_report = _read_json(spec.linear_predictor_report_path)

    raw_matrix = build_benchmark_matrix(input_bundle.responses_long, benchmark_id=benchmark_id)
    extra_item_ids = sorted(
        set(raw_matrix.columns.astype("string").tolist()) - set(spec.selected_item_ids)
    )
    matrix = raw_matrix.reindex(
        index=pd.Index(model_ids, dtype="string"),
        columns=pd.Index(spec.selected_item_ids, dtype="string"),
    ).astype("Float64")

    coefficient_lookup = coefficients.set_index(ITEM_ID)[COEFFICIENT].astype(float)
    missing_coefficients = [
        item_id for item_id in spec.selected_item_ids if item_id not in coefficient_lookup.index
    ]
    if missing_coefficients:
        raise FileNotFoundError(
            f"{benchmark_id} calibration bundle is missing linear coefficients for items "
            f"{missing_coefficients}"
        )
    coefficient_vector = coefficient_lookup.loc[list(spec.selected_item_ids)].to_numpy(dtype=float)

    theta_grid = np.linspace(
        float(theta_scoring["theta_min"]),
        float(theta_scoring["theta_max"]),
        int(theta_scoring["theta_grid_size"]),
        dtype=float,
    )
    selected_item_count = len(spec.selected_item_ids)
    observed_item_count = matrix.notna().sum(axis=1).astype("Int64")
    reduced_item_coverage = observed_item_count.astype(float) / max(1, selected_item_count)
    reduced_subscore = matrix.mean(axis=1, skipna=True) * 100.0

    complete_coverage_mask = observed_item_count.astype(int) == selected_item_count
    linear_prediction = pd.Series(pd.NA, index=matrix.index, dtype="Float64")
    if complete_coverage_mask.any():
        complete_ids = (
            complete_coverage_mask.index[complete_coverage_mask].astype("string").tolist()
        )
        linear_prediction.loc[complete_ids] = (
            matrix.loc[complete_ids].to_numpy(dtype=float) @ coefficient_vector
        )
    missing_reason = pd.Series(pd.NA, index=matrix.index, dtype="string")
    missing_reason.loc[~complete_coverage_mask] = "insufficient_reduced_item_coverage"

    theta_rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        responses = matrix.loc[model_id]
        estimate = estimate_theta_responses(
            responses=responses,
            item_params=item_params,
            theta_method=str(theta_scoring["theta_method"]),
            theta_grid=theta_grid,
            missing_heavy_threshold=float(theta_scoring["missing_heavy_threshold"]),
        )
        theta_rows.append(
            {
                BENCHMARK_ID: benchmark_id,
                MODEL_ID: model_id,
                THETA_HAT: estimate["theta_hat"],
                THETA_SE: estimate["theta_se"],
                THETA_METHOD: str(theta_scoring["theta_method"]),
                OBSERVED_ITEM_COUNT: estimate["observed_item_count"],
                SELECTED_ITEM_COUNT: selected_item_count,
                REDUCED_SCORE: estimate["reduced_score"],
                MISSING_HEAVY: estimate["missing_heavy"],
                SATURATED: estimate["saturated"],
                SATURATION_SIDE: estimate["saturation_side"],
                RESPONSE_PATTERN: estimate["response_pattern"],
            }
        )
        marginal_rows.append(
            {
                BENCHMARK_ID: benchmark_id,
                MODEL_ID: model_id,
                MARGINAL_THETA: estimate["theta_hat"],
                MARGINAL_THETA_SE: estimate["theta_se"],
                THETA_METHOD: str(theta_scoring["theta_method"]),
                REDUCED_SUBSCORE: reduced_subscore.loc[model_id],
                LINEAR_PREDICTION: linear_prediction.loc[model_id],
                LINEAR_PREDICTION_MISSING_REASON: missing_reason.loc[model_id],
                "reduced_item_coverage": reduced_item_coverage.loc[model_id],
                OBSERVED_ITEM_COUNT: int(observed_item_count.loc[model_id]),
                SELECTED_ITEM_COUNT: selected_item_count,
            }
        )

    theta_frame = pd.DataFrame.from_records(theta_rows).astype(
        _empty_theta_estimates_frame().dtypes.to_dict()
    )
    marginal_frame = pd.DataFrame.from_records(marginal_rows).astype(
        _empty_features_marginal_frame().dtypes.to_dict()
    )
    report = {
        "benchmark_id": benchmark_id,
        "warnings": [],
        "selected_item_count": selected_item_count,
        "input_model_count": len(model_ids),
        "extra_item_ids_ignored": extra_item_ids,
        "linear_model_kind": linear_predictor_report["training_diagnostics"]["model_kind"],
        "predicted_linear_count": int(marginal_frame[LINEAR_PREDICTION].notna().sum()),
    }
    if extra_item_ids:
        report["warnings"].append(
            {
                "benchmark_id": benchmark_id,
                "code": "extra_items_ignored",
                "message": (
                    f"{benchmark_id} prediction input included {len(extra_item_ids)} non-selected "
                    "items that were ignored."
                ),
                "severity": "warning",
            }
        )
    return theta_frame, marginal_frame, report


def _build_deployment_joint_features(
    *,
    model_ids: list[str],
    benchmark_ids: list[str],
    features_marginal: pd.DataFrame,
) -> pd.DataFrame:
    if features_marginal.empty:
        return _empty_features_joint_frame(benchmark_ids)

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

    aligned_index = pd.Index(model_ids, dtype="string", name=MODEL_ID)
    theta_wide = theta_wide.reindex(aligned_index)
    sub_wide = sub_wide.reindex(aligned_index)
    lin_wide = lin_wide.reindex(aligned_index)

    required_theta_columns = [f"theta_{benchmark_id}" for benchmark_id in benchmark_ids]
    required_sub_columns = [f"sub_{benchmark_id}" for benchmark_id in benchmark_ids]
    required_lin_columns = [f"lin_{benchmark_id}" for benchmark_id in benchmark_ids]
    complete_mask = (
        theta_wide[required_theta_columns].notna().all(axis=1)
        & sub_wide[required_sub_columns].notna().all(axis=1)
        & lin_wide[required_lin_columns].notna().all(axis=1)
    )
    grand_summary = pd.DataFrame(
        {
            MODEL_ID: pd.Series(model_ids, dtype="string"),
            GRAND_SUB: pd.Series(pd.NA, index=range(len(model_ids)), dtype="Float64"),
            GRAND_LIN: pd.Series(pd.NA, index=range(len(model_ids)), dtype="Float64"),
        }
    )
    complete_ids = complete_mask.index[complete_mask].astype("string").tolist()
    if complete_ids:
        grand_summary = grand_summary.set_index(MODEL_ID)
        grand_summary.loc[complete_ids, GRAND_SUB] = (
            sub_wide.loc[complete_ids, required_sub_columns].mean(axis=1).to_numpy()
        )
        grand_summary.loc[complete_ids, GRAND_LIN] = (
            lin_wide.loc[complete_ids, required_lin_columns].mean(axis=1).to_numpy()
        )
        grand_summary = grand_summary.reset_index()

    rows: list[pd.DataFrame] = []
    for benchmark_id in benchmark_ids:
        benchmark_rows = features_marginal.loc[
            features_marginal[BENCHMARK_ID] == benchmark_id,
            [
                BENCHMARK_ID,
                MODEL_ID,
                MARGINAL_THETA_SE,
                REDUCED_SUBSCORE,
                LINEAR_PREDICTION,
            ],
        ].copy()
        benchmark_rows = benchmark_rows.merge(grand_summary, on=MODEL_ID, how="left")
        benchmark_rows = benchmark_rows.merge(theta_wide.reset_index(), on=MODEL_ID, how="left")
        rows.append(benchmark_rows)

    if not rows:
        return _empty_features_joint_frame(benchmark_ids)
    frame = pd.concat(rows, ignore_index=True)
    return frame.astype(_empty_features_joint_frame(benchmark_ids).dtypes.to_dict())


def _predict_model_type(
    *,
    benchmark_id: str,
    model_ids: list[str],
    feature_frame: pd.DataFrame,
    model_type: str,
    model_payload: dict[str, Any],
    bundle: LoadedCalibrationBundle,
    missing_reason: str,
) -> pd.DataFrame:
    feature_frame = feature_frame.reset_index(drop=True)
    if not model_payload["available"]:
        return pd.DataFrame(
            {
                BENCHMARK_ID: pd.Series([benchmark_id] * len(model_ids), dtype="string"),
                MODEL_ID: pd.Series(model_ids, dtype="string"),
                MODEL_TYPE: pd.Series([model_type] * len(model_ids), dtype="string"),
                PREDICTED_SCORE: pd.Series(pd.NA, index=range(len(model_ids)), dtype="Float64"),
                PREDICTION_AVAILABLE: pd.Series([False] * len(model_ids), dtype=bool),
                PREDICTION_MISSING_REASON: pd.Series(
                    [model_payload["skip_reason"]] * len(model_ids),
                    dtype="string",
                ),
            }
        ).astype(_empty_predictions_frame().dtypes.to_dict())

    fitted_model = _load_prediction_model(bundle, model_payload)
    if list(fitted_model.feature_names) != list(model_payload["feature_columns"]):
        raise ValueError(
            f"{benchmark_id} {model_type} feature schema does not match the calibration bundle"
        )
    complete_mask = feature_frame.loc[:, fitted_model.feature_names].notna().all(axis=1)
    predictions = pd.Series(pd.NA, index=feature_frame.index, dtype="Float64")
    if complete_mask.any():
        predictions.loc[complete_mask] = fitted_model.predict(
            feature_frame.loc[complete_mask, fitted_model.feature_names]
        )
    reason = pd.Series(missing_reason, index=feature_frame.index, dtype="string")
    reason.loc[complete_mask] = pd.NA
    frame = pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(feature_frame[BENCHMARK_ID].to_numpy(), dtype="string"),
            MODEL_ID: pd.Series(feature_frame[MODEL_ID].to_numpy(), dtype="string"),
            MODEL_TYPE: pd.Series([model_type] * len(feature_frame.index), dtype="string"),
            PREDICTED_SCORE: predictions,
            PREDICTION_AVAILABLE: complete_mask.astype(bool).to_numpy(),
            PREDICTION_MISSING_REASON: reason,
        }
    )
    return frame.astype(_empty_predictions_frame().dtypes.to_dict())


def _load_prediction_model(
    bundle: LoadedCalibrationBundle,
    model_payload: dict[str, Any],
) -> FittedGAM:
    model_path = bundle.resolve_path(model_payload["model_path"])
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"required fitted GAM is missing: {model_path}")
    return load_gam(model_path)


def _select_best_available_predictions(
    *,
    predictions: pd.DataFrame,
    benchmark_ids: list[str],
    model_ids: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark_id in benchmark_ids:
        benchmark_predictions = predictions.loc[predictions[BENCHMARK_ID] == benchmark_id].copy()
        for model_id in model_ids:
            model_predictions = benchmark_predictions.loc[
                benchmark_predictions[MODEL_ID] == model_id
            ].copy()
            chosen_row = None
            for preferred_model_type in (JOINT_MODEL, MARGINAL_MODEL):
                candidate_rows = model_predictions.loc[
                    (model_predictions[MODEL_TYPE] == preferred_model_type)
                    & model_predictions[PREDICTION_AVAILABLE]
                ]
                if not candidate_rows.empty:
                    chosen_row = candidate_rows.iloc[0]
                    break
            if chosen_row is None and not model_predictions.empty:
                if not model_predictions.loc[model_predictions[MODEL_TYPE] == JOINT_MODEL].empty:
                    chosen_row = model_predictions.loc[
                        model_predictions[MODEL_TYPE] == JOINT_MODEL
                    ].iloc[0]
                else:
                    chosen_row = model_predictions.iloc[0]
            rows.append(
                {
                    BENCHMARK_ID: benchmark_id,
                    MODEL_ID: model_id,
                    PREDICTED_SCORE: (pd.NA if chosen_row is None else chosen_row[PREDICTED_SCORE]),
                    SELECTED_MODEL_TYPE: (pd.NA if chosen_row is None else chosen_row[MODEL_TYPE]),
                    PREDICTION_AVAILABLE: (
                        False if chosen_row is None else bool(chosen_row[PREDICTION_AVAILABLE])
                    ),
                    PREDICTION_MISSING_REASON: (
                        pd.NA if chosen_row is None else chosen_row[PREDICTION_MISSING_REASON]
                    ),
                }
            )
    return pd.DataFrame.from_records(rows).astype(_empty_best_predictions_frame().dtypes.to_dict())


def _build_prediction_report(
    *,
    calibration_bundle_path: Path,
    predictions: pd.DataFrame,
    predictions_best_available: pd.DataFrame,
    theta_estimates: pd.DataFrame,
    benchmark_reports: dict[str, Any],
    warnings: list[dict[str, Any]],
    input_benchmark_ids: list[str],
    calibrated_benchmark_ids: list[str],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calibration_bundle_manifest": str(calibration_bundle_path),
        "warnings": warnings,
        "counts": {
            "input_benchmark_count": len(input_benchmark_ids),
            "calibrated_benchmark_count": len(calibrated_benchmark_ids),
            "theta_rows": int(len(theta_estimates.index)),
            "prediction_rows": int(len(predictions.index)),
            "best_available_rows": int(len(predictions_best_available.index)),
            "best_available_non_null_predictions": int(
                predictions_best_available[PREDICTED_SCORE].notna().sum()
            ),
        },
        "benchmarks": benchmark_reports,
        "prediction_availability": {
            "by_model_type": {
                model_type: int(
                    predictions.loc[
                        predictions[MODEL_TYPE] == model_type,
                        PREDICTION_AVAILABLE,
                    ].sum()
                )
                for model_type in (MARGINAL_MODEL, JOINT_MODEL)
            },
            "best_available_non_null_by_benchmark": {
                benchmark_id: int(
                    predictions_best_available.loc[
                        (predictions_best_available[BENCHMARK_ID] == benchmark_id)
                        & predictions_best_available[PREDICTED_SCORE].notna()
                    ].shape[0]
                )
                for benchmark_id in calibrated_benchmark_ids
            },
        },
    }


def _write_prediction_artifacts(
    *,
    run_root: Path,
    theta_estimates: pd.DataFrame,
    features_marginal: pd.DataFrame,
    features_joint: pd.DataFrame,
    predictions: pd.DataFrame,
    predictions_best_available: pd.DataFrame,
    prediction_report: dict[str, Any],
) -> dict[str, Path]:
    stage_dir = run_root / "artifacts" / "01_predict"
    return {
        "theta_estimates": write_parquet(theta_estimates, stage_dir / "theta_estimates.parquet"),
        "features_marginal": write_parquet(
            features_marginal,
            stage_dir / "features_marginal.parquet",
        ),
        "features_joint": write_parquet(features_joint, stage_dir / "features_joint.parquet"),
        "predictions": write_parquet(predictions, stage_dir / "predictions.parquet"),
        "predictions_best_available": write_parquet(
            predictions_best_available,
            stage_dir / "predictions_best_available.parquet",
        ),
        "prediction_report": write_json(prediction_report, stage_dir / "prediction_report.json"),
    }


def _validate_reconstruction_payload(
    *,
    root: Path,
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    for model_type in (MARGINAL_MODEL, JOINT_MODEL):
        model_payload = dict(payload[model_type])
        if model_payload["available"]:
            _require_existing(root / model_payload["model_path"])
            _require_existing(root / model_payload["metadata_path"])
        validated[model_type] = model_payload
    return validated


def _resolve_calibration_manifest_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_file():
        return resolved
    candidate = resolved / CALIBRATION_BUNDLE_DIRNAME / "manifest.json"
    if candidate.exists():
        return candidate
    if (resolved / "manifest.json").exists():
        return resolved / "manifest.json"
    raise FileNotFoundError(
        "could not locate a calibration bundle manifest at "
        f"{resolved}, {resolved / 'manifest.json'}, or {candidate}"
    )


def _require_existing(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required calibration bundle artifact is missing: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _empty_theta_estimates_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            THETA_HAT: pd.Series(dtype="Float64"),
            THETA_SE: pd.Series(dtype="Float64"),
            THETA_METHOD: pd.Series(dtype="string"),
            OBSERVED_ITEM_COUNT: pd.Series(dtype="Int64"),
            SELECTED_ITEM_COUNT: pd.Series(dtype="Int64"),
            REDUCED_SCORE: pd.Series(dtype="Float64"),
            MISSING_HEAVY: pd.Series(dtype=bool),
            SATURATED: pd.Series(dtype=bool),
            SATURATION_SIDE: pd.Series(dtype="string"),
            RESPONSE_PATTERN: pd.Series(dtype="string"),
        }
    )


def _empty_features_marginal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            MARGINAL_THETA: pd.Series(dtype="Float64"),
            MARGINAL_THETA_SE: pd.Series(dtype="Float64"),
            THETA_METHOD: pd.Series(dtype="string"),
            REDUCED_SUBSCORE: pd.Series(dtype="Float64"),
            LINEAR_PREDICTION: pd.Series(dtype="Float64"),
            LINEAR_PREDICTION_MISSING_REASON: pd.Series(dtype="string"),
            "reduced_item_coverage": pd.Series(dtype="Float64"),
            OBSERVED_ITEM_COUNT: pd.Series(dtype="Int64"),
            SELECTED_ITEM_COUNT: pd.Series(dtype="Int64"),
        }
    )


def _empty_features_joint_frame(benchmark_ids: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
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


def _empty_predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            MODEL_TYPE: pd.Series(dtype="string"),
            PREDICTED_SCORE: pd.Series(dtype="Float64"),
            PREDICTION_AVAILABLE: pd.Series(dtype=bool),
            PREDICTION_MISSING_REASON: pd.Series(dtype="string"),
        }
    )


def _empty_best_predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            PREDICTED_SCORE: pd.Series(dtype="Float64"),
            SELECTED_MODEL_TYPE: pd.Series(dtype="string"),
            PREDICTION_AVAILABLE: pd.Series(dtype=bool),
            PREDICTION_MISSING_REASON: pd.Series(dtype="string"),
        }
    )


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
