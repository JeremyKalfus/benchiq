"""Stage-09 GAM-based score reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from benchiq.io.load import Bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.logging import update_manifest
from benchiq.reconstruct.features import (
    GRAND_LIN,
    GRAND_SUB,
    MARGINAL_TARGET,
    MARGINAL_THETA,
    MARGINAL_THETA_SE,
    FeatureTableResult,
)
from benchiq.reconstruct.gam import (
    DEFAULT_LAM_GRID,
    cross_validate_gam,
    write_gam_artifacts,
)
from benchiq.reconstruct.linear_predictor import LINEAR_PREDICTION, REDUCED_SUBSCORE
from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID, SPLIT

MODEL_TYPE = "model_type"
ACTUAL_SCORE = "actual_score"
PREDICTED_SCORE = "predicted_score"
BASELINE_PREDICTION = "baseline_prediction"
RESIDUAL = "residual"
BASELINE_RESIDUAL = "baseline_residual"
RMSE = "rmse"
MAE = "mae"
PEARSON_R = "pearson_r"
SPEARMAN_R = "spearman_r"
BASELINE_RMSE = "baseline_rmse"
BASELINE_MAE = "baseline_mae"
ROW_COUNT = "row_count"
SKIPPED = "skipped"
SKIP_REASON = "skip_reason"

MARGINAL_MODEL = "marginal"
JOINT_MODEL = "joint"
DEFAULT_RECONSTRUCT_CV_FOLDS = 5
DEFAULT_RECONSTRUCT_N_SPLINES = 20


@dataclass(slots=True)
class BenchmarkReconstructionResult:
    """Per-benchmark stage-09 reconstruction outputs."""

    benchmark_id: str
    predictions: pd.DataFrame
    reconstruction_report: dict[str, Any]
    artifact_paths: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReconstructionResult:
    """Stage-09 reconstruction outputs."""

    benchmarks: dict[str, BenchmarkReconstructionResult]
    reconstruction_summary: pd.DataFrame
    reconstruction_report: dict[str, Any]
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None


def reconstruct_scores(
    bundle: Bundle,
    feature_result: FeatureTableResult,
    *,
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
    cv_folds: int = DEFAULT_RECONSTRUCT_CV_FOLDS,
    n_splines: int = DEFAULT_RECONSTRUCT_N_SPLINES,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> ReconstructionResult:
    """Fit marginal and joint GAM reconstruction models from stage-08 feature tables."""

    benchmark_ids = list(
        feature_result.feature_report["parameters"].get(
            "benchmark_ids",
            sorted(
                feature_result.features_marginal[BENCHMARK_ID].dropna().astype("string").unique()
            ),
        )
    )
    benchmark_results: dict[str, BenchmarkReconstructionResult] = {}
    summary_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for benchmark_index, benchmark_id in enumerate(benchmark_ids):
        benchmark_result = reconstruct_benchmark(
            feature_result,
            benchmark_id=benchmark_id,
            lam_grid=lam_grid,
            cv_folds=cv_folds,
            n_splines=n_splines,
            random_seed=bundle.config.random_seed + benchmark_index,
        )
        benchmark_results[benchmark_id] = benchmark_result
        summary_rows.extend(benchmark_result.reconstruction_report["summary_rows"])
        for warning in benchmark_result.reconstruction_report["warnings"]:
            warnings.append({"benchmark_id": benchmark_id, **warning})

    reconstruction_summary = _build_reconstruction_summary(summary_rows)
    reconstruction_report = _build_reconstruction_report(
        benchmark_ids=benchmark_ids,
        reconstruction_summary=reconstruction_summary,
        benchmark_results=benchmark_results,
        warnings=warnings,
        lam_grid=lam_grid,
        cv_folds=cv_folds,
        n_splines=n_splines,
    )
    result = ReconstructionResult(
        benchmarks=benchmark_results,
        reconstruction_summary=reconstruction_summary,
        reconstruction_report=reconstruction_report,
    )

    run_root, manifest_path = _resolve_run_root(bundle, out_dir=out_dir, run_id=run_id)
    if run_root is not None:
        artifact_paths = _write_reconstruction_artifacts(result, run_root=run_root)
        result.artifact_paths = artifact_paths
        result.manifest_path = manifest_path
        if manifest_path is not None:
            update_manifest(
                manifest_path,
                {
                    "artifacts": {
                        "09_reconstruct": {
                            "reconstruction_summary": str(artifact_paths["reconstruction_summary"]),
                            "reconstruction_report": str(artifact_paths["reconstruction_report"]),
                            "per_benchmark": {
                                benchmark_id: _stringify_paths(paths)
                                for benchmark_id, paths in sorted(
                                    artifact_paths["per_benchmark"].items()
                                )
                            },
                        }
                    }
                },
            )
    return result


def reconstruct_benchmark(
    feature_result: FeatureTableResult,
    *,
    benchmark_id: str,
    lam_grid: Sequence[float],
    cv_folds: int,
    n_splines: int,
    random_seed: int,
) -> BenchmarkReconstructionResult:
    """Fit marginal and optional joint GAMs for one benchmark."""

    marginal_features = feature_result.features_marginal.loc[
        feature_result.features_marginal[BENCHMARK_ID] == benchmark_id
    ].copy()
    marginal_result = _fit_model_type(
        benchmark_id=benchmark_id,
        model_type=MARGINAL_MODEL,
        feature_frame=marginal_features,
        feature_columns=[MARGINAL_THETA, MARGINAL_THETA_SE, REDUCED_SUBSCORE, LINEAR_PREDICTION],
        lam_grid=lam_grid,
        cv_folds=cv_folds,
        n_splines=n_splines,
        random_seed=random_seed,
        skipped_reason=None,
    )

    if feature_result.feature_report["joint"]["skipped"]:
        joint_result = _skipped_model_result(
            benchmark_id=benchmark_id,
            model_type=JOINT_MODEL,
            feature_frame=_empty_prediction_frame(),
            skipped_reason=feature_result.feature_report["joint"]["skip_reason"],
        )
    else:
        joint_features = feature_result.features_joint.loc[
            feature_result.features_joint[BENCHMARK_ID] == benchmark_id
        ].copy()
        joint_theta_columns = sorted(
            column
            for column in feature_result.features_joint.columns
            if column.startswith("theta_")
        )
        joint_result = _fit_model_type(
            benchmark_id=benchmark_id,
            model_type=JOINT_MODEL,
            feature_frame=joint_features,
            feature_columns=[
                *joint_theta_columns,
                REDUCED_SUBSCORE,
                GRAND_SUB,
                LINEAR_PREDICTION,
                GRAND_LIN,
            ],
            lam_grid=lam_grid,
            cv_folds=cv_folds,
            n_splines=n_splines,
            random_seed=random_seed + 1000,
            skipped_reason=None,
        )

    predictions = pd.concat(
        [marginal_result["predictions"], joint_result["predictions"]],
        ignore_index=True,
    )
    if predictions.empty:
        predictions = _empty_prediction_frame()
    else:
        predictions = predictions.astype(_empty_prediction_frame().dtypes.to_dict())

    summary_rows = marginal_result["summary_rows"] + joint_result["summary_rows"]
    warnings = marginal_result["warnings"] + joint_result["warnings"]
    reconstruction_report = {
        "benchmark_id": benchmark_id,
        "warnings": warnings,
        "marginal": marginal_result["report"],
        "joint": joint_result["report"],
        "summary_rows": summary_rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return BenchmarkReconstructionResult(
        benchmark_id=benchmark_id,
        predictions=predictions,
        reconstruction_report=reconstruction_report,
    )


def _fit_model_type(
    *,
    benchmark_id: str,
    model_type: str,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    lam_grid: Sequence[float],
    cv_folds: int,
    n_splines: int,
    random_seed: int,
    skipped_reason: str | None,
) -> dict[str, Any]:
    if skipped_reason is not None:
        return _skipped_model_result(
            benchmark_id=benchmark_id,
            model_type=model_type,
            feature_frame=feature_frame,
            skipped_reason=skipped_reason,
        )

    if feature_frame.empty:
        return _skipped_model_result(
            benchmark_id=benchmark_id,
            model_type=model_type,
            feature_frame=feature_frame,
            skipped_reason="no_feature_rows_available",
        )

    available = feature_frame.dropna(
        subset=[MODEL_ID, SPLIT, MARGINAL_TARGET, *feature_columns],
    ).copy()
    if available.empty:
        return _skipped_model_result(
            benchmark_id=benchmark_id,
            model_type=model_type,
            feature_frame=feature_frame,
            skipped_reason="no_complete_feature_rows_available",
        )

    split_counts = (
        available[SPLIT]
        .astype("string")
        .value_counts(dropna=False)
        .sort_index()
        .astype(int)
        .to_dict()
    )
    train_rows = available.loc[available[SPLIT] == "train"].copy()
    val_rows = available.loc[available[SPLIT] == "val"].copy()
    test_rows = available.loc[available[SPLIT] == "test"].copy()

    if len(train_rows.index) < 4:
        return _skipped_model_result(
            benchmark_id=benchmark_id,
            model_type=model_type,
            feature_frame=available,
            skipped_reason="insufficient_train_rows_for_gam",
        )

    effective_cv_folds = min(int(cv_folds), int(len(train_rows.index)))
    if effective_cv_folds < 2:
        return _skipped_model_result(
            benchmark_id=benchmark_id,
            model_type=model_type,
            feature_frame=available,
            skipped_reason="insufficient_train_rows_for_cv",
        )
    effective_n_splines = min(int(n_splines), int(len(train_rows.index)))
    if effective_n_splines < 4:
        return _skipped_model_result(
            benchmark_id=benchmark_id,
            model_type=model_type,
            feature_frame=available,
            skipped_reason="insufficient_train_rows_for_splines",
        )

    train_mean = float(train_rows[MARGINAL_TARGET].astype(float).mean())
    cv_result = cross_validate_gam(
        train_rows.loc[:, feature_columns],
        train_rows[MARGINAL_TARGET],
        lam_grid=lam_grid,
        cv_folds=effective_cv_folds,
        random_seed=random_seed,
        feature_names=feature_columns,
        target_name=MARGINAL_TARGET,
        n_splines=effective_n_splines,
        X_test=test_rows.loc[:, feature_columns] if not test_rows.empty else None,
        y_test=test_rows[MARGINAL_TARGET] if not test_rows.empty else None,
    )

    prediction_rows = available.loc[:, [BENCHMARK_ID, MODEL_ID, SPLIT, MARGINAL_TARGET]].copy()
    prediction_rows[MODEL_TYPE] = model_type
    prediction_rows[PREDICTED_SCORE] = cv_result.best_model.predict(
        available.loc[:, feature_columns]
    )
    prediction_rows[BASELINE_PREDICTION] = train_mean
    prediction_rows[ACTUAL_SCORE] = prediction_rows[MARGINAL_TARGET].astype("Float64")
    prediction_rows[RESIDUAL] = prediction_rows[PREDICTED_SCORE].astype(float) - prediction_rows[
        ACTUAL_SCORE
    ].astype(float)
    prediction_rows[BASELINE_RESIDUAL] = prediction_rows[BASELINE_PREDICTION].astype(
        float
    ) - prediction_rows[ACTUAL_SCORE].astype(float)
    predictions = prediction_rows.loc[
        :,
        [
            BENCHMARK_ID,
            MODEL_ID,
            SPLIT,
            MODEL_TYPE,
            ACTUAL_SCORE,
            PREDICTED_SCORE,
            BASELINE_PREDICTION,
            RESIDUAL,
            BASELINE_RESIDUAL,
        ],
    ].copy()
    predictions[MODEL_TYPE] = predictions[MODEL_TYPE].astype("string")
    predictions[ACTUAL_SCORE] = pd.Series(predictions[ACTUAL_SCORE], dtype="Float64")
    predictions[PREDICTED_SCORE] = pd.Series(predictions[PREDICTED_SCORE], dtype="Float64")
    predictions[BASELINE_PREDICTION] = pd.Series(predictions[BASELINE_PREDICTION], dtype="Float64")
    predictions[RESIDUAL] = pd.Series(predictions[RESIDUAL], dtype="Float64")
    predictions[BASELINE_RESIDUAL] = pd.Series(predictions[BASELINE_RESIDUAL], dtype="Float64")
    predictions = predictions.astype(
        {
            BENCHMARK_ID: "string",
            MODEL_ID: "string",
            SPLIT: "string",
            MODEL_TYPE: "string",
        }
    )

    summary_rows = [
        _split_metrics_row(
            benchmark_id=benchmark_id,
            model_type=model_type,
            split_name=split_name,
            prediction_rows=predictions.loc[predictions[SPLIT] == split_name].copy(),
            skipped=False,
            skip_reason=None,
        )
        for split_name in ["train", "val", "test"]
    ]
    warnings: list[dict[str, Any]] = []
    if test_rows.empty:
        warnings.append(
            {
                "code": "missing_test_rows",
                "message": f"{benchmark_id} {model_type} reconstruction had no held-out test rows.",
                "severity": "warning",
            }
        )

    return {
        "predictions": predictions,
        "summary_rows": summary_rows,
        "warnings": warnings,
        "report": {
            "benchmark_id": benchmark_id,
            "model_type": model_type,
            "skipped": False,
            "skip_reason": None,
            "feature_columns": feature_columns,
            "train_row_count": int(len(train_rows.index)),
            "val_row_count": int(len(val_rows.index)),
            "test_row_count": int(len(test_rows.index)),
            "available_row_count": int(len(available.index)),
            "split_counts": split_counts,
            "cv_report": cv_result.cv_report,
            "metrics": {
                row[SPLIT]: {
                    RMSE: row[RMSE],
                    MAE: row[MAE],
                    PEARSON_R: row[PEARSON_R],
                    SPEARMAN_R: row[SPEARMAN_R],
                    BASELINE_RMSE: row[BASELINE_RMSE],
                    BASELINE_MAE: row[BASELINE_MAE],
                    ROW_COUNT: row[ROW_COUNT],
                }
                for row in summary_rows
            },
            "_cv_result": cv_result,
        },
        "cv_result": cv_result,
    }


def _skipped_model_result(
    *,
    benchmark_id: str,
    model_type: str,
    feature_frame: pd.DataFrame,
    skipped_reason: str,
) -> dict[str, Any]:
    split_counts = (
        feature_frame[SPLIT]
        .astype("string")
        .value_counts(dropna=False)
        .sort_index()
        .astype(int)
        .to_dict()
        if not feature_frame.empty and SPLIT in feature_frame.columns
        else {}
    )
    warning = {
        "code": f"{model_type}_reconstruction_skipped",
        "message": f"{benchmark_id} {model_type} reconstruction skipped: {skipped_reason}",
        "severity": "warning",
    }
    summary_rows = [
        {
            BENCHMARK_ID: benchmark_id,
            MODEL_TYPE: model_type,
            SPLIT: "all",
            ROW_COUNT: int(len(feature_frame.index)),
            RMSE: pd.NA,
            MAE: pd.NA,
            PEARSON_R: pd.NA,
            SPEARMAN_R: pd.NA,
            BASELINE_RMSE: pd.NA,
            BASELINE_MAE: pd.NA,
            SKIPPED: True,
            SKIP_REASON: skipped_reason,
        }
    ]
    return {
        "predictions": _empty_prediction_frame(),
        "summary_rows": summary_rows,
        "warnings": [warning],
        "report": {
            "benchmark_id": benchmark_id,
            "model_type": model_type,
            "skipped": True,
            "skip_reason": skipped_reason,
            "feature_columns": [],
            "train_row_count": int(split_counts.get("train", 0)),
            "val_row_count": int(split_counts.get("val", 0)),
            "test_row_count": int(split_counts.get("test", 0)),
            "available_row_count": int(len(feature_frame.index)),
            "split_counts": split_counts,
            "cv_report": None,
            "metrics": {},
        },
        "cv_result": None,
    }


def _split_metrics_row(
    *,
    benchmark_id: str,
    model_type: str,
    split_name: str,
    prediction_rows: pd.DataFrame,
    skipped: bool,
    skip_reason: str | None,
) -> dict[str, Any]:
    if prediction_rows.empty:
        return {
            BENCHMARK_ID: benchmark_id,
            MODEL_TYPE: model_type,
            SPLIT: split_name,
            ROW_COUNT: 0,
            RMSE: pd.NA,
            MAE: pd.NA,
            PEARSON_R: pd.NA,
            SPEARMAN_R: pd.NA,
            BASELINE_RMSE: pd.NA,
            BASELINE_MAE: pd.NA,
            SKIPPED: skipped,
            SKIP_REASON: skip_reason,
        }

    actual = prediction_rows[ACTUAL_SCORE].astype(float).to_numpy()
    predicted = prediction_rows[PREDICTED_SCORE].astype(float).to_numpy()
    baseline = prediction_rows[BASELINE_PREDICTION].astype(float).to_numpy()
    return {
        BENCHMARK_ID: benchmark_id,
        MODEL_TYPE: model_type,
        SPLIT: split_name,
        ROW_COUNT: int(len(prediction_rows.index)),
        RMSE: _rmse(actual, predicted),
        MAE: _mae(actual, predicted),
        PEARSON_R: _correlation(actual, predicted, method="pearson"),
        SPEARMAN_R: _correlation(actual, predicted, method="spearman"),
        BASELINE_RMSE: _rmse(actual, baseline),
        BASELINE_MAE: _mae(actual, baseline),
        SKIPPED: skipped,
        SKIP_REASON: skip_reason,
    }


def _build_reconstruction_summary(summary_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not summary_rows:
        return _empty_reconstruction_summary_frame()
    summary = pd.DataFrame.from_records(summary_rows)
    return summary.astype(_empty_reconstruction_summary_frame().dtypes.to_dict())


def _build_reconstruction_report(
    *,
    benchmark_ids: list[str],
    reconstruction_summary: pd.DataFrame,
    benchmark_results: dict[str, BenchmarkReconstructionResult],
    warnings: list[dict[str, Any]],
    lam_grid: Sequence[float],
    cv_folds: int,
    n_splines: int,
) -> dict[str, Any]:
    marginal_test_rmse = {}
    joint_test_rmse = {}
    joint_skips = {}
    for benchmark_id in benchmark_ids:
        benchmark_report = benchmark_results[benchmark_id].reconstruction_report
        marginal_metrics = benchmark_report["marginal"]["metrics"].get("test", {})
        joint_metrics = benchmark_report["joint"]["metrics"].get("test", {})
        marginal_test_rmse[benchmark_id] = marginal_metrics.get(RMSE)
        joint_test_rmse[benchmark_id] = joint_metrics.get(RMSE)
        if benchmark_report["joint"]["skipped"]:
            joint_skips[benchmark_id] = benchmark_report["joint"]["skip_reason"]

    return {
        "warnings": warnings,
        "parameters": {
            "lam_grid": list(map(float, lam_grid)),
            "cv_folds": int(cv_folds),
            "n_splines": int(n_splines),
            "benchmark_ids": benchmark_ids,
        },
        "counts": {
            "benchmark_count": len(benchmark_ids),
            "summary_rows": int(len(reconstruction_summary.index)),
            "joint_skipped_benchmark_count": len(joint_skips),
        },
        "rmse": {
            "marginal_test_by_benchmark": marginal_test_rmse,
            "joint_test_by_benchmark": joint_test_rmse,
        },
        "joint_skips": joint_skips,
        "benchmarks": {
            benchmark_id: result.reconstruction_report
            for benchmark_id, result in sorted(benchmark_results.items())
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_reconstruction_artifacts(
    result: ReconstructionResult,
    *,
    run_root: Path,
) -> dict[str, Any]:
    stage_dir = run_root / "artifacts" / "09_reconstruct"
    per_benchmark_paths: dict[str, dict[str, Any]] = {}
    for benchmark_id, benchmark_result in sorted(result.benchmarks.items()):
        benchmark_dir = stage_dir / "per_benchmark" / benchmark_id
        benchmark_paths: dict[str, Any] = {
            "predictions": write_parquet(
                benchmark_result.predictions,
                benchmark_dir / "predictions.parquet",
            ),
            "reconstruction_report": write_json(
                _json_safe(_sanitize_benchmark_report(benchmark_result.reconstruction_report)),
                benchmark_dir / "reconstruction_report.json",
            ),
        }
        for model_type in [MARGINAL_MODEL, JOINT_MODEL]:
            model_report = benchmark_result.reconstruction_report[model_type]
            if model_report["skipped"]:
                benchmark_paths[model_type] = {"skipped": True}
                continue
            model_dir = benchmark_dir / model_type
            cv_result = benchmark_result.reconstruction_report[model_type]["_cv_result"]
            model_paths = {
                **write_gam_artifacts(cv_result, out_dir=model_dir),
                "plots": _write_reconstruction_plots(
                    benchmark_result.predictions.loc[
                        benchmark_result.predictions[MODEL_TYPE] == model_type
                    ].copy(),
                    out_dir=model_dir / "plots",
                ),
            }
            benchmark_paths[model_type] = model_paths
        per_benchmark_paths[benchmark_id] = benchmark_paths

    summary_path = write_parquet(
        result.reconstruction_summary,
        stage_dir / "reconstruction_summary.parquet",
    )
    report_path = write_json(
        _json_safe(_report_without_cv_objects(result.reconstruction_report)),
        stage_dir / "reconstruction_report.json",
    )
    return {
        "reconstruction_summary": summary_path,
        "reconstruction_report": report_path,
        "per_benchmark": per_benchmark_paths,
    }


def _write_reconstruction_plots(
    predictions: pd.DataFrame,
    *,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_rows = predictions.copy()
    if plot_rows.empty:
        return {}

    calibration_path = out_dir / "calibration.png"
    figure, axis = plt.subplots(figsize=(6, 4))
    for split_name, split_frame in plot_rows.groupby(SPLIT, sort=True):
        axis.scatter(
            split_frame[ACTUAL_SCORE].astype(float),
            split_frame[PREDICTED_SCORE].astype(float),
            label=str(split_name),
            alpha=0.75,
        )
    all_actual = plot_rows[ACTUAL_SCORE].astype(float)
    min_value = float(all_actual.min())
    max_value = float(all_actual.max())
    axis.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--")
    axis.set_xlabel("actual score")
    axis.set_ylabel("predicted score")
    axis.set_title("calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(calibration_path, dpi=150)
    plt.close(figure)

    predicted_vs_actual_path = out_dir / "predicted_vs_actual.png"
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(
        plot_rows[ACTUAL_SCORE].astype(float),
        plot_rows[PREDICTED_SCORE].astype(float),
        alpha=0.8,
    )
    axis.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--")
    axis.set_xlabel("actual score")
    axis.set_ylabel("predicted score")
    axis.set_title("predicted vs actual")
    figure.tight_layout()
    figure.savefig(predicted_vs_actual_path, dpi=150)
    plt.close(figure)

    residual_path = out_dir / "residual_histogram.png"
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(plot_rows[RESIDUAL].astype(float), bins=16, color="#4C72B0", alpha=0.85)
    axis.set_xlabel("prediction residual")
    axis.set_ylabel("count")
    axis.set_title("residual histogram")
    figure.tight_layout()
    figure.savefig(residual_path, dpi=150)
    plt.close(figure)

    return {
        "calibration_plot": calibration_path,
        "predicted_vs_actual_plot": predicted_vs_actual_path,
        "residual_histogram": residual_path,
    }


def _empty_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            SPLIT: pd.Series(dtype="string"),
            MODEL_TYPE: pd.Series(dtype="string"),
            ACTUAL_SCORE: pd.Series(dtype="Float64"),
            PREDICTED_SCORE: pd.Series(dtype="Float64"),
            BASELINE_PREDICTION: pd.Series(dtype="Float64"),
            RESIDUAL: pd.Series(dtype="Float64"),
            BASELINE_RESIDUAL: pd.Series(dtype="Float64"),
        }
    )


def _empty_reconstruction_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_TYPE: pd.Series(dtype="string"),
            SPLIT: pd.Series(dtype="string"),
            ROW_COUNT: pd.Series(dtype="Int64"),
            RMSE: pd.Series(dtype="Float64"),
            MAE: pd.Series(dtype="Float64"),
            PEARSON_R: pd.Series(dtype="Float64"),
            SPEARMAN_R: pd.Series(dtype="Float64"),
            BASELINE_RMSE: pd.Series(dtype="Float64"),
            BASELINE_MAE: pd.Series(dtype="Float64"),
            SKIPPED: pd.Series(dtype=bool),
            SKIP_REASON: pd.Series(dtype="string"),
        }
    )


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def _correlation(actual: np.ndarray, predicted: np.ndarray, *, method: str) -> float | pd.NA:
    if actual.shape[0] < 2:
        return pd.NA
    if np.allclose(actual, actual[0]) or np.allclose(predicted, predicted[0]):
        return pd.NA
    if method == "pearson":
        return float(pearsonr(actual, predicted).statistic)
    return float(spearmanr(actual, predicted).statistic)


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


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _json_safe(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _report_without_cv_objects(report: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        "warnings": report["warnings"],
        "parameters": report["parameters"],
        "counts": report["counts"],
        "rmse": report["rmse"],
        "joint_skips": report["joint_skips"],
        "generated_at": report["generated_at"],
        "benchmarks": {},
    }
    for benchmark_id, benchmark_report in report["benchmarks"].items():
        sanitized["benchmarks"][benchmark_id] = {
            "benchmark_id": benchmark_report["benchmark_id"],
            "warnings": benchmark_report["warnings"],
            "marginal": _sanitize_model_report(benchmark_report["marginal"]),
            "joint": _sanitize_model_report(benchmark_report["joint"]),
            "summary_rows": benchmark_report["summary_rows"],
            "generated_at": benchmark_report["generated_at"],
        }
    return sanitized


def _sanitize_model_report(model_report: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(model_report)
    sanitized.pop("_cv_result", None)
    return sanitized


def _sanitize_benchmark_report(benchmark_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_report["benchmark_id"],
        "warnings": benchmark_report["warnings"],
        "marginal": _sanitize_model_report(benchmark_report["marginal"]),
        "joint": _sanitize_model_report(benchmark_report["joint"]),
        "summary_rows": benchmark_report["summary_rows"],
        "generated_at": benchmark_report["generated_at"],
    }
