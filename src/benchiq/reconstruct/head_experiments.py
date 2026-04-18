"""Reconstruction-head comparison harness for BenchIQ experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from benchiq.io.write import write_json, write_parquet
from benchiq.reconstruct.features import (
    GRAND_LIN,
    GRAND_SUB,
    MARGINAL_TARGET,
    MARGINAL_THETA,
    MARGINAL_THETA_SE,
    FeatureTableResult,
)
from benchiq.reconstruct.gam import DEFAULT_LAM_GRID, cross_validate_gam
from benchiq.reconstruct.linear_predictor import LINEAR_PREDICTION, REDUCED_SUBSCORE
from benchiq.reconstruct.reconstruction import ACTUAL_SCORE, JOINT_MODEL, MARGINAL_MODEL
from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID, SPLIT

METHOD_GAM = "gam"
METHOD_ELASTIC_NET = "elastic_net"
METHOD_XGBOOST = "xgboost"
DEFAULT_METHODS = (METHOD_GAM, METHOD_ELASTIC_NET, METHOD_XGBOOST)
DEFAULT_SEEDS = (7, 11, 19)
DEFAULT_ELASTIC_ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0)
DEFAULT_ELASTIC_L1_GRID = (0.1, 0.5, 0.9)
DEFAULT_XGBOOST_PARAM_GRID = (
    {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 80, "learning_rate": 0.1, "max_depth": 2},
    {"n_estimators": 80, "learning_rate": 0.1, "max_depth": 3},
)


@dataclass(slots=True)
class ReconstructionHeadExperimentResult:
    """Saved outputs from a reconstruction-head experiment run."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: pd.DataFrame
    report: dict[str, Any]
    artifact_paths: dict[str, Path] = field(default_factory=dict)


def run_reconstruction_head_experiments(
    feature_result: FeatureTableResult,
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
    cv_folds: int = 5,
    n_splines: int = 20,
    out_dir: str | Path | None = None,
) -> ReconstructionHeadExperimentResult:
    """Compare reconstruction heads on the same fixed feature tables."""

    benchmark_ids = sorted(
        feature_result.features_marginal[BENCHMARK_ID].dropna().astype("string").unique().tolist()
    )
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    skip_reasons: dict[str, str] = {}
    for benchmark_id in benchmark_ids:
        for model_type, feature_frame, feature_columns in _feature_sets_for_benchmark(
            feature_result,
            benchmark_id=benchmark_id,
        ):
            if feature_frame.empty:
                skip_reasons[f"{benchmark_id}:{model_type}"] = "no_feature_rows_available"
                continue
            for method in methods:
                resolved_method = _resolve_method(method)
                for seed in seeds:
                    experiment = _fit_head_once(
                        benchmark_id=benchmark_id,
                        model_type=model_type,
                        method=resolved_method,
                        seed=int(seed),
                        feature_frame=feature_frame,
                        feature_columns=feature_columns,
                        lam_grid=lam_grid,
                        cv_folds=cv_folds,
                        n_splines=n_splines,
                    )
                    metrics_rows.extend(experiment["metrics_rows"])
                    prediction_rows.extend(experiment["prediction_rows"])

    metrics = _build_metrics_frame(metrics_rows)
    predictions = _build_predictions_frame(prediction_rows)
    summary = _build_summary_frame(metrics)
    report = _build_report(
        metrics=metrics,
        summary=summary,
        methods=methods,
        seeds=seeds,
        skip_reasons=skip_reasons,
    )
    result = ReconstructionHeadExperimentResult(
        metrics=metrics,
        predictions=predictions,
        summary=summary,
        report=report,
    )
    if out_dir is not None:
        result.artifact_paths = _write_experiment_artifacts(result, out_dir=Path(out_dir))
    return result


def _feature_sets_for_benchmark(
    feature_result: FeatureTableResult,
    *,
    benchmark_id: str,
) -> list[tuple[str, pd.DataFrame, list[str]]]:
    marginal = feature_result.features_marginal.loc[
        feature_result.features_marginal[BENCHMARK_ID] == benchmark_id
    ].copy()
    feature_sets = [
        (
            MARGINAL_MODEL,
            marginal,
            [MARGINAL_THETA, MARGINAL_THETA_SE, REDUCED_SUBSCORE, LINEAR_PREDICTION],
        )
    ]
    if not feature_result.feature_report["joint"]["skipped"]:
        joint = feature_result.features_joint.loc[
            feature_result.features_joint[BENCHMARK_ID] == benchmark_id
        ].copy()
        joint_theta_columns = sorted(
            column
            for column in feature_result.features_joint.columns
            if column.startswith("theta_")
        )
        feature_sets.append(
            (
                JOINT_MODEL,
                joint,
                [
                    *joint_theta_columns,
                    REDUCED_SUBSCORE,
                    GRAND_SUB,
                    LINEAR_PREDICTION,
                    GRAND_LIN,
                ],
            )
        )
    return feature_sets


def _fit_head_once(
    *,
    benchmark_id: str,
    model_type: str,
    method: str,
    seed: int,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    lam_grid: Sequence[float],
    cv_folds: int,
    n_splines: int,
) -> dict[str, Any]:
    available = feature_frame.dropna(
        subset=[MODEL_ID, SPLIT, MARGINAL_TARGET, *feature_columns],
    ).copy()
    if available.empty:
        return {
            "metrics_rows": [
                _skip_metrics_row(
                    benchmark_id=benchmark_id,
                    model_type=model_type,
                    method=method,
                    seed=seed,
                    split_name="all",
                    skip_reason="no_complete_feature_rows_available",
                )
            ],
            "prediction_rows": [],
        }

    train_rows = available.loc[available[SPLIT] == "train"].copy()
    if len(train_rows.index) < 4:
        return {
            "metrics_rows": [
                _skip_metrics_row(
                    benchmark_id=benchmark_id,
                    model_type=model_type,
                    method=method,
                    seed=seed,
                    split_name="all",
                    skip_reason="insufficient_train_rows",
                )
            ],
            "prediction_rows": [],
        }

    effective_cv_folds = min(int(cv_folds), int(len(train_rows.index)))
    if effective_cv_folds < 2:
        return {
            "metrics_rows": [
                _skip_metrics_row(
                    benchmark_id=benchmark_id,
                    model_type=model_type,
                    method=method,
                    seed=seed,
                    split_name="all",
                    skip_reason="insufficient_train_rows_for_cv",
                )
            ],
            "prediction_rows": [],
        }

    train_x = train_rows.loc[:, feature_columns].astype(float).to_numpy()
    train_y = train_rows[MARGINAL_TARGET].astype(float).to_numpy()
    available_x = available.loc[:, feature_columns].astype(float).to_numpy()

    started_at = perf_counter()
    if method == METHOD_GAM:
        fitted = _fit_gam_head(
            train_x=train_x,
            train_y=train_y,
            available_x=available_x,
            lam_grid=lam_grid,
            effective_cv_folds=effective_cv_folds,
            feature_columns=feature_columns,
            n_splines=min(int(n_splines), int(len(train_rows.index))),
            seed=seed,
        )
    elif method == METHOD_ELASTIC_NET:
        fitted = _fit_elastic_net_head(
            train_x=train_x,
            train_y=train_y,
            available_x=available_x,
            effective_cv_folds=effective_cv_folds,
            seed=seed,
        )
    elif method == METHOD_XGBOOST:
        fitted = _fit_xgboost_head(
            train_x=train_x,
            train_y=train_y,
            available_x=available_x,
            effective_cv_folds=effective_cv_folds,
            seed=seed,
        )
    else:  # pragma: no cover - _resolve_method guards this
        raise ValueError(f"unsupported reconstruction method: {method}")
    runtime_seconds = perf_counter() - started_at

    prediction_frame = available.loc[:, [BENCHMARK_ID, MODEL_ID, SPLIT, MARGINAL_TARGET]].copy()
    prediction_frame["model_type"] = model_type
    prediction_frame["method"] = method
    prediction_frame["seed"] = seed
    prediction_frame[ACTUAL_SCORE] = prediction_frame[MARGINAL_TARGET].astype("Float64")
    prediction_frame["predicted_score"] = pd.Series(
        fitted["predictions"],
        dtype="Float64",
    )

    metrics_rows = [
        _metrics_row_from_predictions(
            benchmark_id=benchmark_id,
            model_type=model_type,
            method=method,
            seed=seed,
            split_name=split_name,
            prediction_rows=prediction_frame.loc[prediction_frame[SPLIT] == split_name].copy(),
            runtime_seconds=runtime_seconds,
            hyperparameters=fitted["hyperparameters"],
        )
        for split_name in ("train", "val", "test")
    ]
    prediction_rows = prediction_frame.loc[
        :,
        [
            BENCHMARK_ID,
            MODEL_ID,
            SPLIT,
            "model_type",
            "method",
            "seed",
            ACTUAL_SCORE,
            "predicted_score",
        ],
    ].to_dict(orient="records")
    return {
        "metrics_rows": metrics_rows,
        "prediction_rows": prediction_rows,
    }


def _fit_gam_head(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    available_x: np.ndarray,
    lam_grid: Sequence[float],
    effective_cv_folds: int,
    feature_columns: Sequence[str],
    n_splines: int,
    seed: int,
) -> dict[str, Any]:
    result = cross_validate_gam(
        train_x,
        train_y,
        lam_grid=lam_grid,
        cv_folds=effective_cv_folds,
        random_seed=seed,
        feature_names=feature_columns,
        target_name=MARGINAL_TARGET,
        n_splines=max(4, n_splines),
    )
    return {
        "predictions": result.best_model.predict(available_x),
        "hyperparameters": {"best_lam": result.best_lam, "cv_folds": effective_cv_folds},
    }


def _fit_elastic_net_head(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    available_x: np.ndarray,
    effective_cv_folds: int,
    seed: int,
) -> dict[str, Any]:
    folds = KFold(n_splits=effective_cv_folds, shuffle=True, random_state=seed)
    best_candidate: dict[str, Any] | None = None
    for alpha in DEFAULT_ELASTIC_ALPHA_GRID:
        for l1_ratio in DEFAULT_ELASTIC_L1_GRID:
            val_rmses: list[float] = []
            for train_index, val_index in folds.split(train_x):
                scaler = StandardScaler()
                fold_train_x = scaler.fit_transform(train_x[train_index])
                fold_val_x = scaler.transform(train_x[val_index])
                model = ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    max_iter=20000,
                )
                model.fit(fold_train_x, train_y[train_index])
                val_predictions = model.predict(fold_val_x)
                val_rmses.append(_rmse(train_y[val_index], val_predictions))
            candidate = {
                "alpha": float(alpha),
                "l1_ratio": float(l1_ratio),
                "mean_val_rmse": float(np.mean(val_rmses)),
                "max_val_rmse": float(np.max(val_rmses)),
            }
            if best_candidate is None or (
                candidate["mean_val_rmse"],
                candidate["max_val_rmse"],
                candidate["alpha"],
                candidate["l1_ratio"],
            ) < (
                best_candidate["mean_val_rmse"],
                best_candidate["max_val_rmse"],
                best_candidate["alpha"],
                best_candidate["l1_ratio"],
            ):
                best_candidate = candidate
    assert best_candidate is not None  # pragma: no cover - guarded by grids above
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    available_x_scaled = scaler.transform(available_x)
    model = ElasticNet(
        alpha=best_candidate["alpha"],
        l1_ratio=best_candidate["l1_ratio"],
        max_iter=20000,
    )
    model.fit(train_x_scaled, train_y)
    return {
        "predictions": model.predict(available_x_scaled),
        "hyperparameters": {
            "alpha": best_candidate["alpha"],
            "l1_ratio": best_candidate["l1_ratio"],
            "cv_folds": effective_cv_folds,
        },
    }


def _fit_xgboost_head(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    available_x: np.ndarray,
    effective_cv_folds: int,
    seed: int,
) -> dict[str, Any]:
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "xgboost is required for the tree-based reconstruction comparison"
        ) from exc

    folds = KFold(n_splits=effective_cv_folds, shuffle=True, random_state=seed)
    best_candidate: dict[str, Any] | None = None
    for params in DEFAULT_XGBOOST_PARAM_GRID:
        val_rmses: list[float] = []
        for train_index, val_index in folds.split(train_x):
            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
                subsample=1.0,
                colsample_bytree=1.0,
                verbosity=0,
                **params,
            )
            model.fit(train_x[train_index], train_y[train_index])
            val_predictions = model.predict(train_x[val_index])
            val_rmses.append(_rmse(train_y[val_index], val_predictions))
        normalized_params = {
            key: int(value) if key in {"n_estimators", "max_depth"} else float(value)
            for key, value in params.items()
        }
        candidate = {
            **normalized_params,
            "mean_val_rmse": float(np.mean(val_rmses)),
            "max_val_rmse": float(np.max(val_rmses)),
        }
        if best_candidate is None or (
            candidate["mean_val_rmse"],
            candidate["max_val_rmse"],
            candidate["max_depth"],
            candidate["learning_rate"],
            candidate["n_estimators"],
        ) < (
            best_candidate["mean_val_rmse"],
            best_candidate["max_val_rmse"],
            best_candidate["max_depth"],
            best_candidate["learning_rate"],
            best_candidate["n_estimators"],
        ):
            best_candidate = candidate
    assert best_candidate is not None  # pragma: no cover - guarded by grid above
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        subsample=1.0,
        colsample_bytree=1.0,
        verbosity=0,
        n_estimators=int(best_candidate["n_estimators"]),
        learning_rate=float(best_candidate["learning_rate"]),
        max_depth=int(best_candidate["max_depth"]),
    )
    model.fit(train_x, train_y)
    return {
        "predictions": model.predict(available_x),
        "hyperparameters": {
            "n_estimators": int(best_candidate["n_estimators"]),
            "learning_rate": float(best_candidate["learning_rate"]),
            "max_depth": int(best_candidate["max_depth"]),
            "cv_folds": effective_cv_folds,
        },
    }


def _metrics_row_from_predictions(
    *,
    benchmark_id: str,
    model_type: str,
    method: str,
    seed: int,
    split_name: str,
    prediction_rows: pd.DataFrame,
    runtime_seconds: float,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    if prediction_rows.empty:
        return _skip_metrics_row(
            benchmark_id=benchmark_id,
            model_type=model_type,
            method=method,
            seed=seed,
            split_name=split_name,
            skip_reason="no_rows_for_split",
            runtime_seconds=runtime_seconds,
            hyperparameters=hyperparameters,
        )
    actual = prediction_rows[ACTUAL_SCORE].astype(float).to_numpy()
    predicted = prediction_rows["predicted_score"].astype(float).to_numpy()
    return {
        BENCHMARK_ID: benchmark_id,
        "model_type": model_type,
        "method": method,
        "seed": seed,
        SPLIT: split_name,
        "row_count": int(len(prediction_rows.index)),
        "rmse": _rmse(actual, predicted),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "pearson_r": _safe_correlation(actual, predicted, method="pearson"),
        "spearman_r": _safe_correlation(actual, predicted, method="spearman"),
        "runtime_seconds": float(runtime_seconds),
        "skipped": False,
        "skip_reason": None,
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
    }


def _skip_metrics_row(
    *,
    benchmark_id: str,
    model_type: str,
    method: str,
    seed: int,
    split_name: str,
    skip_reason: str,
    runtime_seconds: float | None = None,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        BENCHMARK_ID: benchmark_id,
        "model_type": model_type,
        "method": method,
        "seed": seed,
        SPLIT: split_name,
        "row_count": 0,
        "rmse": pd.NA,
        "mae": pd.NA,
        "pearson_r": pd.NA,
        "spearman_r": pd.NA,
        "runtime_seconds": runtime_seconds,
        "skipped": True,
        "skip_reason": skip_reason,
        "hyperparameters": json.dumps(hyperparameters or {}, sort_keys=True),
    }


def _build_metrics_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            {
                BENCHMARK_ID: pd.Series(dtype="string"),
                "model_type": pd.Series(dtype="string"),
                "method": pd.Series(dtype="string"),
                "seed": pd.Series(dtype="Int64"),
                SPLIT: pd.Series(dtype="string"),
                "row_count": pd.Series(dtype="Int64"),
                "rmse": pd.Series(dtype="Float64"),
                "mae": pd.Series(dtype="Float64"),
                "pearson_r": pd.Series(dtype="Float64"),
                "spearman_r": pd.Series(dtype="Float64"),
                "runtime_seconds": pd.Series(dtype="Float64"),
                "skipped": pd.Series(dtype="boolean"),
                "skip_reason": pd.Series(dtype="string"),
                "hyperparameters": pd.Series(dtype="string"),
            }
        )
    return (
        pd.DataFrame(rows)
        .astype(
            {
                BENCHMARK_ID: "string",
                "model_type": "string",
                "method": "string",
                "seed": "Int64",
                SPLIT: "string",
                "row_count": "Int64",
                "rmse": "Float64",
                "mae": "Float64",
                "pearson_r": "Float64",
                "spearman_r": "Float64",
                "runtime_seconds": "Float64",
                "skipped": "boolean",
                "skip_reason": "string",
                "hyperparameters": "string",
            }
        )
        .sort_values([BENCHMARK_ID, "model_type", "method", "seed", SPLIT])
        .reset_index(drop=True)
    )


def _build_predictions_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            {
                BENCHMARK_ID: pd.Series(dtype="string"),
                MODEL_ID: pd.Series(dtype="string"),
                SPLIT: pd.Series(dtype="string"),
                "model_type": pd.Series(dtype="string"),
                "method": pd.Series(dtype="string"),
                "seed": pd.Series(dtype="Int64"),
                ACTUAL_SCORE: pd.Series(dtype="Float64"),
                "predicted_score": pd.Series(dtype="Float64"),
            }
        )
    return (
        pd.DataFrame(rows)
        .astype(
            {
                BENCHMARK_ID: "string",
                MODEL_ID: "string",
                SPLIT: "string",
                "model_type": "string",
                "method": "string",
                "seed": "Int64",
                ACTUAL_SCORE: "Float64",
                "predicted_score": "Float64",
            }
        )
        .sort_values([BENCHMARK_ID, "model_type", "method", "seed", MODEL_ID])
        .reset_index(drop=True)
    )


def _build_summary_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    test_metrics = metrics.loc[(metrics[SPLIT] == "test") & (~metrics["skipped"])].copy()
    if test_metrics.empty:
        return pd.DataFrame(
            {
                "model_type": pd.Series(dtype="string"),
                "method": pd.Series(dtype="string"),
                "rmse_mean": pd.Series(dtype="Float64"),
                "rmse_std": pd.Series(dtype="Float64"),
                "mae_mean": pd.Series(dtype="Float64"),
                "pearson_mean": pd.Series(dtype="Float64"),
                "spearman_mean": pd.Series(dtype="Float64"),
                "runtime_mean_seconds": pd.Series(dtype="Float64"),
                "seed_rmse_std": pd.Series(dtype="Float64"),
                "runs": pd.Series(dtype="Int64"),
            }
        )
    grouped = (
        test_metrics.groupby(["model_type", "method"], dropna=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            pearson_mean=("pearson_r", "mean"),
            spearman_mean=("spearman_r", "mean"),
            runtime_mean_seconds=("runtime_seconds", "mean"),
            runs=("rmse", "count"),
        )
        .reset_index()
    )
    seed_spread = (
        test_metrics.groupby(["model_type", "method", "seed"], dropna=False)["rmse"]
        .mean()
        .groupby(level=[0, 1])
        .std()
        .rename("seed_rmse_std")
        .reset_index()
    )
    return (
        grouped.merge(seed_spread, on=["model_type", "method"], how="left")
        .astype(
            {
                "model_type": "string",
                "method": "string",
                "rmse_mean": "Float64",
                "rmse_std": "Float64",
                "mae_mean": "Float64",
                "pearson_mean": "Float64",
                "spearman_mean": "Float64",
                "runtime_mean_seconds": "Float64",
                "seed_rmse_std": "Float64",
                "runs": "Int64",
            }
        )
        .sort_values(["model_type", "rmse_mean", "method"])
        .reset_index(drop=True)
    )


def _build_report(
    *,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    methods: Sequence[str],
    seeds: Sequence[int],
    skip_reasons: dict[str, str],
) -> dict[str, Any]:
    winners = {}
    for model_type in summary["model_type"].dropna().astype("string").unique().tolist():
        model_rows = summary.loc[summary["model_type"] == model_type].copy()
        model_rows = model_rows.dropna(subset=["rmse_mean"])
        if model_rows.empty:
            continue
        best_row = model_rows.sort_values(["rmse_mean", "runtime_mean_seconds", "method"]).iloc[0]
        winners[model_type] = _json_safe(
            {
                "method": best_row["method"],
                "rmse_mean": best_row["rmse_mean"],
                "runtime_mean_seconds": best_row["runtime_mean_seconds"],
            }
        )
    return {
        "methods": [str(_resolve_method(method)) for method in methods],
        "seeds": [int(seed) for seed in seeds],
        "summary_rows": int(len(summary.index)),
        "detailed_rows": int(len(metrics.index)),
        "skip_reasons": skip_reasons,
        "winners_by_model_type": winners,
    }


def _write_experiment_artifacts(
    result: ReconstructionHeadExperimentResult,
    *,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_parquet = write_parquet(result.metrics, out_dir / "head_metrics.parquet")
    predictions_parquet = write_parquet(result.predictions, out_dir / "head_predictions.parquet")
    summary_parquet = write_parquet(result.summary, out_dir / "head_summary.parquet")
    metrics_csv = out_dir / "head_metrics.csv"
    predictions_csv = out_dir / "head_predictions.csv"
    summary_csv = out_dir / "head_summary.csv"
    result.metrics.to_csv(metrics_csv, index=False)
    result.predictions.to_csv(predictions_csv, index=False)
    result.summary.to_csv(summary_csv, index=False)
    report_json = write_json(_json_safe(result.report), out_dir / "head_report.json")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rmse_plot = _plot_summary_metric(
        result.summary,
        value_column="rmse_mean",
        title="held-out rmse by method",
        ylabel="test rmse",
        out_path=plots_dir / "test_rmse_by_method.png",
    )
    runtime_plot = _plot_summary_metric(
        result.summary,
        value_column="runtime_mean_seconds",
        title="runtime by method",
        ylabel="mean runtime (s)",
        out_path=plots_dir / "runtime_by_method.png",
    )
    stability_plot = _plot_summary_metric(
        result.summary,
        value_column="seed_rmse_std",
        title="seed rmse spread by method",
        ylabel="seed rmse std",
        out_path=plots_dir / "seed_spread_by_method.png",
    )
    summary_md = out_dir / "summary.md"
    summary_md.write_text(_summary_markdown(result), encoding="utf-8")
    return {
        "metrics_parquet": metrics_parquet,
        "metrics_csv": metrics_csv,
        "predictions_parquet": predictions_parquet,
        "predictions_csv": predictions_csv,
        "summary_parquet": summary_parquet,
        "summary_csv": summary_csv,
        "report_json": report_json,
        "summary_md": summary_md,
        "rmse_plot": rmse_plot,
        "runtime_plot": runtime_plot,
        "stability_plot": stability_plot,
    }


def _plot_summary_metric(
    summary: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> Path:
    plot_rows = summary.dropna(subset=[value_column]).copy()
    if plot_rows.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    methods = plot_rows["method"].dropna().astype("string").unique().tolist()
    model_types = plot_rows["model_type"].dropna().astype("string").unique().tolist()
    x = np.arange(len(methods), dtype=float)
    width = 0.8 / max(1, len(model_types))
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, model_type in enumerate(model_types):
        model_rows = (
            plot_rows.loc[plot_rows["model_type"] == model_type]
            .set_index("method")
            .reindex(methods)
        )
        ax.bar(
            x + (index - (len(model_types) - 1) / 2.0) * width,
            model_rows[value_column].astype(float).to_numpy(),
            width=width,
            label=model_type,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _summary_markdown(result: ReconstructionHeadExperimentResult) -> str:
    lines = [
        "# reconstruction head comparison",
        "",
        "## winners",
        "",
    ]
    if not result.report["winners_by_model_type"]:
        lines.append("- no valid model comparisons were available")
    else:
        for model_type, payload in sorted(result.report["winners_by_model_type"].items()):
            lines.append(
                "- "
                f"`{model_type}` winner: `{payload['method']}` "
                f"(rmse_mean={_fmt_float(payload['rmse_mean'])}, "
                f"runtime_mean_seconds={_fmt_float(payload['runtime_mean_seconds'])})"
            )
    lines.extend(["", "## summary", ""])
    if result.summary.empty:
        lines.append("- no summary rows")
    else:
        for row in result.summary.to_dict(orient="records"):
            lines.append(
                "- "
                f"`{row['model_type']}` `{row['method']}`: "
                f"rmse_mean={_fmt_float(row['rmse_mean'])}, "
                f"mae_mean={_fmt_float(row['mae_mean'])}, "
                f"pearson_mean={_fmt_float(row['pearson_mean'])}, "
                f"spearman_mean={_fmt_float(row['spearman_mean'])}, "
                f"runtime_mean_seconds={_fmt_float(row['runtime_mean_seconds'])}, "
                f"seed_rmse_std={_fmt_float(row['seed_rmse_std'])}"
            )
    return "\n".join(lines) + "\n"


def _resolve_method(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_")
    if normalized in {METHOD_GAM, METHOD_ELASTIC_NET, METHOD_XGBOOST}:
        return normalized
    raise ValueError(f"unsupported reconstruction method: {method}")


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def _safe_correlation(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    method: str,
) -> float | None:
    if actual.shape[0] < 2:
        return None
    if np.allclose(actual, actual[0]) or np.allclose(predicted, predicted[0]):
        return None
    if method == "pearson":
        statistic = pearsonr(actual, predicted).statistic
    elif method == "spearman":
        statistic = spearmanr(actual, predicted).statistic
    else:  # pragma: no cover - internal callsites are fixed
        raise ValueError(f"unsupported correlation method: {method}")
    return None if np.isnan(statistic) else float(statistic)


def _fmt_float(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


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
