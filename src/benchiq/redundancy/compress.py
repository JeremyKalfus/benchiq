"""Stage-10 benchmark-level redundancy and compressibility analysis."""

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
from benchiq.preprocess.scores import SCORE_FULL, ScoreResult
from benchiq.reconstruct.features import MARGINAL_TARGET, FeatureTableResult
from benchiq.reconstruct.gam import (
    DEFAULT_LAM_GRID,
    GAMCVResult,
    cross_validate_gam,
    rmse_score,
)
from benchiq.reconstruct.reconstruction import MAE, MARGINAL_MODEL, RMSE
from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID, SPLIT

from .corr import (
    CORRELATION,
    OTHER_BENCHMARK_ID,
    build_pairwise_correlation_table,
    summarize_correlation_table,
)
from .factor import FACTOR, LOADING, run_factor_analysis

CROSS_ONLY_TEST_RMSE = "cross_only_test_rmse"
MARGINAL_TEST_RMSE = "marginal_test_rmse"
REDUNDANCY_RMSE_GAIN = "redundancy_rmse_gain"
REDUNDANCY_RATIO = "redundancy_ratio"
CROSS_ONLY_TEST_MAE = "cross_only_test_mae"
CROSS_ONLY_TEST_PEARSON_R = "cross_only_test_pearson_r"
CROSS_ONLY_TEST_SPEARMAN_R = "cross_only_test_spearman_r"
SKIPPED = "skipped"
SKIP_REASON = "skip_reason"

DEFAULT_N_FACTORS_TO_TRY = (1, 2, 3)
DEFAULT_REDUNDANCY_CV_FOLDS = 5
DEFAULT_REDUNDANCY_N_SPLINES = 20


@dataclass(slots=True)
class RedundancyResult:
    """Stage-10 redundancy-analysis outputs."""

    corr_theta: pd.DataFrame
    corr_scores: pd.DataFrame
    factor_loadings: pd.DataFrame
    compressibility: pd.DataFrame
    redundancy_report: dict[str, Any]
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None


def analyze_redundancy(
    bundle: Bundle,
    score_result: ScoreResult,
    theta_result: Any,
    feature_result: FeatureTableResult,
    reconstruction_result: Any,
    *,
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
    cv_folds: int = DEFAULT_REDUNDANCY_CV_FOLDS,
    n_splines: int = DEFAULT_REDUNDANCY_N_SPLINES,
    n_factors_to_try: Sequence[int] = DEFAULT_N_FACTORS_TO_TRY,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> RedundancyResult:
    """Compute benchmark-level correlation, factor, and compressibility diagnostics."""

    benchmark_ids = sorted(
        feature_result.feature_report["parameters"].get(
            "benchmark_ids",
            feature_result.features_marginal[BENCHMARK_ID].dropna().astype("string").unique(),
        )
    )

    corr_theta = build_pairwise_correlation_table(
        theta_result.theta_estimates,
        value_column="theta_hat",
        benchmark_ids=benchmark_ids,
    )
    corr_scores = build_pairwise_correlation_table(
        score_result.scores_full,
        value_column=SCORE_FULL,
        benchmark_ids=benchmark_ids,
    )
    factor_loadings, factor_report = run_factor_analysis(
        theta_result.theta_estimates,
        benchmark_ids=benchmark_ids,
        min_overlap_models_for_redundancy=bundle.config.min_overlap_models_for_redundancy,
        n_factors_to_try=n_factors_to_try,
    )
    compressibility, compressibility_report = _compute_compressibility(
        bundle=bundle,
        feature_result=feature_result,
        reconstruction_result=reconstruction_result,
        benchmark_ids=benchmark_ids,
        lam_grid=lam_grid,
        cv_folds=cv_folds,
        n_splines=n_splines,
    )

    redundancy_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "benchmark_ids": benchmark_ids,
            "lam_grid": [float(value) for value in lam_grid],
            "cv_folds": cv_folds,
            "n_splines": n_splines,
            "n_factors_to_try": [int(value) for value in n_factors_to_try],
            "min_overlap_models_for_redundancy": bundle.config.min_overlap_models_for_redundancy,
        },
        "theta_correlation_summary": summarize_correlation_table(corr_theta),
        "score_correlation_summary": summarize_correlation_table(corr_scores),
        "factor_analysis": factor_report,
        "compressibility": compressibility_report,
    }
    result = RedundancyResult(
        corr_theta=corr_theta,
        corr_scores=corr_scores,
        factor_loadings=factor_loadings,
        compressibility=compressibility,
        redundancy_report=redundancy_report,
    )

    run_root, manifest_path = _resolve_run_root(bundle, out_dir=out_dir, run_id=run_id)
    if run_root is not None:
        artifact_paths = _write_redundancy_artifacts(result, run_root=run_root)
        result.artifact_paths = artifact_paths
        result.manifest_path = manifest_path
        if manifest_path is not None:
            update_manifest(
                manifest_path,
                {
                    "artifacts": {
                        "10_redundancy": {
                            "corr_theta": str(artifact_paths["corr_theta"]),
                            "corr_scores": str(artifact_paths["corr_scores"]),
                            "factor_loadings": str(artifact_paths["factor_loadings"]),
                            "compressibility": str(artifact_paths["compressibility"]),
                            "redundancy_report": str(artifact_paths["redundancy_report"]),
                            "plots": {
                                name: str(path)
                                for name, path in sorted(artifact_paths["plots"].items())
                            },
                        }
                    }
                },
            )
    return result


def _compute_compressibility(
    *,
    bundle: Bundle,
    feature_result: FeatureTableResult,
    reconstruction_result: Any,
    benchmark_ids: list[str],
    lam_grid: Sequence[float],
    cv_folds: int,
    n_splines: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    joint_report = feature_result.feature_report["joint"]
    theta_columns = sorted(
        column for column in feature_result.features_joint.columns if column.startswith("theta_")
    )
    benchmark_reports: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    skipped_benchmarks: dict[str, str] = {}
    warnings: list[dict[str, Any]] = []
    if joint_report["skipped"]:
        skip_reason = f"joint_features_unavailable:{joint_report['skip_reason']}"
        for benchmark_id in benchmark_ids:
            rows.append(_skipped_compressibility_row(benchmark_id, skip_reason))
            benchmark_reports[benchmark_id] = {
                "skipped": True,
                "skip_reason": skip_reason,
                "feature_columns": [],
            }
            skipped_benchmarks[benchmark_id] = skip_reason
        return _compressibility_frame(rows), {
            "benchmark_reports": benchmark_reports,
            "skipped_benchmarks": skipped_benchmarks,
            "warnings": warnings,
        }

    complete_overlap = int(joint_report["complete_feature_model_count"])
    if complete_overlap < bundle.config.min_overlap_models_for_redundancy:
        skip_reason = "overlap_below_redundancy_threshold"
        for benchmark_id in benchmark_ids:
            rows.append(_skipped_compressibility_row(benchmark_id, skip_reason))
            benchmark_reports[benchmark_id] = {
                "skipped": True,
                "skip_reason": skip_reason,
                "feature_columns": [],
                "complete_feature_model_count": complete_overlap,
            }
            skipped_benchmarks[benchmark_id] = skip_reason
        return _compressibility_frame(rows), {
            "benchmark_reports": benchmark_reports,
            "skipped_benchmarks": skipped_benchmarks,
            "warnings": warnings,
        }

    marginal_lookup = _marginal_test_rmse_lookup(reconstruction_result.reconstruction_summary)

    for benchmark_index, benchmark_id in enumerate(benchmark_ids):
        feature_columns = [column for column in theta_columns if column != f"theta_{benchmark_id}"]
        if not feature_columns:
            skip_reason = "no_cross_benchmark_theta_features"
            rows.append(_skipped_compressibility_row(benchmark_id, skip_reason))
            benchmark_reports[benchmark_id] = {
                "skipped": True,
                "skip_reason": skip_reason,
                "feature_columns": feature_columns,
            }
            skipped_benchmarks[benchmark_id] = skip_reason
            continue

        benchmark_frame = feature_result.features_joint.loc[
            feature_result.features_joint[BENCHMARK_ID] == benchmark_id
        ].copy()
        benchmark_frame = benchmark_frame.loc[
            :,
            [MODEL_ID, SPLIT, MARGINAL_TARGET, *feature_columns],
        ].dropna(subset=[MARGINAL_TARGET, *feature_columns])
        if benchmark_frame.empty:
            skip_reason = "no_complete_cross_only_rows"
            rows.append(_skipped_compressibility_row(benchmark_id, skip_reason))
            benchmark_reports[benchmark_id] = {
                "skipped": True,
                "skip_reason": skip_reason,
                "feature_columns": feature_columns,
            }
            skipped_benchmarks[benchmark_id] = skip_reason
            continue

        train_frame = benchmark_frame.loc[benchmark_frame[SPLIT] == "train"].copy()
        test_frame = benchmark_frame.loc[benchmark_frame[SPLIT] == "test"].copy()
        val_frame = benchmark_frame.loc[benchmark_frame[SPLIT] == "val"].copy()
        required_rows = max(cv_folds, 4)
        if len(train_frame.index) < required_rows:
            skip_reason = "insufficient_train_rows_for_cross_only_gam"
            rows.append(_skipped_compressibility_row(benchmark_id, skip_reason))
            benchmark_reports[benchmark_id] = {
                "skipped": True,
                "skip_reason": skip_reason,
                "feature_columns": feature_columns,
                "train_row_count": int(len(train_frame.index)),
            }
            skipped_benchmarks[benchmark_id] = skip_reason
            continue
        if len(test_frame.index) == 0:
            skip_reason = "no_test_rows_for_cross_only_gam"
            rows.append(_skipped_compressibility_row(benchmark_id, skip_reason))
            benchmark_reports[benchmark_id] = {
                "skipped": True,
                "skip_reason": skip_reason,
                "feature_columns": feature_columns,
            }
            skipped_benchmarks[benchmark_id] = skip_reason
            continue
        if len(train_frame.index) <= n_splines:
            skip_reason = "insufficient_train_rows_for_requested_n_splines"
            rows.append(_skipped_compressibility_row(benchmark_id, skip_reason))
            benchmark_reports[benchmark_id] = {
                "skipped": True,
                "skip_reason": skip_reason,
                "feature_columns": feature_columns,
            }
            skipped_benchmarks[benchmark_id] = skip_reason
            continue

        cv_result = cross_validate_gam(
            train_frame.loc[:, feature_columns],
            train_frame[MARGINAL_TARGET],
            lam_grid=lam_grid,
            cv_folds=cv_folds,
            random_seed=bundle.config.random_seed + benchmark_index,
            feature_names=feature_columns,
            target_name=MARGINAL_TARGET,
            n_splines=n_splines,
            X_test=test_frame.loc[:, feature_columns],
            y_test=test_frame[MARGINAL_TARGET],
        )
        split_metrics = _evaluate_cross_only_model(
            cv_result=cv_result,
            benchmark_frame=benchmark_frame,
            feature_columns=feature_columns,
        )
        marginal_rmse = marginal_lookup.get(benchmark_id)
        cross_only_rmse = split_metrics["test"][RMSE]
        row = {
            BENCHMARK_ID: benchmark_id,
            MARGINAL_TEST_RMSE: marginal_rmse,
            CROSS_ONLY_TEST_RMSE: cross_only_rmse,
            REDUNDANCY_RMSE_GAIN: (
                float(marginal_rmse) - float(cross_only_rmse)
                if marginal_rmse is not None and cross_only_rmse is not None
                else None
            ),
            REDUNDANCY_RATIO: (
                float(cross_only_rmse) / float(marginal_rmse)
                if marginal_rmse not in (None, 0) and cross_only_rmse is not None
                else None
            ),
            CROSS_ONLY_TEST_MAE: split_metrics["test"][MAE],
            CROSS_ONLY_TEST_PEARSON_R: split_metrics["test"]["pearson_r"],
            CROSS_ONLY_TEST_SPEARMAN_R: split_metrics["test"]["spearman_r"],
            "feature_count": len(feature_columns),
            "train_row_count": int(len(train_frame.index)),
            "val_row_count": int(len(val_frame.index)),
            "test_row_count": int(len(test_frame.index)),
            "best_lam": float(cv_result.best_lam),
            SKIPPED: False,
            SKIP_REASON: None,
        }
        rows.append(row)
        benchmark_reports[benchmark_id] = {
            "skipped": False,
            "skip_reason": None,
            "feature_columns": feature_columns,
            "train_row_count": int(len(train_frame.index)),
            "val_row_count": int(len(val_frame.index)),
            "test_row_count": int(len(test_frame.index)),
            "best_lam": float(cv_result.best_lam),
            "cv_report": cv_result.cv_report,
            "split_metrics": split_metrics,
        }
        if (
            marginal_rmse is not None
            and cross_only_rmse is not None
            and cross_only_rmse > marginal_rmse
        ):
            warnings.append(
                {
                    "benchmark_id": benchmark_id,
                    "code": "cross_only_worse_than_marginal",
                    "message": (
                        f"Cross-only RMSE ({cross_only_rmse:.4f}) exceeded marginal RMSE "
                        f"({marginal_rmse:.4f})."
                    ),
                    "severity": "warning",
                }
            )

    return _compressibility_frame(rows), {
        "benchmark_reports": benchmark_reports,
        "skipped_benchmarks": skipped_benchmarks,
        "warnings": warnings,
    }


def _evaluate_cross_only_model(
    *,
    cv_result: GAMCVResult,
    benchmark_frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, dict[str, float | int | None]]:
    model = cv_result.best_model
    split_metrics: dict[str, dict[str, float | int | None]] = {}
    for split_name in ("train", "val", "test"):
        split_frame = benchmark_frame.loc[benchmark_frame[SPLIT] == split_name]
        if split_frame.empty:
            split_metrics[split_name] = {
                RMSE: None,
                MAE: None,
                "pearson_r": None,
                "spearman_r": None,
                "row_count": 0,
            }
            continue
        y_true = split_frame[MARGINAL_TARGET].to_numpy(dtype=float)
        y_pred = model.predict(split_frame.loc[:, feature_columns])
        split_metrics[split_name] = {
            RMSE: rmse_score(y_true, y_pred),
            MAE: float(np.mean(np.abs(y_pred - y_true))),
            "pearson_r": _safe_correlation(pearsonr, y_true, y_pred),
            "spearman_r": _safe_correlation(spearmanr, y_true, y_pred),
            "row_count": int(len(split_frame.index)),
        }
    return split_metrics


def _safe_correlation(func: Any, y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return None
    statistic = func(y_true, y_pred).statistic
    return float(statistic) if statistic is not None else None


def _marginal_test_rmse_lookup(summary: pd.DataFrame) -> dict[str, float]:
    lookup: dict[str, float] = {}
    rows = summary.loc[(summary["model_type"] == MARGINAL_MODEL) & (summary[SPLIT] == "test")]
    for row in rows.itertuples(index=False):
        rmse_value = getattr(row, RMSE)
        if pd.notna(rmse_value):
            lookup[str(getattr(row, BENCHMARK_ID))] = float(rmse_value)
    return lookup


def _skipped_compressibility_row(benchmark_id: str, skip_reason: str) -> dict[str, Any]:
    return {
        BENCHMARK_ID: benchmark_id,
        MARGINAL_TEST_RMSE: None,
        CROSS_ONLY_TEST_RMSE: None,
        REDUNDANCY_RMSE_GAIN: None,
        REDUNDANCY_RATIO: None,
        CROSS_ONLY_TEST_MAE: None,
        CROSS_ONLY_TEST_PEARSON_R: None,
        CROSS_ONLY_TEST_SPEARMAN_R: None,
        "feature_count": 0,
        "train_row_count": 0,
        "val_row_count": 0,
        "test_row_count": 0,
        "best_lam": None,
        SKIPPED: True,
        SKIP_REASON: skip_reason,
    }


def _compressibility_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        rows = []
    return pd.DataFrame.from_records(rows).astype(
        {
            BENCHMARK_ID: "string",
            MARGINAL_TEST_RMSE: "Float64",
            CROSS_ONLY_TEST_RMSE: "Float64",
            REDUNDANCY_RMSE_GAIN: "Float64",
            REDUNDANCY_RATIO: "Float64",
            CROSS_ONLY_TEST_MAE: "Float64",
            CROSS_ONLY_TEST_PEARSON_R: "Float64",
            CROSS_ONLY_TEST_SPEARMAN_R: "Float64",
            "feature_count": "Int64",
            "train_row_count": "Int64",
            "val_row_count": "Int64",
            "test_row_count": "Int64",
            "best_lam": "Float64",
            SKIPPED: bool,
            SKIP_REASON: "string",
        }
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


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_redundancy_artifacts(
    result: RedundancyResult,
    *,
    run_root: Path,
) -> dict[str, Any]:
    stage_dir = run_root / "artifacts" / "10_redundancy"
    plots_dir = stage_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "corr_theta": write_parquet(result.corr_theta, stage_dir / "corr_theta.parquet"),
        "corr_scores": write_parquet(result.corr_scores, stage_dir / "corr_scores.parquet"),
        "factor_loadings": write_parquet(
            result.factor_loadings,
            stage_dir / "factor_loadings.parquet",
        ),
        "compressibility": write_parquet(
            result.compressibility,
            stage_dir / "compressibility.parquet",
        ),
        "redundancy_report": write_json(
            _json_safe(result.redundancy_report),
            stage_dir / "redundancy_report.json",
        ),
        "plots": {},
    }
    artifact_paths["plots"]["corr_theta_heatmap"] = _write_correlation_heatmap(
        result.corr_theta,
        path=plots_dir / "corr_theta_heatmap.png",
        title="Theta Correlation Heatmap",
    )
    artifact_paths["plots"]["corr_scores_heatmap"] = _write_correlation_heatmap(
        result.corr_scores,
        path=plots_dir / "corr_scores_heatmap.png",
        title="Score Correlation Heatmap",
    )
    artifact_paths["plots"]["compressibility"] = _write_compressibility_plot(
        result.compressibility,
        path=plots_dir / "compressibility_ratio.png",
    )
    if not result.redundancy_report["factor_analysis"]["skipped"]:
        artifact_paths["plots"]["factor_scree"] = _write_factor_scree_plot(
            result.redundancy_report["factor_analysis"]["candidate_scores"],
            path=plots_dir / "factor_scree.png",
        )
        artifact_paths["plots"]["factor_loadings"] = _write_factor_loadings_plot(
            result.factor_loadings,
            path=plots_dir / "factor_loadings.png",
        )
    return artifact_paths


def _write_correlation_heatmap(
    table: pd.DataFrame,
    *,
    path: Path,
    title: str,
) -> Path:
    matrix = (
        table.pivot(index=BENCHMARK_ID, columns=OTHER_BENCHMARK_ID, values=CORRELATION)
        .astype(float)
        .sort_index()
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.to_numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(matrix.columns)), labels=list(matrix.columns), rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), labels=list(matrix.index))
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_factor_scree_plot(candidate_scores: list[dict[str, Any]], *, path: Path) -> Path:
    scree = pd.DataFrame.from_records(candidate_scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scree["n_factors"], scree["score"], marker="o")
    ax.set_xlabel("Factors")
    ax.set_ylabel("Mean Log-Likelihood")
    ax.set_title("Factor Analysis Candidate Scores")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_factor_loadings_plot(loadings: pd.DataFrame, *, path: Path) -> Path:
    pivoted = loadings.pivot(index=BENCHMARK_ID, columns=FACTOR, values=LOADING).astype(float)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(pivoted.index))
    width = 0.8 / max(len(pivoted.columns), 1)
    for index, factor_name in enumerate(pivoted.columns):
        ax.bar(x + index * width, pivoted[factor_name], width=width, label=factor_name)
    ax.set_xticks(x + width * max(len(pivoted.columns) - 1, 0) / 2, labels=list(pivoted.index))
    ax.set_ylabel("Loading")
    ax.set_title("Factor Loadings")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_compressibility_plot(table: pd.DataFrame, *, path: Path) -> Path:
    plotted = table.loc[~table[SKIPPED]].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    if plotted.empty:
        ax.text(0.5, 0.5, "no compressibility rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.bar(plotted[BENCHMARK_ID].astype(str), plotted[REDUNDANCY_RATIO].astype(float))
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("cross-only / marginal rmse")
        ax.set_title("Benchmark Compressibility Ratio")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
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
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if value is pd.NA or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
