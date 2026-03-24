"""Stage-07 theta estimation and theta standard-error artifact writing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from benchiq.io.load import Bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.irt.fit import IRTResult
from benchiq.irt.info import test_information_2pl
from benchiq.logging import update_manifest
from benchiq.preprocess.stats import build_benchmark_matrix
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID, SPLIT
from benchiq.split.splitters import SplitResult

if TYPE_CHECKING:
    from benchiq.select import SelectResult

THETA_HAT = "theta_hat"
THETA_SE = "theta_se"
THETA_METHOD = "theta_method"
OBSERVED_ITEM_COUNT = "observed_item_count"
SELECTED_ITEM_COUNT = "selected_item_count"
REDUCED_SCORE = "reduced_score"
MISSING_HEAVY = "missing_heavy"
SATURATED = "saturated"
SATURATION_SIDE = "saturation_side"
RESPONSE_PATTERN = "response_pattern"

ThetaMethod = Literal["MAP", "EAP"]


@dataclass(slots=True)
class ThetaResult:
    """Stage-07 theta-estimation outputs."""

    theta_estimates: pd.DataFrame
    theta_report: dict[str, Any]
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None


def estimate_theta_bundle(
    bundle: Bundle,
    split_result: SplitResult,
    select_result: SelectResult,
    irt_result: IRTResult,
    *,
    theta_method: ThetaMethod = "MAP",
    theta_min: float = -4.0,
    theta_max: float = 4.0,
    theta_grid_size: int = 161,
    missing_heavy_threshold: float = 0.5,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> ThetaResult:
    """Estimate theta and theta standard errors for all benchmark-model pairs."""

    if theta_grid_size < 3:
        raise ValueError("theta_grid_size must be at least 3")
    if theta_max <= theta_min:
        raise ValueError("theta_max must be greater than theta_min")

    theta_grid = np.linspace(theta_min, theta_max, theta_grid_size, dtype=float)
    estimate_frames: list[pd.DataFrame] = []
    benchmark_reports: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []

    for benchmark_id in sorted(select_result.benchmarks):
        frame, benchmark_report = estimate_theta_benchmark(
            bundle,
            split_result,
            select_result,
            irt_result,
            benchmark_id=benchmark_id,
            theta_method=theta_method,
            theta_grid=theta_grid,
            missing_heavy_threshold=missing_heavy_threshold,
        )
        estimate_frames.append(frame)
        benchmark_reports[benchmark_id] = benchmark_report
        warnings.extend(benchmark_report["warnings"])

    if estimate_frames:
        theta_estimates = pd.concat(estimate_frames, ignore_index=True)
    else:
        theta_estimates = _empty_theta_estimates_frame()

    theta_report = _build_theta_report(
        theta_estimates=theta_estimates,
        benchmark_reports=benchmark_reports,
        warnings=warnings,
        theta_method=theta_method,
        theta_grid=theta_grid,
        missing_heavy_threshold=missing_heavy_threshold,
    )
    result = ThetaResult(theta_estimates=theta_estimates, theta_report=theta_report)

    run_root, manifest_path = _resolve_run_root(bundle, out_dir=out_dir, run_id=run_id)
    if run_root is not None:
        artifact_paths = _write_theta_artifacts(result, run_root=run_root)
        result.artifact_paths = artifact_paths
        result.manifest_path = manifest_path
        if manifest_path is not None:
            update_manifest(
                manifest_path,
                {
                    "artifacts": {
                        "07_theta": {
                            "theta_estimates": str(artifact_paths["theta_estimates"]),
                            "theta_report": str(artifact_paths["theta_report"]),
                            "plots": {
                                benchmark_id: str(path)
                                for benchmark_id, path in sorted(artifact_paths["plots"].items())
                            },
                        },
                    },
                },
            )
    return result


def estimate_theta_benchmark(
    bundle: Bundle,
    split_result: SplitResult,
    select_result: SelectResult,
    irt_result: IRTResult,
    *,
    benchmark_id: str,
    theta_method: ThetaMethod,
    theta_grid: np.ndarray,
    missing_heavy_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate theta for one benchmark across all split-assigned models."""

    selection = select_result.benchmarks[benchmark_id]
    if selection.selection_report["skipped"] or selection.subset_final.empty:
        return _empty_theta_estimates_frame(), _skipped_benchmark_report(
            benchmark_id,
            skipped_reason=selection.selection_report["skipped_reason"]
            or "no_selected_items_available",
        )

    split_frame = split_result.per_benchmark_splits.get(benchmark_id)
    if split_frame is None or split_frame.empty:
        return _empty_theta_estimates_frame(), _skipped_benchmark_report(
            benchmark_id,
            skipped_reason="no_split_models_available",
        )

    selected_item_ids = selection.subset_final[ITEM_ID].astype("string").tolist()
    item_params = (
        irt_result.benchmarks[benchmark_id]
        .irt_item_params.set_index(ITEM_ID)
        .loc[selected_item_ids]
        .reset_index()
    )
    response_matrix = build_benchmark_matrix(
        bundle.responses_long,
        benchmark_id=benchmark_id,
    ).reindex(
        index=split_frame[MODEL_ID].astype("string").tolist(),
        columns=selected_item_ids,
    )

    estimate_records: list[dict[str, Any]] = []
    benchmark_warnings: list[dict[str, Any]] = []
    for split_row in split_frame.itertuples(index=False):
        model_id = str(getattr(split_row, MODEL_ID))
        split_name = str(getattr(split_row, SPLIT))
        responses = response_matrix.loc[model_id]
        estimate = estimate_theta_responses(
            responses=responses,
            item_params=item_params,
            theta_method=theta_method,
            theta_grid=theta_grid,
            missing_heavy_threshold=missing_heavy_threshold,
        )
        estimate_records.append(
            {
                BENCHMARK_ID: benchmark_id,
                MODEL_ID: model_id,
                SPLIT: split_name,
                THETA_HAT: estimate["theta_hat"],
                THETA_SE: estimate["theta_se"],
                THETA_METHOD: theta_method,
                OBSERVED_ITEM_COUNT: estimate["observed_item_count"],
                SELECTED_ITEM_COUNT: len(selected_item_ids),
                REDUCED_SCORE: estimate["reduced_score"],
                MISSING_HEAVY: estimate["missing_heavy"],
                SATURATED: estimate["saturated"],
                SATURATION_SIDE: estimate["saturation_side"],
                RESPONSE_PATTERN: estimate["response_pattern"],
            }
        )

    theta_estimates = pd.DataFrame.from_records(estimate_records).astype(
        {
            BENCHMARK_ID: "string",
            MODEL_ID: "string",
            SPLIT: "string",
            THETA_HAT: "Float64",
            THETA_SE: "Float64",
            THETA_METHOD: "string",
            OBSERVED_ITEM_COUNT: "Int64",
            SELECTED_ITEM_COUNT: "Int64",
            REDUCED_SCORE: "Float64",
            MISSING_HEAVY: bool,
            SATURATED: bool,
            SATURATION_SIDE: "string",
            RESPONSE_PATTERN: "string",
        }
    )

    saturation_count = int(theta_estimates[SATURATED].sum())
    if saturation_count > 0:
        benchmark_warnings.append(
            {
                "code": "theta_saturation",
                "message": (
                    f"{saturation_count} models hit theta bounds during {theta_method} estimation."
                ),
                "severity": "warning",
            }
        )
    missing_heavy_count = int(theta_estimates[MISSING_HEAVY].sum())
    if missing_heavy_count > 0:
        benchmark_warnings.append(
            {
                "code": "missing_heavy_patterns",
                "message": (
                    f"{missing_heavy_count} models had reduced-response coverage below the "
                    "missing-heavy threshold."
                ),
                "severity": "warning",
            }
        )

    report = {
        "benchmark_id": benchmark_id,
        "skipped": False,
        "skipped_reason": None,
        "warnings": benchmark_warnings,
        "parameters": {
            "theta_method": theta_method,
        },
        "counts": {
            "model_count": int(len(theta_estimates.index)),
            "selected_item_count": len(selected_item_ids),
            "saturated_count": saturation_count,
            "all_correct_count": int((theta_estimates[RESPONSE_PATTERN] == "all_correct").sum()),
            "all_wrong_count": int((theta_estimates[RESPONSE_PATTERN] == "all_wrong").sum()),
            "missing_heavy_count": missing_heavy_count,
            "no_observed_item_count": int(
                (theta_estimates[RESPONSE_PATTERN] == "no_observed_items").sum()
            ),
            "split_counts": theta_estimates.groupby(SPLIT, sort=True).size().astype(int).to_dict(),
        },
        "plots": {
            "theta_distribution": None,
        },
    }
    return theta_estimates, report


def estimate_theta_responses(
    *,
    responses: pd.Series,
    item_params: pd.DataFrame,
    theta_method: ThetaMethod,
    theta_grid: np.ndarray,
    missing_heavy_threshold: float,
) -> dict[str, Any]:
    """Estimate theta for one model's reduced response pattern."""

    observed_mask = responses.notna()
    observed_responses = responses.loc[observed_mask].astype(float).to_numpy()
    observed_item_ids = responses.index[observed_mask].astype("string").tolist()
    observed_item_count = len(observed_item_ids)
    selected_item_count = int(len(responses.index))
    reduced_score = float(np.mean(observed_responses)) if observed_item_count > 0 else None
    missing_heavy = (
        selected_item_count > 0
        and (observed_item_count / selected_item_count) < missing_heavy_threshold
    )

    if observed_item_count == 0:
        theta_hat = 0.0
        theta_se = None
        return {
            "theta_hat": theta_hat,
            "theta_se": theta_se,
            "observed_item_count": 0,
            "reduced_score": reduced_score,
            "missing_heavy": True,
            "saturated": False,
            "saturation_side": "none",
            "response_pattern": "no_observed_items",
        }

    observed_item_params = item_params.set_index(ITEM_ID).loc[observed_item_ids].reset_index()
    discriminations = observed_item_params["discrimination"].astype(float).to_numpy()
    difficulties = observed_item_params["difficulty"].astype(float).to_numpy()

    if theta_method == "MAP":
        theta_hat = _estimate_theta_map(
            responses=observed_responses,
            discriminations=discriminations,
            difficulties=difficulties,
            theta_min=float(theta_grid.min()),
            theta_max=float(theta_grid.max()),
        )
    elif theta_method == "EAP":
        theta_hat = _estimate_theta_eap(
            responses=observed_responses,
            discriminations=discriminations,
            difficulties=difficulties,
            theta_grid=theta_grid,
        )
    else:
        raise ValueError(f"unsupported theta_method: {theta_method}")

    saturated, saturation_side = _saturation_state(
        theta_hat,
        theta_min=float(theta_grid.min()),
        theta_max=float(theta_grid.max()),
    )
    info_value = test_information_2pl(
        theta_hat,
        discriminations=discriminations,
        difficulties=difficulties,
    )
    theta_se = (1.0 / sqrt(info_value)) if info_value > 0.0 else None
    if np.all(observed_responses == 1.0):
        response_pattern = "all_correct"
    elif np.all(observed_responses == 0.0):
        response_pattern = "all_wrong"
    else:
        response_pattern = "mixed"
    return {
        "theta_hat": float(theta_hat),
        "theta_se": None if theta_se is None else float(theta_se),
        "observed_item_count": observed_item_count,
        "reduced_score": reduced_score,
        "missing_heavy": missing_heavy,
        "saturated": saturated,
        "saturation_side": saturation_side,
        "response_pattern": response_pattern,
    }


def _estimate_theta_map(
    *,
    responses: np.ndarray,
    discriminations: np.ndarray,
    difficulties: np.ndarray,
    theta_min: float,
    theta_max: float,
) -> float:
    def objective(theta: float) -> float:
        return -_log_posterior(theta, responses, discriminations, difficulties)

    result = minimize_scalar(
        objective,
        bounds=(theta_min, theta_max),
        method="bounded",
        options={"xatol": 1e-5},
    )
    theta_hat = float(result.x if result.success else 0.0)
    return float(np.clip(theta_hat, theta_min, theta_max))


def _estimate_theta_eap(
    *,
    responses: np.ndarray,
    discriminations: np.ndarray,
    difficulties: np.ndarray,
    theta_grid: np.ndarray,
) -> float:
    log_posterior = np.array(
        [_log_posterior(theta, responses, discriminations, difficulties) for theta in theta_grid],
        dtype=float,
    )
    posterior_weights = np.exp(log_posterior - logsumexp(log_posterior))
    return float(np.sum(theta_grid * posterior_weights))


def _log_posterior(
    theta: float,
    responses: np.ndarray,
    discriminations: np.ndarray,
    difficulties: np.ndarray,
) -> float:
    return _log_likelihood(theta, responses, discriminations, difficulties) + _log_prior(theta)


def _log_likelihood(
    theta: float,
    responses: np.ndarray,
    discriminations: np.ndarray,
    difficulties: np.ndarray,
) -> float:
    logits = discriminations * (float(theta) - difficulties)
    log_prob_correct = -np.logaddexp(0.0, -logits)
    log_prob_wrong = -np.logaddexp(0.0, logits)
    return float((responses * log_prob_correct + (1.0 - responses) * log_prob_wrong).sum())


def _log_prior(theta: float) -> float:
    return float(-0.5 * (theta**2))


def _saturation_state(
    theta_hat: float,
    *,
    theta_min: float,
    theta_max: float,
    epsilon: float = 1e-3,
) -> tuple[bool, str]:
    if theta_hat <= theta_min + epsilon:
        return True, "lower"
    if theta_hat >= theta_max - epsilon:
        return True, "upper"
    return False, "none"


def _build_theta_report(
    *,
    theta_estimates: pd.DataFrame,
    benchmark_reports: dict[str, Any],
    warnings: list[dict[str, Any]],
    theta_method: ThetaMethod,
    theta_grid: np.ndarray,
    missing_heavy_threshold: float,
) -> dict[str, Any]:
    return {
        "theta_method": theta_method,
        "warnings": warnings,
        "parameters": {
            "theta_method": theta_method,
            "theta_min": float(theta_grid.min()),
            "theta_max": float(theta_grid.max()),
            "theta_grid_size": int(len(theta_grid)),
            "prior": "Normal(0,1)",
            "missing_heavy_threshold": missing_heavy_threshold,
        },
        "counts": {
            "row_count": int(len(theta_estimates.index)),
            "benchmark_count": len(benchmark_reports),
            "split_counts": theta_estimates.groupby(SPLIT, sort=True).size().astype(int).to_dict()
            if not theta_estimates.empty
            else {},
            "saturated_count": (
                int(theta_estimates[SATURATED].sum()) if not theta_estimates.empty else 0
            ),
            "all_correct_count": int((theta_estimates[RESPONSE_PATTERN] == "all_correct").sum())
            if not theta_estimates.empty
            else 0,
            "all_wrong_count": int((theta_estimates[RESPONSE_PATTERN] == "all_wrong").sum())
            if not theta_estimates.empty
            else 0,
            "missing_heavy_count": int(theta_estimates[MISSING_HEAVY].sum())
            if not theta_estimates.empty
            else 0,
            "no_observed_item_count": int(
                (theta_estimates[RESPONSE_PATTERN] == "no_observed_items").sum()
            )
            if not theta_estimates.empty
            else 0,
        },
        "benchmarks": benchmark_reports,
        "artifacts": {
            "theta_estimates": None,
            "theta_report": None,
            "plots": {},
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _skipped_benchmark_report(benchmark_id: str, *, skipped_reason: str) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "skipped": True,
        "skipped_reason": skipped_reason,
        "warnings": [],
        "parameters": {},
        "counts": {
            "model_count": 0,
            "selected_item_count": 0,
            "saturated_count": 0,
            "all_correct_count": 0,
            "all_wrong_count": 0,
            "missing_heavy_count": 0,
            "no_observed_item_count": 0,
            "split_counts": {},
        },
        "plots": {
            "theta_distribution": None,
        },
    }


def _write_theta_artifacts(result: ThetaResult, *, run_root: Path) -> dict[str, Any]:
    stage_dir = run_root / "artifacts" / "07_theta"
    plots_dir = stage_dir / "plots"
    plot_paths: dict[str, Path] = {}
    for benchmark_id, benchmark_report in sorted(result.theta_report["benchmarks"].items()):
        benchmark_frame = result.theta_estimates.loc[
            result.theta_estimates[BENCHMARK_ID] == benchmark_id
        ].copy()
        if benchmark_report["skipped"] or benchmark_frame.empty:
            benchmark_report["plots"]["theta_distribution"] = None
            continue
        plot_path = _write_theta_distribution_plot(
            benchmark_frame,
            path=plots_dir / f"{benchmark_id}__theta_distribution.png",
        )
        benchmark_report["plots"]["theta_distribution"] = str(plot_path)
        plot_paths[benchmark_id] = plot_path

    theta_estimates_path = write_parquet(
        result.theta_estimates,
        stage_dir / "theta_estimates.parquet",
    )
    result.theta_report["artifacts"]["theta_estimates"] = str(theta_estimates_path)
    result.theta_report["artifacts"]["plots"] = {
        benchmark_id: str(path) for benchmark_id, path in sorted(plot_paths.items())
    }
    result.theta_report["artifacts"]["theta_report"] = str(stage_dir / "theta_report.json")
    theta_report_path = write_json(result.theta_report, stage_dir / "theta_report.json")
    return {
        "theta_estimates": theta_estimates_path,
        "theta_report": theta_report_path,
        "plots": plot_paths,
    }


def _write_theta_distribution_plot(frame: pd.DataFrame, *, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    split_names = ["train", "val", "test"]
    for axis, split_name in zip(axes, split_names, strict=True):
        split_frame = frame.loc[frame[SPLIT] == split_name]
        values = split_frame[THETA_HAT].dropna().astype(float).to_numpy()
        if len(values) > 0:
            axis.hist(
                values,
                bins=min(10, max(3, ceil(len(values) / 2))),
                color="#4c78a8",
                alpha=0.85,
            )
        axis.set_title(split_name)
        axis.set_xlabel("theta")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("models")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


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


def _empty_theta_estimates_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            SPLIT: pd.Series(dtype="string"),
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
