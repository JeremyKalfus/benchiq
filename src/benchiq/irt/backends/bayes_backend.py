"""Optional Bayesian 2PL backend for stage-05 IRT fitting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from benchiq.irt.backends.common import (
    EXCLUDE_DISCRIMINATION_RANGE,
    WARN_DISCRIMINATION_RANGE,
    IRT2PLResult,
    IRTBackendDependencyError,
    build_ability_frame,
    build_item_params_frame,
)
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID, SCORE

BayesMCMC2PLResult = IRT2PLResult


@dataclass(slots=True)
class BayesPosteriorSummary:
    """Posterior summaries needed to normalize a Bayesian stage-05 fit."""

    discrimination_mean: np.ndarray
    difficulty_mean: np.ndarray
    ability_mean: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class ObservedResponseData:
    """Observed stage-05 responses aligned to requested items and models."""

    response_matrix: pd.DataFrame
    observed_scores: np.ndarray
    item_indices: np.ndarray
    model_indices: np.ndarray


def fit_bayes_mcmc_2pl(
    responses_long: pd.DataFrame,
    *,
    benchmark_id: str,
    item_ids: Sequence[str],
    model_ids: Sequence[str],
    options: dict[str, Any] | None = None,
) -> BayesMCMC2PLResult:
    """Fit a Bayesian 2PL model and return posterior-mean stage-05 outputs."""

    if not item_ids:
        raise ValueError("item_ids must contain at least one preselected item")
    if not model_ids:
        raise ValueError("model_ids must contain at least one train model")

    observed = build_bayes_observed_responses(
        responses_long,
        benchmark_id=benchmark_id,
        item_ids=item_ids,
        model_ids=model_ids,
    )
    sampling_options = _resolve_sampling_options(options or {})

    runtime_start = perf_counter()
    posterior = _sample_posterior_summary(
        observed=observed,
        item_ids=item_ids,
        model_ids=model_ids,
        sampling_options=sampling_options,
    )
    runtime_seconds = perf_counter() - runtime_start

    all_item_params = build_item_params_frame(
        benchmark_id=benchmark_id,
        item_ids=item_ids,
        discrimination=posterior.discrimination_mean,
        difficulty=posterior.difficulty_mean,
        backend_name="bayes_mcmc",
    )
    dropped_pathological_items = (
        all_item_params.loc[all_item_params["pathology_excluded"]].copy().reset_index(drop=True)
    )
    item_params = (
        all_item_params.loc[~all_item_params["pathology_excluded"]].copy().reset_index(drop=True)
    )
    ability_estimates = build_ability_frame(
        benchmark_id=benchmark_id,
        model_ids=model_ids,
        ability_values=posterior.ability_mean,
    )
    fit_report = _build_fit_report(
        benchmark_id=benchmark_id,
        observed=observed,
        item_params=item_params,
        dropped_pathological_items=dropped_pathological_items,
        ability_estimates=ability_estimates,
        runtime_seconds=runtime_seconds,
        sampling_options=sampling_options,
        diagnostics=posterior.diagnostics,
    )
    return BayesMCMC2PLResult(
        item_params=item_params,
        dropped_pathological_items=dropped_pathological_items,
        fit_report=fit_report,
        ability_estimates=ability_estimates,
    )


def build_bayes_observed_responses(
    responses_long: pd.DataFrame,
    *,
    benchmark_id: str,
    item_ids: Sequence[str],
    model_ids: Sequence[str],
) -> ObservedResponseData:
    """Build aligned observed responses for the Bayesian stage-05 path."""

    benchmark_rows = responses_long.loc[
        responses_long[BENCHMARK_ID] == benchmark_id,
        [MODEL_ID, ITEM_ID, SCORE],
    ].copy()
    response_matrix = benchmark_rows.pivot(index=ITEM_ID, columns=MODEL_ID, values=SCORE).reindex(
        index=pd.Index(item_ids, dtype="string"),
        columns=pd.Index(model_ids, dtype="string"),
    )
    values = response_matrix.astype("Float64").to_numpy(dtype=float, copy=True)
    observed_mask = np.isfinite(values)
    if not observed_mask.any():
        raise ValueError("bayes_mcmc backend requires at least one observed binary response")

    item_indices, model_indices = np.where(observed_mask)
    observed_scores = values[observed_mask].astype(int, copy=False)
    return ObservedResponseData(
        response_matrix=response_matrix,
        observed_scores=observed_scores,
        item_indices=np.asarray(item_indices, dtype=int),
        model_indices=np.asarray(model_indices, dtype=int),
    )


def _resolve_sampling_options(options: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "draws",
        "tune",
        "chains",
        "cores",
        "target_accept",
        "random_seed",
        "init",
    }
    unexpected = sorted(set(options) - allowed_keys)
    if unexpected:
        unexpected_joined = ", ".join(unexpected)
        raise ValueError(
            f"unsupported bayes_mcmc backend option(s): {unexpected_joined}; "
            "supported keys are chains, cores, draws, init, random_seed, target_accept, tune"
        )
    return {
        "draws": int(options.get("draws", 1000)),
        "tune": int(options.get("tune", 1000)),
        "chains": int(options.get("chains", 2)),
        "cores": int(options.get("cores", 1)),
        "target_accept": float(options.get("target_accept", 0.9)),
        "random_seed": options.get("random_seed"),
        "init": str(options.get("init", "adapt_diag")),
    }


def _sample_posterior_summary(
    *,
    observed: ObservedResponseData,
    item_ids: Sequence[str],
    model_ids: Sequence[str],
    sampling_options: dict[str, Any],
) -> BayesPosteriorSummary:
    pm, az = _import_bayesian_deps()

    coords = {
        "item": list(item_ids),
        "model": list(model_ids),
        "observation": np.arange(len(observed.observed_scores), dtype=int),
    }
    with pm.Model(coords=coords):
        ability = pm.Normal("ability", mu=0.0, sigma=1.0, dims="model")
        discrimination = pm.LogNormal("discrimination", mu=0.0, sigma=0.5, dims="item")
        difficulty = pm.Normal("difficulty", mu=0.0, sigma=1.0, dims="item")
        logits = discrimination[observed.item_indices] * (
            ability[observed.model_indices] - difficulty[observed.item_indices]
        )
        pm.Bernoulli(
            "responses",
            logit_p=logits,
            observed=observed.observed_scores,
            dims="observation",
        )
        inference_data = pm.sample(
            draws=sampling_options["draws"],
            tune=sampling_options["tune"],
            chains=sampling_options["chains"],
            cores=sampling_options["cores"],
            target_accept=sampling_options["target_accept"],
            init=sampling_options["init"],
            random_seed=sampling_options["random_seed"],
            progressbar=False,
            return_inferencedata=True,
        )

    posterior = inference_data.posterior
    discrimination_mean = _posterior_mean(posterior["discrimination"])
    difficulty_mean = _posterior_mean(posterior["difficulty"])
    ability_mean = _posterior_mean(posterior["ability"])
    diagnostics = _summarize_diagnostics(
        inference_data=inference_data,
        az=az,
        chains=sampling_options["chains"],
    )
    return BayesPosteriorSummary(
        discrimination_mean=discrimination_mean,
        difficulty_mean=difficulty_mean,
        ability_mean=ability_mean,
        diagnostics=diagnostics,
    )


def _posterior_mean(values: Any) -> np.ndarray:
    return np.asarray(values.mean(dim=("chain", "draw")).values, dtype=float).reshape(-1)


def _summarize_diagnostics(*, inference_data: Any, az: Any, chains: int) -> dict[str, Any]:
    divergence_count = 0
    sample_stats = getattr(inference_data, "sample_stats", None)
    if sample_stats is not None and "diverging" in sample_stats:
        divergence_count = int(np.asarray(sample_stats["diverging"].values, dtype=int).sum())

    if chains < 2:
        return {
            "status_available": False,
            "status": None,
            "rhat_max": None,
            "ess_bulk_min": None,
            "divergence_count": divergence_count,
            "warning_code": "mcmc_diagnostics_unavailable",
            "warning_message": (
                "bayes_mcmc ran with fewer than two chains, so rhat-based convergence diagnostics "
                "are unavailable."
            ),
        }

    rhat_values = _flatten_diagnostic_values(az.rhat(inference_data))
    ess_bulk_values = _flatten_diagnostic_values(az.ess(inference_data, method="bulk"))
    rhat_max = float(rhat_values.max()) if rhat_values.size > 0 else None
    ess_bulk_min = float(ess_bulk_values.min()) if ess_bulk_values.size > 0 else None
    warning_code = None
    warning_message = None
    status = "ok"
    if divergence_count > 0:
        status = "warning"
        warning_code = "mcmc_divergences_detected"
        warning_message = (
            f"bayes_mcmc reported {divergence_count} divergent transitions; posterior means are "
            "saved, but convergence diagnostics should be reviewed."
        )
    elif rhat_max is not None and rhat_max > 1.01:
        status = "warning"
        warning_code = "mcmc_rhat_threshold_exceeded"
        warning_message = (
            f"bayes_mcmc max rhat was {rhat_max:.4f}, above the 1.01 threshold; posterior means "
            "are saved, but convergence diagnostics should be reviewed."
        )
    return {
        "status_available": True,
        "status": status,
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min,
        "divergence_count": divergence_count,
        "warning_code": warning_code,
        "warning_message": warning_message,
    }


def _flatten_diagnostic_values(values: Any) -> np.ndarray:
    if hasattr(values, "to_array"):
        array = values.to_array().values
    else:
        array = values
    flattened = np.asarray(array, dtype=float).reshape(-1)
    return flattened[np.isfinite(flattened)]


def _import_bayesian_deps() -> tuple[Any, Any]:
    try:
        import arviz as az
        import pymc as pm
    except ImportError as exc:  # pragma: no cover - exercised by unit tests via monkeypatch
        raise IRTBackendDependencyError(
            "stage-05 backend 'bayes_mcmc' requires optional dependencies `pymc` and `arviz`; "
            "install them with `python -m pip install -e '.[bayes]'` before using "
            "stage_options['05_irt']['backend'] = 'bayes_mcmc'"
        ) from exc
    return pm, az


def _build_fit_report(
    *,
    benchmark_id: str,
    observed: ObservedResponseData,
    item_params: pd.DataFrame,
    dropped_pathological_items: pd.DataFrame,
    ability_estimates: pd.DataFrame,
    runtime_seconds: float,
    sampling_options: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    missing_count = int(observed.response_matrix.isna().sum().sum())
    total_cells = int(observed.response_matrix.size)
    valid_cells = total_cells - missing_count
    warning_items = item_params.loc[item_params["pathology_warning"], ITEM_ID].astype("string")
    excluded_items = dropped_pathological_items[ITEM_ID].astype("string")

    warnings: list[dict[str, Any]] = []
    if diagnostics["warning_code"] is not None:
        warnings.append(
            {
                "code": diagnostics["warning_code"],
                "message": diagnostics["warning_message"],
                "severity": "warning",
                "limitation": False,
            }
        )
    if len(excluded_items.index) > 0:
        warnings.append(
            {
                "code": "pathological_items_dropped",
                "message": (
                    f"{len(excluded_items.index)} pathological items were dropped from "
                    "irt_item_params.parquet and written to dropped_pathological_items.parquet."
                ),
                "severity": "warning",
                "limitation": False,
            }
        )
    return {
        "benchmark_id": benchmark_id,
        "irt_backend": "bayes_mcmc",
        "model": "2pl",
        "parameter_summary": "posterior_mean",
        "skipped": False,
        "skipped_reason": None,
        "warnings": warnings,
        "backend_options": sampling_options,
        "convergence": {
            "status": diagnostics["status"],
            "backend_exposes_flag": False,
            "status_available": bool(diagnostics["status_available"]),
            "diagnostic_basis": "rhat_and_divergence_checks",
            "rhat_threshold": 1.01,
            "rhat_max": diagnostics["rhat_max"],
            "ess_bulk_min": diagnostics["ess_bulk_min"],
            "divergence_count": diagnostics["divergence_count"],
            "warning_code": diagnostics["warning_code"],
        },
        "counts": {
            "train_model_count": int(observed.response_matrix.shape[1]),
            "preselect_item_count": int(observed.response_matrix.shape[0]),
            "valid_response_count": valid_cells,
            "missing_response_count": missing_count,
            "pathology_warning_count": int(len(warning_items.index)),
            "pathology_excluded_count": int(len(excluded_items.index)),
            "retained_item_count": int(len(item_params.index)),
        },
        "pathology": {
            "warning_item_ids": warning_items.tolist(),
            "excluded_item_ids": excluded_items.tolist(),
            "retained_item_ids": item_params[ITEM_ID].astype("string").tolist(),
            "exclusion_is_data_dependent": True,
            "exclusion_note": (
                "posterior-mean pathological exclusion is data-dependent and may be absent on "
                "well-behaved fixtures"
            ),
            "excluded_items": dropped_pathological_items[
                [ITEM_ID, "pathology_excluded_reasons"]
            ].to_dict(orient="records"),
            "warning_thresholds": {
                "discrimination_min": WARN_DISCRIMINATION_RANGE[0],
                "discrimination_max": WARN_DISCRIMINATION_RANGE[1],
            },
            "exclusion_thresholds": {
                "discrimination_min": EXCLUDE_DISCRIMINATION_RANGE[0],
                "discrimination_max": EXCLUDE_DISCRIMINATION_RANGE[1],
                "difficulty_must_be_finite": True,
            },
        },
        "fit_metrics": {
            "runtime_seconds": runtime_seconds,
            "aic": None,
            "bic": None,
            "ability_mean": float(ability_estimates["ability_eap"].mean()),
            "ability_sd": float(ability_estimates["ability_eap"].std(ddof=0)),
            "draws": sampling_options["draws"],
            "tune": sampling_options["tune"],
            "chains": sampling_options["chains"],
        },
        "artifacts": {
            "plots_written": False,
            "plots_reason": "written_by_stage05_artifact_writer",
            "dropped_pathological_items_written": len(excluded_items.index) > 0,
            "item_parameter_scatter": None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
