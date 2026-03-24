"""Core girth-backed 2PL fitting helpers for BenchIQ."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
from girth import tag_missing_data, twopl_mml
from girth.utilities import INVALID_RESPONSE

from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID, SCORE

WARN_DISCRIMINATION_RANGE = (0.1, 5.0)
EXCLUDE_DISCRIMINATION_RANGE = (0.05, 10.0)
BOUNDARY_EPSILON = 1e-3


@dataclass(slots=True)
class Girth2PLResult:
    """Normalized output from a girth 2PL fit."""

    item_params: pd.DataFrame
    dropped_pathological_items: pd.DataFrame
    fit_report: dict[str, Any]
    ability_estimates: pd.DataFrame


def fit_girth_2pl(
    responses_long: pd.DataFrame,
    *,
    benchmark_id: str,
    item_ids: Sequence[str],
    model_ids: Sequence[str],
    options: dict[str, Any] | None = None,
) -> Girth2PLResult:
    """Fit a girth 2PL model on one benchmark's preselected train responses."""

    if not item_ids:
        raise ValueError("item_ids must contain at least one preselected item")
    if not model_ids:
        raise ValueError("model_ids must contain at least one train model")

    response_matrix = build_girth_response_matrix(
        responses_long,
        benchmark_id=benchmark_id,
        item_ids=item_ids,
        model_ids=model_ids,
    )
    runtime_start = perf_counter()
    raw_result = twopl_mml(response_matrix, options=options)
    runtime_seconds = perf_counter() - runtime_start

    all_item_params = _build_item_params_frame(
        benchmark_id=benchmark_id,
        item_ids=item_ids,
        raw_result=raw_result,
    )
    dropped_pathological_items = (
        all_item_params.loc[all_item_params["pathology_excluded"]].copy().reset_index(drop=True)
    )
    item_params = (
        all_item_params.loc[~all_item_params["pathology_excluded"]].copy().reset_index(drop=True)
    )
    ability_estimates = _build_ability_frame(
        benchmark_id=benchmark_id,
        model_ids=model_ids,
        raw_result=raw_result,
    )
    fit_report = _build_fit_report(
        benchmark_id=benchmark_id,
        response_matrix=response_matrix,
        item_params=item_params,
        dropped_pathological_items=dropped_pathological_items,
        ability_estimates=ability_estimates,
        runtime_seconds=runtime_seconds,
        options=options or {},
        raw_result=raw_result,
    )
    return Girth2PLResult(
        item_params=item_params,
        dropped_pathological_items=dropped_pathological_items,
        fit_report=fit_report,
        ability_estimates=ability_estimates,
    )


def build_girth_response_matrix(
    responses_long: pd.DataFrame,
    *,
    benchmark_id: str,
    item_ids: Sequence[str],
    model_ids: Sequence[str],
) -> np.ndarray:
    """Build an items-by-models integer matrix for girth with tagged missing values."""

    benchmark_rows = responses_long.loc[
        responses_long[BENCHMARK_ID] == benchmark_id,
        [MODEL_ID, ITEM_ID, SCORE],
    ].copy()
    pivoted = benchmark_rows.pivot(index=ITEM_ID, columns=MODEL_ID, values=SCORE).reindex(
        index=pd.Index(item_ids, dtype="string"),
        columns=pd.Index(model_ids, dtype="string"),
    )
    tagged = _tag_missing_binary_responses(pivoted)
    if tagged.shape != (len(item_ids), len(model_ids)):
        raise ValueError("response matrix shape does not match requested item/model ids")
    return tagged


def _tag_missing_binary_responses(matrix: pd.DataFrame) -> np.ndarray:
    numeric = matrix.astype("Float64")
    values = numeric.to_numpy(dtype=float, copy=True)
    missing_mask = np.isnan(values)
    values[missing_mask] = INVALID_RESPONSE
    tagged = tag_missing_data(values.astype(int, copy=False), valid_responses=[0, 1])
    return np.asarray(tagged, dtype=int)


def _build_item_params_frame(
    *,
    benchmark_id: str,
    item_ids: Sequence[str],
    raw_result: dict[str, Any],
) -> pd.DataFrame:
    discrimination = np.asarray(raw_result["Discrimination"], dtype=float).reshape(-1)
    difficulty = np.asarray(raw_result["Difficulty"], dtype=float).reshape(-1)

    records: list[dict[str, Any]] = []
    for item_id, a_value, b_value in zip(item_ids, discrimination, difficulty, strict=True):
        warning_flags = _warning_flags(a_value)
        exclusion_flags = _exclusion_flags(a_value, b_value)
        records.append(
            {
                BENCHMARK_ID: benchmark_id,
                ITEM_ID: str(item_id),
                "irt_backend": "girth",
                "discrimination": float(a_value),
                "difficulty": float(b_value),
                "pathology_warning": bool(warning_flags),
                "pathology_warning_reasons": warning_flags,
                "pathology_excluded": bool(exclusion_flags),
                "pathology_excluded_reasons": exclusion_flags,
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame[BENCHMARK_ID] = frame[BENCHMARK_ID].astype("string")
    frame[ITEM_ID] = frame[ITEM_ID].astype("string")
    frame["irt_backend"] = frame["irt_backend"].astype("string")
    frame["discrimination"] = pd.Series(frame["discrimination"], dtype="Float64")
    frame["difficulty"] = pd.Series(frame["difficulty"], dtype="Float64")
    frame["pathology_warning"] = frame["pathology_warning"].astype(bool)
    frame["pathology_excluded"] = frame["pathology_excluded"].astype(bool)
    return frame.sort_values(ITEM_ID).reset_index(drop=True)


def _build_ability_frame(
    *,
    benchmark_id: str,
    model_ids: Sequence[str],
    raw_result: dict[str, Any],
) -> pd.DataFrame:
    ability = np.asarray(raw_result["Ability"], dtype=float).reshape(-1)
    return (
        pd.DataFrame(
            {
                BENCHMARK_ID: pd.Series([benchmark_id] * len(model_ids), dtype="string"),
                MODEL_ID: pd.Series(list(model_ids), dtype="string"),
                "ability_eap": pd.Series(ability, dtype="Float64"),
            }
        )
        .sort_values(MODEL_ID)
        .reset_index(drop=True)
    )


def _build_fit_report(
    *,
    benchmark_id: str,
    response_matrix: np.ndarray,
    item_params: pd.DataFrame,
    dropped_pathological_items: pd.DataFrame,
    ability_estimates: pd.DataFrame,
    runtime_seconds: float,
    options: dict[str, Any],
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    missing_count = int((response_matrix == INVALID_RESPONSE).sum())
    total_cells = int(response_matrix.size)
    valid_cells = total_cells - missing_count
    warning_items = item_params.loc[item_params["pathology_warning"], ITEM_ID].astype("string")
    excluded_items = dropped_pathological_items[ITEM_ID].astype("string")
    warnings = [
        {
            "code": "backend_convergence_status_unavailable",
            "message": (
                "girth does not expose a convergence-status flag; BenchIQ cannot confirm "
                "optimizer convergence from backend outputs."
            ),
            "severity": "warning",
            "limitation": True,
        }
    ]
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
        "irt_backend": "girth",
        "model": "2pl",
        "skipped": False,
        "skipped_reason": None,
        "warnings": warnings,
        "backend_options": options,
        "convergence": {
            "status": None,
            "backend_exposes_flag": False,
            "status_available": False,
            "max_iteration": options.get("max_iteration"),
            "warning_code": "backend_convergence_status_unavailable",
        },
        "counts": {
            "train_model_count": int(response_matrix.shape[1]),
            "preselect_item_count": int(response_matrix.shape[0]),
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
            "aic": raw_result["AIC"],
            "bic": raw_result["BIC"],
            "ability_mean": float(ability_estimates["ability_eap"].mean()),
            "ability_sd": float(ability_estimates["ability_eap"].std(ddof=0)),
        },
        "artifacts": {
            "plots_written": False,
            "plots_reason": "not_implemented_in_t09",
            "dropped_pathological_items_written": len(excluded_items.index) > 0,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _warning_flags(discrimination: float) -> list[str]:
    flags: list[str] = []
    if not np.isfinite(discrimination):
        flags.append("nonfinite_discrimination")
        return flags
    if discrimination < WARN_DISCRIMINATION_RANGE[0]:
        flags.append("low_discrimination_warning")
    if discrimination > WARN_DISCRIMINATION_RANGE[1]:
        flags.append("high_discrimination_warning")
    if discrimination >= WARN_DISCRIMINATION_RANGE[1] - BOUNDARY_EPSILON:
        flags.append("backend_upper_bound_hit")
    return flags


def _exclusion_flags(discrimination: float, difficulty: float) -> list[str]:
    flags: list[str] = []
    if not np.isfinite(discrimination):
        flags.append("nonfinite_discrimination")
    elif discrimination < EXCLUDE_DISCRIMINATION_RANGE[0]:
        flags.append("low_discrimination_excluded")
    elif discrimination > EXCLUDE_DISCRIMINATION_RANGE[1]:
        flags.append("high_discrimination_excluded")
    if not np.isfinite(difficulty):
        flags.append("nonfinite_difficulty")
    return flags
