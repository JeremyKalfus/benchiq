"""Shared stage-05 IRT backend contracts and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID

WARN_DISCRIMINATION_RANGE = (0.1, 5.0)
EXCLUDE_DISCRIMINATION_RANGE = (0.05, 10.0)
BOUNDARY_EPSILON = 1e-3
SUPPORTED_IRT_BACKENDS = ("girth", "bayes_mcmc")


class IRTBackendDependencyError(ImportError):
    """Raised when an opt-in IRT backend is selected without its dependencies."""


class UnknownIRTBackendError(ValueError):
    """Raised when the requested stage-05 backend is not supported."""


@dataclass(slots=True)
class IRT2PLResult:
    """Normalized output from a stage-05 2PL backend."""

    item_params: pd.DataFrame
    dropped_pathological_items: pd.DataFrame
    fit_report: dict[str, Any]
    ability_estimates: pd.DataFrame


def normalize_backend_name(backend: str) -> str:
    """Return a normalized backend name or raise for unsupported values."""

    normalized = str(backend).strip().lower()
    if normalized not in SUPPORTED_IRT_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_IRT_BACKENDS))
        raise UnknownIRTBackendError(
            f"unsupported stage-05 IRT backend: {backend!r}; expected one of {supported}"
        )
    return normalized


def empty_item_params_frame() -> pd.DataFrame:
    """Return the canonical empty stage-05 item-parameter frame."""

    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            ITEM_ID: pd.Series(dtype="string"),
            "irt_backend": pd.Series(dtype="string"),
            "discrimination": pd.Series(dtype="Float64"),
            "difficulty": pd.Series(dtype="Float64"),
            "pathology_warning": pd.Series(dtype=bool),
            "pathology_warning_reasons": pd.Series(dtype=object),
            "pathology_excluded": pd.Series(dtype=bool),
            "pathology_excluded_reasons": pd.Series(dtype=object),
        }
    )


def empty_ability_frame() -> pd.DataFrame:
    """Return the canonical empty stage-05 ability-estimate frame."""

    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            "ability_eap": pd.Series(dtype="Float64"),
        }
    )


def build_item_params_frame(
    *,
    benchmark_id: str,
    item_ids: Sequence[str],
    discrimination: Sequence[float],
    difficulty: Sequence[float],
    backend_name: str,
) -> pd.DataFrame:
    """Build the canonical stage-05 item-parameter table."""

    records: list[dict[str, Any]] = []
    for item_id, a_value, b_value in zip(
        item_ids,
        np.asarray(discrimination, dtype=float).reshape(-1),
        np.asarray(difficulty, dtype=float).reshape(-1),
        strict=True,
    ):
        warning_reasons = warning_flags(float(a_value))
        excluded_reasons = exclusion_flags(float(a_value), float(b_value))
        records.append(
            {
                BENCHMARK_ID: benchmark_id,
                ITEM_ID: str(item_id),
                "irt_backend": backend_name,
                "discrimination": float(a_value),
                "difficulty": float(b_value),
                "pathology_warning": bool(warning_reasons),
                "pathology_warning_reasons": warning_reasons,
                "pathology_excluded": bool(excluded_reasons),
                "pathology_excluded_reasons": excluded_reasons,
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return empty_item_params_frame()
    frame[BENCHMARK_ID] = frame[BENCHMARK_ID].astype("string")
    frame[ITEM_ID] = frame[ITEM_ID].astype("string")
    frame["irt_backend"] = frame["irt_backend"].astype("string")
    frame["discrimination"] = pd.Series(frame["discrimination"], dtype="Float64")
    frame["difficulty"] = pd.Series(frame["difficulty"], dtype="Float64")
    frame["pathology_warning"] = frame["pathology_warning"].astype(bool)
    frame["pathology_excluded"] = frame["pathology_excluded"].astype(bool)
    return frame.sort_values(ITEM_ID).reset_index(drop=True)


def build_ability_frame(
    *,
    benchmark_id: str,
    model_ids: Sequence[str],
    ability_values: Sequence[float],
) -> pd.DataFrame:
    """Build the canonical stage-05 ability-estimate table."""

    return (
        pd.DataFrame(
            {
                BENCHMARK_ID: pd.Series([benchmark_id] * len(model_ids), dtype="string"),
                MODEL_ID: pd.Series(list(model_ids), dtype="string"),
                "ability_eap": pd.Series(
                    np.asarray(ability_values, dtype=float).reshape(-1),
                    dtype="Float64",
                ),
            }
        )
        .sort_values(MODEL_ID)
        .reset_index(drop=True)
    )


def warning_flags(discrimination: float) -> list[str]:
    """Return pathology warning flags for a fitted item."""

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


def exclusion_flags(discrimination: float, difficulty: float) -> list[str]:
    """Return hard pathology exclusions for a fitted item."""

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
