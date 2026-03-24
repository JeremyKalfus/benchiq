"""Analytic 2PL probability and Fisher-information helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID

THETA = "theta"
PROBABILITY_CORRECT = "probability_correct"
FISHER_INFORMATION = "fisher_information"


def probability_2pl(
    theta: np.ndarray | pd.Series | list[float] | float,
    *,
    discrimination: float,
    difficulty: float,
) -> np.ndarray:
    """Return 2PL probabilities for one item across theta values."""

    theta_values = np.asarray(theta, dtype=float)
    logits = discrimination * (theta_values - difficulty)
    return 1.0 / (1.0 + np.exp(-logits))


def fisher_information_2pl(
    theta: np.ndarray | pd.Series | list[float] | float,
    *,
    discrimination: float,
    difficulty: float,
) -> np.ndarray:
    """Return analytic 2PL Fisher information across theta values."""

    probability = probability_2pl(
        theta,
        discrimination=discrimination,
        difficulty=difficulty,
    )
    information = (float(discrimination) ** 2) * probability * (1.0 - probability)
    return np.maximum(information, 0.0)


def test_information_2pl(
    theta: float,
    *,
    discriminations: np.ndarray | pd.Series | list[float],
    difficulties: np.ndarray | pd.Series | list[float],
) -> float:
    """Return summed test information at one theta value."""

    discrimination_values = np.asarray(discriminations, dtype=float)
    difficulty_values = np.asarray(difficulties, dtype=float)
    logits = discrimination_values * (float(theta) - difficulty_values)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    item_information = (discrimination_values**2) * probabilities * (1.0 - probabilities)
    return float(np.maximum(item_information, 0.0).sum())


def build_theta_grid(
    item_params: pd.DataFrame,
    *,
    grid_type: Literal["difficulty_range", "uniform"] = "difficulty_range",
    grid_size: int = 401,
    theta_min: float | None = None,
    theta_max: float | None = None,
) -> np.ndarray:
    """Build a deterministic theta grid for information-based selection."""

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    if grid_type == "difficulty_range":
        finite_difficulties = (
            item_params["difficulty"].astype("Float64").dropna().astype(float).to_numpy()
        )
        if len(finite_difficulties) == 0:
            raise ValueError("difficulty_range theta grid requires at least one finite difficulty")
        resolved_min = float(finite_difficulties.min()) - 1.0
        resolved_max = float(finite_difficulties.max()) + 1.0
    elif grid_type == "uniform":
        if theta_min is None or theta_max is None:
            raise ValueError("uniform theta grid requires theta_min and theta_max")
        resolved_min = float(theta_min)
        resolved_max = float(theta_max)
    else:
        raise ValueError(f"unsupported theta grid type: {grid_type}")

    if resolved_max <= resolved_min:
        midpoint = resolved_min
        resolved_min = midpoint - 1.0
        resolved_max = midpoint + 1.0

    return np.linspace(resolved_min, resolved_max, grid_size, dtype=float)


def build_information_grid(
    item_params: pd.DataFrame,
    *,
    theta_grid: np.ndarray,
) -> pd.DataFrame:
    """Return a theta-by-item information table for retained 2PL items."""

    records: list[dict[str, float | str]] = []
    for item in item_params.itertuples(index=False):
        probabilities = probability_2pl(
            theta_grid,
            discrimination=float(item.discrimination),
            difficulty=float(item.difficulty),
        )
        information = fisher_information_2pl(
            theta_grid,
            discrimination=float(item.discrimination),
            difficulty=float(item.difficulty),
        )
        benchmark_id = str(getattr(item, BENCHMARK_ID))
        item_id = str(getattr(item, ITEM_ID))
        for theta_value, probability, info_value in zip(
            theta_grid,
            probabilities,
            information,
            strict=True,
        ):
            records.append(
                {
                    BENCHMARK_ID: benchmark_id,
                    ITEM_ID: item_id,
                    THETA: float(theta_value),
                    PROBABILITY_CORRECT: float(probability),
                    FISHER_INFORMATION: float(info_value),
                }
            )

    return pd.DataFrame.from_records(records).astype(
        {
            BENCHMARK_ID: "string",
            ITEM_ID: "string",
            THETA: "Float64",
            PROBABILITY_CORRECT: "Float64",
            FISHER_INFORMATION: "Float64",
        }
    )
