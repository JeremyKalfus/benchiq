"""Benchmark-level factor-analysis helpers for stage-10 redundancy analysis."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

from benchiq.irt.theta import THETA_HAT
from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID

FACTOR = "factor"
LOADING = "loading"
ABS_LOADING = "abs_loading"


def run_factor_analysis(
    theta_estimates: pd.DataFrame,
    *,
    benchmark_ids: list[str],
    min_overlap_models_for_redundancy: int,
    n_factors_to_try: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run benchmark-level factor analysis on complete-overlap theta values."""

    if len(benchmark_ids) < 2:
        return _empty_factor_loadings(), {
            "skipped": True,
            "skip_reason": "insufficient_benchmark_count",
            "backend": None,
            "complete_overlap_model_count": 0,
            "selected_n_factors": None,
            "candidate_scores": [],
        }

    theta_wide = (
        theta_estimates.loc[:, [BENCHMARK_ID, MODEL_ID, THETA_HAT]]
        .dropna(subset=[THETA_HAT])
        .pivot(index=MODEL_ID, columns=BENCHMARK_ID, values=THETA_HAT)
        .reindex(columns=benchmark_ids)
        .astype("Float64")
    )
    complete_rows = theta_wide.dropna()
    overlap_count = int(len(complete_rows.index))
    if overlap_count < min_overlap_models_for_redundancy:
        return _empty_factor_loadings(), {
            "skipped": True,
            "skip_reason": "overlap_below_redundancy_threshold",
            "backend": None,
            "complete_overlap_model_count": overlap_count,
            "selected_n_factors": None,
            "candidate_scores": [],
        }

    X = complete_rows.to_numpy(dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds = np.where(stds <= 1e-8, 1.0, stds)
    X = (X - means) / stds

    max_factors = min(len(benchmark_ids) - 1, X.shape[0] - 1)
    candidates = sorted(
        {int(value) for value in n_factors_to_try if 1 <= int(value) <= max_factors}
    )
    if not candidates:
        return _empty_factor_loadings(), {
            "skipped": True,
            "skip_reason": "no_valid_factor_candidates",
            "backend": None,
            "complete_overlap_model_count": overlap_count,
            "selected_n_factors": None,
            "candidate_scores": [],
        }

    candidate_scores: list[dict[str, float | int]] = []
    fitted_models: dict[int, FactorAnalysis] = {}
    for n_factors in candidates:
        model = FactorAnalysis(n_components=n_factors, random_state=0)
        model.fit(X)
        fitted_models[n_factors] = model
        candidate_scores.append(
            {
                "n_factors": n_factors,
                "score": float(model.score(X)),
            }
        )

    best = max(candidate_scores, key=lambda row: (row["score"], -row["n_factors"]))
    best_factors = int(best["n_factors"])
    best_model = fitted_models[best_factors]
    loadings = best_model.components_.T

    records: list[dict[str, object]] = []
    for benchmark_index, benchmark_id in enumerate(benchmark_ids):
        for factor_index in range(best_factors):
            loading = float(loadings[benchmark_index, factor_index])
            records.append(
                {
                    BENCHMARK_ID: benchmark_id,
                    FACTOR: f"factor_{factor_index + 1}",
                    LOADING: loading,
                    ABS_LOADING: abs(loading),
                }
            )

    factor_loadings = pd.DataFrame.from_records(records).astype(
        {
            BENCHMARK_ID: "string",
            FACTOR: "string",
            LOADING: "Float64",
            ABS_LOADING: "Float64",
        }
    )
    return factor_loadings, {
        "skipped": False,
        "skip_reason": None,
        "backend": "sklearn_factor_analysis",
        "complete_overlap_model_count": overlap_count,
        "selected_n_factors": best_factors,
        "candidate_scores": candidate_scores,
    }


def _empty_factor_loadings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            FACTOR: pd.Series(dtype="string"),
            LOADING: pd.Series(dtype="Float64"),
            ABS_LOADING: pd.Series(dtype="Float64"),
        }
    )
