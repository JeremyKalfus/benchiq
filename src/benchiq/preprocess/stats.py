"""Benchmark-wise preprocessing statistics for BenchIQ."""

from __future__ import annotations

from math import ceil, floor

import pandas as pd

from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID, SCORE


def build_benchmark_matrix(
    responses_long: pd.DataFrame,
    *,
    benchmark_id: str,
) -> pd.DataFrame:
    """Pivot one benchmark into a models x items matrix."""

    benchmark_rows = responses_long.loc[
        responses_long[BENCHMARK_ID] == benchmark_id,
        [MODEL_ID, ITEM_ID, SCORE],
    ]
    matrix = benchmark_rows.pivot(index=MODEL_ID, columns=ITEM_ID, values=SCORE)
    matrix = matrix.sort_index().sort_index(axis=1)
    return matrix.astype("Float64")


def compute_item_stats(
    matrix: pd.DataFrame,
    *,
    benchmark_id: str,
) -> pd.DataFrame:
    """Compute per-item mean, sd, point-biserial, and coverage stats."""

    records: list[dict[str, float | int | str | None]] = []
    total_models = len(matrix.index)
    for item_id in matrix.columns.tolist():
        item_scores = matrix[item_id]
        n_responses = int(item_scores.notna().sum())
        rest_matrix = matrix.drop(columns=item_id)
        rest_scores = rest_matrix.sum(axis=1, skipna=True)
        rest_counts = rest_matrix.notna().sum(axis=1)
        valid_mask = item_scores.notna() & (rest_counts > 0)
        point_biserial = _pearson_correlation(item_scores[valid_mask], rest_scores[valid_mask])
        records.append(
            {
                "benchmark_id": benchmark_id,
                "item_id": item_id,
                "n_responses": n_responses,
                "item_coverage": (n_responses / total_models) if total_models else 0.0,
                "mean": _safe_mean(item_scores),
                "sd": _safe_std(item_scores),
                "point_biserial": point_biserial,
            },
        )

    return pd.DataFrame.from_records(records).sort_values("item_id").reset_index(drop=True)


def compute_model_scores(matrix: pd.DataFrame) -> pd.Series:
    """Compute per-model mean score across available items."""

    return matrix.mean(axis=1, skipna=True)


def low_tail_trim_count(
    *,
    n_models_benchmark: int,
    quantile: float,
) -> int:
    """Return the number of low-tail models to trim for a benchmark."""

    if n_models_benchmark <= 0 or quantile <= 0.0:
        return 0
    return floor(n_models_benchmark * quantile)


def select_low_tail_model_ids(
    model_scores: pd.Series,
    *,
    quantile: float,
) -> list[str]:
    """Select the lowest-scoring models to trim from the low tail."""

    trim_count = low_tail_trim_count(
        n_models_benchmark=len(model_scores.index),
        quantile=quantile,
    )
    if trim_count == 0:
        return []

    ranked_scores = (
        pd.DataFrame(
            {
                MODEL_ID: model_scores.index.astype("string"),
                "full_score": model_scores.astype("Float64").to_numpy(),
            },
        )
        .sort_values(["full_score", MODEL_ID], na_position="first")
        .reset_index(drop=True)
    )
    return ranked_scores.head(trim_count)[MODEL_ID].astype("string").tolist()


def compute_model_coverage(
    matrix: pd.DataFrame,
    *,
    benchmark_id: str,
) -> pd.DataFrame:
    """Compute per-model retained-item coverage."""

    total_items = matrix.shape[1]
    answered_items = matrix.notna().sum(axis=1)
    if total_items:
        coverage = answered_items / total_items
    else:
        coverage = pd.Series(0.0, index=matrix.index, dtype="Float64")
    return (
        pd.DataFrame(
            {
                "benchmark_id": benchmark_id,
                "model_id": matrix.index.astype("string"),
                "answered_items": answered_items.astype("Int64").to_numpy(),
                "model_coverage": coverage.to_numpy(),
                "full_score": compute_model_scores(matrix).to_numpy(),
            },
        )
        .sort_values("model_id")
        .reset_index(drop=True)
    )


def effective_min_models_per_item(
    *,
    n_models_benchmark: int,
    min_models_per_item: int,
) -> int:
    """Return the v0.1 coverage floor for item retention."""

    return max(min_models_per_item, ceil(0.2 * n_models_benchmark))


def _safe_mean(series: pd.Series) -> float | None:
    if not series.notna().any():
        return None
    return float(series.mean(skipna=True))


def _safe_std(series: pd.Series) -> float | None:
    if not series.notna().any():
        return None
    return float(series.std(skipna=True, ddof=0))


def _pearson_correlation(lhs: pd.Series, rhs: pd.Series) -> float | None:
    if len(lhs.index) < 2:
        return None
    if lhs.nunique(dropna=True) < 2 or rhs.nunique(dropna=True) < 2:
        return None
    correlation = lhs.astype(float).corr(rhs.astype(float))
    if pd.isna(correlation):
        return None
    return float(correlation)
