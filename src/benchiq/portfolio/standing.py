"""Summary helpers for the narrowed portfolio standing pass."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _ensure_strategy_id(summary: pd.DataFrame) -> pd.DataFrame:
    """Derive the strategy id from optimization summary columns when needed."""

    if summary.empty or "strategy_id" in summary.columns:
        return summary
    required = {"profile_id", "preselection_method"}
    if not required.issubset(summary.columns):
        raise KeyError("strategy_id")
    frame = summary.copy()
    frame["strategy_id"] = (
        frame["profile_id"].astype("string")
        + "__"
        + frame["preselection_method"].astype("string")
    )
    return frame


def build_equal_weight_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one metric row per source so dense sources do not dominate by row count."""

    summary = _ensure_strategy_id(summary)
    if summary.empty:
        return summary
    rows: list[dict[str, Any]] = []
    for strategy_id, group in summary.groupby("strategy_id", sort=True):
        informative = group.dropna(subset=["best_available_test_rmse_mean"]).copy()
        if informative.empty:
            continue
        rows.append(
            {
                "strategy_id": strategy_id,
                "equal_weight_rmse_mean": informative["best_available_test_rmse_mean"].mean(),
                "equal_weight_mae_mean": informative["best_available_test_mae_mean"].mean(),
                "equal_weight_pearson_mean": informative[
                    "best_available_test_pearson_mean"
                ].mean(),
                "equal_weight_spearman_mean": informative[
                    "best_available_test_spearman_mean"
                ].mean(),
                "equal_weight_seed_rmse_std_mean": informative["seed_rmse_std"].mean(),
                "informative_source_count": int(len(informative.index)),
            }
        )
    ranking = pd.DataFrame.from_records(rows)
    if ranking.empty:
        return ranking
    return ranking.sort_values(
        ["equal_weight_rmse_mean", "equal_weight_seed_rmse_std_mean", "strategy_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_leave_one_out_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """Recompute the equal-weight ranking with each optimization source left out in turn."""

    summary = _ensure_strategy_id(summary)
    if summary.empty:
        return summary
    dataset_ids = sorted(summary["dataset_id"].dropna().astype("string").unique().tolist())
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        subset = summary.loc[summary["dataset_id"].astype("string") != dataset_id].copy()
        ranking = build_equal_weight_ranking(subset)
        for rank, row in enumerate(ranking.to_dict(orient="records"), start=1):
            rows.append(
                {
                    "left_out_dataset_id": dataset_id,
                    "rank": rank,
                    **row,
                }
            )
    return pd.DataFrame.from_records(rows)
