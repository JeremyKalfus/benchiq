from __future__ import annotations

import pandas as pd

from benchiq.portfolio.standing import (
    build_equal_weight_ranking,
    build_leave_one_out_ranking,
)


def test_equal_weight_ranking_gives_each_source_one_vote() -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset_id": "source_a__snapshot",
                "strategy_id": "reconstruction_first__deterministic_info",
                "best_available_test_rmse_mean": 1.0,
                "best_available_test_mae_mean": 0.7,
                "best_available_test_pearson_mean": 0.9,
                "best_available_test_spearman_mean": 0.9,
                "seed_rmse_std": 0.1,
            },
            {
                "dataset_id": "source_b__snapshot",
                "strategy_id": "reconstruction_first__deterministic_info",
                "best_available_test_rmse_mean": 2.0,
                "best_available_test_mae_mean": 1.0,
                "best_available_test_pearson_mean": 0.8,
                "best_available_test_spearman_mean": 0.8,
                "seed_rmse_std": 0.2,
            },
            {
                "dataset_id": "source_a__snapshot",
                "strategy_id": "psychometric_default__random_cv",
                "best_available_test_rmse_mean": 1.5,
                "best_available_test_mae_mean": 1.0,
                "best_available_test_pearson_mean": 0.7,
                "best_available_test_spearman_mean": 0.7,
                "seed_rmse_std": 0.3,
            },
            {
                "dataset_id": "source_b__snapshot",
                "strategy_id": "psychometric_default__random_cv",
                "best_available_test_rmse_mean": 1.6,
                "best_available_test_mae_mean": 1.1,
                "best_available_test_pearson_mean": 0.6,
                "best_available_test_spearman_mean": 0.6,
                "seed_rmse_std": 0.3,
            },
        ]
    )
    ranking = build_equal_weight_ranking(summary)
    assert ranking.iloc[0]["strategy_id"] == "reconstruction_first__deterministic_info"
    assert ranking.iloc[0]["equal_weight_rmse_mean"] == 1.5


def test_leave_one_out_ranking_produces_one_winner_per_left_out_source() -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset_id": "source_a__snapshot",
                "strategy_id": "reconstruction_first__deterministic_info",
                "best_available_test_rmse_mean": 1.0,
                "best_available_test_mae_mean": 0.7,
                "best_available_test_pearson_mean": 0.9,
                "best_available_test_spearman_mean": 0.9,
                "seed_rmse_std": 0.1,
            },
            {
                "dataset_id": "source_b__snapshot",
                "strategy_id": "reconstruction_first__deterministic_info",
                "best_available_test_rmse_mean": 2.0,
                "best_available_test_mae_mean": 1.0,
                "best_available_test_pearson_mean": 0.8,
                "best_available_test_spearman_mean": 0.8,
                "seed_rmse_std": 0.2,
            },
            {
                "dataset_id": "source_a__snapshot",
                "strategy_id": "psychometric_default__random_cv",
                "best_available_test_rmse_mean": 1.4,
                "best_available_test_mae_mean": 1.0,
                "best_available_test_pearson_mean": 0.7,
                "best_available_test_spearman_mean": 0.7,
                "seed_rmse_std": 0.3,
            },
            {
                "dataset_id": "source_b__snapshot",
                "strategy_id": "psychometric_default__random_cv",
                "best_available_test_rmse_mean": 1.6,
                "best_available_test_mae_mean": 1.1,
                "best_available_test_pearson_mean": 0.6,
                "best_available_test_spearman_mean": 0.6,
                "seed_rmse_std": 0.3,
            },
        ]
    )
    leave_one_out = build_leave_one_out_ranking(summary)
    winners = leave_one_out.loc[leave_one_out["rank"] == 1]
    assert set(winners["left_out_dataset_id"]) == {"source_a__snapshot", "source_b__snapshot"}
