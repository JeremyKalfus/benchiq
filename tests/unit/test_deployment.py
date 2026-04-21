import pandas as pd

from benchiq.deployment import _select_best_available_predictions


def test_select_best_available_predictions_prefers_calibrated_model_type() -> None:
    predictions = pd.DataFrame(
        [
            {
                "benchmark_id": "b1",
                "model_id": "m1",
                "model_type": "marginal",
                "predicted_score": 10.0,
                "prediction_available": True,
                "prediction_missing_reason": pd.NA,
            },
            {
                "benchmark_id": "b1",
                "model_id": "m1",
                "model_type": "joint",
                "predicted_score": 20.0,
                "prediction_available": True,
                "prediction_missing_reason": pd.NA,
            },
        ]
    )

    best = _select_best_available_predictions(
        predictions=predictions,
        benchmark_ids=["b1"],
        model_ids=["m1"],
        preferred_model_type_by_benchmark={"b1": "marginal"},
    )

    assert best.iloc[0]["selected_model_type"] == "marginal"
    assert float(best.iloc[0]["predicted_score"]) == 10.0


def test_select_best_available_predictions_falls_back_when_preferred_model_is_missing() -> None:
    predictions = pd.DataFrame(
        [
            {
                "benchmark_id": "b1",
                "model_id": "m1",
                "model_type": "marginal",
                "predicted_score": 10.0,
                "prediction_available": True,
                "prediction_missing_reason": pd.NA,
            },
            {
                "benchmark_id": "b1",
                "model_id": "m1",
                "model_type": "joint",
                "predicted_score": pd.NA,
                "prediction_available": False,
                "prediction_missing_reason": "missing_required_joint_features",
            },
        ]
    )

    best = _select_best_available_predictions(
        predictions=predictions,
        benchmark_ids=["b1"],
        model_ids=["m1"],
        preferred_model_type_by_benchmark={"b1": "joint"},
    )

    assert best.iloc[0]["selected_model_type"] == "marginal"
    assert float(best.iloc[0]["predicted_score"]) == 10.0
