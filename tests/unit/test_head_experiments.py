import importlib
import json

import numpy as np
import pandas as pd
import pytest

from benchiq.reconstruct import FeatureTableResult, run_reconstruction_head_experiments
from benchiq.reconstruct.head_experiments import (
    ReconstructionHeadExperimentResult,
    _write_experiment_artifacts,
)
from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID, SPLIT


def test_run_reconstruction_head_experiments_writes_comparison_artifacts(tmp_path) -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost is not installed in this environment")

    feature_result = _make_feature_result()
    result = run_reconstruction_head_experiments(
        feature_result,
        methods=("gam", "elastic_net", "xgboost"),
        seeds=(3, 5),
        lam_grid=(0.1, 1.0),
        cv_folds=3,
        n_splines=5,
        out_dir=tmp_path / "head-experiments",
    )

    assert not result.metrics.empty
    assert not result.predictions.empty
    assert not result.summary.empty
    assert set(result.summary["method"].astype(str)) == {"gam", "elastic_net", "xgboost"}
    assert result.artifact_paths["summary_md"].exists()
    assert result.artifact_paths["rmse_plot"].exists()
    assert result.artifact_paths["runtime_plot"].exists()
    assert result.artifact_paths["stability_plot"].exists()
    assert "marginal" in result.report["winners_by_model_type"]


def test_write_experiment_artifacts_serializes_missing_winner_metrics(tmp_path) -> None:
    result = ReconstructionHeadExperimentResult(
        metrics=pd.DataFrame(
            [
                {
                    "benchmark_id": "b1",
                    "model_type": "joint",
                    "method": "gam",
                    "seed": 7,
                    "split": "test",
                    "rmse": pd.NA,
                }
            ]
        ),
        predictions=pd.DataFrame(
            [
                {
                    BENCHMARK_ID: "b1",
                    MODEL_ID: "m1",
                    "model_type": "joint",
                    "method": "gam",
                    "seed": 7,
                    SPLIT: "test",
                    "actual_score": pd.NA,
                    "predicted_score": pd.NA,
                }
            ]
        ),
        summary=pd.DataFrame(
            [
                {
                    "model_type": "joint",
                    "method": "gam",
                    "rmse_mean": pd.NA,
                    "mae_mean": pd.NA,
                    "pearson_mean": pd.NA,
                    "spearman_mean": pd.NA,
                    "runtime_mean_seconds": 0.1,
                    "seed_rmse_std": pd.NA,
                }
            ]
        ),
        report={
            "methods": ["gam"],
            "seeds": [7],
            "summary_rows": 1,
            "detailed_rows": 1,
            "skip_reasons": {"b1:joint": "joint_feature_values_missing"},
            "winners_by_model_type": {
                "joint": {
                    "method": "gam",
                    "rmse_mean": pd.NA,
                    "runtime_mean_seconds": 0.1,
                }
            },
        },
    )

    artifact_paths = _write_experiment_artifacts(result, out_dir=tmp_path / "head-experiments")

    payload = json.loads(artifact_paths["report_json"].read_text(encoding="utf-8"))
    assert payload["winners_by_model_type"]["joint"]["rmse_mean"] is None
    assert artifact_paths["summary_md"].exists()


def _make_feature_result() -> FeatureTableResult:
    rng = np.random.default_rng(17)
    model_ids = [f"m{index:02d}" for index in range(1, 61)]
    rows: list[dict[str, object]] = []
    for index, model_id in enumerate(model_ids):
        theta = np.linspace(-2.0, 2.0, len(model_ids))[index]
        split_name = "train" if index < 30 else "val" if index < 45 else "test"
        rows.append(
            {
                BENCHMARK_ID: "b1",
                MODEL_ID: model_id,
                SPLIT: split_name,
                "score_full_b": float(
                    55.0 + 10.0 * np.tanh(theta) + 2.5 * theta**2 + rng.normal(0.0, 0.35)
                ),
                "theta_b": float(theta + rng.normal(0.0, 0.08)),
                "theta_se_b": float(0.25 + 0.03 * abs(theta)),
                "sub_b": float(50.0 + 18.0 * np.tanh(theta) + rng.normal(0.0, 0.7)),
                "lin_b": float(50.0 + 16.0 * np.tanh(theta) + rng.normal(0.0, 0.7)),
                "theta_method": "MAP",
            }
        )
    features_marginal = pd.DataFrame(rows).astype(
        {
            BENCHMARK_ID: "string",
            MODEL_ID: "string",
            SPLIT: "string",
            "score_full_b": "Float64",
            "theta_b": "Float64",
            "theta_se_b": "Float64",
            "sub_b": "Float64",
            "lin_b": "Float64",
            "theta_method": "string",
        }
    )
    return FeatureTableResult(
        features_marginal=features_marginal,
        features_joint=pd.DataFrame(),
        feature_report={
            "parameters": {"benchmark_ids": ["b1"]},
            "joint": {
                "skipped": True,
                "skip_reason": "joint_features_disabled_for_unit_test",
            },
            "warnings": [],
            "counts": {},
        },
    )
