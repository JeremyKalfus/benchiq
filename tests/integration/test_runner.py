import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import benchiq


def test_runner_executes_full_pipeline_and_partial_rerun_deterministically(tmp_path) -> None:
    responses_path = _write_synthetic_bundle(tmp_path)
    config = benchiq.BenchIQConfig(
        allow_low_n=True,
        max_item_mean=0.99,
        min_abs_point_biserial=0.0,
        min_models_per_benchmark=15,
        warn_models_per_benchmark=20,
        min_items_after_filtering=5,
        min_models_per_item=10,
        min_overlap_models_for_joint=15,
        min_overlap_models_for_redundancy=15,
        random_seed=7,
    )
    stage_options = {
        "04_subsample": {
            "k_preselect": 4,
            "n_iter": 4,
            "cv_folds": 3,
            "checkpoint_interval": 2,
            "lam_grid": (0.1, 1.0),
        },
        "06_select": {
            "k_final": 3,
            "theta_grid_size": 101,
        },
        "07_theta": {
            "theta_grid_size": 81,
        },
        "09_reconstruct": {
            "lam_grid": (0.1, 1.0),
            "cv_folds": 3,
            "n_splines": 5,
        },
        "10_redundancy": {
            "lam_grid": (0.1, 1.0),
            "cv_folds": 3,
            "n_splines": 5,
            "n_factors_to_try": (1, 2),
        },
    }

    runner = benchiq.BenchIQRunner(
        config=config,
        out_dir=tmp_path / "out",
        run_id="runner-toy",
        stage_options=stage_options,
    )
    run_result = runner.run(responses_path)

    assert run_result.executed_stages == (
        "00_bundle",
        "01_preprocess",
        "02_scores",
        "03_splits",
        "04_subsample",
        "05_irt",
        "06_select",
        "07_theta",
        "08_linear",
        "08_features",
        "09_reconstruct",
        "10_redundancy",
    )
    summary = run_result.summary()
    assert {"executed_stages", "warnings", "skip_reasons", "metrics", "stage_records"} <= set(
        summary
    )
    assert summary["metrics"]["selected_items_by_benchmark"]
    assert "theta_correlation_summary" in summary["metrics"]
    assert "factor_analysis" in summary["metrics"]

    subset_b1 = run_result.load_artifact("artifacts/06_select/per_benchmark/b1/subset_final")
    assert isinstance(subset_b1, pd.DataFrame)
    assert len(subset_b1.index) == 3
    metrics_report = run_result.load_artifact("artifacts/reports/metrics")
    assert metrics_report["executed_stages"] == list(run_result.executed_stages)

    manifest = json.loads(run_result.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["stages"]) >= set(run_result.executed_stages)
    assert manifest["artifacts"]["reports"]["run_summary"].endswith("reports/run_summary.md")

    rerun_result = runner.run(start_at="06_select")
    assert rerun_result.executed_stages == (
        "06_select",
        "07_theta",
        "08_linear",
        "08_features",
        "09_reconstruct",
        "10_redundancy",
    )
    rerun_subset_b1 = rerun_result.load_artifact(
        "artifacts/06_select/per_benchmark/b1/subset_final",
    )
    assert subset_b1[["item_id"]].equals(rerun_subset_b1[["item_id"]])

    rerun_summary = rerun_result.summary()
    for benchmark_id, rmse in summary["metrics"]["marginal_test_rmse_by_benchmark"].items():
        rerun_metric = rerun_summary["metrics"]["marginal_test_rmse_by_benchmark"][benchmark_id]
        assert rerun_metric == pytest.approx(
            rmse,
        )
    for benchmark_id, rmse in summary["metrics"]["joint_test_rmse_by_benchmark"].items():
        if rmse is None:
            assert rerun_summary["metrics"]["joint_test_rmse_by_benchmark"][benchmark_id] is None
        else:
            rerun_metric = rerun_summary["metrics"]["joint_test_rmse_by_benchmark"][benchmark_id]
            assert rerun_metric == pytest.approx(
                rmse,
            )


def test_runner_serializes_missing_joint_metrics_when_joint_path_is_skipped(tmp_path) -> None:
    responses_path = _write_synthetic_bundle(tmp_path)
    runner = benchiq.BenchIQRunner(
        config=benchiq.BenchIQConfig(
            allow_low_n=True,
            max_item_mean=0.99,
            min_abs_point_biserial=0.0,
            min_models_per_benchmark=15,
            warn_models_per_benchmark=20,
            min_items_after_filtering=5,
            min_models_per_item=10,
            min_overlap_models_for_joint=1000,
            min_overlap_models_for_redundancy=1000,
            random_seed=7,
        ),
        out_dir=tmp_path / "out",
        run_id="runner-skip-joint",
        stage_options={
            "04_subsample": {
                "k_preselect": 4,
                "n_iter": 4,
                "cv_folds": 3,
                "checkpoint_interval": 2,
                "lam_grid": (0.1, 1.0),
            },
            "06_select": {
                "k_final": 3,
                "theta_grid_size": 101,
            },
            "07_theta": {
                "theta_grid_size": 81,
            },
            "09_reconstruct": {
                "lam_grid": (0.1, 1.0),
                "cv_folds": 3,
                "n_splines": 5,
            },
        },
    )

    run_result = runner.run(responses_path, stop_after="09_reconstruct")

    metrics_report = json.loads(
        (run_result.run_root / "reports" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics_report["metrics"]["joint_test_rmse_by_benchmark"]
    assert all(
        value is None
        for value in metrics_report["metrics"]["joint_test_rmse_by_benchmark"].values()
    )
    joint_stage_metrics = metrics_report["stage_records"]["09_reconstruct"]["metrics"][
        "joint_test_rmse_by_benchmark"
    ]
    assert joint_stage_metrics
    assert all(value is None for value in joint_stage_metrics.values())


def _write_synthetic_bundle(tmp_path) -> Path:
    rng = np.random.default_rng(123)
    benchmark_ids = ["b1", "b2", "b3"]
    model_ids = [f"m{index:03d}" for index in range(1, 61)]
    item_count = 7
    shared = np.linspace(-2.2, 2.2, len(model_ids))
    rows: list[dict[str, object]] = []

    for benchmark_index, benchmark_id in enumerate(benchmark_ids):
        unique = rng.normal(0.0, 1.0, size=len(model_ids))
        theta = 0.85 * shared + 0.15 * unique + rng.normal(0.0, 0.05, size=len(model_ids))
        difficulties = np.linspace(-1.5, 1.5, item_count) + benchmark_index * 0.1
        discriminations = np.linspace(0.9, 1.5, item_count)
        for item_index in range(item_count):
            difficulty = difficulties[item_index]
            discrimination = discriminations[item_index]
            probabilities = 1.0 / (1.0 + np.exp(-discrimination * (theta - difficulty)))
            outcomes = rng.binomial(1, probabilities)
            for model_id, score in zip(model_ids, outcomes, strict=True):
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "item_id": f"{benchmark_id}_item_{item_index + 1}",
                        "model_id": model_id,
                        "score": int(score),
                    }
                )

    responses = pd.DataFrame(rows).astype(
        {
            "benchmark_id": "string",
            "item_id": "string",
            "model_id": "string",
            "score": "Int64",
        }
    )
    path = tmp_path / "runner_fixture.csv"
    responses.to_csv(path, index=False)
    return path
