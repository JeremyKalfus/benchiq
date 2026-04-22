import json
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner

from benchiq.cli.main import main


def test_top_level_help_lists_public_commands() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    for command_name in ("validate", "calibrate", "predict", "run"):
        assert command_name in result.output


def test_validate_writes_validation_bundle_and_exits_zero(tmp_path) -> None:
    responses_path = _write_synthetic_bundle(tmp_path)
    config_path = _write_config(tmp_path)
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(
        main,
        [
            "validate",
            "--responses",
            str(responses_path),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    validation_root = out_dir / "validate"
    assert (validation_root / "manifest.json").exists()
    assert (validation_root / "artifacts" / "00_canonical" / "responses_long.parquet").exists()
    assert (
        validation_root
        / "artifacts"
        / "01_preprocess"
        / "per_benchmark"
        / "b1"
        / "preprocess_report.json"
    ).exists()
    report_path = validation_root / "reports" / "validation_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["preprocess_validation"]["refused_benchmarks"] == {}
    assert f"run location: {validation_root}" in result.output


def test_validate_writes_failure_artifacts_and_exits_nonzero_on_invalid_input(tmp_path) -> None:
    responses = pd.DataFrame(
        {
            "benchmark_id": ["b1"],
            "item_id": ["i1"],
            "model_id": ["m1"],
            "score": [2],
        }
    )
    responses_path = tmp_path / "invalid.csv"
    responses.to_csv(responses_path, index=False)
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(
        main,
        [
            "validate",
            "--responses",
            str(responses_path),
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code != 0
    assert "v0.1 requires dichotomous item scores" in result.output
    validation_root = out_dir / "validate"
    report_path = validation_root / "reports" / "validation_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["schema_validation"]["errors"][0]["code"] == "invalid_score_values"


def test_run_defaults_to_stage_09_without_redundancy(tmp_path) -> None:
    responses_path = _write_synthetic_bundle(tmp_path)
    config_path = _write_config(tmp_path)
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--responses",
            str(responses_path),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--run-id",
            "cli-toy",
        ],
    )

    assert result.exit_code == 0, result.output
    run_root = out_dir / "cli-toy"
    assert (run_root / "manifest.json").exists()
    assert (run_root / "reports" / "run_summary.md").exists()
    assert (run_root / "reports" / "metrics.json").exists()
    assert (
        run_root / "artifacts" / "09_reconstruct" / "per_benchmark" / "b1" / "predictions.parquet"
    ).exists()
    assert not (run_root / "artifacts" / "10_redundancy").exists()
    summary = json.loads((run_root / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert summary["executed_stages"][-1] == "09_reconstruct"
    assert "10_redundancy" not in summary["executed_stages"]
    assert "theta_correlation_summary" not in summary["metrics"]
    assert "factor_analysis" not in summary["metrics"]
    assert f"run location: {run_root}" in result.output
    assert "selected items by benchmark:" in result.output
    assert "marginal test rmse by benchmark:" in result.output


def test_run_with_redundancy_writes_stage_10_outputs_and_metrics(tmp_path) -> None:
    responses_path = _write_synthetic_bundle(tmp_path)
    config_path = _write_config(tmp_path)
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--responses",
            str(responses_path),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--run-id",
            "cli-toy-redundancy",
            "--with-redundancy",
        ],
    )

    assert result.exit_code == 0, result.output
    run_root = out_dir / "cli-toy-redundancy"
    assert (run_root / "artifacts" / "10_redundancy").exists()
    summary = json.loads((run_root / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert summary["executed_stages"][-1] == "10_redundancy"
    assert summary["metrics"]["theta_correlation_summary"]
    assert summary["metrics"]["factor_analysis"]


def test_run_exits_nonzero_on_invalid_input(tmp_path) -> None:
    responses = pd.DataFrame(
        {
            "benchmark_id": ["b1"],
            "item_id": ["i1"],
            "model_id": ["m1"],
            "score": [2],
        }
    )
    responses_path = tmp_path / "invalid.csv"
    responses.to_csv(responses_path, index=False)
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--responses",
            str(responses_path),
            "--out",
            str(out_dir),
            "--run-id",
            "cli-invalid",
        ],
    )

    assert result.exit_code != 0
    assert "run failed" in result.output
    assert "v0.1 requires dichotomous item scores" in result.output


def _write_config(tmp_path: Path) -> Path:
    config = {
        "config": {
            "allow_low_n": True,
            "max_item_mean": 0.99,
            "min_abs_point_biserial": 0.0,
            "min_models_per_benchmark": 15,
            "warn_models_per_benchmark": 20,
            "min_items_after_filtering": 5,
            "min_models_per_item": 10,
            "min_overlap_models_for_joint": 15,
            "min_overlap_models_for_redundancy": 15,
            "random_seed": 7,
        },
        "stage_options": {
            "04_subsample": {
                "k_preselect": 4,
                "n_iter": 4,
                "cv_folds": 3,
                "checkpoint_interval": 2,
                "lam_grid": [0.1, 1.0],
            },
            "06_select": {
                "k_final": 3,
                "theta_grid_size": 101,
            },
            "07_theta": {
                "theta_grid_size": 81,
            },
            "09_reconstruct": {
                "lam_grid": [0.1, 1.0],
                "cv_folds": 3,
                "n_splines": 5,
            },
            "10_redundancy": {
                "lam_grid": [0.1, 1.0],
                "cv_folds": 3,
                "n_splines": 5,
                "n_factors_to_try": [1, 2],
            },
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _write_synthetic_bundle(tmp_path: Path) -> Path:
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
    path = tmp_path / "cli_fixture.csv"
    responses.to_csv(path, index=False)
    return path
