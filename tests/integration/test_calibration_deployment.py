import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import benchiq
from benchiq.cli.main import main


def test_calibrate_then_predict_reuses_saved_models_without_retraining(tmp_path) -> None:
    responses_path = _write_synthetic_bundle(tmp_path)
    config = _build_config()
    stage_options = _build_stage_options()

    calibration_result = benchiq.calibrate(
        responses_path,
        config,
        out_dir=tmp_path / "out",
        run_id="calibration-toy",
        stage_options=stage_options,
    )

    assert (calibration_result.calibration_root / "manifest.json").exists()
    reconstruction_report = json.loads(
        (
            calibration_result.calibration_root / "reconstruction_report.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        calibration_result.calibration_root
        / "per_benchmark"
        / "b1"
        / "reconstruction"
        / "marginal"
        / "gam_model.pkl"
    ).exists()
    assert (
        calibration_result.calibration_manifest["benchmarks"]["b1"]["preferred_model_type"]
        in {"marginal", "joint"}
    )
    assert calibration_result.calibration_manifest["benchmarks"]["b1"]["irt_backend"] == "girth"
    assert reconstruction_report["preferred_model_type_by_benchmark"]["b1"] in {"marginal", "joint"}
    assert reconstruction_report["benchmarks"]["b1"]["preferred_model"]["model_type"] in {
        "marginal",
        "joint",
    }

    reduced_responses_path = _write_reduced_selected_responses(
        tmp_path,
        responses_path=responses_path,
        calibration_manifest=calibration_result.calibration_manifest,
    )
    prediction_result = benchiq.predict(
        calibration_result.calibration_root,
        reduced_responses_path,
        out_dir=tmp_path / "predictions",
        run_id="predict-toy",
    )

    assert (
        prediction_result.run_root / "artifacts" / "01_predict" / "predictions.parquet"
    ).exists()
    assert (
        prediction_result.run_root
        / "artifacts"
        / "01_predict"
        / "predictions_best_available.parquet"
    ).exists()
    assert prediction_result.predictions_best_available["predicted_score"].notna().any()
    assert prediction_result.predictions_best_available["selected_model_type"].notna().all()

    expected_predictions = (
        pd.concat(
            [
                benchmark_result.predictions.loc[
                    :,
                    ["benchmark_id", "model_id", "model_type", "predicted_score"],
                ]
                for benchmark_result in calibration_result.run_result.stage_results[
                    "09_reconstruct"
                ].benchmarks.values()
            ],
            ignore_index=True,
        )
        .sort_values(["benchmark_id", "model_id", "model_type"])
        .reset_index(drop=True)
    )
    observed_predictions = (
        prediction_result.predictions.loc[
            prediction_result.predictions["prediction_available"],
            ["benchmark_id", "model_id", "model_type", "predicted_score"],
        ]
        .sort_values(["benchmark_id", "model_id", "model_type"])
        .reset_index(drop=True)
    )

    merged = expected_predictions.merge(
        observed_predictions,
        on=["benchmark_id", "model_id", "model_type"],
        suffixes=("_expected", "_observed"),
    )
    assert len(merged.index) == len(expected_predictions.index)
    np.testing.assert_allclose(
        merged["predicted_score_expected"].astype(float).to_numpy(),
        merged["predicted_score_observed"].astype(float).to_numpy(),
        atol=1e-8,
    )


def test_cli_calibrate_and_predict_write_artifacts(tmp_path) -> None:
    responses_path = _write_synthetic_bundle(tmp_path)
    config_path = _write_config_file(tmp_path)
    out_dir = tmp_path / "out"

    calibrate_result = CliRunner().invoke(
        main,
        [
            "calibrate",
            "--responses",
            str(responses_path),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--run-id",
            "cli-calibration",
        ],
    )

    assert calibrate_result.exit_code == 0, calibrate_result.output
    calibration_root = out_dir / "cli-calibration" / "calibration_bundle"
    assert (calibration_root / "manifest.json").exists()
    assert "calibration completed" in calibrate_result.output

    calibration_manifest = json.loads(
        (calibration_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert calibration_manifest["benchmarks"]["b1"]["irt_backend"] == "girth"
    reduced_responses_path = _write_reduced_selected_responses(
        tmp_path,
        responses_path=responses_path,
        calibration_manifest=calibration_manifest,
    )

    predict_result = CliRunner().invoke(
        main,
        [
            "predict",
            "--bundle",
            str(out_dir / "cli-calibration"),
            "--responses",
            str(reduced_responses_path),
            "--out",
            str(tmp_path / "predict_out"),
            "--run-id",
            "cli-predict",
        ],
    )

    assert predict_result.exit_code == 0, predict_result.output
    predict_root = tmp_path / "predict_out" / "cli-predict"
    assert (predict_root / "manifest.json").exists()
    assert (
        predict_root / "artifacts" / "01_predict" / "predictions_best_available.parquet"
    ).exists()
    assert "prediction completed" in predict_result.output


def test_load_calibration_bundle_fails_when_required_model_file_is_missing(tmp_path) -> None:
    responses_path = _write_synthetic_bundle(tmp_path)
    calibration_result = benchiq.calibrate(
        responses_path,
        _build_config(),
        out_dir=tmp_path / "out",
        run_id="missing-artifact",
        stage_options=_build_stage_options(),
    )
    missing_path = (
        calibration_result.calibration_root
        / "per_benchmark"
        / "b1"
        / "reconstruction"
        / "marginal"
        / "gam_model.pkl"
    )
    missing_path.unlink()

    with pytest.raises(FileNotFoundError, match="required calibration bundle artifact"):
        benchiq.load_calibration_bundle(calibration_result.calibration_root)


def _build_config() -> benchiq.BenchIQConfig:
    return benchiq.BenchIQConfig(
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


def _build_stage_options() -> dict[str, dict[str, object]]:
    return {
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
    }


def _write_config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "config": _build_config().model_dump(mode="json"),
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
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_reduced_selected_responses(
    tmp_path: Path,
    *,
    responses_path: Path,
    calibration_manifest: dict[str, object],
) -> Path:
    responses = pd.read_csv(responses_path)
    selected_frames: list[pd.DataFrame] = []
    for benchmark_id, benchmark_payload in calibration_manifest["benchmarks"].items():
        selected_item_ids = set(benchmark_payload["selected_item_ids"])
        selected_frames.append(
            responses.loc[
                (responses["benchmark_id"] == benchmark_id)
                & responses["item_id"].isin(selected_item_ids)
            ].copy()
        )
    reduced = pd.concat(selected_frames, ignore_index=True)
    path = tmp_path / "reduced_selected_responses.csv"
    reduced.to_csv(path, index=False)
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
    path = tmp_path / "calibration_fixture.csv"
    responses.to_csv(path, index=False)
    return path
