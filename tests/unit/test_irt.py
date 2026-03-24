import json

import numpy as np
import pandas as pd
import pytest
from girth.synthetic import create_synthetic_irt_dichotomous
from scipy.stats import spearmanr

import benchiq
from benchiq.irt import fit_irt_bundle
from benchiq.irt.backends import fit_girth_2pl, girth_backend
from benchiq.preprocess import compute_scores, preprocess_bundle
from benchiq.split import split_models
from benchiq.subsample import subsample_bundle


def test_fit_girth_2pl_recovers_rank_order_on_synthetic_data() -> None:
    difficulty = np.array([-1.2, -0.7, -0.2, 0.4, 0.9, 1.3], dtype=float)
    discrimination = np.array([0.7, 0.9, 1.1, 1.4, 1.8, 2.1], dtype=float)
    thetas = np.linspace(-2.5, 2.5, 300)
    responses = create_synthetic_irt_dichotomous(
        difficulty=difficulty,
        discrimination=discrimination,
        thetas=thetas,
        seed=5,
    )

    rows: list[dict[str, object]] = []
    item_ids = [f"i{index + 1}" for index in range(responses.shape[0])]
    model_ids = [f"m{index + 1:03d}" for index in range(responses.shape[1])]
    for item_index, item_id in enumerate(item_ids):
        for model_index, model_id in enumerate(model_ids):
            rows.append(
                {
                    "benchmark_id": "b1",
                    "item_id": item_id,
                    "model_id": model_id,
                    "score": int(responses[item_index, model_index]),
                }
            )

    result = fit_girth_2pl(
        pd.DataFrame(rows),
        benchmark_id="b1",
        item_ids=item_ids,
        model_ids=model_ids,
        options={"max_iteration": 40},
    )

    estimated = result.item_params.sort_values("item_id").reset_index(drop=True)
    diff_corr = spearmanr(difficulty, estimated["difficulty"].to_numpy()).statistic
    disc_corr = spearmanr(discrimination, estimated["discrimination"].to_numpy()).statistic

    assert diff_corr is not None and diff_corr > 0.8
    assert disc_corr is not None and disc_corr > 0.7
    assert not estimated["pathology_excluded"].any()
    assert result.fit_report["counts"]["retained_item_count"] == len(item_ids)
    assert result.fit_report["warnings"][0]["code"] == "backend_convergence_status_unavailable"
    assert result.fit_report["convergence"]["status_available"] is False


def test_fit_girth_2pl_artifacts_convergence_limitation_and_drops_pathological_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"benchmark_id": "b1", "item_id": "i1", "model_id": "m1", "score": 1},
        {"benchmark_id": "b1", "item_id": "i1", "model_id": "m2", "score": 0},
        {"benchmark_id": "b1", "item_id": "i2", "model_id": "m1", "score": 1},
        {"benchmark_id": "b1", "item_id": "i2", "model_id": "m2", "score": 1},
        {"benchmark_id": "b1", "item_id": "i3", "model_id": "m1", "score": 0},
        {"benchmark_id": "b1", "item_id": "i3", "model_id": "m2", "score": 1},
    ]

    def fake_twopl_mml(dataset, options=None):  # type: ignore[no-untyped-def]
        return {
            "Discrimination": np.array([0.8, 0.01, 6.0], dtype=float),
            "Difficulty": np.array([0.2, np.inf, -0.3], dtype=float),
            "Ability": np.array([0.1, -0.2], dtype=float),
            "AIC": {"final": 12.0, "null": 14.0, "delta": 2.0},
            "BIC": {"final": 13.0, "null": 15.0, "delta": 2.0},
        }

    monkeypatch.setattr(girth_backend, "twopl_mml", fake_twopl_mml)

    result = fit_girth_2pl(
        pd.DataFrame(rows),
        benchmark_id="b1",
        item_ids=["i1", "i2", "i3"],
        model_ids=["m1", "m2"],
    )

    assert result.item_params["item_id"].tolist() == ["i1", "i3"]
    assert result.dropped_pathological_items["item_id"].tolist() == ["i2"]
    assert result.dropped_pathological_items["pathology_excluded_reasons"].iloc[0] == [
        "low_discrimination_excluded",
        "nonfinite_difficulty",
    ]
    assert result.fit_report["counts"]["retained_item_count"] == 2
    assert result.fit_report["counts"]["pathology_excluded_count"] == 1
    assert result.fit_report["pathology"]["excluded_item_ids"] == ["i2"]
    assert result.fit_report["pathology"]["retained_item_ids"] == ["i1", "i3"]
    assert result.fit_report["warnings"][0]["code"] == "backend_convergence_status_unavailable"
    assert result.fit_report["warnings"][1]["code"] == "pathological_items_dropped"
    assert result.fit_report["artifacts"]["dropped_pathological_items_written"] is True


def test_fit_irt_bundle_writes_artifacts_and_expected_columns(tmp_path) -> None:
    difficulty = np.array([-1.0, -0.5, 0.0, 0.4, 0.8, 1.2], dtype=float)
    discrimination = np.array([0.8, 1.0, 1.2, 1.3, 1.5, 1.8], dtype=float)
    thetas = np.linspace(-2.0, 2.0, 36)
    responses = create_synthetic_irt_dichotomous(
        difficulty=difficulty,
        discrimination=discrimination,
        thetas=thetas,
        seed=11,
    )

    rows: list[dict[str, object]] = []
    for item_index in range(responses.shape[0]):
        item_id = f"i{item_index + 1}"
        for model_index in range(responses.shape[1]):
            model_id = f"m{model_index + 1:02d}"
            rows.append(
                {
                    "benchmark_id": "b1",
                    "item_id": item_id,
                    "model_id": model_id,
                    "score": int(responses[item_index, model_index]),
                }
            )

    responses_path = tmp_path / "responses.csv"
    pd.DataFrame(rows).to_csv(responses_path, index=False)

    config = benchiq.BenchIQConfig(
        allow_low_n=True,
        drop_low_tail_models_quantile=0.0,
        min_models_per_benchmark=1,
        warn_models_per_benchmark=1,
        min_items_after_filtering=1,
        min_models_per_item=1,
        min_item_coverage=1.0,
        min_item_sd=0.0,
        max_item_mean=1.0,
        min_abs_point_biserial=0.0,
        min_overlap_models_for_joint=6,
        p_test=0.2,
        p_val=0.25,
        n_strata_bins=3,
        random_seed=3,
    )
    bundle = benchiq.load_bundle(
        responses_path,
        config=config,
        out_dir=tmp_path / "out",
        run_id="irt-toy",
    )
    preprocess_result = preprocess_bundle(bundle)
    score_result = compute_scores(bundle, preprocess_result)
    split_result = split_models(bundle, score_result)
    subsample_result = subsample_bundle(
        bundle,
        preprocess_result,
        score_result,
        split_result,
        k_preselect=4,
        n_iter=4,
        cv_folds=4,
        checkpoint_interval=2,
        lam_grid=(0.1, 1.0),
    )

    irt_result = fit_irt_bundle(
        bundle,
        split_result,
        subsample_result,
        backend_options={"max_iteration": 30},
    )

    benchmark_result = irt_result.benchmarks["b1"]
    expected_columns = {
        "benchmark_id",
        "item_id",
        "irt_backend",
        "discrimination",
        "difficulty",
        "pathology_warning",
        "pathology_warning_reasons",
        "pathology_excluded",
        "pathology_excluded_reasons",
    }
    assert set(benchmark_result.irt_item_params.columns) == expected_columns
    assert len(benchmark_result.irt_item_params.index) == 4
    assert benchmark_result.dropped_pathological_items.empty
    assert benchmark_result.irt_fit_report["irt_backend"] == "girth"
    assert benchmark_result.irt_fit_report["convergence"]["backend_exposes_flag"] is False
    assert benchmark_result.irt_fit_report["convergence"]["status_available"] is False
    assert (
        benchmark_result.irt_fit_report["warnings"][0]["code"]
        == "backend_convergence_status_unavailable"
    )
    assert benchmark_result.irt_fit_report["counts"]["train_model_count"] > 0

    stage_dir = tmp_path / "out" / "irt-toy" / "artifacts" / "05_irt" / "per_benchmark" / "b1"
    assert (stage_dir / "irt_item_params.parquet").exists()
    assert (stage_dir / "dropped_pathological_items.parquet").exists()
    assert (stage_dir / "irt_fit_report.json").exists()
    assert (stage_dir / "ability_estimates.parquet").exists()

    report = json.loads((stage_dir / "irt_fit_report.json").read_text(encoding="utf-8"))
    assert report["counts"]["preselect_item_count"] == 4
    assert report["skipped"] is False
    dropped = pd.read_parquet(stage_dir / "dropped_pathological_items.parquet")
    assert dropped.empty
