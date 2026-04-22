import json

import numpy as np
import pandas as pd
from girth.synthetic import create_synthetic_irt_dichotomous

import benchiq
from benchiq.irt import BenchmarkIRTResult, IRTResult, fit_irt_bundle
from benchiq.irt.info import FISHER_INFORMATION, fisher_information_2pl
from benchiq.preprocess import compute_scores, preprocess_bundle
from benchiq.select import select_benchmark, select_bundle
from benchiq.split import split_models
from benchiq.subsample import subsample_bundle


def test_fisher_information_is_nonnegative_and_peaks_near_difficulty() -> None:
    theta = np.linspace(-3.0, 3.0, 1201)
    difficulty = 0.7
    information = fisher_information_2pl(
        theta,
        discrimination=1.4,
        difficulty=difficulty,
    )

    assert np.all(information >= 0.0)
    peak_theta = float(theta[int(np.argmax(information))])
    assert abs(peak_theta - difficulty) < 0.02


def test_select_benchmark_picks_expected_items_across_theta_bins() -> None:
    item_params = pd.DataFrame(
        {
            "benchmark_id": pd.Series(["b1"] * 4, dtype="string"),
            "item_id": pd.Series(["i_low", "i_midlow", "i_midhigh", "i_high"], dtype="string"),
            "irt_backend": pd.Series(["bayes_mcmc"] * 4, dtype="string"),
            "discrimination": pd.Series([1.8, 0.8, 0.7, 1.7], dtype="Float64"),
            "difficulty": pd.Series([-1.6, -0.3, 0.4, 1.5], dtype="Float64"),
            "pathology_warning": pd.Series([False] * 4, dtype=bool),
            "pathology_warning_reasons": pd.Series([[] for _ in range(4)], dtype=object),
            "pathology_excluded": pd.Series([False] * 4, dtype=bool),
            "pathology_excluded_reasons": pd.Series([[] for _ in range(4)], dtype=object),
        }
    )
    benchmark_result = BenchmarkIRTResult(
        benchmark_id="b1",
        irt_item_params=item_params,
        dropped_pathological_items=item_params.iloc[0:0].copy(),
        irt_fit_report={"skipped": False},
        ability_estimates=pd.DataFrame(),
    )
    irt_result = IRTResult(benchmarks={"b1": benchmark_result})

    selection_result = select_benchmark(
        irt_result,
        benchmark_id="b1",
        k_final=2,
        n_bins=2,
        theta_grid_type="uniform",
        theta_min=-2.5,
        theta_max=2.5,
        theta_grid_size=401,
    )

    assert selection_result.subset_final["item_id"].tolist() == ["i_low", "i_high"]
    assert not selection_result.subset_final["item_id"].duplicated().any()
    assert len(selection_result.subset_final.index) == 2
    assert (selection_result.info_grid[FISHER_INFORMATION] >= 0.0).all()


def test_select_bundle_writes_artifacts_and_expected_plot(tmp_path) -> None:
    difficulty = np.array([-1.1, -0.6, -0.1, 0.4, 0.9, 1.3], dtype=float)
    discrimination = np.array([0.8, 1.0, 1.1, 1.3, 1.5, 1.9], dtype=float)
    thetas = np.linspace(-2.2, 2.2, 40)
    responses = create_synthetic_irt_dichotomous(
        difficulty=difficulty,
        discrimination=discrimination,
        thetas=thetas,
        seed=23,
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
        random_seed=5,
    )
    bundle = benchiq.load_bundle(
        responses_path,
        config=config,
        out_dir=tmp_path / "out",
        run_id="select-toy",
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
    select_result = select_bundle(
        bundle,
        irt_result,
        k_final=2,
        n_bins=2,
        theta_grid_size=201,
    )

    benchmark_result = select_result.benchmarks["b1"]
    assert len(benchmark_result.subset_final.index) == 2
    assert not benchmark_result.subset_final["item_id"].duplicated().any()
    assert len(benchmark_result.info_grid.index) == 4 * 201
    assert benchmark_result.selection_report["counts"]["selected_item_count"] == 2

    stage_dir = tmp_path / "out" / "select-toy" / "artifacts" / "06_select" / "per_benchmark" / "b1"
    assert (stage_dir / "subset_final.parquet").exists()
    assert (stage_dir / "info_grid.parquet").exists()
    assert (stage_dir / "selection_report.json").exists()
    assert (stage_dir / "plots" / "expected_test_information.png").exists()

    report = json.loads((stage_dir / "selection_report.json").read_text(encoding="utf-8"))
    assert report["skipped"] is False
    assert report["artifacts"]["plots_written"] is True
    assert report["artifacts"]["plots_reason"] is None
    assert report["artifacts"]["expected_test_information_plot"].endswith(
        "plots/expected_test_information.png"
    )
    subset = pd.read_parquet(stage_dir / "subset_final.parquet")
    assert len(subset.index) == 2
    assert not subset["item_id"].duplicated().any()
