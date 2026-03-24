import json

import numpy as np
import pandas as pd
from girth.synthetic import create_synthetic_irt_dichotomous
from scipy.stats import spearmanr

import benchiq
from benchiq.irt import estimate_theta_bundle, fit_irt_bundle
from benchiq.irt.theta import estimate_theta_responses
from benchiq.preprocess import compute_scores, preprocess_bundle
from benchiq.select import select_bundle
from benchiq.split import split_models
from benchiq.subsample import subsample_bundle


def test_estimate_theta_responses_handles_edge_patterns_and_eap() -> None:
    item_params = pd.DataFrame(
        {
            "item_id": pd.Series(["i1", "i2", "i3", "i4"], dtype="string"),
            "discrimination": pd.Series([1.0, 1.2, 1.3, 1.5], dtype="Float64"),
            "difficulty": pd.Series([-1.0, -0.2, 0.5, 1.1], dtype="Float64"),
        }
    )
    theta_grid = np.linspace(-4.0, 4.0, 161)

    all_wrong = estimate_theta_responses(
        responses=pd.Series([0, 0, 0, 0], index=item_params["item_id"]),
        item_params=item_params,
        theta_method="MAP",
        theta_grid=theta_grid,
        missing_heavy_threshold=0.5,
    )
    all_correct = estimate_theta_responses(
        responses=pd.Series([1, 1, 1, 1], index=item_params["item_id"]),
        item_params=item_params,
        theta_method="MAP",
        theta_grid=theta_grid,
        missing_heavy_threshold=0.5,
    )
    missing_heavy = estimate_theta_responses(
        responses=pd.Series([1, np.nan, np.nan, np.nan], index=item_params["item_id"]),
        item_params=item_params,
        theta_method="MAP",
        theta_grid=theta_grid,
        missing_heavy_threshold=0.5,
    )
    eap_mixed = estimate_theta_responses(
        responses=pd.Series([0, 1, 1, 0], index=item_params["item_id"]),
        item_params=item_params,
        theta_method="EAP",
        theta_grid=theta_grid,
        missing_heavy_threshold=0.5,
    )
    no_observed = estimate_theta_responses(
        responses=pd.Series([np.nan, np.nan, np.nan, np.nan], index=item_params["item_id"]),
        item_params=item_params,
        theta_method="MAP",
        theta_grid=theta_grid,
        missing_heavy_threshold=0.5,
    )

    assert np.isfinite(all_wrong["theta_hat"])
    assert np.isfinite(all_correct["theta_hat"])
    assert np.isfinite(all_wrong["theta_se"])
    assert np.isfinite(all_correct["theta_se"])
    assert all_wrong["response_pattern"] == "all_wrong"
    assert all_correct["response_pattern"] == "all_correct"
    assert all_wrong["theta_hat"] < eap_mixed["theta_hat"] < all_correct["theta_hat"]
    assert missing_heavy["missing_heavy"] is True
    assert np.isfinite(missing_heavy["theta_hat"])
    assert no_observed["response_pattern"] == "no_observed_items"
    assert no_observed["theta_hat"] == 0.0
    assert no_observed["theta_se"] is None


def test_estimate_theta_bundle_writes_artifacts_and_monotone_thetas(tmp_path) -> None:
    difficulty = np.array([-1.2, -0.7, -0.2, 0.3, 0.8, 1.2], dtype=float)
    discrimination = np.array([0.8, 1.0, 1.2, 1.3, 1.5, 1.8], dtype=float)
    thetas = np.linspace(-2.5, 2.5, 60)
    responses = create_synthetic_irt_dichotomous(
        difficulty=difficulty,
        discrimination=discrimination,
        thetas=thetas,
        seed=29,
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
        n_strata_bins=4,
        random_seed=11,
    )
    bundle = benchiq.load_bundle(
        responses_path,
        config=config,
        out_dir=tmp_path / "out",
        run_id="theta-toy",
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
        k_final=3,
        n_bins=3,
        theta_grid_size=201,
    )
    theta_result = estimate_theta_bundle(
        bundle,
        split_result,
        select_result,
        irt_result,
        theta_method="MAP",
    )

    theta_estimates = theta_result.theta_estimates
    assert {"train", "val", "test"} <= set(theta_estimates["split"].astype(str))
    assert theta_estimates["theta_hat"].dropna().map(np.isfinite).all()

    valid_rows = theta_estimates.dropna(subset=["theta_hat", "reduced_score"])
    rho = spearmanr(
        valid_rows["reduced_score"].astype(float).to_numpy(),
        valid_rows["theta_hat"].astype(float).to_numpy(),
    ).statistic
    assert rho is not None and rho > 0.6

    stage_dir = tmp_path / "out" / "theta-toy" / "artifacts" / "07_theta"
    assert (stage_dir / "theta_estimates.parquet").exists()
    assert (stage_dir / "theta_report.json").exists()
    assert (stage_dir / "plots" / "b1__theta_distribution.png").exists()

    report = json.loads((stage_dir / "theta_report.json").read_text(encoding="utf-8"))
    assert report["parameters"]["theta_method"] == "MAP"
    assert report["counts"]["benchmark_count"] == 1
    assert report["counts"]["split_counts"]["train"] > 0
    assert report["counts"]["split_counts"]["val"] > 0
    assert report["counts"]["split_counts"]["test"] > 0
