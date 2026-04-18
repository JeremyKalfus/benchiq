import json

import pandas as pd

import benchiq
from benchiq.preprocess import compute_scores, preprocess_bundle
from benchiq.split import split_models
from benchiq.subsample import subsample_bundle


def test_subsample_bundle_writes_artifacts_and_respects_k_preselect(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    thresholds = {"i1": 2, "i2": 4, "i3": 6, "i4": 8, "i5": 10, "i6": 12}
    for model_index in range(1, 13):
        model_id = f"m{model_index:02d}"
        for item_id, threshold in thresholds.items():
            rows.append(
                {
                    "model_id": model_id,
                    "benchmark_id": "b1",
                    "item_id": item_id,
                    "score": 1 if model_index >= threshold else 0,
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
        min_overlap_models_for_joint=4,
        p_test=0.25,
        p_val=0.25,
        n_strata_bins=2,
        random_seed=9,
    )
    bundle = benchiq.load_bundle(
        responses_path,
        config=config,
        out_dir=tmp_path / "out",
        run_id="subsample-toy",
    )
    preprocess_result = preprocess_bundle(bundle)
    score_result = compute_scores(bundle, preprocess_result)
    split_result = split_models(bundle, score_result)

    subsample_result = subsample_bundle(
        bundle,
        preprocess_result,
        score_result,
        split_result,
        k_preselect=3,
        n_iter=5,
        cv_folds=4,
        checkpoint_interval=2,
        lam_grid=(0.1, 1.0),
    )

    benchmark_result = subsample_result.benchmarks["b1"]
    assert len(benchmark_result.preselect_items.index) == 3
    assert benchmark_result.preselect_items["item_id"].nunique() == 3
    assert benchmark_result.subsample_report["counts"]["valid_iterations"] == 5
    assert benchmark_result.subsample_report["counts"]["failed_iterations"] == 0
    assert len(benchmark_result.cv_results.index) == 20
    assert benchmark_result.subsample_report["best_iteration"]["iteration_id"] in range(5)

    stage_dir = (
        tmp_path / "out" / "subsample-toy" / "artifacts" / "04_subsample" / "per_benchmark" / "b1"
    )
    assert (stage_dir / "preselect_items.parquet").exists()
    assert (stage_dir / "cv_results.parquet").exists()
    assert (stage_dir / "subsample_report.json").exists()
    assert (stage_dir / "progress.json").exists()

    report = json.loads((stage_dir / "subsample_report.json").read_text(encoding="utf-8"))
    assert report["parameters"]["effective_k_preselect"] == 3
    assert report["counts"]["cv_rows"] == 20

    progress = json.loads((stage_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_iterations"] == 5
    assert progress["best_iteration"]["iteration_id"] in range(5)


def test_subsample_bundle_counts_failed_iterations_from_missing_reduced_scores(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    dense_thresholds = {"i_dense_a": 2, "i_dense_b": 6}
    for model_index in range(1, 9):
        model_id = f"m{model_index:02d}"
        for item_id, threshold in dense_thresholds.items():
            rows.append(
                {
                    "model_id": model_id,
                    "benchmark_id": "b1",
                    "item_id": item_id,
                    "score": 1 if model_index >= threshold else 0,
                }
            )

    rows.extend(
        [
            {
                "model_id": "m01",
                "benchmark_id": "b1",
                "item_id": "i_sparse",
                "score": 0,
            },
            {
                "model_id": "m02",
                "benchmark_id": "b1",
                "item_id": "i_sparse",
                "score": 1,
            },
        ]
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
        min_item_coverage=0.5,
        min_item_sd=0.0,
        max_item_mean=1.0,
        min_abs_point_biserial=0.0,
        min_overlap_models_for_joint=3,
        p_test=0.25,
        p_val=0.25,
        n_strata_bins=2,
        random_seed=4,
    )
    bundle = benchiq.load_bundle(responses_path, config=config)
    preprocess_result = preprocess_bundle(bundle)
    score_result = compute_scores(bundle, preprocess_result)
    split_result = split_models(bundle, score_result)

    subsample_result = subsample_bundle(
        bundle,
        preprocess_result,
        score_result,
        split_result,
        k_preselect=1,
        n_iter=6,
        cv_folds=4,
        checkpoint_interval=1,
        lam_grid=(0.1, 1.0),
    )

    benchmark_result = subsample_result.benchmarks["b1"]
    assert benchmark_result.subsample_report["counts"]["failed_iterations"] > 0
    assert benchmark_result.subsample_report["counts"]["valid_iterations"] > 0
    assert (benchmark_result.cv_results["status"] == "failed").any()
    assert (
        benchmark_result.cv_results.loc[
            benchmark_result.cv_results["status"] == "failed",
            "failed_reason",
        ]
        == "too_few_valid_models_for_cv"
    ).any()
    assert len(benchmark_result.preselect_items.index) == 1


def test_subsample_bundle_supports_deterministic_information_ranking(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    thresholds = {"i1": 2, "i2": 4, "i3": 6, "i4": 8, "i5": 10, "i6": 12}
    responses_path = tmp_path / "responses.csv"
    for model_index in range(1, 13):
        model_id = f"m{model_index:02d}"
        for item_id, threshold in thresholds.items():
            rows.append(
                {
                    "model_id": model_id,
                    "benchmark_id": "b1",
                    "item_id": item_id,
                    "score": 1 if model_index >= threshold else 0,
                }
            )
    pd.DataFrame(rows).to_csv(responses_path, index=False)

    selected_item_sets: list[list[str]] = []
    for random_seed in (9, 21):
        bundle = benchiq.load_bundle(
            responses_path,
            config=benchiq.BenchIQConfig(
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
                min_overlap_models_for_joint=4,
                p_test=0.25,
                p_val=0.25,
                n_strata_bins=2,
                random_seed=random_seed,
            ),
            out_dir=tmp_path / "out",
            run_id=f"deterministic-{random_seed}",
        )
        preprocess_result = preprocess_bundle(bundle)
        score_result = compute_scores(bundle, preprocess_result)
        split_result = split_models(bundle, score_result)
        subsample_result = subsample_bundle(
            bundle,
            preprocess_result,
            score_result,
            split_result,
            method="deterministic_info",
            k_preselect=3,
            cv_folds=4,
            lam_grid=(0.1, 1.0),
        )
        benchmark_result = subsample_result.benchmarks["b1"]
        selected_items = (
            benchmark_result.preselect_items["item_id"]
            .dropna()
            .astype("string")
            .sort_values()
            .tolist()
        )
        selected_item_sets.append(selected_items)
        assert benchmark_result.subsample_report["method"] == "deterministic_info"
        assert benchmark_result.subsample_report["counts"]["valid_iterations"] == 1
        assert benchmark_result.subsample_report["counts"]["failed_iterations"] == 0
        assert len(selected_items) == 3
        assert benchmark_result.ranking_table["selected"].sum() == 3

        stage_dir = (
            tmp_path
            / "out"
            / f"deterministic-{random_seed}"
            / "artifacts"
            / "04_subsample"
            / "per_benchmark"
            / "b1"
        )
        assert (stage_dir / "ranking_table.parquet").exists()

    assert selected_item_sets[0] == selected_item_sets[1]
