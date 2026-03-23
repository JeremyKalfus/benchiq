import json

import pandas as pd

import benchiq
from benchiq.preprocess import compute_scores, preprocess_bundle


def test_compute_scores_writes_artifacts_and_expected_percent_tables(tmp_path) -> None:
    responses = pd.DataFrame(
        {
            "model_id": [
                "m1",
                "m1",
                "m2",
                "m2",
                "m3",
                "m3",
                "m1",
                "m1",
                "m2",
                "m2",
                "m3",
                "m3",
            ],
            "benchmark_id": [
                "b1",
                "b1",
                "b1",
                "b1",
                "b1",
                "b1",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
            ],
            "item_id": [
                "i1",
                "i2",
                "i1",
                "i2",
                "i1",
                "i2",
                "j1",
                "j2",
                "j1",
                "j2",
                "j1",
                "j2",
            ],
            "score": [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0],
        },
    )
    responses_path = tmp_path / "responses.csv"
    responses.to_csv(responses_path, index=False)

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
        min_overlap_models_for_joint=2,
    )
    bundle = benchiq.load_bundle(
        responses_path,
        config=config,
        out_dir=tmp_path / "out",
        run_id="scores-toy",
    )
    preprocess_result = preprocess_bundle(bundle)

    score_result = compute_scores(bundle, preprocess_result)

    scores_full = score_result.scores_full.set_index(["benchmark_id", "model_id"])
    assert scores_full.loc[("b1", "m1"), "score_full"] == 50.0
    assert scores_full.loc[("b1", "m2"), "score_full"] == 0.0
    assert scores_full.loc[("b1", "m3"), "score_full"] == 100.0
    assert scores_full.loc[("b2", "m1"), "score_full"] == 100.0
    assert scores_full.loc[("b2", "m2"), "score_full"] == 50.0
    assert scores_full.loc[("b2", "m3"), "score_full"] == 50.0
    assert scores_full["score_missing_reason"].isna().all()

    scores_grand = score_result.scores_grand.set_index("model_id")
    assert scores_grand.loc["m1", "grand_mean_score"] == 75.0
    assert scores_grand.loc["m2", "grand_mean_score"] == 25.0
    assert scores_grand.loc["m3", "grand_mean_score"] == 75.0

    stage_dir = tmp_path / "out" / "scores-toy" / "artifacts" / "02_scores"
    assert (stage_dir / "scores_full.parquet").exists()
    assert (stage_dir / "scores_grand.parquet").exists()
    assert (stage_dir / "score_report.json").exists()

    report = json.loads((stage_dir / "score_report.json").read_text(encoding="utf-8"))
    assert report["counts"]["scores_full_valid"] == 6
    assert report["counts"]["scores_grand_rows"] == 3
    assert report["grand_scores"]["skipped"] is False
    assert report["warnings"] == []


def test_compute_scores_records_low_coverage_missingness_and_grand_skip(tmp_path) -> None:
    responses = pd.DataFrame(
        {
            "model_id": [
                "m1",
                "m1",
                "m2",
                "m2",
                "m3",
                "m3",
                "m1",
                "m1",
                "m2",
                "m3",
                "m3",
            ],
            "benchmark_id": [
                "b1",
                "b1",
                "b1",
                "b1",
                "b1",
                "b1",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
            ],
            "item_id": [
                "i1",
                "i2",
                "i1",
                "i2",
                "i1",
                "i2",
                "j1",
                "j2",
                "j1",
                "j1",
                "j2",
            ],
            "score": [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        },
    )
    responses_path = tmp_path / "responses.csv"
    responses.to_csv(responses_path, index=False)

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
        min_overlap_models_for_joint=3,
    )
    bundle = benchiq.load_bundle(responses_path, config=config)
    preprocess_result = preprocess_bundle(bundle)

    score_result = compute_scores(bundle, preprocess_result)

    scores_full = score_result.scores_full.set_index(["benchmark_id", "model_id"])
    assert pd.isna(scores_full.loc[("b2", "m2"), "score_full"])
    assert scores_full.loc[("b2", "m2"), "score_missing_reason"] == "insufficient_item_coverage"
    assert score_result.scores_grand.empty
    assert score_result.score_report["benchmarks"]["b2"]["missing_low_coverage_count"] == 1
    assert score_result.score_report["grand_scores"]["complete_overlap_model_count"] == 2
    assert (
        score_result.score_report["grand_scores"]["skip_reason"] == "overlap_below_joint_threshold"
    )
    assert any(
        "grand scores skipped" in warning for warning in score_result.score_report["warnings"]
    )


def test_compute_scores_marks_refused_benchmarks_explicitly(tmp_path) -> None:
    responses = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m2", "m2"],
            "benchmark_id": ["b1", "b1", "b1", "b1"],
            "item_id": ["i1", "i2", "i1", "i2"],
            "score": [1, 1, 1, 0],
        },
    )
    responses_path = tmp_path / "responses.csv"
    responses.to_csv(responses_path, index=False)

    config = benchiq.BenchIQConfig(
        allow_low_n=True,
        min_models_per_benchmark=1,
        warn_models_per_benchmark=1,
        min_items_after_filtering=2,
        min_models_per_item=1,
        min_item_coverage=0.5,
    )
    bundle = benchiq.load_bundle(responses_path, config=config)
    preprocess_result = preprocess_bundle(bundle)

    score_result = compute_scores(bundle, preprocess_result)

    assert score_result.scores_full["score_full"].isna().all()
    assert (score_result.scores_full["score_missing_reason"] == "benchmark_refused").all()
    assert score_result.score_report["benchmarks"]["b1"]["missing_benchmark_refused_count"] == 2
    assert score_result.score_report["grand_scores"]["skip_reason"] == "no_complete_bundle_overlap"
