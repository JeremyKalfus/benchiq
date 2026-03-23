import json

import pandas as pd
import pytest

import benchiq
from benchiq.preprocess import preprocess_bundle
from benchiq.preprocess.filters import apply_item_filter_flags
from benchiq.preprocess.stats import compute_item_stats, select_low_tail_model_ids


def _manual_pearson_correlation(lhs: list[float], rhs: list[float]) -> float:
    lhs_mean = sum(lhs) / len(lhs)
    rhs_mean = sum(rhs) / len(rhs)
    numerator = sum((left - lhs_mean) * (right - rhs_mean) for left, right in zip(lhs, rhs))
    lhs_ss = sum((left - lhs_mean) ** 2 for left in lhs)
    rhs_ss = sum((right - rhs_mean) ** 2 for right in rhs)
    return numerator / ((lhs_ss * rhs_ss) ** 0.5)


def test_preprocess_bundle_filters_expected_items_and_writes_artifacts(tmp_path) -> None:
    records = {
        "m1": {"i_keep": 0, "i_keep_2": 0, "i_easy": 1, "i_low_disc": 0, "i_sparse": 0},
        "m2": {"i_keep": 0, "i_keep_2": 0, "i_easy": 1, "i_low_disc": 0},
        "m3": {"i_keep": 0, "i_keep_2": 0, "i_easy": 1, "i_low_disc": 0},
        "m4": {"i_keep": 0, "i_keep_2": 0, "i_easy": 1, "i_low_disc": 0},
        "m5": {"i_keep": 1, "i_easy": 1, "i_low_disc": 1},
        "m6": {"i_keep": 1, "i_keep_2": 1, "i_easy": 1, "i_low_disc": 0},
        "m7": {"i_keep": 1, "i_keep_2": 1, "i_easy": 1, "i_low_disc": 0},
        "m8": {"i_keep": 1, "i_keep_2": 1, "i_easy": 1, "i_low_disc": 0, "i_sparse": 1},
    }
    responses = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "benchmark_id": "b1",
                "item_id": item_id,
                "score": score,
            }
            for model_id, model_scores in records.items()
            for item_id, score in model_scores.items()
        ],
    )
    responses_path = tmp_path / "responses.csv"
    responses.to_csv(responses_path, index=False)

    config = benchiq.BenchIQConfig(
        allow_low_n=True,
        drop_low_tail_models_quantile=0.0,
        min_models_per_benchmark=1,
        warn_models_per_benchmark=1,
        min_items_after_filtering=1,
        min_models_per_item=3,
        min_item_coverage=0.75,
    )
    bundle = benchiq.load_bundle(
        responses_path,
        config=config,
        out_dir=tmp_path / "out",
        run_id="preprocess-toy",
    )

    result = preprocess_bundle(bundle)

    benchmark_result = result.benchmarks["b1"]
    assert benchmark_result.filtered_items["item_id"].tolist() == ["i_keep", "i_keep_2"]
    assert benchmark_result.filtered_models["model_id"].tolist() == [
        "m1",
        "m2",
        "m3",
        "m4",
        "m6",
        "m7",
        "m8",
    ]
    assert not benchmark_result.refused

    stage_dir = tmp_path / "out" / "preprocess-toy" / "artifacts" / "01_preprocess"
    assert (stage_dir / "summary.parquet").exists()
    assert (stage_dir / "per_benchmark" / "b1" / "filtered_items.parquet").exists()
    assert (stage_dir / "per_benchmark" / "b1" / "filtered_models.parquet").exists()
    assert (stage_dir / "per_benchmark" / "b1" / "item_stats.parquet").exists()
    assert (stage_dir / "per_benchmark" / "b1" / "preprocess_report.json").exists()

    report = json.loads(
        (stage_dir / "per_benchmark" / "b1" / "preprocess_report.json").read_text(encoding="utf-8"),
    )
    assert report["counts"]["dropped_near_ceiling_items"] == 1
    assert report["counts"]["dropped_low_discrimination_items"] == 2
    assert report["counts"]["dropped_low_coverage_items"] == 1
    assert report["counts"]["coverage_models_dropped"] == 1


def test_preprocess_bundle_refuses_when_items_too_small_after_filtering(tmp_path) -> None:
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

    result = preprocess_bundle(bundle)

    summary = result.summary.iloc[0]
    assert bool(summary["refused"])
    assert summary["refusal_reasons"] == "too_few_items_after_filtering"


def test_preprocess_bundle_requires_allow_low_n_for_small_benchmark(tmp_path) -> None:
    responses = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m2", "m2", "m3", "m3"],
            "benchmark_id": ["b1", "b1", "b1", "b1", "b1", "b1"],
            "item_id": ["i1", "i2", "i1", "i2", "i1", "i2"],
            "score": [0, 0, 1, 1, 1, 1],
        },
    )
    responses_path = tmp_path / "responses.csv"
    responses.to_csv(responses_path, index=False)

    config = benchiq.BenchIQConfig(
        allow_low_n=False,
        drop_low_tail_models_quantile=0.0,
        min_models_per_benchmark=1,
        warn_models_per_benchmark=5,
        min_items_after_filtering=1,
        min_models_per_item=1,
        min_item_coverage=0.5,
        min_item_sd=0.0,
        max_item_mean=1.0,
        min_abs_point_biserial=0.0,
    )
    bundle = benchiq.load_bundle(responses_path, config=config)

    result = preprocess_bundle(bundle)

    summary = result.summary.iloc[0]
    assert bool(summary["refused"])
    assert summary["refusal_reasons"] == "allow_low_n_required"


@pytest.mark.parametrize("n_models", [2, 10, 999])
def test_select_low_tail_model_ids_skips_tiny_benchmarks(n_models: int) -> None:
    model_scores = pd.Series(
        [float(index) for index in range(n_models)],
        index=[f"m{index}" for index in range(n_models)],
        dtype="Float64",
    )

    assert select_low_tail_model_ids(model_scores, quantile=0.001) == []


def test_select_low_tail_model_ids_trims_when_quantile_supports_one_model() -> None:
    model_scores = pd.Series(
        [float(index) for index in range(1000)],
        index=[f"m{index:04d}" for index in range(1000)],
        dtype="Float64",
    )

    assert select_low_tail_model_ids(model_scores, quantile=0.001) == ["m0000"]


def test_preprocess_bundle_default_low_tail_does_not_trim_tiny_benchmark(tmp_path) -> None:
    responses = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m1", "m2", "m2", "m2"],
            "benchmark_id": ["b1", "b1", "b1", "b1", "b1", "b1"],
            "item_id": ["i1", "i2", "i3", "i1", "i2", "i3"],
            "score": [0, 0, 0, 1, 1, 1],
        },
    )
    responses_path = tmp_path / "responses.csv"
    responses.to_csv(responses_path, index=False)

    config = benchiq.BenchIQConfig(
        allow_low_n=True,
        min_models_per_benchmark=1,
        warn_models_per_benchmark=1,
        min_items_after_filtering=1,
        min_models_per_item=1,
        min_item_coverage=0.5,
        min_item_sd=0.0,
        max_item_mean=1.0,
        min_abs_point_biserial=0.0,
    )
    bundle = benchiq.load_bundle(responses_path, config=config)

    result = preprocess_bundle(bundle)

    benchmark_result = result.benchmarks["b1"]
    assert benchmark_result.preprocess_report["counts"]["low_tail_models_dropped"] == 0
    assert benchmark_result.preprocess_report["dropped_model_ids"]["low_tail"] == []
    assert benchmark_result.filtered_models["model_id"].tolist() == ["m1", "m2"]


def test_compute_item_stats_matches_manual_point_biserial_without_missing_data() -> None:
    matrix = pd.DataFrame(
        {
            "i1": pd.Series([0, 0, 1, 1, 1], index=["m1", "m2", "m3", "m4", "m5"], dtype="Float64"),
            "i2": pd.Series([0, 1, 0, 1, 1], index=["m1", "m2", "m3", "m4", "m5"], dtype="Float64"),
            "i3": pd.Series([0, 0, 1, 1, 0], index=["m1", "m2", "m3", "m4", "m5"], dtype="Float64"),
        },
    )

    item_stats = compute_item_stats(matrix, benchmark_id="b1")
    point_biserial = float(
        item_stats.loc[item_stats["item_id"] == "i1", "point_biserial"].iloc[0],
    )

    assert point_biserial == pytest.approx(
        _manual_pearson_correlation(
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0, 1.0],
        ),
    )


def test_compute_item_stats_matches_part_whole_point_biserial_with_missing_data() -> None:
    matrix = pd.DataFrame(
        {
            "i1": pd.Series([0, 0, 1, 1], index=["m1", "m2", "m3", "m4"], dtype="Float64"),
            "i2": pd.Series([0, 0, 1, 1], index=["m1", "m2", "m3", "m4"], dtype="Float64"),
            "i3": pd.Series([0, 1, pd.NA, 1], index=["m1", "m2", "m3", "m4"], dtype="Float64"),
        },
    )

    item_stats = compute_item_stats(matrix, benchmark_id="b1")
    point_biserial = float(
        item_stats.loc[item_stats["item_id"] == "i1", "point_biserial"].iloc[0],
    )

    assert point_biserial == pytest.approx(0.7071067811865475)


def test_apply_item_filter_flags_uses_spec_threshold_boundaries() -> None:
    config = benchiq.BenchIQConfig()
    item_stats = pd.DataFrame(
        [
            {
                "benchmark_id": "b1",
                "item_id": "sd_edge",
                "n_responses": 50,
                "item_coverage": 1.0,
                "mean": 0.50,
                "sd": 0.01,
                "point_biserial": 0.20,
            },
            {
                "benchmark_id": "b1",
                "item_id": "mean_edge",
                "n_responses": 50,
                "item_coverage": 1.0,
                "mean": 0.95,
                "sd": 0.20,
                "point_biserial": 0.20,
            },
            {
                "benchmark_id": "b1",
                "item_id": "disc_below",
                "n_responses": 50,
                "item_coverage": 1.0,
                "mean": 0.50,
                "sd": 0.20,
                "point_biserial": 0.049,
            },
            {
                "benchmark_id": "b1",
                "item_id": "disc_edge",
                "n_responses": 50,
                "item_coverage": 1.0,
                "mean": 0.50,
                "sd": 0.20,
                "point_biserial": 0.05,
            },
            {
                "benchmark_id": "b1",
                "item_id": "coverage_below",
                "n_responses": 49,
                "item_coverage": 0.98,
                "mean": 0.50,
                "sd": 0.20,
                "point_biserial": 0.20,
            },
        ],
    )

    flagged = apply_item_filter_flags(
        item_stats,
        min_item_sd=config.min_item_sd,
        max_item_mean=config.max_item_mean,
        min_abs_point_biserial=config.min_abs_point_biserial,
        min_models_per_item=config.min_models_per_item,
    )
    indexed = flagged.set_index("item_id")

    assert bool(indexed.loc["sd_edge", "drop_low_variance"])
    assert bool(indexed.loc["mean_edge", "drop_near_ceiling"])
    assert bool(indexed.loc["disc_below", "drop_low_discrimination"])
    assert not bool(indexed.loc["disc_edge", "drop_low_discrimination"])
    assert bool(indexed.loc["coverage_below", "drop_item_coverage"])
    assert bool(indexed.loc["disc_edge", "retained_after_item_filters"])
