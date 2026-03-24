import json

import pandas as pd

import benchiq
from benchiq.preprocess import compute_scores, preprocess_bundle
from benchiq.split import split_models


def test_split_models_writes_artifacts_and_uses_global_test_split(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    for model_index in range(1, 11):
        model_id = f"m{model_index:02d}"
        b1_scores = {
            "i1": 1 if model_index >= 4 else 0,
            "i2": 1 if model_index >= 7 else 0,
        }
        b2_scores = {
            "j1": 1 if model_index >= 3 else 0,
            "j2": 1 if model_index >= 8 else 0,
        }
        for item_id, score in b1_scores.items():
            rows.append(
                {
                    "model_id": model_id,
                    "benchmark_id": "b1",
                    "item_id": item_id,
                    "score": score,
                },
            )
        for item_id, score in b2_scores.items():
            rows.append(
                {
                    "model_id": model_id,
                    "benchmark_id": "b2",
                    "item_id": item_id,
                    "score": score,
                },
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
        p_test=0.2,
        p_val=0.25,
        n_strata_bins=2,
        random_seed=7,
    )
    bundle = benchiq.load_bundle(
        responses_path,
        config=config,
        out_dir=tmp_path / "out",
        run_id="split-toy",
    )
    preprocess_result = preprocess_bundle(bundle)
    score_result = compute_scores(bundle, preprocess_result)

    split_result = split_models(bundle, score_result)

    global_splits = split_result.splits_models.set_index("model_id")
    test_model_ids = set(global_splits.index[global_splits["global_split"] == "test"].tolist())
    assert split_result.split_report["global_test"]["enabled"] is True
    assert split_result.split_report["global_test"]["assigned_test_count"] == 2
    assert len(test_model_ids) == 2
    assert (global_splits["global_split"] == "benchmark_local_only").sum() == 0

    b1_splits = split_result.per_benchmark_splits["b1"]
    b2_splits = split_result.per_benchmark_splits["b2"]
    assert len(b1_splits.index) == 10
    assert len(b2_splits.index) == 10
    assert set(b1_splits.loc[b1_splits["split"] == "test", "model_id"].tolist()) == test_model_ids
    assert set(b2_splits.loc[b2_splits["split"] == "test", "model_id"].tolist()) == test_model_ids
    assert int((b1_splits["split"] == "val").sum()) == 2
    assert int((b2_splits["split"] == "val").sum()) == 2
    assert int((b1_splits["split"] == "train").sum()) == 6
    assert int((b2_splits["split"] == "train").sum()) == 6
    assert not b1_splits["model_id"].duplicated().any()
    assert not b2_splits["model_id"].duplicated().any()

    stage_dir = tmp_path / "out" / "split-toy" / "artifacts" / "03_splits"
    assert (stage_dir / "splits_models.parquet").exists()
    assert (stage_dir / "per_benchmark" / "b1" / "splits_models.parquet").exists()
    assert (stage_dir / "per_benchmark" / "b2" / "splits_models.parquet").exists()
    assert (stage_dir / "split_report.json").exists()

    report = json.loads((stage_dir / "split_report.json").read_text(encoding="utf-8"))
    assert report["global_test"]["enabled"] is True
    assert report["global_test"]["skip_reason"] is None


def test_split_models_falls_back_to_benchmark_local_when_global_overlap_is_too_small(
    tmp_path,
) -> None:
    rows: list[dict[str, object]] = []
    for model_index in range(1, 7):
        model_id = f"m{model_index:02d}"
        b1_scores = {
            "i1": 1 if model_index >= 3 else 0,
            "i2": 1 if model_index >= 5 else 0,
        }
        b2_scores = {
            "j1": 1 if model_index >= 2 else 0,
            "j2": None if model_index >= 5 else 1 if model_index >= 4 else 0,
        }
        for item_id, score in b1_scores.items():
            rows.append(
                {
                    "model_id": model_id,
                    "benchmark_id": "b1",
                    "item_id": item_id,
                    "score": score,
                },
            )
        for item_id, score in b2_scores.items():
            if score is None:
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "benchmark_id": "b2",
                    "item_id": item_id,
                    "score": score,
                },
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
        min_overlap_models_for_joint=5,
        p_test=0.2,
        p_val=0.25,
        n_strata_bins=2,
        random_seed=3,
    )
    bundle = benchiq.load_bundle(responses_path, config=config)
    preprocess_result = preprocess_bundle(bundle)
    score_result = compute_scores(bundle, preprocess_result)

    split_result = split_models(bundle, score_result)

    assert split_result.split_report["global_test"]["enabled"] is False
    assert (
        split_result.split_report["global_test"]["skip_reason"] == "overlap_below_joint_threshold"
    )
    assert int((split_result.splits_models["global_split"] == "test").sum()) == 0
    assert (split_result.splits_models["global_split"] == "benchmark_local_only").all()
    assert int((split_result.per_benchmark_splits["b1"]["split"] == "test").sum()) == 0
    assert int((split_result.per_benchmark_splits["b2"]["split"] == "test").sum()) == 0
    assert (
        split_result.split_report["benchmarks"]["b1"]["fallback_reason"] == "benchmark_local_only"
    )
    assert (
        split_result.split_report["benchmarks"]["b2"]["fallback_reason"] == "benchmark_local_only"
    )
