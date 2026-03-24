import json
from itertools import product

import numpy as np
import pandas as pd

import benchiq
from benchiq.preprocess import compute_scores, preprocess_bundle
from benchiq.reconstruct import (
    fit_linear_predictor_bundle,
    fit_no_intercept_linear_predictor,
)
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID, SPLIT
from benchiq.select import BenchmarkSelectResult, SelectResult
from benchiq.split import SplitResult


def test_fit_no_intercept_linear_predictor_recovers_coefficients() -> None:
    X_train = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    expected = np.asarray([10.0, 20.0, 30.0], dtype=float)
    y_train = X_train @ expected

    result = fit_no_intercept_linear_predictor(
        X_train,
        y_train,
        feature_names=["i1", "i2", "i3"],
    )

    assert result["model_kind"] == "ols_no_intercept"
    assert result["fallback_used"] is False
    assert result["fallback_reason"] is None
    assert result["chosen_alpha"] == 0.0
    assert result["ridge_candidates"] == []
    assert result["train_rmse"] < 1e-12
    np.testing.assert_allclose(result["coefficients"], expected, atol=1e-10)


def test_fit_no_intercept_linear_predictor_falls_back_to_ridge() -> None:
    X_train = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ],
        dtype=float,
    )
    y_train = np.asarray([15.0, 30.0, 45.0, 60.0], dtype=float)

    result = fit_no_intercept_linear_predictor(
        X_train,
        y_train,
        feature_names=["i1", "i2"],
        ridge_alpha_grid=(1e-6, 1e-3, 1.0),
    )

    assert result["model_kind"] == "ridge_no_intercept"
    assert result["fallback_used"] is True
    assert result["fallback_reason"] == "rank_deficient_design_matrix"
    assert result["chosen_alpha"] in {1e-6, 1e-3, 1.0}
    assert len(result["ridge_candidates"]) == 3
    assert result["train_rmse"] >= 0.0
    assert np.isfinite(result["coefficients"]).all()


def test_fit_linear_predictor_bundle_writes_artifacts_and_subscores(tmp_path) -> None:
    item_ids = ["i1", "i2", "i3", "i4"]
    rows: list[dict[str, object]] = []
    for model_index, pattern in enumerate(product([0, 1], repeat=4), start=1):
        model_id = f"m{model_index:02d}"
        for item_id, score in zip(item_ids, pattern, strict=True):
            stored_score: int | None = int(score)
            if model_id == "m16" and item_id == "i4":
                stored_score = None
            rows.append(
                {
                    BENCHMARK_ID: "b1",
                    ITEM_ID: item_id,
                    MODEL_ID: model_id,
                    "score": stored_score,
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
        min_item_coverage=0.75,
        min_item_sd=0.0,
        max_item_mean=1.0,
        min_abs_point_biserial=0.0,
        min_overlap_models_for_joint=1,
        random_seed=7,
    )
    bundle = benchiq.load_bundle(
        responses_path,
        config=config,
        out_dir=tmp_path / "out",
        run_id="linear-toy",
    )
    preprocess_result = preprocess_bundle(bundle)
    score_result = compute_scores(bundle, preprocess_result)

    split_rows = []
    for model_index in range(1, 17):
        model_id = f"m{model_index:02d}"
        split = "train" if model_index <= 12 else "val" if model_index <= 14 else "test"
        split_rows.append(
            {
                BENCHMARK_ID: "b1",
                MODEL_ID: model_id,
                SPLIT: split,
            }
        )
    benchmark_split = pd.DataFrame(split_rows).astype(
        {
            BENCHMARK_ID: "string",
            MODEL_ID: "string",
            SPLIT: "string",
        }
    )
    split_result = SplitResult(
        splits_models=pd.DataFrame({MODEL_ID: benchmark_split[MODEL_ID].copy()}),
        per_benchmark_splits={"b1": benchmark_split},
        split_report={},
    )

    subset_final = pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(["b1"] * len(item_ids), dtype="string"),
            ITEM_ID: pd.Series(item_ids, dtype="string"),
        }
    )
    select_result = SelectResult(
        benchmarks={
            "b1": BenchmarkSelectResult(
                benchmark_id="b1",
                subset_final=subset_final,
                info_grid=pd.DataFrame(),
                selection_report={
                    "skipped": False,
                    "skipped_reason": None,
                    "warnings": [],
                },
            )
        }
    )

    linear_result = fit_linear_predictor_bundle(
        bundle,
        score_result,
        split_result,
        select_result,
    )

    benchmark_result = linear_result.benchmarks["b1"]
    model_outputs = benchmark_result.model_outputs.sort_values(MODEL_ID).reset_index(drop=True)
    coefficients = benchmark_result.coefficients.sort_values(ITEM_ID).reset_index(drop=True)

    assert {"sub_b", "lin_b"} <= set(model_outputs.columns)
    predicted_rows = model_outputs.dropna(subset=["lin_b"])
    np.testing.assert_allclose(
        predicted_rows["lin_b"].astype(float).to_numpy(),
        predicted_rows["score_full"].astype(float).to_numpy(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        coefficients["coefficient"].astype(float).to_numpy(),
        np.asarray([25.0, 25.0, 25.0, 25.0], dtype=float),
        atol=1e-10,
    )
    assert benchmark_result.linear_predictor_report["training_diagnostics"]["model_kind"] == (
        "ols_no_intercept"
    )
    assert (
        benchmark_result.linear_predictor_report["training_diagnostics"]["fallback_used"] is False
    )
    assert benchmark_result.linear_predictor_report["counts"]["training_eligible_count"] == 12

    missing_row = model_outputs.loc[model_outputs[MODEL_ID] == "m16"].iloc[0]
    assert pd.isna(missing_row["lin_b"])
    assert missing_row["linear_prediction_missing_reason"] == "insufficient_reduced_item_coverage"
    assert float(missing_row["reduced_item_coverage"]) == 0.75
    assert float(missing_row["sub_b"]) == 100.0

    stage_dir = (
        tmp_path / "out" / "linear-toy" / "artifacts" / "08_features" / "per_benchmark" / "b1"
    )
    assert (stage_dir / "model_outputs.parquet").exists()
    assert (stage_dir / "coefficients.parquet").exists()
    assert (stage_dir / "linear_predictor_report.json").exists()
    assert (
        tmp_path / "out" / "linear-toy" / "artifacts" / "08_features" / "feature_report.json"
    ).exists()

    report = json.loads((stage_dir / "linear_predictor_report.json").read_text(encoding="utf-8"))
    assert report["parameters"]["selected_item_count"] == 4
    assert report["parameters"]["ridge_alpha_grid"] == [1e-06, 0.0001, 0.01, 0.1, 1.0, 10.0]
    assert report["training_diagnostics"]["model_kind"] == "ols_no_intercept"
    assert report["training_diagnostics"]["train_row_count"] == 12
    assert report["counts"]["predicted_model_count"] == 15

    feature_report = json.loads(
        (
            tmp_path / "out" / "linear-toy" / "artifacts" / "08_features" / "feature_report.json"
        ).read_text(encoding="utf-8")
    )
    assert feature_report["counts"]["benchmark_count"] == 1
    assert feature_report["counts"]["skipped_benchmark_count"] == 0
