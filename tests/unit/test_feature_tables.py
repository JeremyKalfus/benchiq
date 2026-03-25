import json

import pandas as pd

import benchiq
from benchiq.irt.theta import THETA_HAT, THETA_METHOD, THETA_SE, ThetaResult
from benchiq.preprocess.scores import GRAND_MEAN_SCORE, SCORE_FULL, ScoreResult
from benchiq.reconstruct import build_feature_tables
from benchiq.reconstruct.linear_predictor import (
    LINEAR_PREDICTION,
    REDUCED_SUBSCORE,
    BenchmarkLinearPredictorResult,
    LinearPredictorResult,
)
from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID, SPLIT
from benchiq.split import SplitResult


def test_build_feature_tables_writes_marginal_and_joint_artifacts(tmp_path) -> None:
    bundle = _load_feature_bundle(tmp_path)
    score_result = _make_score_result(joint_enabled=True)
    split_result = _make_split_result()
    theta_result = _make_theta_result()
    linear_result = _make_linear_result()

    feature_result = build_feature_tables(
        bundle,
        score_result,
        split_result,
        theta_result,
        linear_result,
    )

    features_marginal = feature_result.features_marginal.sort_values(
        [BENCHMARK_ID, MODEL_ID]
    ).reset_index(drop=True)
    features_joint = feature_result.features_joint.sort_values(
        [BENCHMARK_ID, MODEL_ID]
    ).reset_index(drop=True)

    assert len(features_marginal.index) == 6
    assert {"score_full_b", "theta_b", "theta_se_b", "sub_b", "lin_b"} <= set(
        features_marginal.columns
    )
    assert len(features_joint.index) == 6
    assert {"theta_b1", "theta_b2", "grand_sub", "grand_lin"} <= set(features_joint.columns)

    b1_test_row = features_marginal.loc[
        (features_marginal[BENCHMARK_ID] == "b1") & (features_marginal[MODEL_ID] == "m3")
    ].iloc[0]
    b2_same_model_row = features_marginal.loc[
        (features_marginal[BENCHMARK_ID] == "b2") & (features_marginal[MODEL_ID] == "m3")
    ].iloc[0]
    assert b1_test_row[SPLIT] == "test"
    assert b2_same_model_row[SPLIT] == "val"
    assert float(b1_test_row["lin_b"]) == 56.0

    b1_joint_row = features_joint.loc[
        (features_joint[BENCHMARK_ID] == "b1") & (features_joint[MODEL_ID] == "m3")
    ].iloc[0]
    assert b1_joint_row[SPLIT] == "test"
    assert float(b1_joint_row["theta_b1"]) == 0.8
    assert float(b1_joint_row["theta_b2"]) == 0.3
    assert float(b1_joint_row["lin_b"]) == 56.0
    assert float(b1_joint_row["grand_sub"]) == 57.5
    assert float(b1_joint_row["grand_lin"]) == 58.0

    stage_dir = tmp_path / "out" / "features-toy" / "artifacts" / "08_features"
    assert (stage_dir / "features_marginal.parquet").exists()
    assert (stage_dir / "features_joint.parquet").exists()
    assert (stage_dir / "feature_report.json").exists()

    report = json.loads((stage_dir / "feature_report.json").read_text(encoding="utf-8"))
    assert report["joint"]["skipped"] is False
    assert report["joint"]["complete_feature_model_count"] == 3
    assert report["counts"]["features_marginal_rows"] == 6
    assert report["counts"]["features_joint_rows"] == 6


def test_build_feature_tables_artifacts_joint_skip_reason_when_unavailable(tmp_path) -> None:
    bundle = _load_feature_bundle(tmp_path, run_id="features-skip")
    score_result = _make_score_result(joint_enabled=False)
    split_result = _make_split_result()
    theta_result = _make_theta_result()
    linear_result = _make_linear_result()

    feature_result = build_feature_tables(
        bundle,
        score_result,
        split_result,
        theta_result,
        linear_result,
    )

    assert len(feature_result.features_marginal.index) == 6
    assert feature_result.features_joint.empty
    assert feature_result.feature_report["joint"]["skipped"] is True
    assert feature_result.feature_report["joint"]["skip_reason"] == "overlap_below_joint_threshold"

    stage_dir = tmp_path / "out" / "features-skip" / "artifacts" / "08_features"
    assert (stage_dir / "features_marginal.parquet").exists()
    assert not (stage_dir / "features_joint.parquet").exists()

    report = json.loads((stage_dir / "feature_report.json").read_text(encoding="utf-8"))
    assert report["joint"]["skipped"] is True
    assert report["joint"]["skip_reason"] == "overlap_below_joint_threshold"


def _load_feature_bundle(tmp_path, *, run_id: str = "features-toy"):
    responses = pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(
                ["b1", "b1", "b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2", "b2", "b2"],
                dtype="string",
            ),
            "item_id": pd.Series(
                ["i1", "i1", "i2", "i2", "i1", "i2", "j1", "j1", "j2", "j2", "j1", "j2"],
                dtype="string",
            ),
            MODEL_ID: pd.Series(
                ["m1", "m2", "m1", "m2", "m3", "m3", "m1", "m2", "m1", "m2", "m3", "m3"],
                dtype="string",
            ),
            "score": pd.Series([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0], dtype="Int64"),
        }
    )
    responses_path = tmp_path / f"{run_id}.csv"
    responses.to_csv(responses_path, index=False)
    config = benchiq.BenchIQConfig(
        allow_low_n=True,
        min_overlap_models_for_joint=2,
    )
    return benchiq.load_bundle(
        responses_path,
        config=config,
        out_dir=tmp_path / "out",
        run_id=run_id,
    )


def _make_score_result(*, joint_enabled: bool) -> ScoreResult:
    scores_full = pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(["b1", "b1", "b1", "b2", "b2", "b2"], dtype="string"),
            MODEL_ID: pd.Series(["m1", "m2", "m3", "m1", "m2", "m3"], dtype="string"),
            SCORE_FULL: pd.Series([62.0, 58.0, 54.0, 68.0, 63.0, 59.0], dtype="Float64"),
        }
    )
    if joint_enabled:
        scores_grand = pd.DataFrame(
            {
                MODEL_ID: pd.Series(["m1", "m2", "m3"], dtype="string"),
                GRAND_MEAN_SCORE: pd.Series([65.0, 60.5, 56.5], dtype="Float64"),
                "benchmark_count": pd.Series([2, 2, 2], dtype="Int64"),
            }
        )
        grand_report = {
            "complete_overlap_model_count": 3,
            "required_benchmark_count": 2,
            "required_benchmark_ids": ["b1", "b2"],
            "min_overlap_models_for_joint": 2,
            "overlap_model_ids": ["m1", "m2", "m3"],
            "skipped": False,
            "skip_reason": None,
        }
    else:
        scores_grand = pd.DataFrame(
            {
                MODEL_ID: pd.Series(dtype="string"),
                GRAND_MEAN_SCORE: pd.Series(dtype="Float64"),
                "benchmark_count": pd.Series(dtype="Int64"),
            }
        )
        grand_report = {
            "complete_overlap_model_count": 1,
            "required_benchmark_count": 2,
            "required_benchmark_ids": ["b1", "b2"],
            "min_overlap_models_for_joint": 2,
            "overlap_model_ids": [],
            "skipped": True,
            "skip_reason": "overlap_below_joint_threshold",
        }

    return ScoreResult(
        scores_full=scores_full,
        scores_grand=scores_grand,
        score_report={
            "grand_scores": grand_report,
            "warnings": [],
        },
    )


def _make_split_result() -> SplitResult:
    b1 = pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(["b1", "b1", "b1"], dtype="string"),
            MODEL_ID: pd.Series(["m1", "m2", "m3"], dtype="string"),
            SPLIT: pd.Series(["train", "val", "test"], dtype="string"),
        }
    )
    b2 = pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(["b2", "b2", "b2"], dtype="string"),
            MODEL_ID: pd.Series(["m1", "m2", "m3"], dtype="string"),
            SPLIT: pd.Series(["train", "test", "val"], dtype="string"),
        }
    )
    return SplitResult(
        splits_models=pd.DataFrame({MODEL_ID: pd.Series(["m1", "m2", "m3"], dtype="string")}),
        per_benchmark_splits={"b1": b1, "b2": b2},
        split_report={"warnings": []},
    )


def _make_theta_result() -> ThetaResult:
    theta_estimates = pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(["b1", "b1", "b1", "b2", "b2", "b2"], dtype="string"),
            MODEL_ID: pd.Series(["m1", "m2", "m3", "m1", "m2", "m3"], dtype="string"),
            SPLIT: pd.Series(["train", "val", "test", "train", "test", "val"], dtype="string"),
            THETA_HAT: pd.Series([0.2, 0.5, 0.8, -0.1, 0.1, 0.3], dtype="Float64"),
            THETA_SE: pd.Series([0.4, 0.5, 0.6, 0.45, 0.55, 0.65], dtype="Float64"),
            THETA_METHOD: pd.Series(["MAP"] * 6, dtype="string"),
        }
    )
    return ThetaResult(
        theta_estimates=theta_estimates,
        theta_report={"warnings": [], "parameters": {"theta_method": "MAP"}},
    )


def _make_linear_result() -> LinearPredictorResult:
    def benchmark_result(
        benchmark_id: str, subs: list[float], lins: list[float]
    ) -> BenchmarkLinearPredictorResult:
        model_outputs = pd.DataFrame(
            {
                BENCHMARK_ID: pd.Series([benchmark_id] * 3, dtype="string"),
                MODEL_ID: pd.Series(["m1", "m2", "m3"], dtype="string"),
                REDUCED_SUBSCORE: pd.Series(subs, dtype="Float64"),
                LINEAR_PREDICTION: pd.Series(lins, dtype="Float64"),
            }
        )
        return BenchmarkLinearPredictorResult(
            benchmark_id=benchmark_id,
            model_outputs=model_outputs,
            coefficients=pd.DataFrame(),
            linear_predictor_report={"warnings": []},
        )

    return LinearPredictorResult(
        benchmarks={
            "b1": benchmark_result("b1", [61.0, 57.0, 55.0], [63.0, 59.0, 56.0]),
            "b2": benchmark_result("b2", [71.0, 65.0, 60.0], [69.0, 64.0, 60.0]),
        },
        feature_report={"warnings": []},
    )
