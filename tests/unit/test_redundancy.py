import json

import numpy as np
import pandas as pd

import benchiq
from benchiq.irt.theta import ThetaResult
from benchiq.preprocess.scores import GRAND_MEAN_SCORE, SCORE_FULL, ScoreResult
from benchiq.reconstruct import FeatureTableResult, reconstruct_scores
from benchiq.redundancy import analyze_redundancy
from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID, SPLIT


def test_redundancy_artifacts_and_metrics_strengthen_with_shared_structure(tmp_path) -> None:
    low = _run_redundancy_fixture(
        tmp_path,
        run_id="redundancy-low",
        shared_strength=0.15,
    )
    high = _run_redundancy_fixture(
        tmp_path,
        run_id="redundancy-high",
        shared_strength=0.85,
    )

    low_report = low.redundancy_report
    high_report = high.redundancy_report

    assert (
        high_report["theta_correlation_summary"]["mean_abs_correlation"]
        > low_report["theta_correlation_summary"]["mean_abs_correlation"]
    )
    assert (
        high_report["score_correlation_summary"]["mean_abs_correlation"]
        > low_report["score_correlation_summary"]["mean_abs_correlation"]
    )

    low_ratio = (
        low.compressibility.loc[~low.compressibility["skipped"], "redundancy_ratio"]
        .astype(float)
        .mean()
    )
    high_ratio = (
        high.compressibility.loc[~high.compressibility["skipped"], "redundancy_ratio"]
        .astype(float)
        .mean()
    )
    assert high_ratio < low_ratio
    assert high.redundancy_report["factor_analysis"]["skipped"] is False
    assert int(high.redundancy_report["factor_analysis"]["selected_n_factors"]) >= 1

    stage_dir = tmp_path / "out" / "redundancy-high" / "artifacts" / "10_redundancy"
    assert (stage_dir / "corr_theta.parquet").exists()
    assert (stage_dir / "corr_scores.parquet").exists()
    assert (stage_dir / "factor_loadings.parquet").exists()
    assert (stage_dir / "compressibility.parquet").exists()
    assert (stage_dir / "redundancy_report.json").exists()
    assert (stage_dir / "plots" / "corr_theta_heatmap.png").exists()
    assert (stage_dir / "plots" / "corr_scores_heatmap.png").exists()
    assert (stage_dir / "plots" / "factor_scree.png").exists()
    assert (stage_dir / "plots" / "factor_loadings.png").exists()

    report = json.loads((stage_dir / "redundancy_report.json").read_text(encoding="utf-8"))
    assert report["factor_analysis"]["skipped"] is False
    assert report["compressibility"]["skipped_benchmarks"] == {}


def test_redundancy_skips_factor_and_compressibility_when_overlap_is_too_small(tmp_path) -> None:
    result = _run_redundancy_fixture(
        tmp_path,
        run_id="redundancy-skip",
        shared_strength=0.7,
        min_overlap_models_for_redundancy=200,
    )

    assert result.factor_loadings.empty
    assert bool(result.compressibility["skipped"].all())
    assert result.redundancy_report["factor_analysis"]["skip_reason"] == (
        "overlap_below_redundancy_threshold"
    )
    assert set(result.redundancy_report["compressibility"]["skipped_benchmarks"]) == {
        "b1",
        "b2",
        "b3",
    }

    stage_dir = tmp_path / "out" / "redundancy-skip" / "artifacts" / "10_redundancy"
    report = json.loads((stage_dir / "redundancy_report.json").read_text(encoding="utf-8"))
    assert report["factor_analysis"]["skip_reason"] == "overlap_below_redundancy_threshold"
    assert report["compressibility"]["skipped_benchmarks"]["b1"] == (
        "overlap_below_redundancy_threshold"
    )
    assert not (stage_dir / "plots" / "factor_scree.png").exists()


def _run_redundancy_fixture(
    tmp_path,
    *,
    run_id: str,
    shared_strength: float,
    min_overlap_models_for_redundancy: int = 20,
):
    benchmark_ids = ["b1", "b2", "b3"]
    model_ids, split_by_model = _make_model_splits()
    bundle = _load_bundle(
        tmp_path,
        run_id=run_id,
        benchmark_ids=benchmark_ids,
        model_ids=model_ids,
        min_overlap_models_for_redundancy=min_overlap_models_for_redundancy,
    )
    feature_result = _make_feature_result(
        benchmark_ids=benchmark_ids,
        model_ids=model_ids,
        split_by_model=split_by_model,
        shared_strength=shared_strength,
    )
    reconstruction_result = reconstruct_scores(
        bundle,
        feature_result,
        lam_grid=(0.1, 1.0, 10.0),
        cv_folds=4,
        n_splines=6,
    )
    score_result = _make_score_result(feature_result)
    theta_result = _make_theta_result(feature_result)
    return analyze_redundancy(
        bundle,
        score_result,
        theta_result,
        feature_result,
        reconstruction_result,
        lam_grid=(0.1, 1.0, 10.0),
        cv_folds=4,
        n_splines=6,
    )


def _make_model_splits() -> tuple[list[str], dict[str, str]]:
    model_ids = [f"m{index:03d}" for index in range(1, 91)]
    split_by_model = {}
    for index, model_id in enumerate(model_ids):
        if index < 45:
            split_by_model[model_id] = "train"
        elif index < 68:
            split_by_model[model_id] = "val"
        else:
            split_by_model[model_id] = "test"
    return model_ids, split_by_model


def _load_bundle(
    tmp_path,
    *,
    run_id: str,
    benchmark_ids: list[str],
    model_ids: list[str],
    min_overlap_models_for_redundancy: int,
):
    rows: list[dict[str, object]] = []
    for benchmark_index, benchmark_id in enumerate(benchmark_ids):
        for model_index, model_id in enumerate(model_ids):
            rows.append(
                {
                    BENCHMARK_ID: benchmark_id,
                    "item_id": f"{benchmark_id}_item",
                    MODEL_ID: model_id,
                    "score": int((model_index + benchmark_index) % 2),
                }
            )
    responses_path = tmp_path / f"{run_id}.csv"
    pd.DataFrame(rows).to_csv(responses_path, index=False)
    return benchiq.load_bundle(
        responses_path,
        config=benchiq.BenchIQConfig(
            allow_low_n=True,
            min_overlap_models_for_joint=20,
            min_overlap_models_for_redundancy=min_overlap_models_for_redundancy,
        ),
        out_dir=tmp_path / "out",
        run_id=run_id,
    )


def _make_feature_result(
    *,
    benchmark_ids: list[str],
    model_ids: list[str],
    split_by_model: dict[str, str],
    shared_strength: float,
) -> FeatureTableResult:
    rng = np.random.default_rng(17)
    shared = np.linspace(-2.5, 2.5, len(model_ids))
    thetas: dict[str, np.ndarray] = {}
    theta_ses: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    subscores: dict[str, np.ndarray] = {}
    linear_predictions: dict[str, np.ndarray] = {}
    benchmark_latents: dict[str, np.ndarray] = {}
    for benchmark_index, benchmark_id in enumerate(benchmark_ids):
        unique = rng.normal(0.0, 1.0, size=len(model_ids))
        theta = (
            shared_strength * shared
            + (1.0 - shared_strength) * unique
            + rng.normal(0.0, 0.08, size=len(model_ids))
        )
        benchmark_latents[benchmark_id] = theta
        thetas[benchmark_id] = theta
        theta_ses[benchmark_id] = 0.25 + 0.03 * np.abs(theta)
    for benchmark_id in benchmark_ids:
        other_thetas = np.mean(
            [thetas[other_id] for other_id in benchmark_ids if other_id != benchmark_id],
            axis=0,
        )
        subscores[benchmark_id] = (
            50.0 + 16.0 * np.tanh(thetas[benchmark_id]) + rng.normal(0.0, 0.9, size=len(model_ids))
        )
        linear_predictions[benchmark_id] = (
            49.0 + 15.0 * np.tanh(thetas[benchmark_id]) + rng.normal(0.0, 1.1, size=len(model_ids))
        )
        scores[benchmark_id] = (
            50.0
            + 8.0 * np.tanh(thetas[benchmark_id])
            + 15.0 * shared_strength * other_thetas
            + rng.normal(0.0, 0.35, size=len(model_ids))
        )

    grand_sub = np.mean([subscores[benchmark_id] for benchmark_id in benchmark_ids], axis=0)
    grand_lin = np.mean(
        [linear_predictions[benchmark_id] for benchmark_id in benchmark_ids],
        axis=0,
    )

    marginal_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    for index, model_id in enumerate(model_ids):
        split_name = split_by_model[model_id]
        for benchmark_id in benchmark_ids:
            marginal_rows.append(
                {
                    BENCHMARK_ID: benchmark_id,
                    MODEL_ID: model_id,
                    SPLIT: split_name,
                    "score_full_b": float(scores[benchmark_id][index]),
                    "theta_b": float(thetas[benchmark_id][index]),
                    "theta_se_b": float(theta_ses[benchmark_id][index]),
                    "sub_b": float(subscores[benchmark_id][index]),
                    "lin_b": float(linear_predictions[benchmark_id][index]),
                    "theta_method": "MAP",
                }
            )
            joint_row = {
                BENCHMARK_ID: benchmark_id,
                MODEL_ID: model_id,
                SPLIT: split_name,
                "score_full_b": float(scores[benchmark_id][index]),
                "theta_se_b": float(theta_ses[benchmark_id][index]),
                "sub_b": float(subscores[benchmark_id][index]),
                "lin_b": float(linear_predictions[benchmark_id][index]),
                "grand_sub": float(grand_sub[index]),
                "grand_lin": float(grand_lin[index]),
            }
            for other_id in benchmark_ids:
                joint_row[f"theta_{other_id}"] = float(thetas[other_id][index])
            joint_rows.append(joint_row)

    features_marginal = pd.DataFrame(marginal_rows).astype(
        {
            BENCHMARK_ID: "string",
            MODEL_ID: "string",
            SPLIT: "string",
            "score_full_b": "Float64",
            "theta_b": "Float64",
            "theta_se_b": "Float64",
            "sub_b": "Float64",
            "lin_b": "Float64",
            "theta_method": "string",
        }
    )
    features_joint = pd.DataFrame(joint_rows).astype(
        {
            BENCHMARK_ID: "string",
            MODEL_ID: "string",
            SPLIT: "string",
            "score_full_b": "Float64",
            "theta_se_b": "Float64",
            "sub_b": "Float64",
            "lin_b": "Float64",
            "grand_sub": "Float64",
            "grand_lin": "Float64",
            **{f"theta_{benchmark_id}": "Float64" for benchmark_id in benchmark_ids},
        }
    )
    return FeatureTableResult(
        features_marginal=features_marginal,
        features_joint=features_joint,
        feature_report={
            "parameters": {"benchmark_ids": benchmark_ids},
            "joint": {
                "skipped": False,
                "skip_reason": None,
                "complete_feature_model_count": len(model_ids),
            },
        },
    )


def _make_score_result(feature_result: FeatureTableResult) -> ScoreResult:
    scores_full = feature_result.features_marginal.loc[
        :,
        [BENCHMARK_ID, MODEL_ID, "score_full_b"],
    ].rename(columns={"score_full_b": SCORE_FULL})
    scores_grand = (
        scores_full.groupby(MODEL_ID, as_index=False)[SCORE_FULL]
        .mean()
        .rename(columns={SCORE_FULL: GRAND_MEAN_SCORE})
    )
    scores_grand["benchmark_count"] = len(
        feature_result.feature_report["parameters"]["benchmark_ids"]
    )
    scores_grand = scores_grand.astype(
        {
            MODEL_ID: "string",
            GRAND_MEAN_SCORE: "Float64",
            "benchmark_count": "Int64",
        }
    )
    return ScoreResult(
        scores_full=scores_full.astype(
            {
                BENCHMARK_ID: "string",
                MODEL_ID: "string",
                SCORE_FULL: "Float64",
            }
        ),
        scores_grand=scores_grand,
        score_report={"grand_scores": {"skipped": False, "skip_reason": None}, "warnings": []},
    )


def _make_theta_result(feature_result: FeatureTableResult) -> ThetaResult:
    theta_estimates = feature_result.features_marginal.loc[
        :,
        [BENCHMARK_ID, MODEL_ID, SPLIT, "theta_b", "theta_se_b", "theta_method"],
    ].rename(columns={"theta_b": "theta_hat", "theta_se_b": "theta_se"})
    theta_estimates = theta_estimates.astype(
        {
            BENCHMARK_ID: "string",
            MODEL_ID: "string",
            SPLIT: "string",
            "theta_hat": "Float64",
            "theta_se": "Float64",
            "theta_method": "string",
        }
    )
    return ThetaResult(
        theta_estimates=theta_estimates,
        theta_report={"warnings": [], "parameters": {"theta_method": "MAP"}},
    )
