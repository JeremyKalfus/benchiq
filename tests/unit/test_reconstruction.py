import json

import numpy as np
import pandas as pd

import benchiq
from benchiq.reconstruct import FeatureTableResult, reconstruct_scores
from benchiq.reconstruct.reconstruction import _select_preferred_model
from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID, SPLIT


def test_reconstruct_scores_writes_artifacts_and_marginal_beats_baseline(tmp_path) -> None:
    model_ids, split_by_model = _make_model_splits()
    bundle = _load_reconstruct_bundle(tmp_path, model_ids=model_ids, run_id="reconstruct-skip")
    feature_result = _make_feature_result(
        model_ids=model_ids,
        split_by_model=split_by_model,
        joint_enabled=False,
    )

    reconstruction_result = reconstruct_scores(
        bundle,
        feature_result,
        lam_grid=(0.1, 1.0, 10.0),
        cv_folds=4,
        n_splines=6,
    )

    summary = reconstruction_result.reconstruction_summary
    for benchmark_id in ["b1", "b2"]:
        test_row = summary.loc[
            (summary[BENCHMARK_ID] == benchmark_id)
            & (summary["model_type"] == "marginal")
            & (summary[SPLIT] == "test")
        ].iloc[0]
        assert float(test_row["rmse"]) < float(test_row["baseline_rmse"])

    assert reconstruction_result.reconstruction_report["joint_skips"] == {
        "b1": "overlap_below_joint_threshold",
        "b2": "overlap_below_joint_threshold",
    }

    stage_dir = tmp_path / "out" / "reconstruct-skip" / "artifacts" / "09_reconstruct"
    assert (stage_dir / "reconstruction_summary.parquet").exists()
    assert (stage_dir / "reconstruction_report.json").exists()
    for benchmark_id in ["b1", "b2"]:
        benchmark_dir = stage_dir / "per_benchmark" / benchmark_id
        assert (benchmark_dir / "predictions.parquet").exists()
        assert (benchmark_dir / "reconstruction_report.json").exists()
        assert (benchmark_dir / "marginal" / "gam_model.json").exists()
        assert (benchmark_dir / "marginal" / "plots" / "predicted_vs_actual.png").exists()
        assert not (benchmark_dir / "joint").exists()

    report = json.loads((stage_dir / "reconstruction_report.json").read_text(encoding="utf-8"))
    assert report["joint_skips"]["b1"] == "overlap_below_joint_threshold"
    assert report["joint_skips"]["b2"] == "overlap_below_joint_threshold"
    assert report["preferred_model_type_by_benchmark"]["b1"] == "marginal"
    assert report["benchmarks"]["b1"]["preferred_model"]["model_type"] == "marginal"


def test_reconstruct_scores_joint_improves_on_correlated_bundle(tmp_path) -> None:
    model_ids, split_by_model = _make_model_splits()
    bundle = _load_reconstruct_bundle(tmp_path, model_ids=model_ids, run_id="reconstruct-joint")
    feature_result = _make_feature_result(
        model_ids=model_ids,
        split_by_model=split_by_model,
        joint_enabled=True,
    )

    reconstruction_result = reconstruct_scores(
        bundle,
        feature_result,
        lam_grid=(0.1, 1.0, 10.0),
        cv_folds=4,
        n_splines=6,
    )

    summary = reconstruction_result.reconstruction_summary
    for benchmark_id in ["b1", "b2"]:
        marginal_test = summary.loc[
            (summary[BENCHMARK_ID] == benchmark_id)
            & (summary["model_type"] == "marginal")
            & (summary[SPLIT] == "test")
        ].iloc[0]
        joint_test = summary.loc[
            (summary[BENCHMARK_ID] == benchmark_id)
            & (summary["model_type"] == "joint")
            & (summary[SPLIT] == "test")
        ].iloc[0]
        assert float(joint_test["rmse"]) < float(marginal_test["rmse"])

    assert reconstruction_result.reconstruction_report["joint_skips"] == {}

    stage_dir = tmp_path / "out" / "reconstruct-joint" / "artifacts" / "09_reconstruct"
    for benchmark_id in ["b1", "b2"]:
        benchmark_dir = stage_dir / "per_benchmark" / benchmark_id
        predictions = pd.read_parquet(benchmark_dir / "predictions.parquet")
        assert set(predictions["model_type"].astype(str)) == {"marginal", "joint"}
        assert (benchmark_dir / "joint" / "gam_model.json").exists()
        assert (benchmark_dir / "joint" / "plots" / "predicted_vs_actual.png").exists()
        assert (benchmark_dir / "joint" / "plots" / "calibration.png").exists()
        assert (benchmark_dir / "joint" / "plots" / "residual_histogram.png").exists()

    report = json.loads((stage_dir / "reconstruction_report.json").read_text(encoding="utf-8"))
    assert report["joint_skips"] == {}
    assert report["rmse"]["marginal_test_by_benchmark"]["b1"] is not None
    assert report["rmse"]["joint_test_by_benchmark"]["b1"] is not None
    assert report["preferred_model_type_by_benchmark"]["b1"] == "joint"
    assert report["benchmarks"]["b1"]["preferred_model"]["model_type"] == "joint"


def test_select_preferred_model_uses_validation_rmse_and_marginal_tie_break() -> None:
    preferred = _select_preferred_model(
        {
            "marginal": {
                "skipped": False,
                "val_row_count": 4,
                "metrics": {"val": {"rmse": 4.0, "mae": 3.0}},
                "cv_report": {
                    "best_lam": 0.1,
                    "best_mean_val_rmse": 4.0,
                    "lam_summaries": [{"lam": 0.1, "mean_test_rmse": 4.2}],
                },
            },
            "joint": {
                "skipped": False,
                "val_row_count": 4,
                "metrics": {"val": {"rmse": 4.0, "mae": 3.5}},
                "cv_report": {
                    "best_lam": 0.1,
                    "best_mean_val_rmse": 4.0,
                    "lam_summaries": [{"lam": 0.1, "mean_test_rmse": 3.8}],
                },
            },
        }
    )

    assert preferred["model_type"] == "marginal"
    assert preferred["selection_metric"] == "validation_rmse"
    assert preferred["validation_rmse_by_model_type"]["marginal"] == 4.0
    assert preferred["validation_rmse_by_model_type"]["joint"] == 4.0


def test_select_preferred_model_uses_cv_mean_test_when_validation_is_tiny() -> None:
    preferred = _select_preferred_model(
        {
            "marginal": {
                "skipped": False,
                "val_row_count": 2,
                "metrics": {"val": {"rmse": 8.0, "mae": 7.5}},
                "cv_report": {
                    "best_lam": 0.1,
                    "best_mean_val_rmse": 4.6,
                    "lam_summaries": [{"lam": 0.1, "mean_test_rmse": 7.2}],
                },
            },
            "joint": {
                "skipped": False,
                "val_row_count": 2,
                "metrics": {"val": {"rmse": 11.0, "mae": 10.1}},
                "cv_report": {
                    "best_lam": 0.1,
                    "best_mean_val_rmse": 4.5,
                    "lam_summaries": [{"lam": 0.1, "mean_test_rmse": 5.2}],
                },
            },
        }
    )

    assert preferred["model_type"] == "joint"
    assert preferred["selection_metric"] == "cv_mean_test_rmse"
    assert preferred["validation_row_count_by_model_type"]["marginal"] == 2
    assert preferred["cv_mean_test_rmse_by_model_type"]["joint"] == 5.2


def test_select_preferred_model_keeps_validation_selector_at_threshold() -> None:
    preferred = _select_preferred_model(
        {
            "marginal": {
                "skipped": False,
                "val_row_count": 4,
                "metrics": {"val": {"rmse": 4.0, "mae": 3.0}},
                "cv_report": {
                    "best_lam": 0.1,
                    "best_mean_val_rmse": 5.0,
                    "lam_summaries": [{"lam": 0.1, "mean_test_rmse": 6.0}],
                },
            },
            "joint": {
                "skipped": False,
                "val_row_count": 4,
                "metrics": {"val": {"rmse": 5.0, "mae": 4.0}},
                "cv_report": {
                    "best_lam": 0.1,
                    "best_mean_val_rmse": 4.0,
                    "lam_summaries": [{"lam": 0.1, "mean_test_rmse": 3.0}],
                },
            },
        }
    )

    assert preferred["model_type"] == "marginal"
    assert preferred["selection_metric"] == "validation_rmse"


def _make_model_splits() -> tuple[list[str], dict[str, str]]:
    model_ids = [f"m{index:02d}" for index in range(1, 81)]
    split_by_model = {}
    for model_index, model_id in enumerate(model_ids):
        if model_index < 40:
            split_by_model[model_id] = "train"
        elif model_index < 60:
            split_by_model[model_id] = "val"
        else:
            split_by_model[model_id] = "test"
    return model_ids, split_by_model


def _load_reconstruct_bundle(tmp_path, *, model_ids: list[str], run_id: str):
    rows: list[dict[str, object]] = []
    for benchmark_id in ["b1", "b2"]:
        for model_index, model_id in enumerate(model_ids):
            rows.append(
                {
                    BENCHMARK_ID: benchmark_id,
                    "item_id": f"{benchmark_id}_item",
                    MODEL_ID: model_id,
                    "score": int((model_index + (benchmark_id == "b2")) % 2),
                }
            )
    responses_path = tmp_path / f"{run_id}.csv"
    pd.DataFrame(rows).to_csv(responses_path, index=False)
    return benchiq.load_bundle(
        responses_path,
        config=benchiq.BenchIQConfig(allow_low_n=True, min_overlap_models_for_joint=2),
        out_dir=tmp_path / "out",
        run_id=run_id,
    )


def _make_feature_result(
    *,
    model_ids: list[str],
    split_by_model: dict[str, str],
    joint_enabled: bool,
) -> FeatureTableResult:
    rng = np.random.default_rng(11 if joint_enabled else 5)
    z = np.linspace(-2.4, 2.4, len(model_ids))
    theta_b1 = z + rng.normal(0.0, 0.12, size=len(model_ids))
    latent_cross = rng.normal(0.0, 1.0, size=len(model_ids))
    theta_b2 = 0.2 * z + 0.9 * latent_cross + rng.normal(0.0, 0.08, size=len(model_ids))
    theta_se_b1 = 0.3 + 0.04 * np.abs(theta_b1)
    theta_se_b2 = 0.3 + 0.04 * np.abs(theta_b2)
    sub_b1 = 50.0 + 18.0 * np.tanh(theta_b1) + rng.normal(0.0, 1.0, size=len(model_ids))
    sub_b2 = 50.0 + 18.0 * np.tanh(theta_b2) + rng.normal(0.0, 1.0, size=len(model_ids))
    lin_b1 = 48.0 + 16.0 * np.tanh(theta_b1) + rng.normal(0.0, 1.2, size=len(model_ids))
    lin_b2 = 48.0 + 16.0 * np.tanh(theta_b2) + rng.normal(0.0, 1.2, size=len(model_ids))

    score_b1 = (
        50.0
        + 3.0 * np.tanh(theta_b1)
        + 18.0 * theta_b2
        + rng.normal(0.0, 0.25, size=len(model_ids))
    )
    score_b2 = (
        50.0
        + 3.0 * np.tanh(theta_b2)
        + 18.0 * theta_b1
        + rng.normal(0.0, 0.25, size=len(model_ids))
    )

    marginal_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    grand_sub = (sub_b1 + sub_b2) / 2.0
    grand_lin = (lin_b1 + lin_b2) / 2.0
    for index, model_id in enumerate(model_ids):
        split_name = split_by_model[model_id]
        marginal_rows.extend(
            [
                {
                    BENCHMARK_ID: "b1",
                    MODEL_ID: model_id,
                    SPLIT: split_name,
                    "score_full_b": float(score_b1[index]),
                    "theta_b": float(theta_b1[index]),
                    "theta_se_b": float(theta_se_b1[index]),
                    "sub_b": float(sub_b1[index]),
                    "lin_b": float(lin_b1[index]),
                    "theta_method": "MAP",
                },
                {
                    BENCHMARK_ID: "b2",
                    MODEL_ID: model_id,
                    SPLIT: split_name,
                    "score_full_b": float(score_b2[index]),
                    "theta_b": float(theta_b2[index]),
                    "theta_se_b": float(theta_se_b2[index]),
                    "sub_b": float(sub_b2[index]),
                    "lin_b": float(lin_b2[index]),
                    "theta_method": "MAP",
                },
            ]
        )
        if joint_enabled:
            joint_rows.extend(
                [
                    {
                        BENCHMARK_ID: "b1",
                        MODEL_ID: model_id,
                        SPLIT: split_name,
                        "score_full_b": float(score_b1[index]),
                        "theta_se_b": float(theta_se_b1[index]),
                        "sub_b": float(sub_b1[index]),
                        "lin_b": float(lin_b1[index]),
                        "grand_sub": float(grand_sub[index]),
                        "grand_lin": float(grand_lin[index]),
                        "theta_b1": float(theta_b1[index]),
                        "theta_b2": float(theta_b2[index]),
                    },
                    {
                        BENCHMARK_ID: "b2",
                        MODEL_ID: model_id,
                        SPLIT: split_name,
                        "score_full_b": float(score_b2[index]),
                        "theta_se_b": float(theta_se_b2[index]),
                        "sub_b": float(sub_b2[index]),
                        "lin_b": float(lin_b2[index]),
                        "grand_sub": float(grand_sub[index]),
                        "grand_lin": float(grand_lin[index]),
                        "theta_b1": float(theta_b1[index]),
                        "theta_b2": float(theta_b2[index]),
                    },
                ]
            )

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
    if joint_enabled:
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
                "theta_b1": "Float64",
                "theta_b2": "Float64",
            }
        )
        joint_report = {
            "written": True,
            "skipped": False,
            "skip_reason": None,
            "warnings": [],
        }
    else:
        features_joint = pd.DataFrame(
            {
                BENCHMARK_ID: pd.Series(dtype="string"),
                MODEL_ID: pd.Series(dtype="string"),
                SPLIT: pd.Series(dtype="string"),
                "score_full_b": pd.Series(dtype="Float64"),
                "theta_se_b": pd.Series(dtype="Float64"),
                "sub_b": pd.Series(dtype="Float64"),
                "lin_b": pd.Series(dtype="Float64"),
                "grand_sub": pd.Series(dtype="Float64"),
                "grand_lin": pd.Series(dtype="Float64"),
                "theta_b1": pd.Series(dtype="Float64"),
                "theta_b2": pd.Series(dtype="Float64"),
            }
        )
        joint_report = {
            "written": False,
            "skipped": True,
            "skip_reason": "overlap_below_joint_threshold",
            "warnings": [],
        }

    return FeatureTableResult(
        features_marginal=features_marginal,
        features_joint=features_joint,
        feature_report={
            "parameters": {"benchmark_ids": ["b1", "b2"], "min_overlap_models_for_joint": 2},
            "joint": joint_report,
        },
    )
