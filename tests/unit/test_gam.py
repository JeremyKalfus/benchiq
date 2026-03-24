import json
from math import isfinite

import numpy as np
import pandas as pd

from benchiq.reconstruct import (
    cross_validate_gam,
    fit_gam,
    load_gam,
    make_kfold_assignments,
    rmse_score,
)


def test_fit_gam_roundtrip_serialization_and_rmse(tmp_path) -> None:
    x_values = np.linspace(-2.5, 2.5, 80)
    X = pd.DataFrame({"reduced_score": x_values})
    y = np.sin(x_values) + 0.15 * x_values

    fitted_model = fit_gam(
        X,
        y,
        lam=0.1,
        n_splines=12,
        feature_names=["reduced_score"],
        target_name="full_score",
    )

    predictions = fitted_model.predict(X)
    assert rmse_score(y, predictions) < 0.02

    manifest_path = tmp_path / "manifest.json"
    artifact_paths = fitted_model.save(
        tmp_path / "gam-fit",
        manifest_path=manifest_path,
        cv_folds=4,
        selection_metric="rmse",
    )

    reloaded_model = load_gam(artifact_paths["gam_model"])
    reloaded_predictions = reloaded_model.predict(X)
    np.testing.assert_allclose(predictions, reloaded_predictions, atol=1e-8)

    metadata = json.loads(artifact_paths["gam_model_metadata"].read_text(encoding="utf-8"))
    assert metadata["backend"] == "pygam"
    assert metadata["feature_names"] == ["reduced_score"]
    assert metadata["target_name"] == "full_score"
    assert metadata["selected_lam"] == 0.1
    assert metadata["python_version"]
    assert metadata["pygam_version"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "t07_gam" in manifest["artifacts"]
    assert manifest["stage_metadata"]["t07_gam"]["serialization"]["selected_lam"] == 0.1


def test_cross_validate_gam_selects_finite_lam_and_beats_mean_baseline(tmp_path) -> None:
    rng = np.random.default_rng(7)
    train_x = np.linspace(-3.0, 3.0, 120)
    train_y = np.sin(1.5 * train_x) + 0.2 * train_x + rng.normal(0.0, 0.05, size=120)
    test_x = np.linspace(-2.75, 2.75, 36)
    test_y = np.sin(1.5 * test_x) + 0.2 * test_x

    result = cross_validate_gam(
        pd.DataFrame({"reduced_score": train_x}),
        train_y,
        lam_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        cv_folds=4,
        random_seed=11,
        feature_names=["reduced_score"],
        target_name="full_score",
        n_splines=14,
        X_test=pd.DataFrame({"reduced_score": test_x}),
        y_test=test_y,
        out_dir=tmp_path / "gam-cv",
        manifest_path=tmp_path / "manifest.json",
    )

    assert result.best_lam in {0.01, 0.1, 1.0, 10.0, 100.0}
    assert isfinite(result.best_lam)
    assert result.cv_results.shape[0] == 20
    assert result.cv_report["best_mean_val_rmse"] < result.cv_report["best_mean_baseline_val_rmse"]
    assert result.cv_report["best_model_metadata"]["selected_lam"] == result.best_lam
    assert result.cv_report["holdout_row_count"] == 36

    assert (tmp_path / "gam-cv" / "gam_model.pkl").exists()
    assert (tmp_path / "gam-cv" / "gam_model.json").exists()
    assert (tmp_path / "gam-cv" / "cv_results.parquet").exists()
    assert (tmp_path / "gam-cv" / "cv_report.json").exists()

    loaded_model = load_gam(tmp_path / "gam-cv" / "gam_model.pkl")
    holdout_predictions = loaded_model.predict(pd.DataFrame({"reduced_score": test_x}))
    assert rmse_score(test_y, holdout_predictions) < 0.25

    report = json.loads((tmp_path / "gam-cv" / "cv_report.json").read_text(encoding="utf-8"))
    assert report["selection_metric"] == "rmse"
    assert len(report["lam_summaries"]) == 5


def test_make_kfold_assignments_handles_tiny_but_valid_fold_counts() -> None:
    folds = make_kfold_assignments(6, cv_folds=3, random_seed=5)

    assert len(folds) == 3
    all_validation_rows = sorted(int(index) for fold in folds for index in fold.val_index.tolist())
    assert all_validation_rows == [0, 1, 2, 3, 4, 5]
    for fold in folds:
        assert np.intersect1d(fold.train_index, fold.val_index).size == 0
