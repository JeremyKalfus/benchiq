"""Stable GAM wrapper and cross-validation helpers for BenchIQ."""

from __future__ import annotations

import pickle
import platform
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from sklearn.model_selection import KFold

from benchiq.io.write import write_json, write_parquet
from benchiq.logging import update_manifest

DEFAULT_LAM_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True, slots=True)
class FoldAssignment:
    """Index assignments for one validation fold."""

    fold_id: int
    train_index: np.ndarray
    val_index: np.ndarray


@dataclass(slots=True)
class FittedGAM:
    """Thin, serializable wrapper around a fitted pyGAM model."""

    model: LinearGAM
    feature_names: tuple[str, ...]
    target_name: str
    selected_lam: float
    n_splines: int
    train_row_count: int
    fit_warnings: list[str] = field(default_factory=list)

    def predict(self, X: Any) -> np.ndarray:
        """Return numeric predictions for the provided features."""

        feature_matrix = _coerce_feature_matrix(X)
        if feature_matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                "predict received a feature matrix with the wrong number of columns",
            )
        predictions = np.asarray(self.model.predict(feature_matrix), dtype=float)
        if predictions.ndim != 1:
            raise ValueError("pygam predict did not return a one-dimensional output")
        return predictions

    def to_metadata(
        self,
        *,
        cv_folds: int | None = None,
        selection_metric: str | None = None,
        test_row_count: int | None = None,
        skipped_lam_settings: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a json-safe metadata summary for the fitted model."""

        statistics = getattr(self.model, "statistics_", {})
        metadata = {
            "backend": "pygam",
            "feature_names": list(self.feature_names),
            "target_name": self.target_name,
            "n_features": len(self.feature_names),
            "n_splines": self.n_splines,
            "selected_lam": self.selected_lam,
            "train_row_count": self.train_row_count,
            "test_row_count": test_row_count,
            "cv_folds": cv_folds,
            "selection_metric": selection_metric,
            "fit_warnings": list(self.fit_warnings),
            "skipped_lam_settings": list(skipped_lam_settings or []),
            "python_version": platform.python_version(),
            "benchiq_version": _package_version("benchiq", default="0.1.0a0"),
            "pygam_version": _package_version("pygam", default="unavailable"),
            "pickle_protocol": pickle.HIGHEST_PROTOCOL,
            "model_hyperparameters": {
                "fit_intercept": bool(getattr(self.model, "fit_intercept", True)),
                "lam": self.selected_lam,
                "max_iter": int(getattr(self.model, "max_iter", 0)),
                "terms": str(getattr(self.model, "terms", "unknown")),
                "tol": float(getattr(self.model, "tol", 0.0)),
            },
            "fit_statistics": {
                "aic": statistics.get("AIC"),
                "edof": statistics.get("edof"),
                "gcv": statistics.get("GCV"),
                "loglikelihood": statistics.get("loglikelihood"),
                "scale": statistics.get("scale"),
                "pseudo_r2": statistics.get("pseudo_r2"),
            },
        }
        return _to_jsonable(metadata)

    def save(
        self,
        out_dir: str | Path,
        *,
        metadata_filename: str = "gam_model.json",
        model_filename: str = "gam_model.pkl",
        manifest_path: str | Path | None = None,
        stage_key: str = "t07_gam",
        cv_folds: int | None = None,
        selection_metric: str | None = None,
        test_row_count: int | None = None,
        skipped_lam_settings: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Path]:
        """Persist the fitted wrapper and metadata to disk."""

        resolved_dir = Path(out_dir)
        resolved_dir.mkdir(parents=True, exist_ok=True)
        model_path = resolved_dir / model_filename
        metadata_path = write_json(
            self.to_metadata(
                cv_folds=cv_folds,
                selection_metric=selection_metric,
                test_row_count=test_row_count,
                skipped_lam_settings=skipped_lam_settings,
            ),
            resolved_dir / metadata_filename,
        )
        with model_path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        artifact_paths = {
            "gam_model": model_path,
            "gam_model_metadata": metadata_path,
        }
        if manifest_path is not None:
            update_manifest(
                manifest_path,
                {
                    "artifacts": {
                        stage_key: {
                            name: str(path) for name, path in sorted(artifact_paths.items())
                        },
                    },
                    "stage_metadata": {
                        stage_key: {
                            "serialization": self.to_metadata(
                                cv_folds=cv_folds,
                                selection_metric=selection_metric,
                                test_row_count=test_row_count,
                                skipped_lam_settings=skipped_lam_settings,
                            ),
                        },
                    },
                },
            )
        return artifact_paths


@dataclass(slots=True)
class GAMCVResult:
    """Cross-validation outputs for smoothing selection."""

    best_model: FittedGAM
    best_lam: float
    cv_results: pd.DataFrame
    cv_report: dict[str, Any]
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    manifest_path: Path | None = None


def fit_gam(
    X: Any,
    y: Any,
    *,
    lam: float,
    n_splines: int = 20,
    feature_names: Sequence[str] | None = None,
    target_name: str = "target",
    max_iter: int = 100,
    tol: float = 1e-4,
) -> FittedGAM:
    """Fit a pyGAM regressor on numeric features."""

    _validate_lam(lam)
    if n_splines < 4:
        raise ValueError("n_splines must be at least 4")

    feature_matrix = _coerce_feature_matrix(X)
    target = _coerce_target(y, expected_rows=feature_matrix.shape[0])
    resolved_feature_names = _resolve_feature_names(
        feature_names,
        n_features=feature_matrix.shape[1],
    )

    model = LinearGAM(
        _build_terms(feature_matrix.shape[1], n_splines=n_splines),
        lam=lam,
        max_iter=max_iter,
        tol=tol,
    )
    model.fit(feature_matrix, target)
    return FittedGAM(
        model=model,
        feature_names=resolved_feature_names,
        target_name=target_name,
        selected_lam=float(lam),
        n_splines=int(n_splines),
        train_row_count=int(feature_matrix.shape[0]),
    )


def load_gam(path: str | Path) -> FittedGAM:
    """Load a serialized fitted GAM wrapper from disk."""

    resolved_path = Path(path)
    with resolved_path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, FittedGAM):
        raise TypeError("loaded object is not a FittedGAM instance")
    return model


def make_kfold_assignments(
    n_rows: int,
    *,
    cv_folds: int,
    random_seed: int = 0,
) -> list[FoldAssignment]:
    """Build deterministic shuffled k-fold assignments."""

    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if n_rows < cv_folds:
        raise ValueError("cv_folds cannot exceed the number of rows")

    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    return [
        FoldAssignment(
            fold_id=fold_index,
            train_index=train_index.astype(int, copy=False),
            val_index=val_index.astype(int, copy=False),
        )
        for fold_index, (train_index, val_index) in enumerate(splitter.split(np.arange(n_rows)))
    ]


def rmse_score(y_true: Any, y_pred: Any) -> float:
    """Compute root-mean-squared error."""

    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    if true_values.shape[0] != predicted_values.shape[0]:
        raise ValueError("rmse inputs must have the same number of rows")
    return float(np.sqrt(np.mean((predicted_values - true_values) ** 2)))


def cross_validate_gam(
    X: Any,
    y: Any,
    *,
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
    cv_folds: int = 5,
    random_seed: int = 0,
    feature_names: Sequence[str] | None = None,
    target_name: str = "target",
    n_splines: int = 20,
    max_iter: int = 100,
    tol: float = 1e-4,
    fold_assignments: Sequence[FoldAssignment] | None = None,
    X_test: Any | None = None,
    y_test: Any | None = None,
    out_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    stage_key: str = "t07_gam",
) -> GAMCVResult:
    """Select a smoothing value by k-fold RMSE and fit the final model."""

    feature_matrix = _coerce_feature_matrix(X)
    target = _coerce_target(y, expected_rows=feature_matrix.shape[0])
    resolved_feature_names = _resolve_feature_names(
        feature_names,
        n_features=feature_matrix.shape[1],
    )
    resolved_lam_grid = _normalize_lam_grid(lam_grid)

    if fold_assignments is None:
        resolved_folds = make_kfold_assignments(
            feature_matrix.shape[0],
            cv_folds=cv_folds,
            random_seed=random_seed,
        )
    else:
        resolved_folds = _validate_fold_assignments(
            fold_assignments,
            n_rows=feature_matrix.shape[0],
        )
        cv_folds = len(resolved_folds)

    holdout_features = None
    holdout_target = None
    if X_test is not None or y_test is not None:
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided together")
        holdout_features = _coerce_feature_matrix(X_test)
        holdout_target = _coerce_target(y_test, expected_rows=holdout_features.shape[0])
        if holdout_features.shape[1] != feature_matrix.shape[1]:
            raise ValueError("X_test must have the same number of columns as X")

    fold_records: list[dict[str, Any]] = []
    skipped_lam_settings: list[dict[str, Any]] = []
    lam_summaries: list[dict[str, Any]] = []

    for lam in resolved_lam_grid:
        lam_records: list[dict[str, Any]] = []
        lam_failed = False
        for fold in resolved_folds:
            train_features = feature_matrix[fold.train_index]
            train_target = target[fold.train_index]
            val_features = feature_matrix[fold.val_index]
            val_target = target[fold.val_index]
            try:
                fitted_model = fit_gam(
                    train_features,
                    train_target,
                    lam=lam,
                    n_splines=n_splines,
                    feature_names=resolved_feature_names,
                    target_name=target_name,
                    max_iter=max_iter,
                    tol=tol,
                )
            except Exception as exc:  # pragma: no cover - defensive against backend drift
                skipped_lam_settings.append(
                    {
                        "lam": lam,
                        "fold_id": fold.fold_id,
                        "reason": str(exc),
                    },
                )
                lam_failed = True
                break

            val_predictions = fitted_model.predict(val_features)
            val_rmse = rmse_score(val_target, val_predictions)
            baseline_prediction = np.full(
                shape=val_target.shape[0],
                fill_value=float(train_target.mean()),
                dtype=float,
            )
            baseline_rmse = rmse_score(val_target, baseline_prediction)
            test_rmse = None
            if holdout_features is not None and holdout_target is not None:
                test_rmse = rmse_score(holdout_target, fitted_model.predict(holdout_features))

            lam_records.append(
                {
                    "lam": lam,
                    "fold_id": fold.fold_id,
                    "train_rows": int(fold.train_index.shape[0]),
                    "val_rows": int(fold.val_index.shape[0]),
                    "val_rmse": val_rmse,
                    "baseline_val_rmse": baseline_rmse,
                    "test_rmse": test_rmse,
                },
            )

        if lam_failed:
            continue

        fold_records.extend(lam_records)
        val_rmses = [record["val_rmse"] for record in lam_records]
        baseline_rmses = [record["baseline_val_rmse"] for record in lam_records]
        lam_summaries.append(
            {
                "lam": lam,
                "mean_val_rmse": float(np.mean(val_rmses)),
                "max_val_rmse": float(np.max(val_rmses)),
                "mean_baseline_val_rmse": float(np.mean(baseline_rmses)),
                "fold_count": len(lam_records),
                "mean_test_rmse": (
                    float(np.mean([record["test_rmse"] for record in lam_records]))
                    if holdout_features is not None and holdout_target is not None
                    else None
                ),
            },
        )

    if not lam_summaries:
        raise RuntimeError("all smoothing settings failed during cross-validation")

    best_summary = min(
        lam_summaries,
        key=lambda summary: (summary["mean_val_rmse"], summary["max_val_rmse"], summary["lam"]),
    )
    best_model = fit_gam(
        feature_matrix,
        target,
        lam=float(best_summary["lam"]),
        n_splines=n_splines,
        feature_names=resolved_feature_names,
        target_name=target_name,
        max_iter=max_iter,
        tol=tol,
    )
    cv_results = pd.DataFrame(fold_records).astype(
        {
            "lam": "Float64",
            "fold_id": "Int64",
            "train_rows": "Int64",
            "val_rows": "Int64",
            "val_rmse": "Float64",
            "baseline_val_rmse": "Float64",
            "test_rmse": "Float64",
        },
    )
    best_holdout_rmse = None
    if holdout_features is not None and holdout_target is not None:
        best_holdout_rmse = rmse_score(holdout_target, best_model.predict(holdout_features))
    cv_report = _build_cv_report(
        best_model=best_model,
        cv_folds=cv_folds,
        lam_grid=resolved_lam_grid,
        lam_summaries=lam_summaries,
        skipped_lam_settings=skipped_lam_settings,
        holdout_row_count=0 if holdout_target is None else int(holdout_target.shape[0]),
        best_holdout_rmse=best_holdout_rmse,
    )
    result = GAMCVResult(
        best_model=best_model,
        best_lam=float(best_summary["lam"]),
        cv_results=cv_results.sort_values(["lam", "fold_id"]).reset_index(drop=True),
        cv_report=cv_report,
    )
    if out_dir is not None:
        artifact_paths = write_gam_artifacts(
            result,
            out_dir=out_dir,
            manifest_path=manifest_path,
            stage_key=stage_key,
        )
        result.artifact_paths = artifact_paths
        if manifest_path is not None:
            result.manifest_path = Path(manifest_path)
    return result


def write_gam_artifacts(
    result: GAMCVResult,
    *,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
    stage_key: str = "t07_gam",
) -> dict[str, Path]:
    """Write cross-validation results and the selected model to disk."""

    resolved_dir = Path(out_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    model_artifacts = result.best_model.save(
        resolved_dir,
        manifest_path=manifest_path,
        stage_key=stage_key,
        cv_folds=result.cv_report["cv_folds"],
        selection_metric=result.cv_report["selection_metric"],
        test_row_count=result.cv_report["holdout_row_count"],
        skipped_lam_settings=result.cv_report["skipped_lam_settings"],
    )
    artifact_paths = {
        **model_artifacts,
        "cv_results": write_parquet(result.cv_results, resolved_dir / "cv_results.parquet"),
        "cv_report": write_json(result.cv_report, resolved_dir / "cv_report.json"),
    }
    if manifest_path is not None:
        update_manifest(
            manifest_path,
            {
                "artifacts": {
                    stage_key: {name: str(path) for name, path in sorted(artifact_paths.items())},
                },
                "stage_metadata": {
                    stage_key: result.cv_report,
                },
            },
        )
    return artifact_paths


def _build_terms(n_features: int, *, n_splines: int) -> Any:
    terms = s(0, n_splines=n_splines)
    for feature_index in range(1, n_features):
        terms += s(feature_index, n_splines=n_splines)
    return terms


def _coerce_feature_matrix(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        matrix = X.to_numpy(dtype=float, copy=True)
    else:
        matrix = np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("X must be convertible to a two-dimensional numeric matrix")
    if matrix.shape[0] == 0:
        raise ValueError("X must contain at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError("X must not contain missing or infinite values")
    return matrix


def _coerce_target(y: Any, *, expected_rows: int) -> np.ndarray:
    if isinstance(y, pd.Series):
        target = y.to_numpy(dtype=float, copy=True)
    else:
        target = np.asarray(y, dtype=float).reshape(-1)
    if target.ndim != 1:
        raise ValueError("y must be convertible to a one-dimensional numeric vector")
    if target.shape[0] != expected_rows:
        raise ValueError("X and y must contain the same number of rows")
    if not np.isfinite(target).all():
        raise ValueError("y must not contain missing or infinite values")
    return target


def _resolve_feature_names(
    feature_names: Sequence[str] | None,
    *,
    n_features: int,
) -> tuple[str, ...]:
    if feature_names is None:
        return tuple(f"x{index}" for index in range(n_features))
    if len(feature_names) != n_features:
        raise ValueError("feature_names must match the number of feature columns")
    return tuple(str(name) for name in feature_names)


def _normalize_lam_grid(lam_grid: Sequence[float]) -> list[float]:
    if len(lam_grid) == 0:
        raise ValueError("lam_grid must contain at least one smoothing value")
    normalized = []
    for lam in lam_grid:
        _validate_lam(lam)
        normalized.append(float(lam))
    return normalized


def _validate_lam(lam: float) -> None:
    if not isfinite(float(lam)) or float(lam) <= 0.0:
        raise ValueError("lam must be a finite positive number")


def _validate_fold_assignments(
    fold_assignments: Sequence[FoldAssignment],
    *,
    n_rows: int,
) -> list[FoldAssignment]:
    if not fold_assignments:
        raise ValueError("fold_assignments must contain at least one fold")
    resolved_folds: list[FoldAssignment] = []
    for fold in fold_assignments:
        train_index = np.asarray(fold.train_index, dtype=int).reshape(-1)
        val_index = np.asarray(fold.val_index, dtype=int).reshape(-1)
        if train_index.size == 0 or val_index.size == 0:
            raise ValueError("each fold must contain both train and validation rows")
        if train_index.min() < 0 or val_index.min() < 0:
            raise ValueError("fold indices must be non-negative")
        if train_index.max() >= n_rows or val_index.max() >= n_rows:
            raise ValueError("fold indices exceed the available row count")
        if np.intersect1d(train_index, val_index).size > 0:
            raise ValueError("train and validation indices must be disjoint within each fold")
        resolved_folds.append(
            FoldAssignment(
                fold_id=int(fold.fold_id),
                train_index=train_index,
                val_index=val_index,
            ),
        )
    return resolved_folds


def _build_cv_report(
    *,
    best_model: FittedGAM,
    cv_folds: int,
    lam_grid: Sequence[float],
    lam_summaries: Sequence[dict[str, Any]],
    skipped_lam_settings: Sequence[dict[str, Any]],
    holdout_row_count: int,
    best_holdout_rmse: float | None,
) -> dict[str, Any]:
    best_summary = min(
        lam_summaries,
        key=lambda summary: (summary["mean_val_rmse"], summary["max_val_rmse"], summary["lam"]),
    )
    return _to_jsonable(
        {
            "backend": "pygam",
            "cv_folds": cv_folds,
            "selection_metric": "rmse",
            "lam_grid": list(lam_grid),
            "best_lam": best_model.selected_lam,
            "best_mean_val_rmse": best_summary["mean_val_rmse"],
            "best_max_val_rmse": best_summary["max_val_rmse"],
            "best_mean_baseline_val_rmse": best_summary["mean_baseline_val_rmse"],
            "best_holdout_rmse": best_holdout_rmse,
            "holdout_row_count": holdout_row_count,
            "lam_summaries": list(lam_summaries),
            "skipped_lam_settings": list(skipped_lam_settings),
            "best_model_metadata": best_model.to_metadata(
                cv_folds=cv_folds,
                selection_metric="rmse",
                test_row_count=holdout_row_count,
                skipped_lam_settings=skipped_lam_settings,
            ),
        },
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _package_version(package_name: str, *, default: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return default
