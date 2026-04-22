"""Saved IRT backend comparison helpers and winner-decision logic."""

from __future__ import annotations

import json
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import benchiq
from benchiq.calibration import write_calibration_bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.irt.r_baseline import DEFAULT_PARITY_GATE_THRESHOLDS, run_r_baseline_comparison
from benchiq.reconstruct.reconstruction import JOINT_MODEL, MARGINAL_MODEL
from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID, SPLIT

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "irt_backend_comparison"
DEFAULT_BACKENDS = ("girth", "bayes_mcmc")
DEFAULT_SEEDS = (7, 11, 19)
DEFAULT_DEPLOYMENT_SEED = 7
DEFAULT_RUNTIME_MULTIPLIER_LIMIT = 5.0
# keep the saved comparison faithful to the real winner rule. the parity gate already
# stops the bayesian branch early when it cannot clear the baseline thresholds.
DEFAULT_RUN_TIMEOUT_SECONDS = 0
DEFAULT_BAYES_BACKEND_OPTIONS = {
    "draws": 1,
    "tune": 1,
    "chains": 1,
    "cores": 1,
    "target_accept": 0.9,
}
FIXTURE_DATASET_ID = "compact_validation_fixture"
LARGE_DATASET_ID = "large_release_default_subset"
SYNTHETIC_DENSE_DATASET_ID = "synthetic_dense_overlap"
SYNTHETIC_SPARSE_DATASET_ID = "synthetic_sparse_overlap"


@dataclass(slots=True, frozen=True)
class BackendComparisonDataset:
    """One fixed bundle used in the IRT backend comparison pass."""

    dataset_id: str
    label: str
    source_path: Path
    config_payload: dict[str, Any]
    stage_options: dict[str, dict[str, Any]]
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class IRTBackendComparisonResult:
    """Saved outputs and decision payload from the backend comparison harness."""

    out_dir: Path
    run_index: pd.DataFrame
    per_run_metrics: pd.DataFrame
    dataset_summary: pd.DataFrame
    backend_summary: pd.DataFrame
    deployment_summary: pd.DataFrame
    parity_summary: pd.DataFrame
    report: dict[str, Any]
    artifact_paths: dict[str, Path]


def compare_irt_backends(
    *,
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    backends: Sequence[str] = DEFAULT_BACKENDS,
    deployment_seed: int = DEFAULT_DEPLOYMENT_SEED,
    bayes_backend_options: Mapping[str, Any] | None = None,
    runtime_multiplier_limit: float = DEFAULT_RUNTIME_MULTIPLIER_LIMIT,
    run_timeout_seconds: int = DEFAULT_RUN_TIMEOUT_SECONDS,
    parity_gate_thresholds: Mapping[str, float] | None = None,
    datasets: Sequence[BackendComparisonDataset] | None = None,
) -> IRTBackendComparisonResult:
    """Run the fixed-bundle comparison and return the saved decision artifacts."""

    resolved_out_dir = Path(out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    resolved_datasets = list(build_backend_comparison_datasets() if datasets is None else datasets)
    if not resolved_datasets:
        raise ValueError("backend comparison requires at least one dataset")

    resolved_backends = tuple(str(backend) for backend in backends)
    resolved_seeds = tuple(int(seed) for seed in seeds)
    resolved_parity_thresholds = dict(
        DEFAULT_PARITY_GATE_THRESHOLDS if parity_gate_thresholds is None else parity_gate_thresholds
    )
    resolved_bayes_backend_options = dict(DEFAULT_BAYES_BACKEND_OPTIONS)
    if bayes_backend_options is not None:
        resolved_bayes_backend_options.update(dict(bayes_backend_options))

    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    deployment_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []

    workdir = resolved_out_dir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    for backend in resolved_backends:
        parity_row = _parity_row(
            out_dir=resolved_out_dir / "parity" / backend,
            backend=backend,
            backend_options=(
                resolved_bayes_backend_options
                if backend == "bayes_mcmc"
                else {"max_iteration": 60}
            ),
            gate_thresholds=resolved_parity_thresholds,
        )
        parity_rows.append(parity_row)
        if not parity_row["gate_passed"]:
            for dataset in resolved_datasets:
                for seed in resolved_seeds:
                    run_rows.append(
                        _skipped_run_row(
                            dataset=dataset,
                            backend=backend,
                            seed=seed,
                            reason="skipped after parity-gate failure",
                        )
                    )
                    if seed == deployment_seed:
                        deployment_rows.append(
                            {
                                **_failed_deployment_row(
                                    dataset=dataset,
                                    backend=backend,
                                    seed=seed,
                                    error_type="SkippedAfterParityFailure",
                                    error_message="skipped after parity-gate failure",
                                ),
                                "status": "skipped",
                            }
                        )
            continue
        for dataset in resolved_datasets:
            if not dataset.source_path.exists():
                raise FileNotFoundError(
                    f"backend comparison source is missing for `{dataset.dataset_id}`: "
                    f"{dataset.source_path}"
                )
            for seed in resolved_seeds:
                run_signature = f"{dataset.dataset_id}__{backend}__seed_{seed}"
                run_root = workdir / dataset.dataset_id / backend
                config = _config_for_seed(dataset, seed=seed)
                stage_options = _stage_options_for_backend(
                    dataset,
                    backend=backend,
                    seed=seed,
                    bayes_backend_options=resolved_bayes_backend_options,
                )
                wall_start = perf_counter()
                try:
                    run_result = _run_with_timeout(
                        timeout_seconds=run_timeout_seconds,
                        source_path=dataset.source_path,
                        config=config,
                        out_dir=run_root,
                        run_id=run_signature,
                        stage_options=stage_options,
                    )
                except Exception as exc:
                    wall_runtime = perf_counter() - wall_start
                    run_rows.append(
                        {
                            "run_signature": run_signature,
                            "run_id": run_signature,
                            "dataset_id": dataset.dataset_id,
                            "dataset_label": dataset.label,
                            "backend": backend,
                            "seed": seed,
                            "status": "failed",
                            "run_root": None,
                            "metrics_path": None,
                            "run_runtime_seconds": wall_runtime,
                            "stage05_runtime_seconds": None,
                            "stage09_runtime_seconds": None,
                            "warning_count": None,
                            "error_type": exc.__class__.__name__,
                            "error_message": str(exc),
                        }
                    )
                    if seed == deployment_seed:
                        deployment_rows.append(
                            _failed_deployment_row(
                                dataset=dataset,
                                backend=backend,
                                seed=seed,
                                error_type=exc.__class__.__name__,
                                error_message=str(exc),
                            )
                        )
                    continue

                summary = run_result.summary()
                stage_records = summary["stage_records"]
                run_rows.append(
                    {
                        "run_signature": run_signature,
                        "run_id": run_signature,
                        "dataset_id": dataset.dataset_id,
                        "dataset_label": dataset.label,
                        "backend": backend,
                        "seed": seed,
                        "status": "ok",
                        "run_root": str(run_result.run_root),
                        "metrics_path": str(run_result.run_root / "reports" / "metrics.json"),
                        "run_runtime_seconds": _sum_stage_runtime(stage_records),
                        "stage05_runtime_seconds": _stage_runtime(stage_records, "05_irt"),
                        "stage09_runtime_seconds": _stage_runtime(stage_records, "09_reconstruct"),
                        "warning_count": int(len(summary["warnings"])),
                        "error_type": None,
                        "error_message": None,
                    }
                )
                metric_rows.extend(
                    _benchmark_metric_rows(
                        dataset=dataset,
                        backend=backend,
                        seed=seed,
                        run_signature=run_signature,
                        run_result=run_result,
                    )
                )
                if seed == deployment_seed:
                    deployment_rows.append(
                        _deployment_row(
                            dataset=dataset,
                            backend=backend,
                            seed=seed,
                            run_signature=run_signature,
                            run_result=run_result,
                            stage_options=stage_options,
                            out_dir=resolved_out_dir / "deployment" / dataset.dataset_id / backend,
                        )
                    )

    run_index = _records_frame(run_rows)
    per_run_metrics = _records_frame(metric_rows)
    dataset_summary = _dataset_summary_frame(run_index=run_index, per_run_metrics=per_run_metrics)
    backend_summary = _backend_summary_frame(
        run_index=run_index,
        per_run_metrics=per_run_metrics,
        dataset_summary=dataset_summary,
    )
    deployment_summary = _records_frame(deployment_rows)
    parity_summary = _records_frame(parity_rows)
    report = _decision_report(
        run_index=run_index,
        dataset_summary=dataset_summary,
        backend_summary=backend_summary,
        deployment_summary=deployment_summary,
        parity_summary=parity_summary,
        seeds=resolved_seeds,
        runtime_multiplier_limit=runtime_multiplier_limit,
        run_timeout_seconds=run_timeout_seconds,
        parity_gate_thresholds=resolved_parity_thresholds,
    )
    artifact_paths = write_backend_comparison_artifacts(
        out_dir=resolved_out_dir,
        run_index=run_index,
        per_run_metrics=per_run_metrics,
        dataset_summary=dataset_summary,
        backend_summary=backend_summary,
        deployment_summary=deployment_summary,
        parity_summary=parity_summary,
        report=report,
    )
    return IRTBackendComparisonResult(
        out_dir=resolved_out_dir,
        run_index=run_index,
        per_run_metrics=per_run_metrics,
        dataset_summary=dataset_summary,
        backend_summary=backend_summary,
        deployment_summary=deployment_summary,
        parity_summary=parity_summary,
        report=report,
        artifact_paths=artifact_paths,
    )


def build_backend_comparison_datasets() -> list[BackendComparisonDataset]:
    """Return the fixed bundle family used by the IRT backend comparison pass."""

    tiny_config = json.loads(
        (REPO_ROOT / "tests" / "data" / "tiny_example" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    reconstruction_profile = benchiq.build_reconstruction_first_profile(random_seed=7)
    return [
        BackendComparisonDataset(
            dataset_id=FIXTURE_DATASET_ID,
            label="compact validation fixture",
            source_path=REPO_ROOT / "tests" / "data" / "compact_validation" / "responses_long.csv",
            config_payload=tiny_config["config"],
            stage_options=tiny_config["stage_options"],
            notes=("fast continuity fixture for stage-shape and score quality checks",),
        ),
        BackendComparisonDataset(
            dataset_id=LARGE_DATASET_ID,
            label="large release-default subset",
            source_path=REPO_ROOT
            / "out"
            / "release_bundle_source"
            / "release_default_subset_responses_long.parquet",
            config_payload=reconstruction_profile.config.model_dump(mode="json"),
            stage_options={
                "04_subsample": {
                    "method": "deterministic_info",
                    "k_preselect": 350,
                    "n_iter": 24,
                    "cv_folds": 5,
                    "checkpoint_interval": 8,
                    "lam_grid": [0.1, 1.0],
                },
                "05_irt": {"backend": "girth", "backend_options": None},
                "06_select": {
                    "k_final": 250,
                    "n_bins": 250,
                    "theta_grid_size": 251,
                },
                "07_theta": {"theta_method": "MAP", "theta_grid_size": 251},
                "09_reconstruct": {
                    "lam_grid": [0.1, 1.0],
                    "cv_folds": 5,
                    "n_splines": 10,
                },
                "10_redundancy": {
                    "lam_grid": [0.1, 1.0],
                    "cv_folds": 5,
                    "n_splines": 10,
                    "n_factors_to_try": [1, 2, 3],
                },
            },
            notes=("primary held-out real-data decision bundle",),
        ),
        BackendComparisonDataset(
            dataset_id=SYNTHETIC_DENSE_DATASET_ID,
            label="synthetic dense-overlap bundle",
            source_path=REPO_ROOT
            / "reports"
            / "generalization_optimization"
            / "generated_bundles"
            / SYNTHETIC_DENSE_DATASET_ID
            / "responses_long.parquet",
            config_payload={
                "allow_low_n": True,
                "drop_low_tail_models_quantile": 0.001,
                "min_item_sd": 0.01,
                "max_item_mean": 0.95,
                "min_abs_point_biserial": 0.05,
                "min_models_per_benchmark": 180,
                "warn_models_per_benchmark": 220,
                "min_items_after_filtering": 60,
                "min_models_per_item": 60,
                "min_item_coverage": 0.80,
                "min_overlap_models_for_joint": 180,
                "min_overlap_models_for_redundancy": 180,
                "random_seed": 7,
            },
            stage_options=_synthetic_stage_options(k_preselect=70, k_final=45),
            notes=("dense-overlap stress bundle reused from the saved generalization pass",),
        ),
        BackendComparisonDataset(
            dataset_id=SYNTHETIC_SPARSE_DATASET_ID,
            label="synthetic sparse-overlap bundle",
            source_path=REPO_ROOT
            / "reports"
            / "generalization_optimization"
            / "generated_bundles"
            / SYNTHETIC_SPARSE_DATASET_ID
            / "responses_long.parquet",
            config_payload={
                "allow_low_n": True,
                "drop_low_tail_models_quantile": 0.001,
                "min_item_sd": 0.01,
                "max_item_mean": 0.95,
                "min_abs_point_biserial": 0.05,
                "min_models_per_benchmark": 120,
                "warn_models_per_benchmark": 160,
                "min_items_after_filtering": 45,
                "min_models_per_item": 40,
                "min_item_coverage": 0.80,
                "min_overlap_models_for_joint": 90,
                "min_overlap_models_for_redundancy": 90,
                "random_seed": 7,
            },
            stage_options=_synthetic_stage_options(k_preselect=55, k_final=35),
            notes=("sparse-overlap stress bundle reused from the saved generalization pass",),
        ),
    ]


def write_backend_comparison_artifacts(
    *,
    out_dir: str | Path,
    run_index: pd.DataFrame,
    per_run_metrics: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    backend_summary: pd.DataFrame,
    deployment_summary: pd.DataFrame,
    parity_summary: pd.DataFrame,
    report: Mapping[str, Any],
) -> dict[str, Path]:
    """Write the stable report family for the backend comparison pass."""

    resolved_out_dir = Path(out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_index_parquet": write_parquet(run_index, resolved_out_dir / "run_index.parquet"),
        "per_run_metrics_parquet": write_parquet(
            per_run_metrics,
            resolved_out_dir / "per_run_metrics.parquet",
        ),
        "summary_parquet": write_parquet(dataset_summary, resolved_out_dir / "summary.parquet"),
        "backend_summary_parquet": write_parquet(
            backend_summary,
            resolved_out_dir / "backend_summary.parquet",
        ),
        "deployment_summary_parquet": write_parquet(
            deployment_summary,
            resolved_out_dir / "deployment_summary.parquet",
        ),
        "parity_summary_parquet": write_parquet(
            parity_summary,
            resolved_out_dir / "parity_summary.parquet",
        ),
    }
    _write_csv(run_index, resolved_out_dir / "run_index.csv")
    _write_csv(per_run_metrics, resolved_out_dir / "per_run_metrics.csv")
    _write_csv(dataset_summary, resolved_out_dir / "summary.csv")
    _write_csv(backend_summary, resolved_out_dir / "backend_summary.csv")
    _write_csv(deployment_summary, resolved_out_dir / "deployment_summary.csv")
    _write_csv(parity_summary, resolved_out_dir / "parity_summary.csv")
    paths["report_json"] = write_json(dict(report), resolved_out_dir / "report.json")
    summary_md = resolved_out_dir / "summary.md"
    summary_md.write_text(
        _summary_markdown(
            dataset_summary=dataset_summary,
            backend_summary=backend_summary,
            deployment_summary=deployment_summary,
            parity_summary=parity_summary,
            report=report,
        ),
        encoding="utf-8",
    )
    paths["summary_md"] = summary_md
    return paths


def _synthetic_stage_options(*, k_preselect: int, k_final: int) -> dict[str, dict[str, Any]]:
    return {
        "04_subsample": {
            "method": "deterministic_info",
            "k_preselect": k_preselect,
            "n_iter": 12,
            "cv_folds": 5,
            "checkpoint_interval": 4,
            "lam_grid": [0.1, 1.0],
        },
        "05_irt": {"backend": "girth", "backend_options": None},
        "06_select": {"k_final": k_final, "theta_grid_size": 151},
        "07_theta": {"theta_grid_size": 151},
        "09_reconstruct": {
            "lam_grid": [0.1, 1.0],
            "cv_folds": 5,
            "n_splines": 8,
        },
        "10_redundancy": {
            "lam_grid": [0.1, 1.0],
            "cv_folds": 5,
            "n_splines": 8,
            "n_factors_to_try": [1, 2, 3],
        },
    }


def _config_for_seed(dataset: BackendComparisonDataset, *, seed: int) -> dict[str, Any]:
    config = json.loads(json.dumps(dataset.config_payload))
    config["random_seed"] = seed
    return config


def _stage_options_for_backend(
    dataset: BackendComparisonDataset,
    *,
    backend: str,
    seed: int,
    bayes_backend_options: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    stage_options = json.loads(json.dumps(dataset.stage_options))
    stage_options.setdefault("05_irt", {})
    if backend == "girth":
        stage_options["05_irt"] = {"backend": "girth", "backend_options": {"max_iteration": 60}}
        return stage_options
    backend_options = dict(bayes_backend_options)
    backend_options.setdefault("random_seed", seed)
    stage_options["05_irt"] = {"backend": "bayes_mcmc", "backend_options": backend_options}
    return stage_options


def _parity_row(
    *,
    out_dir: Path,
    backend: str,
    backend_options: Mapping[str, Any],
    gate_thresholds: Mapping[str, float],
) -> dict[str, Any]:
    result = run_r_baseline_comparison(
        out_dir=out_dir,
        backend=backend,
        backend_options=backend_options,
        gate_thresholds=gate_thresholds,
    )
    metrics = result.report.get("metrics") or {}
    theta_metrics = metrics.get("theta") or {}
    icc_metrics = metrics.get("icc") or {}
    return {
        "backend": backend,
        "status": result.report["status"],
        "gate_passed": bool(result.report["gate"]["passed"]),
        "skip_reason": result.report.get("skip_reason"),
        "theta_pearson": theta_metrics.get("pearson"),
        "theta_spearman": theta_metrics.get("spearman"),
        "icc_mean_rmse": icc_metrics.get("mean_rmse"),
        "summary_path": str(result.summary_path),
        "report_json_path": str(out_dir / "irt_r_baseline_summary.json"),
        "failure_count": int(result.report["gate"]["failure_count"]),
    }


def _benchmark_metric_rows(
    *,
    dataset: BackendComparisonDataset,
    backend: str,
    seed: int,
    run_signature: str,
    run_result,
) -> list[dict[str, Any]]:
    reconstruction_summary = run_result.stage_results["09_reconstruct"].reconstruction_summary
    rows: list[dict[str, Any]] = []
    for benchmark_id in (
        reconstruction_summary[BENCHMARK_ID].dropna().astype("string").unique().tolist()
    ):
        marginal_rmse = _split_metric(
            reconstruction_summary,
            benchmark_id=benchmark_id,
            model_type=MARGINAL_MODEL,
            split_name="test",
        )
        joint_rmse = _split_metric(
            reconstruction_summary,
            benchmark_id=benchmark_id,
            model_type=JOINT_MODEL,
            split_name="test",
        )
        rows.append(
            {
                "run_signature": run_signature,
                "dataset_id": dataset.dataset_id,
                "dataset_label": dataset.label,
                "backend": backend,
                "seed": seed,
                BENCHMARK_ID: benchmark_id,
                "marginal_test_rmse": marginal_rmse,
                "joint_test_rmse": joint_rmse,
                "best_available_test_rmse": joint_rmse if joint_rmse is not None else marginal_rmse,
                "best_available_model_type": (
                    JOINT_MODEL
                    if joint_rmse is not None
                    else (MARGINAL_MODEL if marginal_rmse is not None else None)
                ),
                "joint_available": joint_rmse is not None,
            }
        )
    return rows


def _deployment_row(
    *,
    dataset: BackendComparisonDataset,
    backend: str,
    seed: int,
    run_signature: str,
    run_result,
    stage_options: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reduced_path = out_dir / "reduced_test_responses.parquet"
    reduced_responses = _deployment_reduced_test_responses(run_result)
    reduced_responses.to_parquet(reduced_path, index=False)
    try:
        calibration_root, _, _ = write_calibration_bundle(run_result, stage_options=stage_options)
        prediction_result = benchiq.predict(
            calibration_root,
            reduced_path,
            out_dir=out_dir,
            run_id=f"{run_signature}__predict",
        )
    except Exception as exc:
        return _failed_deployment_row(
            dataset=dataset,
            backend=backend,
            seed=seed,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        )

    comparison = _deployment_comparison_frame(
        run_result=run_result,
        prediction_result=prediction_result,
    )
    prediction_available_rate = float(comparison["prediction_available"].mean())
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_label": dataset.label,
        "backend": backend,
        "seed": seed,
        "status": "ok",
        "prediction_available_rate": prediction_available_rate,
        "deployment_rmse": _rmse(
            comparison["actual_score"].astype(float).to_numpy(),
            comparison["predicted_score"].astype(float).to_numpy(),
        ),
        "calibration_root": str(calibration_root),
        "prediction_root": str(prediction_result.run_root),
        "error_type": None,
        "error_message": None,
    }


def _failed_deployment_row(
    *,
    dataset: BackendComparisonDataset,
    backend: str,
    seed: int,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_label": dataset.label,
        "backend": backend,
        "seed": seed,
        "status": "failed",
        "prediction_available_rate": None,
        "deployment_rmse": None,
        "calibration_root": None,
        "prediction_root": None,
        "error_type": error_type,
        "error_message": error_message,
    }


def _skipped_run_row(
    *,
    dataset: BackendComparisonDataset,
    backend: str,
    seed: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "run_signature": f"{dataset.dataset_id}__{backend}__seed_{seed}",
        "run_id": f"{dataset.dataset_id}__{backend}__seed_{seed}",
        "dataset_id": dataset.dataset_id,
        "dataset_label": dataset.label,
        "backend": backend,
        "seed": seed,
        "status": "skipped",
        "run_root": None,
        "metrics_path": None,
        "run_runtime_seconds": None,
        "stage05_runtime_seconds": None,
        "stage09_runtime_seconds": None,
        "warning_count": None,
        "error_type": "SkippedRun",
        "error_message": reason,
    }


def _deployment_reduced_test_responses(run_result) -> pd.DataFrame:
    bundle = run_result.stage_results["00_bundle"]
    split_result = run_result.stage_results["03_splits"]
    select_result = run_result.stage_results["06_select"]
    test_model_ids = {
        str(model_id)
        for split_frame in split_result.per_benchmark_splits.values()
        for model_id in split_frame.loc[split_frame["split"] == "test", MODEL_ID]
        .astype("string")
        .tolist()
    }
    selected_by_benchmark = {
        benchmark_id: set(
            benchmark_result.subset_final["item_id"].dropna().astype("string").tolist()
        )
        for benchmark_id, benchmark_result in select_result.benchmarks.items()
    }
    reduced = bundle.responses_long.loc[
        bundle.responses_long[MODEL_ID].astype("string").isin(sorted(test_model_ids))
    ].copy()
    keep_mask = reduced.apply(
        lambda row: row["item_id"] in selected_by_benchmark.get(str(row["benchmark_id"]), set()),
        axis=1,
    )
    return reduced.loc[keep_mask].reset_index(drop=True)


def _deployment_comparison_frame(*, run_result, prediction_result) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark_id, benchmark_result in sorted(
        run_result.stage_results["09_reconstruct"].benchmarks.items()
    ):
        test_rows = benchmark_result.predictions.loc[
            benchmark_result.predictions[SPLIT] == "test"
        ].copy()
        for model_id in test_rows[MODEL_ID].dropna().astype("string").unique().tolist():
            model_rows = test_rows.loc[test_rows[MODEL_ID] == model_id].copy()
            joint_rows = model_rows.loc[model_rows["model_type"] == JOINT_MODEL].copy()
            chosen_rows = (
                joint_rows
                if not joint_rows.empty
                else model_rows.loc[model_rows["model_type"] == MARGINAL_MODEL].copy()
            )
            if chosen_rows.empty:
                continue
            row = chosen_rows.iloc[0]
            rows.append(
                {
                    BENCHMARK_ID: benchmark_id,
                    MODEL_ID: model_id,
                    "actual_score": row["actual_score"],
                    "reference_model_type": row["model_type"],
                }
            )
    reference = pd.DataFrame.from_records(rows)
    deployment = prediction_result.predictions_best_available.loc[
        :,
        [BENCHMARK_ID, MODEL_ID, "predicted_score", "selected_model_type"],
    ].rename(columns={"selected_model_type": "deployment_model_type"})
    merged = reference.merge(deployment, on=[BENCHMARK_ID, MODEL_ID], how="left")
    merged["prediction_available"] = merged["predicted_score"].notna()
    return merged.astype(
        {
            BENCHMARK_ID: "string",
            MODEL_ID: "string",
            "actual_score": "Float64",
            "predicted_score": "Float64",
            "prediction_available": bool,
        }
    )


def _dataset_summary_frame(
    *,
    run_index: pd.DataFrame,
    per_run_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (backend, dataset_id), group in run_index.groupby(["backend", "dataset_id"], dropna=False):
        dataset_label = str(group["dataset_label"].dropna().iloc[0])
        metric_group = per_run_metrics.loc[
            (per_run_metrics["backend"] == backend) & (per_run_metrics["dataset_id"] == dataset_id)
        ].copy()
        informative = metric_group.loc[metric_group["best_available_test_rmse"].notna()].copy()
        seed_means = (
            informative.groupby("seed", dropna=False)["best_available_test_rmse"].mean()
            if not informative.empty
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "backend": backend,
                "dataset_id": dataset_id,
                "dataset_label": dataset_label,
                "run_count": int(len(group.index)),
                "successful_run_count": int(group["status"].eq("ok").sum()),
                "failed_run_count": int(group["status"].ne("ok").sum()),
                "failure_rate": float(group["status"].ne("ok").mean()),
                "informative_benchmark_rows": int(len(informative.index)),
                "best_available_test_rmse_mean": (
                    float(informative["best_available_test_rmse"].mean())
                    if not informative.empty
                    else None
                ),
                "best_available_test_rmse_std": (
                    float(informative["best_available_test_rmse"].std(ddof=1))
                    if len(informative.index) > 1
                    else None
                ),
                "joint_available_rate": (
                    float(informative["joint_available"].mean()) if not informative.empty else None
                ),
                "run_runtime_mean_seconds": _mean_or_none(group["run_runtime_seconds"]),
                "stage05_runtime_mean_seconds": _mean_or_none(group["stage05_runtime_seconds"]),
                "seed_rmse_std": (
                    float(seed_means.std(ddof=1)) if len(seed_means.index) > 1 else None
                ),
                "informative_dataset": not informative.empty,
            }
        )
    return _records_frame(rows).sort_values(["dataset_id", "backend"]).reset_index(drop=True)


def _backend_summary_frame(
    *,
    run_index: pd.DataFrame,
    per_run_metrics: pd.DataFrame,
    dataset_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for backend, backend_runs in run_index.groupby("backend", dropna=False):
        backend_datasets = dataset_summary.loc[dataset_summary["backend"] == backend].copy()
        informative = backend_datasets.loc[backend_datasets["informative_dataset"]].copy()
        seed_series = (
            run_index.loc[run_index["backend"] == backend]
            .merge(
                dataset_summary.loc[
                    :,
                    ["backend", "dataset_id", "informative_dataset"],
                ],
                on=["backend", "dataset_id"],
                how="left",
            )
            .loc[
                lambda frame: (
                    frame["informative_dataset"].fillna(False) & frame["status"].eq("ok")
                )
            ]
            .groupby("seed", dropna=False)["run_runtime_seconds"]
            .mean()
        )
        rows.append(
            {
                "backend": backend,
                "run_count": int(len(backend_runs.index)),
                "failure_rate": float(backend_runs["status"].ne("ok").mean()),
                "informative_bundle_count": int(len(informative.index)),
                "equal_weight_informative_rmse_mean": (
                    float(informative["best_available_test_rmse_mean"].mean())
                    if not informative.empty
                    else None
                ),
                "seed_rmse_std": _seed_metric_std(
                    per_run_metrics=per_run_metrics,
                    dataset_summary=dataset_summary,
                    backend=str(backend),
                ),
                "large_bundle_runtime_mean_seconds": _dataset_value_or_none(
                    dataset_summary,
                    backend=str(backend),
                    dataset_id=LARGE_DATASET_ID,
                    column="run_runtime_mean_seconds",
                ),
                "large_bundle_stage05_runtime_mean_seconds": _dataset_value_or_none(
                    dataset_summary,
                    backend=str(backend),
                    dataset_id=LARGE_DATASET_ID,
                    column="stage05_runtime_mean_seconds",
                ),
                "mean_successful_run_runtime_seconds": _mean_or_none(
                    backend_runs.loc[backend_runs["status"] == "ok", "run_runtime_seconds"]
                ),
                "successful_seed_runtime_std": (
                    float(seed_series.std(ddof=1)) if len(seed_series.index) > 1 else None
                ),
            }
        )
    return _records_frame(rows).sort_values("backend").reset_index(drop=True)


def _decision_report(
    *,
    run_index: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    backend_summary: pd.DataFrame,
    deployment_summary: pd.DataFrame,
    parity_summary: pd.DataFrame,
    seeds: Sequence[int],
    runtime_multiplier_limit: float,
    run_timeout_seconds: int,
    parity_gate_thresholds: Mapping[str, float],
) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    for backend_row in backend_summary.to_dict(orient="records"):
        backend = str(backend_row["backend"])
        candidate_rows.append(
            {
                **backend_row,
                "disqualifications": _disqualifications_for_backend(
                    backend=backend,
                    backend_summary=backend_summary,
                    dataset_summary=dataset_summary,
                    run_index=run_index,
                    deployment_summary=deployment_summary,
                    parity_summary=parity_summary,
                    runtime_multiplier_limit=runtime_multiplier_limit,
                ),
            }
        )

    winner_payload = _winner_payload(candidate_rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "runtime_multiplier_limit": runtime_multiplier_limit,
        "run_timeout_seconds": run_timeout_seconds,
        "parity_gate_thresholds": dict(parity_gate_thresholds),
        "winner": winner_payload,
        "candidates": candidate_rows,
    }


def _disqualifications_for_backend(
    *,
    backend: str,
    backend_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    run_index: pd.DataFrame,
    deployment_summary: pd.DataFrame,
    parity_summary: pd.DataFrame,
    runtime_multiplier_limit: float,
) -> list[str]:
    reasons: list[str] = []
    parity_row = parity_summary.loc[parity_summary["backend"] == backend]
    if parity_row.empty or not bool(parity_row.iloc[0]["gate_passed"]):
        reasons.append("failed the aligned R parity gate")

    large_deployment = deployment_summary.loc[
        (deployment_summary["backend"] == backend)
        & (deployment_summary["dataset_id"] == LARGE_DATASET_ID)
    ]
    if large_deployment.empty or large_deployment.iloc[0]["status"] != "ok":
        reasons.append("failed calibration or prediction on the large release-default subset")
    else:
        prediction_available_rate = _float_or_none(
            large_deployment.iloc[0]["prediction_available_rate"]
        )
        if prediction_available_rate is None or prediction_available_rate < 1.0:
            reasons.append(
                "prediction_available_rate fell below 1.0 on the large release-default subset"
            )

    other_rows = backend_summary.loc[backend_summary["backend"] != backend].copy()
    if not other_rows.empty:
        other_backend = str(other_rows.iloc[0]["backend"])
        other_runtime = _float_or_none(other_rows.iloc[0]["large_bundle_runtime_mean_seconds"])
        this_runtime = _float_or_none(
            backend_summary.loc[backend_summary["backend"] == backend].iloc[0][
                "large_bundle_runtime_mean_seconds"
            ]
        )
        if (
            other_runtime is not None
            and other_runtime > 0.0
            and this_runtime is not None
            and this_runtime > runtime_multiplier_limit * other_runtime
        ):
            reasons.append(
                f"large-bundle runtime exceeded {runtime_multiplier_limit:.1f}x "
                f"the {other_backend} runtime"
            )

    for dataset_id in dataset_summary["dataset_id"].dropna().astype(str).unique().tolist():
        this_row = dataset_summary.loc[
            (dataset_summary["backend"] == backend) & (dataset_summary["dataset_id"] == dataset_id)
        ]
        other_row = dataset_summary.loc[
            (dataset_summary["backend"] != backend) & (dataset_summary["dataset_id"] == dataset_id)
        ]
        if this_row.empty or other_row.empty:
            continue
        if (
            int(this_row.iloc[0]["successful_run_count"]) == 0
            and int(other_row.iloc[0]["successful_run_count"]) > 0
        ):
            reasons.append(
                f"failed pipeline runs on `{dataset_id}` while the other backend completed"
            )

    for dataset_id in deployment_summary["dataset_id"].dropna().astype(str).unique().tolist():
        this_row = deployment_summary.loc[
            (deployment_summary["backend"] == backend)
            & (deployment_summary["dataset_id"] == dataset_id)
        ]
        other_row = deployment_summary.loc[
            (deployment_summary["backend"] != backend)
            & (deployment_summary["dataset_id"] == dataset_id)
        ]
        if this_row.empty or other_row.empty:
            continue
        if this_row.iloc[0]["status"] != "ok" and other_row.iloc[0]["status"] == "ok":
            reasons.append(
                f"failed calibration or prediction on `{dataset_id}` "
                "while the other backend succeeded"
            )
    return sorted(set(reasons))


def _winner_payload(candidate_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not candidate_rows:
        raise ValueError("backend comparison produced no candidate rows")
    eligible = [row for row in candidate_rows if not row["disqualifications"]]
    if not eligible:
        girth_row = next(
            (row for row in candidate_rows if row["backend"] == "girth"),
            candidate_rows[0],
        )
        return {
            "backend": girth_row["backend"],
            "eligible": False,
            "reason": "no candidate cleared every gate; keep girth as the shipped default",
        }
    winner = sorted(
        eligible,
        key=lambda row: (
            float("inf")
            if _float_or_none(row["equal_weight_informative_rmse_mean"]) is None
            else float(row["equal_weight_informative_rmse_mean"]),
            (
                float("inf")
                if _float_or_none(row["seed_rmse_std"]) is None
                else float(row["seed_rmse_std"])
            ),
            float(row["failure_rate"]),
            float("inf")
            if _float_or_none(row["large_bundle_runtime_mean_seconds"]) is None
            else float(row["large_bundle_runtime_mean_seconds"]),
            0 if row["backend"] == "girth" else 1,
        ),
    )[0]
    return {
        "backend": winner["backend"],
        "eligible": True,
        "reason": (
            "winner selected by held-out equal-weight rmse, then seed rmse "
            "stability, failure rate, and runtime"
        ),
    }


def _summary_markdown(
    *,
    dataset_summary: pd.DataFrame,
    backend_summary: pd.DataFrame,
    deployment_summary: pd.DataFrame,
    parity_summary: pd.DataFrame,
    report: Mapping[str, Any],
) -> str:
    lines = [
        "# irt backend comparison",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- winner: `{report['winner']['backend']}`",
        f"- winner_eligible: `{report['winner']['eligible']}`",
        f"- winner_reason: `{report['winner']['reason']}`",
        "",
        "## candidates",
        "",
    ]
    for row in report["candidates"]:
        lines.append(
            "- "
            + f"`{row['backend']}`: "
            + "equal_weight_informative_rmse_mean="
            + f"{_fmt_float(row['equal_weight_informative_rmse_mean'])}, "
            + f"seed_rmse_std={_fmt_float(row['seed_rmse_std'])}, "
            + f"failure_rate={_fmt_float(row['failure_rate'])}, "
            + "large_bundle_runtime_mean_seconds="
            + f"{_fmt_float(row['large_bundle_runtime_mean_seconds'])}, "
            + f"disqualifications={row['disqualifications']}"
        )
    lines.extend(["", "## dataset summary", ""])
    for row in dataset_summary.to_dict(orient="records"):
        lines.append(
            "- "
            + f"`{row['dataset_id']}` / `{row['backend']}`: "
            + f"rmse_mean={_fmt_float(row['best_available_test_rmse_mean'])}, "
            + f"seed_rmse_std={_fmt_float(row['seed_rmse_std'])}, "
            + f"failure_rate={_fmt_float(row['failure_rate'])}, "
            + f"informative_dataset={row['informative_dataset']}"
        )
    lines.extend(["", "## parity", ""])
    for row in parity_summary.to_dict(orient="records"):
        lines.append(
            "- "
            + f"`{row['backend']}`: status={row['status']}, gate_passed={row['gate_passed']}, "
            + f"theta_pearson={_fmt_float(row['theta_pearson'])}, "
            + f"theta_spearman={_fmt_float(row['theta_spearman'])}, "
            + f"icc_mean_rmse={_fmt_float(row['icc_mean_rmse'])}"
        )
    lines.extend(["", "## deployment", ""])
    for row in deployment_summary.to_dict(orient="records"):
        lines.append(
            "- "
            + f"`{row['dataset_id']}` / `{row['backend']}`: status={row['status']}, "
            + f"prediction_available_rate={_fmt_float(row['prediction_available_rate'])}, "
            + f"deployment_rmse={_fmt_float(row['deployment_rmse'])}"
        )
    return "\n".join(lines) + "\n"


def _records_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def _run_with_timeout(
    *,
    timeout_seconds: int,
    source_path: Path,
    config: Mapping[str, Any],
    out_dir: Path,
    run_id: str,
    stage_options: Mapping[str, Mapping[str, Any]],
):
    if timeout_seconds <= 0:
        return benchiq.run(
            source_path,
            config=config,
            out_dir=out_dir,
            run_id=run_id,
            stage_options=stage_options,
        )

    def _timeout_handler(signum, frame):  # type: ignore[no-untyped-def]
        raise TimeoutError(f"backend comparison run exceeded {timeout_seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return benchiq.run(
            source_path,
            config=config,
            out_dir=out_dir,
            run_id=run_id,
            stage_options=stage_options,
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _split_metric(
    reconstruction_summary: pd.DataFrame,
    *,
    benchmark_id: str,
    model_type: str,
    split_name: str,
) -> float | None:
    rows = reconstruction_summary.loc[
        (reconstruction_summary[BENCHMARK_ID] == benchmark_id)
        & (reconstruction_summary["model_type"] == model_type)
        & (reconstruction_summary[SPLIT] == split_name)
    ].copy()
    if rows.empty or rows["rmse"].isna().all():
        return None
    return float(rows["rmse"].dropna().iloc[0])


def _sum_stage_runtime(stage_records: Mapping[str, Mapping[str, Any]]) -> float:
    return float(
        sum(float(stage_record["duration_seconds"]) for stage_record in stage_records.values())
    )


def _stage_runtime(stage_records: Mapping[str, Mapping[str, Any]], stage_name: str) -> float | None:
    record = stage_records.get(stage_name)
    if record is None:
        return None
    return float(record["duration_seconds"])


def _mean_or_none(values: pd.Series) -> float | None:
    numeric = values.dropna()
    if numeric.empty:
        return None
    return float(numeric.astype(float).mean())


def _dataset_value_or_none(
    dataset_summary: pd.DataFrame,
    *,
    backend: str,
    dataset_id: str,
    column: str,
) -> float | None:
    rows = dataset_summary.loc[
        (dataset_summary["backend"] == backend) & (dataset_summary["dataset_id"] == dataset_id)
    ]
    if rows.empty:
        return None
    return _float_or_none(rows.iloc[0][column])


def _seed_metric_std(
    *,
    per_run_metrics: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    backend: str,
) -> float | None:
    if per_run_metrics.empty:
        return None
    frame = per_run_metrics.loc[per_run_metrics["backend"] == backend].copy()
    if frame.empty:
        return None
    informative_ids = set(
        dataset_summary.loc[
            (dataset_summary["backend"] == backend) & dataset_summary["informative_dataset"],
            "dataset_id",
        ]
        .astype(str)
        .tolist()
    )
    frame = frame.loc[frame["dataset_id"].astype(str).isin(informative_ids)].copy()
    frame = frame.loc[frame["best_available_test_rmse"].notna()].copy()
    if frame.empty:
        return None
    seed_means = (
        frame.groupby(["seed", "dataset_id"], dropna=False)["best_available_test_rmse"]
        .mean()
        .groupby(level=0)
        .mean()
    )
    if len(seed_means.index) <= 1:
        return None
    return float(seed_means.std(ddof=1))


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _fmt_float(value: Any) -> str:
    coerced = _float_or_none(value)
    if coerced is None:
        return "NA"
    return f"{coerced:.4f}"


__all__ = [
    "BackendComparisonDataset",
    "IRTBackendComparisonResult",
    "build_backend_comparison_datasets",
    "compare_irt_backends",
    "write_backend_comparison_artifacts",
]
