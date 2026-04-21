"""First-class calibration bundle export for BenchIQ."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from benchiq.config import BenchIQConfig
from benchiq.io.load import Bundle
from benchiq.io.write import write_json
from benchiq.logging import update_manifest
from benchiq.reconstruct.reconstruction import JOINT_MODEL, MARGINAL_MODEL
from benchiq.runner import BenchIQRunner, RunResult
from benchiq.schema.tables import ITEM_ID

CALIBRATION_BUNDLE_DIRNAME = "calibration_bundle"
CALIBRATION_BUNDLE_KIND = "benchiq_calibration_bundle"
CALIBRATION_BUNDLE_VERSION = "0.1"


@dataclass(slots=True)
class CalibrationResult:
    """Top-level calibration result handle."""

    run_result: RunResult
    calibration_root: Path
    calibration_manifest_path: Path
    calibration_manifest: dict[str, Any]


def calibrate(
    bundle_or_responses_path: Bundle | str | Path,
    config: BenchIQConfig | Mapping[str, Any] | None = None,
    out_dir: str | Path | None = None,
    *,
    items_path: str | Path | None = None,
    models_path: str | Path | None = None,
    run_id: str | None = None,
    stage_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> CalibrationResult:
    """Run the calibration path and publish a reusable fitted bundle."""

    runner = BenchIQRunner(
        config=config,
        out_dir=out_dir,
        run_id=run_id,
        stage_options=stage_options,
    )
    run_result = runner.run(
        bundle_or_responses_path,
        items_path,
        models_path,
        stop_after="09_reconstruct",
    )
    calibration_root, manifest_path, manifest = write_calibration_bundle(
        run_result,
        stage_options=runner.stage_options,
    )
    return CalibrationResult(
        run_result=run_result,
        calibration_root=calibration_root,
        calibration_manifest_path=manifest_path,
        calibration_manifest=manifest,
    )


def write_calibration_bundle(
    run_result: RunResult,
    *,
    stage_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Export the reusable fitted artifacts from a completed calibration run."""

    required_stages = (
        "00_bundle",
        "05_irt",
        "06_select",
        "07_theta",
        "08_linear",
        "09_reconstruct",
    )
    missing_stages = [
        stage_name for stage_name in required_stages if stage_name not in run_result.stage_results
    ]
    if missing_stages:
        raise ValueError(
            "calibration bundle export requires completed stages through 09_reconstruct; "
            f"missing stages: {missing_stages}"
        )

    bundle = run_result.stage_results["00_bundle"]
    select_result = run_result.stage_results["06_select"]
    irt_result = run_result.stage_results["05_irt"]
    theta_result = run_result.stage_results["07_theta"]
    reconstruction_result = run_result.stage_results["09_reconstruct"]

    calibration_root = run_result.run_root / CALIBRATION_BUNDLE_DIRNAME
    calibration_root.mkdir(parents=True, exist_ok=True)

    config_path = write_json(
        bundle.config.model_dump(mode="json"),
        calibration_root / "config_resolved.json",
    )
    reconstruction_summary_path = _copy_artifact(
        run_result.run_root / "artifacts" / "09_reconstruct" / "reconstruction_summary.parquet",
        calibration_root / "reconstruction_summary.parquet",
    )
    reconstruction_report_path = _copy_artifact(
        run_result.run_root / "artifacts" / "09_reconstruct" / "reconstruction_report.json",
        calibration_root / "reconstruction_report.json",
    )

    theta_parameters = dict(theta_result.theta_report["parameters"])
    benchmark_manifest: dict[str, Any] = {}
    for benchmark_id in sorted(select_result.benchmarks):
        subset_result = select_result.benchmarks[benchmark_id]
        irt_benchmark = irt_result.benchmarks[benchmark_id]
        reconstruction_benchmark = reconstruction_result.benchmarks[benchmark_id]

        benchmark_dir = calibration_root / "per_benchmark" / benchmark_id
        subset_final_path = _copy_artifact(
            run_result.run_root
            / "artifacts"
            / "06_select"
            / "per_benchmark"
            / benchmark_id
            / "subset_final.parquet",
            benchmark_dir / "subset_final.parquet",
        )
        selection_report_path = _copy_artifact(
            run_result.run_root
            / "artifacts"
            / "06_select"
            / "per_benchmark"
            / benchmark_id
            / "selection_report.json",
            benchmark_dir / "selection_report.json",
        )
        irt_item_params_path = _copy_artifact(
            run_result.run_root
            / "artifacts"
            / "05_irt"
            / "per_benchmark"
            / benchmark_id
            / "irt_item_params.parquet",
            benchmark_dir / "irt_item_params.parquet",
        )
        irt_fit_report_path = _copy_artifact(
            run_result.run_root
            / "artifacts"
            / "05_irt"
            / "per_benchmark"
            / benchmark_id
            / "irt_fit_report.json",
            benchmark_dir / "irt_fit_report.json",
        )
        linear_coefficients_path = _copy_artifact(
            run_result.run_root
            / "artifacts"
            / "08_features"
            / "per_benchmark"
            / benchmark_id
            / "coefficients.parquet",
            benchmark_dir / "linear_predictor_coefficients.parquet",
        )
        linear_report_path = _copy_artifact(
            run_result.run_root
            / "artifacts"
            / "08_features"
            / "per_benchmark"
            / benchmark_id
            / "linear_predictor_report.json",
            benchmark_dir / "linear_predictor_report.json",
        )
        benchmark_reconstruction_report_path = _copy_artifact(
            run_result.run_root
            / "artifacts"
            / "09_reconstruct"
            / "per_benchmark"
            / benchmark_id
            / "reconstruction_report.json",
            benchmark_dir / "reconstruction_report.json",
        )

        theta_scoring_metadata = {
            "benchmark_id": benchmark_id,
            "theta_method": theta_parameters["theta_method"],
            "theta_min": theta_parameters["theta_min"],
            "theta_max": theta_parameters["theta_max"],
            "theta_grid_size": theta_parameters["theta_grid_size"],
            "missing_heavy_threshold": theta_parameters["missing_heavy_threshold"],
            "selected_item_count": int(len(subset_result.subset_final.index)),
            "selected_item_ids": subset_result.subset_final[ITEM_ID].astype("string").tolist(),
        }
        theta_scoring_metadata_path = write_json(
            theta_scoring_metadata,
            benchmark_dir / "theta_scoring_metadata.json",
        )

        reconstruction_manifest: dict[str, Any] = {}
        for model_type in (MARGINAL_MODEL, JOINT_MODEL):
            model_report = reconstruction_benchmark.reconstruction_report[model_type]
            if model_report["skipped"]:
                reconstruction_manifest[model_type] = {
                    "available": False,
                    "skip_reason": model_report["skip_reason"],
                }
                continue

            source_model_dir = (
                run_result.run_root
                / "artifacts"
                / "09_reconstruct"
                / "per_benchmark"
                / benchmark_id
                / model_type
            )
            target_model_dir = benchmark_dir / "reconstruction" / model_type
            model_path = _copy_artifact(
                source_model_dir / "gam_model.pkl",
                target_model_dir / "gam_model.pkl",
            )
            metadata_path = _copy_artifact(
                source_model_dir / "gam_model.json",
                target_model_dir / "gam_model.json",
            )
            cv_results_path = _copy_artifact(
                source_model_dir / "cv_results.parquet",
                target_model_dir / "cv_results.parquet",
            )
            cv_report_path = _copy_artifact(
                source_model_dir / "cv_report.json",
                target_model_dir / "cv_report.json",
            )
            reconstruction_manifest[model_type] = {
                "available": True,
                "feature_columns": list(model_report["feature_columns"]),
                "model_path": _relative_path(model_path, calibration_root),
                "metadata_path": _relative_path(metadata_path, calibration_root),
                "cv_results_path": _relative_path(cv_results_path, calibration_root),
                "cv_report_path": _relative_path(cv_report_path, calibration_root),
            }

        benchmark_manifest[benchmark_id] = {
            "selected_item_count": int(len(subset_result.subset_final.index)),
            "selected_item_ids": subset_result.subset_final[ITEM_ID].astype("string").tolist(),
            "preferred_model_type": reconstruction_benchmark.reconstruction_report[
                "preferred_model"
            ]["model_type"],
            "subset_final_path": _relative_path(subset_final_path, calibration_root),
            "selection_report_path": _relative_path(selection_report_path, calibration_root),
            "irt_item_params_path": _relative_path(irt_item_params_path, calibration_root),
            "irt_fit_report_path": _relative_path(irt_fit_report_path, calibration_root),
            "theta_scoring_metadata_path": _relative_path(
                theta_scoring_metadata_path,
                calibration_root,
            ),
            "linear_predictor_coefficients_path": _relative_path(
                linear_coefficients_path,
                calibration_root,
            ),
            "linear_predictor_report_path": _relative_path(linear_report_path, calibration_root),
            "reconstruction_report_path": _relative_path(
                benchmark_reconstruction_report_path,
                calibration_root,
            ),
            "reconstruction": reconstruction_manifest,
            "linear_prediction_requires_complete_coverage": True,
            "irt_backend": irt_benchmark.irt_fit_report["irt_backend"],
        }

    bundle_summary_path = calibration_root / "bundle_summary.md"
    bundle_summary_path.write_text(
        _bundle_summary_markdown(
            run_result=run_result,
            calibration_root=calibration_root,
            benchmark_manifest=benchmark_manifest,
        ),
        encoding="utf-8",
    )

    manifest = {
        "kind": CALIBRATION_BUNDLE_KIND,
        "bundle_version": CALIBRATION_BUNDLE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run_root": _relative_path(run_result.run_root, calibration_root),
        "source_run_manifest": _relative_path(run_result.manifest_path, calibration_root),
        "config": bundle.config.model_dump(mode="json"),
        "stage_options": _json_safe_stage_options(stage_options),
        "theta_scoring_defaults": theta_parameters,
        "artifacts": {
            "config_resolved": _relative_path(config_path, calibration_root),
            "reconstruction_summary": _relative_path(reconstruction_summary_path, calibration_root),
            "reconstruction_report": _relative_path(reconstruction_report_path, calibration_root),
            "bundle_summary": _relative_path(bundle_summary_path, calibration_root),
        },
        "benchmarks": benchmark_manifest,
    }
    manifest_path = write_json(manifest, calibration_root / "manifest.json")

    update_manifest(
        run_result.manifest_path,
        {
            "artifacts": {
                CALIBRATION_BUNDLE_DIRNAME: {
                    "manifest": str(manifest_path),
                    "bundle_summary": str(bundle_summary_path),
                }
            }
        },
    )
    return calibration_root, manifest_path, manifest


def _copy_artifact(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"required calibration artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def _json_safe_stage_options(
    stage_options: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if stage_options is None:
        return {}
    payload: dict[str, dict[str, Any]] = {}
    for stage_name, options in sorted(stage_options.items()):
        payload[stage_name] = {}
        for key, value in sorted(options.items()):
            if isinstance(value, tuple):
                payload[stage_name][key] = list(value)
            else:
                payload[stage_name][key] = value
    return payload


def _bundle_summary_markdown(
    *,
    run_result: RunResult,
    calibration_root: Path,
    benchmark_manifest: Mapping[str, Mapping[str, Any]],
) -> str:
    summary = run_result.summary()
    lines = [
        "# calibration bundle summary",
        "",
        f"- calibration root: `{calibration_root}`",
        f"- source run id: `{summary['run_id']}`",
        f"- source run root: `{run_result.run_root}`",
        f"- benchmark count: {len(benchmark_manifest)}",
        "",
        "## selected items",
        "",
    ]
    for benchmark_id, benchmark_payload in sorted(benchmark_manifest.items()):
        lines.append(
            f"- `{benchmark_id}`: {benchmark_payload['selected_item_count']} selected items"
        )
    lines.extend(["", "## held-out reconstruction", ""])
    marginal = summary["metrics"].get("marginal_test_rmse_by_benchmark", {})
    joint = summary["metrics"].get("joint_test_rmse_by_benchmark", {})
    lines.append(f"- marginal test rmse by benchmark: {marginal}")
    lines.append(f"- joint test rmse by benchmark: {joint}")
    lines.append("")
    return "\n".join(lines)
