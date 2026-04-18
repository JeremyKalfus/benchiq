"""Artifact-first CLI entrypoints for BenchIQ."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from benchiq.calibration import calibrate as run_calibration
from benchiq.cli.commands_metabench import (
    DEFAULT_PROFILE,
    render_metabench_failure,
    render_metabench_success,
    run_metabench_validation,
)
from benchiq.config import BenchIQConfig
from benchiq.deployment import predict as run_prediction
from benchiq.io import Bundle, load_bundle
from benchiq.io.write import write_json
from benchiq.logging import update_manifest
from benchiq.preprocess import preprocess_bundle
from benchiq.runner import RunResult
from benchiq.runner import run as run_pipeline
from benchiq.schema.checks import SchemaValidationError, ValidationReport
from benchiq.validate import validate as validate_bundle

try:  # pragma: no cover - python >=3.11 in local verification, but keep 3.10-safe
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on 3.10
    tomllib = None


@click.group()
def main() -> None:
    """Run the artifact-first BenchIQ CLI."""


@main.command("validate")
@click.option(
    "--responses",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to responses_long as csv or parquet.",
)
@click.option(
    "--items",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional path to canonical items as csv or parquet.",
)
@click.option(
    "--models",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional path to canonical models as csv or parquet.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional BenchIQ config file (.json or .toml).",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Explicit output directory. Validation writes to OUT/validate/.",
)
def validate_command(
    *,
    responses: Path,
    items: Path | None,
    models: Path | None,
    config_path: Path | None,
    out_dir: Path,
) -> None:
    """Canonicalize inputs and write validation artifacts."""

    settings = _load_cli_settings(config_path)
    run_root = out_dir / "validate"
    try:
        bundle = load_bundle(
            responses,
            items,
            models,
            config=settings.config,
            out_dir=out_dir,
            run_id="validate",
        )
        schema_report = validate_bundle(bundle, bundle.config)
        preprocess_result = preprocess_bundle(bundle)
        payload = _build_validation_payload(
            bundle=bundle,
            schema_report=schema_report,
            preprocess_result=preprocess_result,
        )
        _write_validation_artifacts(
            run_root=run_root,
            config=bundle.config,
            payload=payload,
            manifest_path=bundle.manifest_path,
        )
    except SchemaValidationError as exc:
        payload = _build_validation_failure_payload(
            message=str(exc),
            report=exc.report,
            attempted_paths={
                "responses": responses,
                "items": items,
                "models": models,
            },
        )
        _write_validation_artifacts(
            run_root=run_root,
            config=BenchIQConfig.model_validate({} if settings.config is None else settings.config),
            payload=payload,
            manifest_path=None,
        )
        click.echo(_render_validation_failure(run_root=run_root, payload=payload), err=True)
        raise click.exceptions.Exit(1) from exc
    except Exception as exc:
        payload = _build_validation_failure_payload(
            message=str(exc),
            report=None,
            attempted_paths={
                "responses": responses,
                "items": items,
                "models": models,
            },
        )
        _write_validation_artifacts(
            run_root=run_root,
            config=BenchIQConfig.model_validate({} if settings.config is None else settings.config),
            payload=payload,
            manifest_path=None,
        )
        click.echo(_render_validation_failure(run_root=run_root, payload=payload), err=True)
        raise click.exceptions.Exit(1) from exc

    if not payload["ok"]:
        click.echo(_render_validation_failure(run_root=run_root, payload=payload), err=True)
        raise click.exceptions.Exit(1)

    click.echo(_render_validation_success(run_root=run_root, payload=payload))


@main.command("run")
@click.option(
    "--responses",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to responses_long as csv or parquet.",
)
@click.option(
    "--items",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional path to canonical items as csv or parquet.",
)
@click.option(
    "--models",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional path to canonical models as csv or parquet.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional BenchIQ config file (.json or .toml).",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Explicit output directory. Runs write to OUT/<run-id>/.",
)
@click.option("--run-id", help="Optional explicit run id for stable output paths.")
def run_command(
    *,
    responses: Path,
    items: Path | None,
    models: Path | None,
    config_path: Path | None,
    out_dir: Path,
    run_id: str | None,
) -> None:
    """Execute the full deterministic BenchIQ pipeline."""

    settings = _load_cli_settings(config_path)
    resolved_run_id = run_id or _default_run_id()
    try:
        run_result = run_pipeline(
            responses,
            settings.config,
            out_dir=out_dir,
            items_path=items,
            models_path=models,
            run_id=resolved_run_id,
            stage_options=settings.stage_options,
        )
    except SchemaValidationError as exc:
        click.echo(_render_run_failure(exc, run_root=out_dir / resolved_run_id), err=True)
        raise click.exceptions.Exit(1) from exc
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_render_run_success(run_result))


@main.command("calibrate")
@click.option(
    "--responses",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to responses_long as csv or parquet.",
)
@click.option(
    "--items",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional path to canonical items as csv or parquet.",
)
@click.option(
    "--models",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional path to canonical models as csv or parquet.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional BenchIQ config file (.json or .toml).",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Explicit output directory. Calibration writes to OUT/<run-id>/.",
)
@click.option("--run-id", help="Optional explicit run id for stable output paths.")
def calibrate_command(
    *,
    responses: Path,
    items: Path | None,
    models: Path | None,
    config_path: Path | None,
    out_dir: Path,
    run_id: str | None,
) -> None:
    """Fit the reusable calibration stack without retraining at prediction time."""

    settings = _load_cli_settings(config_path)
    resolved_run_id = run_id or _default_run_id()
    try:
        calibration_result = run_calibration(
            responses,
            settings.config,
            out_dir=out_dir,
            items_path=items,
            models_path=models,
            run_id=resolved_run_id,
            stage_options=settings.stage_options,
        )
    except SchemaValidationError as exc:
        click.echo(_render_calibration_failure(exc, run_root=out_dir / resolved_run_id), err=True)
        raise click.exceptions.Exit(1) from exc
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_render_calibration_success(calibration_result))


@main.command("predict")
@click.option(
    "--bundle",
    "bundle_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help=(
        "Path to a calibration bundle directory, its manifest.json, "
        "or a calibration run root containing calibration_bundle/."
    ),
)
@click.option(
    "--responses",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to reduced responses_long as csv or parquet.",
)
@click.option(
    "--items",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional path to canonical items as csv or parquet.",
)
@click.option(
    "--models",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional path to canonical models as csv or parquet.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Explicit output directory. Predictions write to OUT/<run-id>/.",
)
@click.option("--run-id", help="Optional explicit run id for stable output paths.")
def predict_command(
    *,
    bundle_path: Path,
    responses: Path,
    items: Path | None,
    models: Path | None,
    out_dir: Path,
    run_id: str | None,
) -> None:
    """Load a saved calibration bundle and predict full benchmark scores."""

    resolved_run_id = run_id or _default_run_id()
    try:
        prediction_result = run_prediction(
            bundle_path,
            responses,
            out_dir=out_dir,
            run_id=resolved_run_id,
            items_path=items,
            models_path=models,
        )
    except SchemaValidationError as exc:
        click.echo(_render_prediction_failure(exc, run_root=out_dir / resolved_run_id), err=True)
        raise click.exceptions.Exit(1) from exc
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_render_prediction_success(prediction_result))


@main.group("metabench")
def metabench_group() -> None:
    """Run strict metabench-validation workflows."""


@metabench_group.command("run")
@click.option(
    "--responses",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Optional local metabench-style responses_long csv/parquet. "
        "Defaults to the bundled reduced fixture."
    ),
)
@click.option(
    "--items",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional local items table to pair with --responses.",
)
@click.option(
    "--models",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional local models table to pair with --responses.",
)
@click.option(
    "--expected",
    "expected_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional expected-metrics json override.",
)
@click.option(
    "--profile",
    type=click.Choice(["reduced", "full"], case_sensitive=False),
    default=DEFAULT_PROFILE,
    show_default=True,
    help=(
        "Validation profile. Reduced is the bundled ci-sized fixture; "
        "full is for manual local snapshots."
    ),
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Explicit output directory for the metabench validation run.",
)
@click.option("--run-id", help="Optional explicit run id for stable output paths.")
def metabench_run_command(
    *,
    responses: Path | None,
    items: Path | None,
    models: Path | None,
    expected_path: Path | None,
    profile: str,
    out_dir: Path,
    run_id: str | None,
) -> None:
    """Execute the strict metabench-validation harness."""

    try:
        result = run_metabench_validation(
            out_dir=out_dir,
            responses_path=responses,
            items_path=items,
            models_path=models,
            expected_path=expected_path,
            profile=profile.lower(),
            run_id=run_id,
        )
    except SchemaValidationError as exc:
        run_root = None if run_id is None else out_dir / run_id
        click.echo(render_metabench_failure(exc, run_root=run_root), err=True)
        raise click.exceptions.Exit(1) from exc
    except Exception as exc:
        fallback_root = None if run_id is None else out_dir / run_id
        click.echo(render_metabench_failure(exc, run_root=fallback_root), err=True)
        raise click.exceptions.Exit(1) from exc

    click.echo(render_metabench_success(result))
    if not result.validation_report["passed"]:
        raise click.exceptions.Exit(1)


@dataclass(slots=True, frozen=True)
class CLISettings:
    """Resolved CLI settings from the optional config file."""

    config: dict[str, Any] | None
    stage_options: dict[str, dict[str, Any]] | None


def _load_cli_settings(config_path: Path | None) -> CLISettings:
    if config_path is None:
        return CLISettings(config=None, stage_options=None)

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix == ".toml":
        if tomllib is None:
            raise click.ClickException(
                "config .toml files require Python 3.11+ in this environment; use .json instead",
            )
        payload = tomllib.loads(text)
    else:
        raise click.ClickException("config file must use a .json or .toml extension")

    if not isinstance(payload, dict):
        raise click.ClickException("config file must parse to a mapping of BenchIQ settings")
    if "config" in payload or "stage_options" in payload:
        extra_keys = sorted(set(payload) - {"config", "stage_options"})
        if extra_keys:
            raise click.ClickException(
                "nested cli config files only support the keys `config` and `stage_options`",
            )
        resolved_config = payload.get("config")
        if resolved_config is not None and not isinstance(resolved_config, dict):
            raise click.ClickException("config file key `config` must map to BenchIQ settings")
        resolved_stage_options = payload.get("stage_options")
        if resolved_stage_options is not None and not isinstance(resolved_stage_options, dict):
            raise click.ClickException(
                "config file key `stage_options` must map to stage options",
            )
        return CLISettings(
            config=resolved_config,
            stage_options=resolved_stage_options,
        )
    return CLISettings(config=payload, stage_options=None)


def _build_validation_payload(
    *,
    bundle: Bundle,
    schema_report: ValidationReport,
    preprocess_result: Any,
) -> dict[str, Any]:
    refused_benchmarks = {
        benchmark_id: benchmark_result.preprocess_report["refusal_reasons"]
        for benchmark_id, benchmark_result in sorted(preprocess_result.benchmarks.items())
        if benchmark_result.refused
    }
    preprocess_warnings = [
        {
            "benchmark_id": benchmark_id,
            "message": warning,
        }
        for benchmark_id, benchmark_result in sorted(preprocess_result.benchmarks.items())
        for warning in benchmark_result.preprocess_report["warnings"]
    ]
    return {
        "ok": schema_report.ok and not refused_benchmarks,
        "run_id": bundle.run_id or "validate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "warning_count": schema_report.counts.warning_count + len(preprocess_warnings),
            "error_count": schema_report.counts.error_count,
            "refused_benchmark_count": len(refused_benchmarks),
        },
        "schema_validation": schema_report.to_dict(),
        "preprocess_validation": {
            "summary": preprocess_result.summary.to_dict(orient="records"),
            "refused_benchmarks": refused_benchmarks,
            "warnings": preprocess_warnings,
        },
    }


def _build_validation_failure_payload(
    *,
    message: str,
    report: ValidationReport | None,
    attempted_paths: dict[str, Path | None],
) -> dict[str, Any]:
    structured_report = (report or ValidationReport()).to_dict()
    structured_report["ok"] = False
    return {
        "ok": False,
        "run_id": "validate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "warning_count": structured_report["counts"]["warning_count"],
            "error_count": structured_report["counts"]["error_count"],
            "refused_benchmark_count": 0,
        },
        "load_error": message,
        "attempted_paths": {
            key: None if value is None else str(value.resolve())
            for key, value in sorted(attempted_paths.items())
        },
        "schema_validation": structured_report,
        "preprocess_validation": {
            "summary": [],
            "refused_benchmarks": {},
            "warnings": [],
        },
    }


def _write_validation_artifacts(
    *,
    run_root: Path,
    config: BenchIQConfig,
    payload: dict[str, Any],
    manifest_path: Path | None,
) -> None:
    reports_dir = run_root / "reports"
    config_path = write_json(config.model_dump(mode="json"), run_root / "config_resolved.json")
    report_path = write_json(payload, reports_dir / "validation_report.json")
    summary_path = reports_dir / "validation_summary.md"
    summary_path.write_text(_validation_summary_markdown(payload), encoding="utf-8")

    update_payload = {
        "run_id": payload["run_id"],
        "timestamp": payload["generated_at"],
        "resolved_config": config.model_dump(mode="json"),
        "validation": {
            "ok": payload["ok"],
            "counts": payload["counts"],
            "refused_benchmarks": payload["preprocess_validation"]["refused_benchmarks"],
        },
        "artifacts": {
            "reports": {
                "validation_report": str(report_path),
                "validation_summary": str(summary_path),
            },
            "config_resolved": str(config_path),
        },
    }
    update_manifest(manifest_path or run_root / "manifest.json", update_payload)


def _validation_summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# validation summary",
        "",
        f"- ok: {payload['ok']}",
        f"- warning count: {payload['counts']['warning_count']}",
        f"- error count: {payload['counts']['error_count']}",
        f"- refused benchmark count: {payload['counts']['refused_benchmark_count']}",
        "",
    ]
    load_error = payload.get("load_error")
    if load_error is not None:
        lines.extend(["## load error", "", f"- {load_error}", ""])
    schema_errors = payload["schema_validation"]["errors"]
    lines.extend(["## schema errors", ""])
    if not schema_errors:
        lines.append("- none")
    else:
        for error in schema_errors:
            table_name = error.get("table_name") or "bundle"
            lines.append(f"- `{table_name}` `{error['code']}`: {error['message']}")
    lines.extend(["", "## refused benchmarks", ""])
    refused = payload["preprocess_validation"]["refused_benchmarks"]
    if not refused:
        lines.append("- none")
    else:
        for benchmark_id, reasons in sorted(refused.items()):
            lines.append(f"- `{benchmark_id}`: {', '.join(reasons)}")
    lines.append("")
    return "\n".join(lines)


def _render_validation_success(*, run_root: Path, payload: dict[str, Any]) -> str:
    refused = payload["preprocess_validation"]["refused_benchmarks"]
    refused_summary = "none" if not refused else json.dumps(refused, sort_keys=True)
    return "\n".join(
        [
            "validation completed",
            f"run location: {run_root}",
            f"warnings: {payload['counts']['warning_count']}",
            f"refused benchmarks: {refused_summary}",
        ]
    )


def _render_validation_failure(*, run_root: Path, payload: dict[str, Any]) -> str:
    lines = [
        "validation failed",
        f"run location: {run_root}",
    ]
    if payload.get("load_error") is not None:
        lines.append(f"load error: {payload['load_error']}")
    for error in payload["schema_validation"]["errors"]:
        table_name = error.get("table_name") or "bundle"
        lines.append(f"error [{error['code']}] {table_name}: {error['message']}")
    for benchmark_id, reasons in sorted(
        payload["preprocess_validation"]["refused_benchmarks"].items()
    ):
        lines.append(f"refused [{benchmark_id}]: {', '.join(reasons)}")
    return "\n".join(lines)


def _render_run_success(run_result: RunResult) -> str:
    summary = run_result.summary()
    metrics = summary["metrics"]
    return "\n".join(
        [
            "run completed",
            f"run id: {summary['run_id']}",
            f"run location: {run_result.run_root}",
            f"executed stages: {', '.join(run_result.executed_stages)}",
            f"warnings: {summary['warning_count']}",
            "selected items by benchmark: "
            f"{json.dumps(metrics.get('selected_items_by_benchmark', {}), sort_keys=True)}",
            "marginal test rmse by benchmark: "
            f"{json.dumps(metrics.get('marginal_test_rmse_by_benchmark', {}), sort_keys=True)}",
            "joint test rmse by benchmark: "
            f"{json.dumps(metrics.get('joint_test_rmse_by_benchmark', {}), sort_keys=True)}",
        ]
    )


def _render_calibration_success(calibration_result: Any) -> str:
    summary = calibration_result.run_result.summary()
    metrics = summary["metrics"]
    return "\n".join(
        [
            "calibration completed",
            f"run id: {summary['run_id']}",
            f"run location: {calibration_result.run_result.run_root}",
            f"calibration bundle: {calibration_result.calibration_root}",
            f"warnings: {summary['warning_count']}",
            "selected items by benchmark: "
            f"{json.dumps(metrics.get('selected_items_by_benchmark', {}), sort_keys=True)}",
            "marginal test rmse by benchmark: "
            f"{json.dumps(metrics.get('marginal_test_rmse_by_benchmark', {}), sort_keys=True)}",
        ]
    )


def _render_prediction_success(prediction_result: Any) -> str:
    report = prediction_result.prediction_report
    benchmark_counts = json.dumps(
        report["prediction_availability"]["best_available_non_null_by_benchmark"],
        sort_keys=True,
    )
    return "\n".join(
        [
            "prediction completed",
            f"run location: {prediction_result.run_root}",
            f"calibration bundle: {prediction_result.calibration_bundle_path}",
            "best available non-null predictions: "
            f"{report['counts']['best_available_non_null_predictions']}",
            f"best available non-null by benchmark: {benchmark_counts}",
        ]
    )


def _render_run_failure(exc: SchemaValidationError, *, run_root: Path) -> str:
    lines = [
        "run failed",
        f"run location: {run_root}",
        f"error: {exc}",
    ]
    if exc.report is not None:
        for error in exc.report.errors:
            table_name = error.table_name or "bundle"
            lines.append(f"error [{error.code}] {table_name}: {error.message}")
    return "\n".join(lines)


def _render_calibration_failure(exc: SchemaValidationError, *, run_root: Path) -> str:
    lines = [
        "calibration failed",
        f"run location: {run_root}",
        f"error: {exc}",
    ]
    if exc.report is not None:
        for error in exc.report.errors:
            table_name = error.table_name or "bundle"
            lines.append(f"error [{error.code}] {table_name}: {error.message}")
    return "\n".join(lines)


def _render_prediction_failure(exc: SchemaValidationError, *, run_root: Path) -> str:
    lines = [
        "prediction failed",
        f"run location: {run_root}",
        f"error: {exc}",
    ]
    if exc.report is not None:
        for error in exc.report.errors:
            table_name = error.table_name or "bundle"
            lines.append(f"error [{error.code}] {table_name}: {error.message}")
    return "\n".join(lines)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":  # pragma: no cover
    main()
