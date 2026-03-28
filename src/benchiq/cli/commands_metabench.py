"""Metabench validation-mode helpers for the BenchIQ CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from benchiq.config import BenchIQConfig
from benchiq.io.write import write_json
from benchiq.logging import update_manifest
from benchiq.reconstruct.features import (
    GRAND_LIN,
    GRAND_SUB,
    MARGINAL_THETA,
    MARGINAL_THETA_SE,
)
from benchiq.reconstruct.linear_predictor import LINEAR_PREDICTION, REDUCED_SUBSCORE
from benchiq.runner import RunResult
from benchiq.runner import run as run_pipeline
from benchiq.schema.checks import SchemaValidationError
from benchiq.schema.tables import BENCHMARK_ID, SPLIT

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE_PATH = REPO_ROOT / "tests" / "data" / "metabench_validation" / "responses_long.csv"
DEFAULT_EXPECTED_PATH = REPO_ROOT / "tests" / "regression" / "expected" / "metabench_metrics.json"
DEFAULT_RUN_ID = "metabench-validation"
DEFAULT_PROFILE = "reduced"


@dataclass(slots=True, frozen=True)
class MetabenchPreset:
    """Strict metabench-validation preset metadata."""

    profile: str
    config: BenchIQConfig
    stage_options: dict[str, dict[str, Any]]
    notes: list[str]


@dataclass(slots=True)
class MetabenchValidationResult:
    """Structured result for ``benchiq metabench run``."""

    run_result: RunResult
    validation_report: dict[str, Any]
    report_path: Path
    summary_path: Path
    expected_path: Path | None


def build_metabench_preset(profile: str = DEFAULT_PROFILE) -> MetabenchPreset:
    """Return the strict validation preset for the selected profile."""

    if profile == "reduced":
        return MetabenchPreset(
            profile=profile,
            config=BenchIQConfig(
                allow_low_n=True,
                drop_low_tail_models_quantile=0.001,
                min_item_sd=0.01,
                max_item_mean=0.95,
                min_abs_point_biserial=0.05,
                min_models_per_benchmark=15,
                warn_models_per_benchmark=20,
                min_items_after_filtering=5,
                min_models_per_item=10,
                min_item_coverage=0.8,
                min_overlap_models_for_joint=15,
                min_overlap_models_for_redundancy=15,
                p_test=0.10,
                p_val=0.10,
                n_strata_bins=10,
                random_seed=7,
            ),
            stage_options={
                "04_subsample": {
                    "k_preselect": 4,
                    "n_iter": 8,
                    "cv_folds": 5,
                    "checkpoint_interval": 2,
                    "lam_grid": (0.1, 1.0),
                },
                "05_irt": {"backend_options": None},
                "06_select": {
                    "k_final": 3,
                    "n_bins": 3,
                    "theta_grid_size": 101,
                },
                "07_theta": {"theta_method": "MAP", "theta_grid_size": 81},
                "09_reconstruct": {
                    "lam_grid": (0.1, 1.0),
                    "cv_folds": 5,
                    "n_splines": 5,
                },
                "10_redundancy": {
                    "lam_grid": (0.1, 1.0),
                    "cv_folds": 5,
                    "n_splines": 5,
                    "n_factors_to_try": (1, 2),
                },
            },
            notes=[
                "reduced fixture profile uses metabench-style thresholds "
                "with ci-sized sample limits",
                "reduced fixture profile keeps p_test=0.10 and 5-fold subsampling structure",
                "full metabench snapshot is not bundled in the repo; "
                "see docs/design/metabench_validation.md",
            ],
        )
    if profile == "full":
        return MetabenchPreset(
            profile=profile,
            config=BenchIQConfig(
                allow_low_n=False,
                drop_low_tail_models_quantile=0.001,
                min_item_sd=0.01,
                max_item_mean=0.95,
                min_abs_point_biserial=0.05,
                min_models_per_benchmark=100,
                warn_models_per_benchmark=200,
                min_items_after_filtering=50,
                min_models_per_item=50,
                min_item_coverage=0.8,
                min_overlap_models_for_joint=75,
                min_overlap_models_for_redundancy=75,
                p_test=0.10,
                p_val=0.10,
                n_strata_bins=10,
                random_seed=7,
            ),
            stage_options={
                "04_subsample": {
                    "k_preselect": 350,
                    "n_iter": 10000,
                    "cv_folds": 5,
                    "checkpoint_interval": 50,
                    "lam_grid": (0.1, 1.0),
                },
                "05_irt": {"backend_options": None},
                "06_select": {
                    "k_final": 250,
                    "n_bins": 250,
                    "theta_grid_size": 251,
                },
                "07_theta": {"theta_method": "MAP", "theta_grid_size": 251},
                "09_reconstruct": {
                    "lam_grid": (0.1, 1.0),
                    "cv_folds": 5,
                    "n_splines": 10,
                },
                "10_redundancy": {
                    "lam_grid": (0.1, 1.0),
                    "cv_folds": 5,
                    "n_splines": 10,
                    "n_factors_to_try": (1, 2, 3),
                },
            },
            notes=[
                "full profile is intended for a local metabench snapshot "
                "and is not exercised in ci",
                "selection defaults approximate metabench quantile coverage via n_bins=250",
                "BenchIQ remains generic arbitrary-benchmark software; "
                "metabench mode is a validation harness",
            ],
        )
    raise ValueError(f"unsupported metabench profile: {profile}")


def run_metabench_validation(
    *,
    out_dir: str | Path,
    responses_path: str | Path | None = None,
    items_path: str | Path | None = None,
    models_path: str | Path | None = None,
    expected_path: str | Path | None = None,
    profile: str = DEFAULT_PROFILE,
    run_id: str | None = None,
) -> MetabenchValidationResult:
    """Run the pipeline in metabench validation mode and write a validation report."""

    preset = build_metabench_preset(profile)
    resolved_responses = (
        Path(responses_path) if responses_path is not None else DEFAULT_FIXTURE_PATH
    )
    resolved_items = None if items_path is None else Path(items_path)
    resolved_models = None if models_path is None else Path(models_path)
    resolved_expected = _resolve_expected_path(
        expected_path=expected_path,
        using_default_fixture=responses_path is None,
        profile=profile,
    )
    if not resolved_responses.exists():
        raise FileNotFoundError(
            f"metabench validation responses file not found: {resolved_responses}"
        )

    resolved_run_id = run_id or DEFAULT_RUN_ID
    run_result = run_pipeline(
        resolved_responses,
        preset.config,
        out_dir=out_dir,
        items_path=resolved_items,
        models_path=resolved_models,
        run_id=resolved_run_id,
        stage_options=preset.stage_options,
    )
    report_payload = _build_validation_report(
        run_result=run_result,
        preset=preset,
        responses_path=resolved_responses,
        items_path=resolved_items,
        models_path=resolved_models,
        expected_path=resolved_expected,
    )
    report_path, summary_path = _write_validation_report(
        run_result=run_result,
        report_payload=report_payload,
    )
    return MetabenchValidationResult(
        run_result=run_result,
        validation_report=report_payload,
        report_path=report_path,
        summary_path=summary_path,
        expected_path=resolved_expected,
    )


def render_metabench_success(result: MetabenchValidationResult) -> str:
    """Render a concise terminal summary for a successful validation run."""

    report = result.validation_report
    evaluation = report["evaluation"]
    return "\n".join(
        [
            "metabench validation completed",
            f"profile: {report['mode']['profile']}",
            f"fixture: {report['fixture']['name']}",
            f"run location: {result.run_result.run_root}",
            f"validation report: {result.report_path}",
            f"artifacts passed: {report['artifact_checks']['passed']}",
            f"metrics passed: {evaluation['passed']}",
            f"checked metrics: {evaluation['checked_metric_count']}",
            f"warnings: {result.run_result.summary()['warning_count']}",
        ]
    )


def render_metabench_failure(exc: Exception, *, run_root: Path | None = None) -> str:
    """Render a concise terminal summary for a failed validation run."""

    lines = ["metabench validation failed"]
    if run_root is not None:
        lines.append(f"run location: {run_root}")
    lines.append(f"error: {exc}")
    if isinstance(exc, SchemaValidationError) and exc.report is not None:
        for error in exc.report.errors:
            table_name = error.table_name or "bundle"
            lines.append(f"error [{error.code}] {table_name}: {error.message}")
    return "\n".join(lines)


def _resolve_expected_path(
    *,
    expected_path: str | Path | None,
    using_default_fixture: bool,
    profile: str,
) -> Path | None:
    if expected_path is not None:
        return Path(expected_path)
    if using_default_fixture and profile == "reduced":
        return DEFAULT_EXPECTED_PATH
    return None


def _build_validation_report(
    *,
    run_result: RunResult,
    preset: MetabenchPreset,
    responses_path: Path,
    items_path: Path | None,
    models_path: Path | None,
    expected_path: Path | None,
) -> dict[str, Any]:
    actual_metrics = _collect_actual_metrics(run_result)
    required_artifacts = _load_required_artifacts(expected_path)
    artifact_checks = _evaluate_artifacts(
        run_result=run_result, required_artifacts=required_artifacts
    )
    expected_payload = _load_expected_payload(expected_path)
    evaluation = _evaluate_metrics(
        actual_metrics=actual_metrics,
        expected_payload=expected_payload,
    )
    passed = artifact_checks["passed"] and evaluation["passed"]
    return {
        "mode": {
            "name": "metabench_validation",
            "profile": preset.profile,
            "strict_mode": True,
            "generic_mode_note": (
                "generic arbitrary-benchmark mode remains `benchiq run`; metabench mode is "
                "only the methodological validation harness"
            ),
        },
        "fixture": {
            "name": (
                "bundled_reduced_fixture"
                if responses_path == DEFAULT_FIXTURE_PATH
                else "user_supplied"
            ),
            "responses_path": str(responses_path),
            "items_path": None if items_path is None else str(items_path),
            "models_path": None if models_path is None else str(models_path),
        },
        "preset": {
            "config": preset.config.model_dump(mode="json"),
            "stage_options": _json_safe(preset.stage_options),
            "notes": preset.notes,
        },
        "run": {
            "run_id": run_result.summary()["run_id"],
            "run_root": str(run_result.run_root),
            "executed_stages": list(run_result.executed_stages),
        },
        "artifact_checks": artifact_checks,
        "metrics": actual_metrics,
        "expected": {
            "path": None if expected_path is None else str(expected_path),
            "available": expected_payload is not None,
        },
        "evaluation": evaluation,
        "warnings": run_result.summary()["warnings"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
    }


def _collect_actual_metrics(run_result: RunResult) -> dict[str, Any]:
    preprocess_result = run_result.stage_results["01_preprocess"]
    split_result = run_result.stage_results["03_splits"]
    subsample_result = run_result.stage_results["04_subsample"]
    feature_result = run_result.stage_results["08_features"]
    reconstruction_result = run_result.stage_results["09_reconstruct"]

    preprocess_rows = preprocess_result.summary.to_dict(orient="records")
    preprocess_retained = {
        row["benchmark_id"]: int(row["retained_items"]) for row in preprocess_rows
    }
    split_counts_by_benchmark: dict[str, dict[str, int]] = {}
    for benchmark_id, split_frame in sorted(split_result.per_benchmark_splits.items()):
        split_counts_by_benchmark[benchmark_id] = {
            split_name: int((split_frame[SPLIT] == split_name).sum())
            for split_name in ("train", "val", "test")
        }
    preselect_count_by_benchmark = {
        benchmark_id: int(len(benchmark_result.preselect_items.index))
        for benchmark_id, benchmark_result in sorted(subsample_result.benchmarks.items())
    }
    subsample_cv_rows_by_benchmark = {
        benchmark_id: int(len(benchmark_result.cv_results.index))
        for benchmark_id, benchmark_result in sorted(subsample_result.benchmarks.items())
    }
    reconstruction_summary = reconstruction_result.reconstruction_summary.copy()
    marginal_test_rows = reconstruction_summary.loc[
        (reconstruction_summary["model_type"] == "marginal")
        & (reconstruction_summary["split"] == "test")
    ]
    joint_test_rows = reconstruction_summary.loc[
        (reconstruction_summary["model_type"] == "joint")
        & (reconstruction_summary["split"] == "test")
    ]
    return {
        "preprocess_retained_items_by_benchmark": preprocess_retained,
        "global_test_enabled": bool(split_result.split_report["global_test"]["enabled"]),
        "global_test_model_count": int(
            split_result.split_report["counts"]["global_test_model_count"]
        ),
        "split_counts_by_benchmark": split_counts_by_benchmark,
        "subsample_preselect_count_by_benchmark": preselect_count_by_benchmark,
        "subsample_cv_rows_by_benchmark": subsample_cv_rows_by_benchmark,
        "feature_columns": {
            "marginal_has_theta": MARGINAL_THETA in feature_result.features_marginal.columns,
            "marginal_has_theta_se": (
                MARGINAL_THETA_SE in feature_result.features_marginal.columns
            ),
            "marginal_has_reduced_subscore": (
                REDUCED_SUBSCORE in feature_result.features_marginal.columns
            ),
            "marginal_has_linear_prediction": LINEAR_PREDICTION
            in feature_result.features_marginal.columns,
            "joint_has_grand_sub": GRAND_SUB in feature_result.features_joint.columns,
            "joint_has_grand_lin": GRAND_LIN in feature_result.features_joint.columns,
        },
        "reconstruction_marginal_test_rmse_by_benchmark": {
            row[BENCHMARK_ID]: float(row["rmse"])
            for row in marginal_test_rows.to_dict(orient="records")
        },
        "reconstruction_joint_test_rmse_by_benchmark": {
            row[BENCHMARK_ID]: float(row["rmse"])
            for row in joint_test_rows.to_dict(orient="records")
        },
    }


def _load_required_artifacts(expected_path: Path | None) -> list[str]:
    payload = _load_expected_payload(expected_path)
    if payload is None:
        return []
    return [str(path) for path in payload.get("required_artifacts", [])]


def _evaluate_artifacts(
    *,
    run_result: RunResult,
    required_artifacts: list[str],
) -> dict[str, Any]:
    artifact_paths = {path.lstrip("/"): run_result.run_root / path for path in required_artifacts}
    missing = [path for path, resolved in artifact_paths.items() if not resolved.exists()]
    return {
        "required_artifacts": required_artifacts,
        "missing_artifacts": missing,
        "passed": not missing,
    }


def _load_expected_payload(expected_path: Path | None) -> dict[str, Any] | None:
    if expected_path is None or not expected_path.exists():
        return None
    return json.loads(expected_path.read_text(encoding="utf-8"))


def _evaluate_metrics(
    *,
    actual_metrics: dict[str, Any],
    expected_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if expected_payload is None:
        return {
            "passed": True,
            "checked_metric_count": 0,
            "checks": [],
            "failures": [],
            "note": "no expected metrics file was available; artifact checks only",
        }

    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_metrics = expected_payload.get("metric_tolerances", {})
    _compare_metric_tree(
        metric_path=(),
        actual=actual_metrics,
        expected=expected_metrics,
        checks=checks,
        failures=failures,
    )
    return {
        "passed": not failures,
        "checked_metric_count": len(checks),
        "checks": checks,
        "failures": failures,
        "note": expected_payload.get("note"),
    }


def _compare_metric_tree(
    *,
    metric_path: tuple[str, ...],
    actual: Any,
    expected: Any,
    checks: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    if isinstance(expected, Mapping) and "target" in expected:
        target = expected["target"]
        abs_tolerance = expected.get("abs_tolerance")
        passed = _compare_leaf(actual=actual, target=target, abs_tolerance=abs_tolerance)
        check = {
            "metric": ".".join(metric_path),
            "actual": _json_safe(actual),
            "target": _json_safe(target),
            "abs_tolerance": _json_safe(abs_tolerance),
            "passed": passed,
        }
        checks.append(check)
        if not passed:
            failures.append(check)
        return

    if not isinstance(expected, Mapping):
        raise ValueError(f"expected metric tree leaf must be a mapping with target: {metric_path}")

    for key, expected_value in expected.items():
        actual_value = None
        if isinstance(actual, Mapping):
            actual_value = actual.get(key)
        _compare_metric_tree(
            metric_path=(*metric_path, str(key)),
            actual=actual_value,
            expected=expected_value,
            checks=checks,
            failures=failures,
        )


def _compare_leaf(*, actual: Any, target: Any, abs_tolerance: Any) -> bool:
    if abs_tolerance is None:
        return actual == target
    if actual is None:
        return False
    return abs(float(actual) - float(target)) <= float(abs_tolerance)


def _write_validation_report(
    *,
    run_result: RunResult,
    report_payload: dict[str, Any],
) -> tuple[Path, Path]:
    reports_dir = run_result.run_root / "reports"
    report_path = write_json(report_payload, reports_dir / "metabench_validation_report.json")
    summary_path = reports_dir / "metabench_validation_summary.md"
    summary_path.write_text(_summary_markdown(report_payload), encoding="utf-8")
    refreshed_artifacts = _evaluate_artifacts(
        run_result=run_result,
        required_artifacts=report_payload["artifact_checks"]["required_artifacts"],
    )
    report_payload["artifact_checks"] = refreshed_artifacts
    report_payload["passed"] = (
        refreshed_artifacts["passed"] and report_payload["evaluation"]["passed"]
    )
    report_path = write_json(report_payload, reports_dir / "metabench_validation_report.json")
    summary_path.write_text(_summary_markdown(report_payload), encoding="utf-8")
    update_manifest(
        run_result.manifest_path,
        {
            "artifacts": {
                "reports": {
                    "metabench_validation_report": str(report_path),
                    "metabench_validation_summary": str(summary_path),
                }
            },
            "metabench_validation": {
                "passed": report_payload["passed"],
                "profile": report_payload["mode"]["profile"],
                "expected_path": report_payload["expected"]["path"],
            },
        },
    )
    return report_path, summary_path


def _summary_markdown(report_payload: dict[str, Any]) -> str:
    lines = [
        "# metabench validation summary",
        "",
        f"- passed: {report_payload['passed']}",
        f"- profile: `{report_payload['mode']['profile']}`",
        f"- fixture: `{report_payload['fixture']['name']}`",
        f"- run root: `{report_payload['run']['run_root']}`",
        f"- checked metrics: {report_payload['evaluation']['checked_metric_count']}",
        "",
        "## failures",
        "",
    ]
    failures = report_payload["evaluation"]["failures"]
    if not failures:
        lines.append("- none")
    else:
        for failure in failures:
            lines.append(
                f"- `{failure['metric']}` actual={failure['actual']} "
                f"target={failure['target']} tol={failure['abs_tolerance']}"
            )
    lines.extend(["", "## notes", ""])
    for note in report_payload["preset"]["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
