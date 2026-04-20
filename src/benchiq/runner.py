"""Deterministic runner orchestration for the BenchIQ v0.1 pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from benchiq.config import BenchIQConfig
from benchiq.io.load import Bundle, load_bundle
from benchiq.io.write import write_json, write_stage0_bundle
from benchiq.irt import estimate_theta_bundle, fit_irt_bundle
from benchiq.logging import update_manifest
from benchiq.preprocess import compute_scores, preprocess_bundle
from benchiq.reconstruct import (
    build_feature_tables,
    fit_linear_predictor_bundle,
    reconstruct_scores,
)
from benchiq.redundancy import analyze_redundancy
from benchiq.select import select_bundle
from benchiq.split import split_models
from benchiq.subsample import subsample_bundle

StageName = str

STAGE_ORDER: tuple[StageName, ...] = (
    "00_bundle",
    "01_preprocess",
    "02_scores",
    "03_splits",
    "04_subsample",
    "05_irt",
    "06_select",
    "07_theta",
    "08_linear",
    "08_features",
    "09_reconstruct",
    "10_redundancy",
)

STAGE_ALIASES: dict[str, StageName] = {
    "bundle": "00_bundle",
    "preprocess": "01_preprocess",
    "scores": "02_scores",
    "splits": "03_splits",
    "subsample": "04_subsample",
    "irt": "05_irt",
    "select": "06_select",
    "theta": "07_theta",
    "linear": "08_linear",
    "features": "08_features",
    "reconstruct": "09_reconstruct",
    "redundancy": "10_redundancy",
    **{stage_name: stage_name for stage_name in STAGE_ORDER},
}

STAGE_DEPENDENCIES: dict[StageName, tuple[StageName, ...]] = {
    "00_bundle": (),
    "01_preprocess": ("00_bundle",),
    "02_scores": ("00_bundle", "01_preprocess"),
    "03_splits": ("00_bundle", "02_scores"),
    "04_subsample": ("00_bundle", "01_preprocess", "02_scores", "03_splits"),
    "05_irt": ("00_bundle", "03_splits", "04_subsample"),
    "06_select": ("00_bundle", "05_irt"),
    "07_theta": ("00_bundle", "03_splits", "06_select", "05_irt"),
    "08_linear": ("00_bundle", "02_scores", "03_splits", "06_select"),
    "08_features": ("00_bundle", "02_scores", "03_splits", "07_theta", "08_linear"),
    "09_reconstruct": ("00_bundle", "08_features"),
    "10_redundancy": ("00_bundle", "02_scores", "07_theta", "08_features", "09_reconstruct"),
}

DEFAULT_STAGE_OPTIONS: dict[str, dict[str, Any]] = {
    "04_subsample": {
        "method": "deterministic_info",
        "k_preselect": None,
        "n_iter": 2000,
        "cv_folds": 5,
        "checkpoint_interval": 25,
    },
    "05_irt": {"backend_options": None},
    "06_select": {"k_final": 10},
    "07_theta": {"theta_method": "MAP"},
    "09_reconstruct": {},
    "10_redundancy": {},
}


@dataclass(slots=True)
class RunResult:
    """Top-level runner handle with artifact access helpers."""

    run_root: Path
    manifest_path: Path
    executed_stages: tuple[StageName, ...]
    stage_results: dict[StageName, Any]
    top_level_summary: dict[str, Any]
    artifact_index: dict[str, Path]

    def load_artifact(self, name: str) -> pd.DataFrame | dict[str, Any] | str:
        """Load an artifact by flattened manifest key or suffix."""

        path = self._resolve_artifact_path(name)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return path.read_text(encoding="utf-8")

    def paths(self) -> dict[str, Path]:
        """Return the flattened artifact path index."""

        return dict(sorted(self.artifact_index.items()))

    def summary(self) -> dict[str, Any]:
        """Return the top-level run summary."""

        return self.top_level_summary

    def _resolve_artifact_path(self, name: str) -> Path:
        normalized = name.strip()
        if normalized in self.artifact_index:
            return self.artifact_index[normalized]
        candidates = [
            path
            for key, path in self.artifact_index.items()
            if key.endswith(normalized) or normalized.endswith(key)
        ]
        if not candidates:
            raise KeyError(f"unknown artifact name: {name}")
        if len(candidates) > 1:
            raise KeyError(f"artifact name is ambiguous: {name}")
        return candidates[0]


class BenchIQRunner:
    """Deterministic orchestrator for the BenchIQ v0.1 pipeline."""

    def __init__(
        self,
        *,
        config: BenchIQConfig | Mapping[str, Any] | None = None,
        out_dir: str | Path | None = None,
        run_id: str | None = None,
        stage_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = BenchIQConfig.model_validate({} if config is None else config)
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.run_id = run_id
        self.stage_options = _merge_stage_options(stage_options)
        self._stage_results: dict[StageName, Any] = {}
        self._last_source: tuple[Any, Any, Any] | None = None

    def run(
        self,
        bundle_or_responses_path: Bundle | str | Path | None = None,
        items_path: str | Path | None = None,
        models_path: str | Path | None = None,
        *,
        start_at: str | None = None,
        stop_after: str | None = None,
    ) -> RunResult:
        """Execute the deterministic stage DAG from start_at through stop_after."""

        start_stage = _resolve_stage_name(start_at or STAGE_ORDER[0])
        stop_stage = _resolve_stage_name(stop_after or STAGE_ORDER[-1])
        execution_order = _slice_stage_order(start_stage, stop_stage)

        if bundle_or_responses_path is not None:
            self._last_source = (bundle_or_responses_path, items_path, models_path)
        elif self._last_source is None:
            raise ValueError("runner requires a bundle or input paths on the first call")
        else:
            bundle_or_responses_path, items_path, models_path = self._last_source

        self._clear_downstream_cache(start_stage)
        executed_stages: list[StageName] = []
        stage_manifest_records: dict[str, Any] = {}

        if "00_bundle" in execution_order:
            started_at = datetime.now(timezone.utc)
            tic = perf_counter()
            bundle = self._prepare_bundle(
                bundle_or_responses_path=bundle_or_responses_path,
                items_path=items_path,
                models_path=models_path,
            )
            duration = perf_counter() - tic
            self._stage_results["00_bundle"] = bundle
            stage_manifest_records["00_bundle"] = self._record_stage_manifest(
                bundle,
                stage_name="00_bundle",
                stage_result=bundle,
                started_at=started_at,
                duration_seconds=duration,
            )
            executed_stages.append("00_bundle")
        else:
            bundle = self._require_stage_result("00_bundle")

        for stage_name in execution_order:
            if stage_name == "00_bundle":
                continue
            for dependency in STAGE_DEPENDENCIES[stage_name]:
                self._require_stage_result(dependency)
            started_at = datetime.now(timezone.utc)
            tic = perf_counter()
            stage_result = self._execute_stage(stage_name)
            duration = perf_counter() - tic
            self._stage_results[stage_name] = stage_result
            stage_manifest_records[stage_name] = self._record_stage_manifest(
                bundle,
                stage_name=stage_name,
                stage_result=stage_result,
                started_at=started_at,
                duration_seconds=duration,
            )
            executed_stages.append(stage_name)

        run_root, manifest_path = _resolve_runner_root(
            bundle, out_dir=self.out_dir, run_id=self.run_id
        )
        summary_payload = _build_top_level_summary(
            bundle=bundle,
            stage_results=self._stage_results,
            executed_stages=executed_stages,
            stage_manifest_records=stage_manifest_records,
        )
        report_paths = _write_runner_reports(
            run_root=run_root,
            bundle=bundle,
            summary_payload=summary_payload,
        )
        update_manifest(
            manifest_path,
            {
                "artifacts": {
                    "reports": {name: str(path) for name, path in sorted(report_paths.items())},
                    "config_resolved": str(run_root / "config_resolved.json"),
                },
                "stages": stage_manifest_records,
                "runner": {
                    "executed_stages": list(executed_stages),
                    "last_run_at": datetime.now(timezone.utc).isoformat(),
                },
            },
        )
        artifact_index = _build_artifact_index(manifest_path)
        return RunResult(
            run_root=run_root,
            manifest_path=manifest_path,
            executed_stages=tuple(executed_stages),
            stage_results={stage: self._stage_results[stage] for stage in self._stage_results},
            top_level_summary=summary_payload,
            artifact_index=artifact_index,
        )

    def _prepare_bundle(
        self,
        *,
        bundle_or_responses_path: Bundle | str | Path,
        items_path: str | Path | None,
        models_path: str | Path | None,
    ) -> Bundle:
        if isinstance(bundle_or_responses_path, Bundle):
            bundle = _bundle_with_config(bundle_or_responses_path, self.config)
            run_root, manifest_path = _resolve_runner_root(
                bundle,
                out_dir=self.out_dir,
                run_id=self.run_id,
            )
            if bundle.manifest_path is None:
                artifact_paths, manifest_path = write_stage0_bundle(
                    bundle,
                    run_root.parent,
                    run_id=run_root.name,
                )
                bundle.artifact_paths = artifact_paths
                bundle.manifest_path = manifest_path
                bundle.run_id = run_root.name
            elif bundle.manifest_path != manifest_path:
                artifact_paths, manifest_path = write_stage0_bundle(
                    bundle,
                    run_root.parent,
                    run_id=run_root.name,
                )
                bundle.artifact_paths = artifact_paths
                bundle.manifest_path = manifest_path
                bundle.run_id = run_root.name
            return bundle

        if self.out_dir is None:
            raise ValueError("runner requires out_dir when loading from input paths")

        resolved_run_id = self.run_id or _default_run_id()
        self.run_id = resolved_run_id
        return load_bundle(
            bundle_or_responses_path,
            items_path,
            models_path,
            config=self.config,
            out_dir=self.out_dir,
            run_id=resolved_run_id,
        )

    def _execute_stage(self, stage_name: StageName) -> Any:
        bundle = self._require_stage_result("00_bundle")
        if stage_name == "01_preprocess":
            return preprocess_bundle(bundle)
        if stage_name == "02_scores":
            return compute_scores(bundle, self._require_stage_result("01_preprocess"))
        if stage_name == "03_splits":
            return split_models(bundle, self._require_stage_result("02_scores"))
        if stage_name == "04_subsample":
            options = self.stage_options["04_subsample"]
            return subsample_bundle(
                bundle,
                self._require_stage_result("01_preprocess"),
                self._require_stage_result("02_scores"),
                self._require_stage_result("03_splits"),
                **options,
            )
        if stage_name == "05_irt":
            return fit_irt_bundle(
                bundle,
                self._require_stage_result("03_splits"),
                self._require_stage_result("04_subsample"),
                **self.stage_options["05_irt"],
            )
        if stage_name == "06_select":
            return select_bundle(
                bundle,
                self._require_stage_result("05_irt"),
                **self.stage_options["06_select"],
            )
        if stage_name == "07_theta":
            return estimate_theta_bundle(
                bundle,
                self._require_stage_result("03_splits"),
                self._require_stage_result("06_select"),
                self._require_stage_result("05_irt"),
                **self.stage_options["07_theta"],
            )
        if stage_name == "08_linear":
            return fit_linear_predictor_bundle(
                bundle,
                self._require_stage_result("02_scores"),
                self._require_stage_result("03_splits"),
                self._require_stage_result("06_select"),
            )
        if stage_name == "08_features":
            return build_feature_tables(
                bundle,
                self._require_stage_result("02_scores"),
                self._require_stage_result("03_splits"),
                self._require_stage_result("07_theta"),
                self._require_stage_result("08_linear"),
            )
        if stage_name == "09_reconstruct":
            return reconstruct_scores(
                bundle,
                self._require_stage_result("08_features"),
                **self.stage_options["09_reconstruct"],
            )
        if stage_name == "10_redundancy":
            return analyze_redundancy(
                bundle,
                self._require_stage_result("02_scores"),
                self._require_stage_result("07_theta"),
                self._require_stage_result("08_features"),
                self._require_stage_result("09_reconstruct"),
                **self.stage_options["10_redundancy"],
            )
        raise ValueError(f"unsupported stage: {stage_name}")

    def _record_stage_manifest(
        self,
        bundle: Bundle,
        *,
        stage_name: StageName,
        stage_result: Any,
        started_at: datetime,
        duration_seconds: float,
    ) -> dict[str, Any]:
        record = _json_safe(
            {
                "stage": stage_name,
                "dependencies": list(STAGE_DEPENDENCIES[stage_name]),
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": duration_seconds,
                "status": "completed",
                "warnings": _stage_warnings(stage_name, stage_result),
                "skip_reasons": _stage_skip_reasons(stage_name, stage_result),
                "metrics": _stage_metrics(stage_name, stage_result),
                "artifact_paths": _stringify_paths(getattr(stage_result, "artifact_paths", {})),
            }
        )
        if bundle.manifest_path is not None:
            update_manifest(bundle.manifest_path, {"stages": {stage_name: record}})
        return record

    def _require_stage_result(self, stage_name: StageName) -> Any:
        if stage_name not in self._stage_results:
            raise ValueError(
                "stage "
                f"{stage_name} is unavailable; run dependencies first or avoid start_at="
                f"{stage_name}"
            )
        return self._stage_results[stage_name]

    def _clear_downstream_cache(self, start_stage: StageName) -> None:
        start_index = STAGE_ORDER.index(start_stage)
        for stage_name in STAGE_ORDER[start_index:]:
            self._stage_results.pop(stage_name, None)


def run(
    bundle_or_responses_path: Bundle | str | Path,
    config: BenchIQConfig | Mapping[str, Any] | None = None,
    out_dir: str | Path | None = None,
    *,
    items_path: str | Path | None = None,
    models_path: str | Path | None = None,
    run_id: str | None = None,
    start_at: str | None = None,
    stop_after: str | None = None,
    stage_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> RunResult:
    """Convenience wrapper for BenchIQRunner."""

    runner = BenchIQRunner(
        config=config,
        out_dir=out_dir,
        run_id=run_id,
        stage_options=stage_options,
    )
    return runner.run(
        bundle_or_responses_path,
        items_path,
        models_path,
        start_at=start_at,
        stop_after=stop_after,
    )


def _resolve_stage_name(stage_name: str) -> StageName:
    try:
        return STAGE_ALIASES[stage_name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown stage name: {stage_name}") from exc


def _slice_stage_order(start_stage: StageName, stop_stage: StageName) -> list[StageName]:
    start_index = STAGE_ORDER.index(start_stage)
    stop_index = STAGE_ORDER.index(stop_stage)
    if stop_index < start_index:
        raise ValueError("stop_after must not precede start_at")
    return list(STAGE_ORDER[start_index : stop_index + 1])


def _merge_stage_options(
    overrides: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    merged = {stage: dict(values) for stage, values in DEFAULT_STAGE_OPTIONS.items()}
    for stage_name, values in (overrides or {}).items():
        resolved_stage = _resolve_stage_name(stage_name)
        merged.setdefault(resolved_stage, {})
        merged[resolved_stage].update(values)
    for stage_name in ("09_reconstruct", "10_redundancy"):
        merged.setdefault(stage_name, {})
    return merged


def _bundle_with_config(bundle: Bundle, config: BenchIQConfig) -> Bundle:
    cloned = Bundle(
        responses_long=bundle.responses_long,
        items=bundle.items,
        models=bundle.models,
        config=config,
        report=bundle.report,
        canonicalization_report=bundle.canonicalization_report,
        sources=bundle.sources,
        artifact_paths=dict(bundle.artifact_paths),
        manifest_path=bundle.manifest_path,
        run_id=bundle.run_id,
    )
    return cloned


def _resolve_runner_root(
    bundle: Bundle,
    *,
    out_dir: Path | None,
    run_id: str | None,
) -> tuple[Path, Path]:
    if out_dir is not None:
        resolved_run_id = run_id or bundle.run_id or _default_run_id()
        return out_dir / resolved_run_id, out_dir / resolved_run_id / "manifest.json"
    if bundle.manifest_path is None:
        raise ValueError("runner requires out_dir when bundle has no manifest_path")
    return bundle.manifest_path.parent, bundle.manifest_path


def _write_runner_reports(
    *,
    run_root: Path,
    bundle: Bundle,
    summary_payload: dict[str, Any],
) -> dict[str, Path]:
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_json(
        bundle.config.model_dump(mode="json"), run_root / "config_resolved.json"
    )
    metrics_path = write_json(_json_safe(summary_payload), reports_dir / "metrics.json")
    warnings_path = reports_dir / "warnings.md"
    warnings_path.write_text(_warnings_markdown(summary_payload["warnings"]), encoding="utf-8")
    run_summary_path = reports_dir / "run_summary.md"
    run_summary_path.write_text(_run_summary_markdown(summary_payload), encoding="utf-8")
    return {
        "config_resolved": config_path,
        "metrics": metrics_path,
        "warnings": warnings_path,
        "run_summary": run_summary_path,
    }


def _warnings_markdown(warnings: list[dict[str, Any]]) -> str:
    if not warnings:
        return "# warnings\n\nnone\n"
    lines = ["# warnings", ""]
    for warning in warnings:
        prefix = f"- `{warning['stage']}`"
        if warning.get("benchmark_id") is not None:
            prefix += f" `{warning['benchmark_id']}`"
        lines.append(f"{prefix}: {warning['message']}")
    lines.append("")
    return "\n".join(lines)


def _run_summary_markdown(summary_payload: dict[str, Any]) -> str:
    selected_items = json.dumps(
        summary_payload["metrics"].get("selected_items_by_benchmark", {}),
        sort_keys=True,
    )
    marginal_rmse = json.dumps(
        summary_payload["metrics"].get("marginal_test_rmse_by_benchmark", {}),
        sort_keys=True,
    )
    joint_rmse = json.dumps(
        summary_payload["metrics"].get("joint_test_rmse_by_benchmark", {}),
        sort_keys=True,
    )
    lines = [
        "# run summary",
        "",
        f"- run_id: `{summary_payload['run_id']}`",
        f"- executed stages: {', '.join(summary_payload['executed_stages'])}",
        f"- warning count: {summary_payload['warning_count']}",
        "",
        "## skip reasons",
        "",
    ]
    if not summary_payload["skip_reasons"]:
        lines.append("- none")
    else:
        for stage_name, reasons in sorted(summary_payload["skip_reasons"].items()):
            lines.append(f"- `{stage_name}`: {json.dumps(reasons, sort_keys=True)}")
    lines.extend(
        [
            "",
            "## headline metrics",
            "",
            f"- selected items by benchmark: {selected_items}",
            f"- marginal test rmse by benchmark: {marginal_rmse}",
            f"- joint test rmse by benchmark: {joint_rmse}",
            "",
        ]
    )
    return "\n".join(lines)


def _build_top_level_summary(
    *,
    bundle: Bundle,
    stage_results: Mapping[StageName, Any],
    executed_stages: list[StageName],
    stage_manifest_records: Mapping[StageName, Any],
) -> dict[str, Any]:
    warnings = _json_safe(_aggregate_warnings(stage_results))
    skip_reasons = _json_safe(_aggregate_skip_reasons(stage_results))
    metrics = _json_safe(_aggregate_metrics(stage_results))
    return _json_safe(
        {
            "run_id": bundle.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "executed_stages": executed_stages,
            "stage_records": dict(stage_manifest_records),
            "warning_count": len(warnings),
            "warnings": warnings,
            "skip_reasons": skip_reasons,
            "metrics": metrics,
        }
    )


def _aggregate_warnings(stage_results: Mapping[StageName, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for stage_name, stage_result in stage_results.items():
        stage_warnings = _stage_warnings(stage_name, stage_result)
        warnings.extend(stage_warnings)
    return warnings


def _aggregate_skip_reasons(stage_results: Mapping[StageName, Any]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for stage_name, stage_result in stage_results.items():
        reasons = _stage_skip_reasons(stage_name, stage_result)
        if reasons:
            aggregated[stage_name] = reasons
    return aggregated


def _aggregate_metrics(stage_results: Mapping[StageName, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    select_result = stage_results.get("06_select")
    if select_result is not None:
        metrics["selected_items_by_benchmark"] = {
            benchmark_id: int(len(benchmark_result.subset_final.index))
            for benchmark_id, benchmark_result in sorted(select_result.benchmarks.items())
        }
    reconstruction_result = stage_results.get("09_reconstruct")
    if reconstruction_result is not None:
        metrics["marginal_test_rmse_by_benchmark"] = reconstruction_result.reconstruction_report[
            "rmse"
        ]["marginal_test_by_benchmark"]
        metrics["joint_test_rmse_by_benchmark"] = reconstruction_result.reconstruction_report[
            "rmse"
        ]["joint_test_by_benchmark"]
        metrics["joint_reconstruction_skips"] = reconstruction_result.reconstruction_report[
            "joint_skips"
        ]
    redundancy_result = stage_results.get("10_redundancy")
    if redundancy_result is not None:
        metrics["theta_correlation_summary"] = redundancy_result.redundancy_report[
            "theta_correlation_summary"
        ]
        metrics["score_correlation_summary"] = redundancy_result.redundancy_report[
            "score_correlation_summary"
        ]
        metrics["factor_analysis"] = redundancy_result.redundancy_report["factor_analysis"]
        metrics["compressibility_by_benchmark"] = {
            row["benchmark_id"]: {
                "redundancy_ratio": row["redundancy_ratio"],
                "redundancy_rmse_gain": row["redundancy_rmse_gain"],
                "cross_only_test_rmse": row["cross_only_test_rmse"],
                "marginal_test_rmse": row["marginal_test_rmse"],
                "skipped": row["skipped"],
                "skip_reason": row["skip_reason"],
            }
            for row in redundancy_result.compressibility.to_dict(orient="records")
        }
    split_result = stage_results.get("03_splits")
    if split_result is not None:
        metrics["global_test_enabled"] = bool(split_result.split_report["global_test"]["enabled"])
    return metrics


def _stage_warnings(stage_name: StageName, stage_result: Any) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if stage_name == "00_bundle":
        for issue in stage_result.report.warnings:
            warnings.append(
                {
                    "stage": stage_name,
                    "benchmark_id": None,
                    "message": issue.message,
                }
            )
        return warnings
    if stage_name == "01_preprocess":
        for benchmark_id, benchmark_result in sorted(stage_result.benchmarks.items()):
            for warning in benchmark_result.preprocess_report["warnings"]:
                warnings.append(
                    {
                        "stage": stage_name,
                        "benchmark_id": benchmark_id,
                        "message": str(warning),
                    }
                )
        return warnings
    if stage_name == "02_scores":
        return [
            {"stage": stage_name, "benchmark_id": None, "message": str(warning)}
            for warning in stage_result.score_report["warnings"]
        ]
    if stage_name == "03_splits":
        return [
            {"stage": stage_name, "benchmark_id": None, "message": str(warning)}
            for warning in stage_result.split_report["warnings"]
        ]
    if stage_name == "04_subsample":
        for benchmark_id, benchmark_result in sorted(stage_result.benchmarks.items()):
            for warning in benchmark_result.subsample_report["warnings"]:
                warnings.append(
                    {
                        "stage": stage_name,
                        "benchmark_id": benchmark_id,
                        "message": str(warning),
                    }
                )
        return warnings
    if stage_name == "05_irt":
        for benchmark_id, benchmark_result in sorted(stage_result.benchmarks.items()):
            for warning in benchmark_result.irt_fit_report["warnings"]:
                warnings.append(
                    {
                        "stage": stage_name,
                        "benchmark_id": benchmark_id,
                        "message": warning.get("message", str(warning)),
                    }
                )
        return warnings
    if stage_name == "06_select":
        for benchmark_id, benchmark_result in sorted(stage_result.benchmarks.items()):
            for warning in benchmark_result.selection_report["warnings"]:
                warnings.append(
                    {
                        "stage": stage_name,
                        "benchmark_id": benchmark_id,
                        "message": warning.get("message", str(warning)),
                    }
                )
        return warnings
    if stage_name == "07_theta":
        for warning in stage_result.theta_report["warnings"]:
            warnings.append(
                {
                    "stage": stage_name,
                    "benchmark_id": warning.get("benchmark_id"),
                    "message": warning.get("message", str(warning)),
                }
            )
        return warnings
    if stage_name == "08_linear":
        for warning in stage_result.feature_report["warnings"]:
            warnings.append(
                {
                    "stage": stage_name,
                    "benchmark_id": warning.get("benchmark_id"),
                    "message": warning.get("message", str(warning)),
                }
            )
        return warnings
    if stage_name == "08_features":
        for warning in stage_result.feature_report["warnings"]:
            warnings.append(
                {
                    "stage": stage_name,
                    "benchmark_id": warning.get("benchmark_id"),
                    "message": warning.get("message", str(warning)),
                }
            )
        return warnings
    if stage_name == "09_reconstruct":
        for warning in stage_result.reconstruction_report["warnings"]:
            warnings.append(
                {
                    "stage": stage_name,
                    "benchmark_id": warning.get("benchmark_id"),
                    "message": warning.get("message", str(warning)),
                }
            )
        return warnings
    if stage_name == "10_redundancy":
        for warning in stage_result.redundancy_report["compressibility"]["warnings"]:
            warnings.append(
                {
                    "stage": stage_name,
                    "benchmark_id": warning.get("benchmark_id"),
                    "message": warning.get("message", str(warning)),
                }
            )
        return warnings
    return warnings


def _stage_skip_reasons(stage_name: StageName, stage_result: Any) -> dict[str, Any] | None:
    if stage_name == "01_preprocess":
        reasons = {
            benchmark_id: benchmark_result.preprocess_report["refusal_reasons"]
            for benchmark_id, benchmark_result in sorted(stage_result.benchmarks.items())
            if benchmark_result.refused
        }
        return reasons or None
    if stage_name == "02_scores":
        grand_skip = stage_result.score_report["grand_scores"]["skip_reason"]
        return None if grand_skip is None else {"grand_scores": grand_skip}
    if stage_name == "04_subsample":
        reasons = {
            benchmark_id: benchmark_result.subsample_report["skipped_reason"]
            for benchmark_id, benchmark_result in sorted(stage_result.benchmarks.items())
            if benchmark_result.subsample_report["skipped"]
        }
        return reasons or None
    if stage_name == "05_irt":
        reasons = {
            benchmark_id: benchmark_result.irt_fit_report["skipped_reason"]
            for benchmark_id, benchmark_result in sorted(stage_result.benchmarks.items())
            if benchmark_result.irt_fit_report["skipped"]
        }
        return reasons or None
    if stage_name == "06_select":
        reasons = {
            benchmark_id: benchmark_result.selection_report["skipped_reason"]
            for benchmark_id, benchmark_result in sorted(stage_result.benchmarks.items())
            if benchmark_result.selection_report["skipped"]
        }
        return reasons or None
    if stage_name == "08_features":
        joint_skip = stage_result.feature_report["joint"]["skip_reason"]
        return None if joint_skip is None else {"joint_features": joint_skip}
    if stage_name == "09_reconstruct":
        return stage_result.reconstruction_report["joint_skips"] or None
    if stage_name == "10_redundancy":
        reasons = stage_result.redundancy_report["compressibility"]["skipped_benchmarks"]
        factor_skip = stage_result.redundancy_report["factor_analysis"]["skip_reason"]
        if factor_skip is not None:
            reasons = dict(reasons)
            reasons["factor_analysis"] = factor_skip
        return reasons or None
    return None


def _stage_metrics(stage_name: StageName, stage_result: Any) -> dict[str, Any]:
    if stage_name == "00_bundle":
        return {
            "responses_row_count": int(len(stage_result.responses_long.index)),
            "item_count": int(len(stage_result.items.index)),
            "model_count": int(len(stage_result.models.index)),
        }
    if stage_name == "01_preprocess":
        return {
            "summary_rows": int(len(stage_result.summary.index)),
            "retained_items_by_benchmark": {
                row["benchmark_id"]: row["retained_items"]
                for row in stage_result.summary.to_dict(orient="records")
            },
        }
    if stage_name == "02_scores":
        return {
            "scores_full_rows": int(len(stage_result.scores_full.index)),
            "grand_score_rows": int(len(stage_result.scores_grand.index)),
        }
    if stage_name == "03_splits":
        return dict(stage_result.split_report["counts"])
    if stage_name == "04_subsample":
        return {
            benchmark_id: {
                "method": benchmark_result.subsample_report["method"],
                "best_iteration": benchmark_result.subsample_report["best_iteration"],
            }
            for benchmark_id, benchmark_result in sorted(stage_result.benchmarks.items())
        }
    if stage_name == "05_irt":
        return {
            benchmark_id: benchmark_result.irt_fit_report["counts"]
            for benchmark_id, benchmark_result in sorted(stage_result.benchmarks.items())
        }
    if stage_name == "06_select":
        return {
            benchmark_id: benchmark_result.selection_report["counts"]
            for benchmark_id, benchmark_result in sorted(stage_result.benchmarks.items())
        }
    if stage_name == "07_theta":
        return dict(stage_result.theta_report["counts"])
    if stage_name == "08_linear":
        return dict(stage_result.feature_report["counts"])
    if stage_name == "08_features":
        return dict(stage_result.feature_report["counts"])
    if stage_name == "09_reconstruct":
        return {
            "marginal_test_rmse_by_benchmark": stage_result.reconstruction_report["rmse"][
                "marginal_test_by_benchmark"
            ],
            "joint_test_rmse_by_benchmark": stage_result.reconstruction_report["rmse"][
                "joint_test_by_benchmark"
            ],
        }
    if stage_name == "10_redundancy":
        return {
            "theta_correlation_summary": stage_result.redundancy_report[
                "theta_correlation_summary"
            ],
            "score_correlation_summary": stage_result.redundancy_report[
                "score_correlation_summary"
            ],
            "factor_analysis": stage_result.redundancy_report["factor_analysis"],
        }
    return {}


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _build_artifact_index(manifest_path: Path) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index: dict[str, Path] = {
        "manifest": manifest_path,
    }

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{prefix}/{key}" if prefix else str(key)
                visit(child, item)
            return
        if isinstance(value, str):
            path = Path(value)
            if path.suffix:
                index[prefix] = path

    visit("artifacts", manifest.get("artifacts", {}))
    return index


def _json_safe(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
