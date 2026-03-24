"""Benchmark-local IRT fitting and stage-05 artifact writing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from benchiq.io.load import Bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.irt.backends.girth_backend import fit_girth_2pl
from benchiq.logging import update_manifest
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID
from benchiq.split.splitters import SplitResult
from benchiq.subsample.random_cv import SubsampleResult


@dataclass(slots=True)
class BenchmarkIRTResult:
    """Per-benchmark stage-05 IRT outputs."""

    benchmark_id: str
    irt_item_params: pd.DataFrame
    dropped_pathological_items: pd.DataFrame
    irt_fit_report: dict[str, Any]
    ability_estimates: pd.DataFrame
    artifact_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(slots=True)
class IRTResult:
    """Stage-05 IRT outputs."""

    benchmarks: dict[str, BenchmarkIRTResult]
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None


def fit_irt_bundle(
    bundle: Bundle,
    split_result: SplitResult,
    subsample_result: SubsampleResult,
    *,
    backend_options: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> IRTResult:
    """Fit benchmark-specific 2PL models on preselected train responses."""

    benchmark_results: dict[str, BenchmarkIRTResult] = {}
    for benchmark_id in sorted(subsample_result.benchmarks):
        benchmark_results[benchmark_id] = fit_irt_benchmark(
            bundle,
            split_result,
            subsample_result,
            benchmark_id=benchmark_id,
            backend_options=backend_options,
        )

    result = IRTResult(benchmarks=benchmark_results)
    run_root, manifest_path = _resolve_run_root(bundle, out_dir=out_dir, run_id=run_id)
    if run_root is not None:
        artifact_paths = _write_irt_artifacts(result, run_root=run_root)
        result.artifact_paths = artifact_paths
        result.manifest_path = manifest_path
        if manifest_path is not None:
            update_manifest(
                manifest_path,
                {
                    "artifacts": {
                        "05_irt": {
                            "per_benchmark": {
                                benchmark_id: {
                                    name: str(path) for name, path in sorted(paths.items())
                                }
                                for benchmark_id, paths in sorted(
                                    artifact_paths["per_benchmark"].items()
                                )
                            },
                        },
                    },
                },
            )
    return result


def fit_irt_benchmark(
    bundle: Bundle,
    split_result: SplitResult,
    subsample_result: SubsampleResult,
    *,
    benchmark_id: str,
    backend_options: dict[str, Any] | None = None,
) -> BenchmarkIRTResult:
    """Fit one benchmark-specific 2PL model using the T08 preselected items."""

    subsample_benchmark = subsample_result.benchmarks[benchmark_id]
    preselect_items = (
        subsample_benchmark.preselect_items[ITEM_ID]
        .dropna()
        .astype("string")
        .sort_values()
        .reset_index(drop=True)
        .tolist()
    )
    if subsample_benchmark.subsample_report["skipped"] or not preselect_items:
        return _skipped_benchmark_result(
            benchmark_id,
            skipped_reason=subsample_benchmark.subsample_report["skipped_reason"]
            or "no_preselect_items",
            preselect_items=preselect_items,
        )

    split_frame = split_result.per_benchmark_splits.get(benchmark_id)
    if split_frame is None or split_frame.empty:
        return _skipped_benchmark_result(
            benchmark_id,
            skipped_reason="no_split_models_available",
            preselect_items=preselect_items,
        )
    train_model_ids = (
        split_frame.loc[split_frame["split"] == "train", MODEL_ID]
        .dropna()
        .astype("string")
        .sort_values()
        .reset_index(drop=True)
        .tolist()
    )
    if not train_model_ids:
        return _skipped_benchmark_result(
            benchmark_id,
            skipped_reason="no_train_models_available",
            preselect_items=preselect_items,
        )

    fit_result = fit_girth_2pl(
        bundle.responses_long,
        benchmark_id=benchmark_id,
        item_ids=preselect_items,
        model_ids=train_model_ids,
        options=backend_options,
    )
    return BenchmarkIRTResult(
        benchmark_id=benchmark_id,
        irt_item_params=fit_result.item_params,
        dropped_pathological_items=fit_result.dropped_pathological_items,
        irt_fit_report=fit_result.fit_report,
        ability_estimates=fit_result.ability_estimates,
    )


def _skipped_benchmark_result(
    benchmark_id: str,
    *,
    skipped_reason: str,
    preselect_items: list[str],
) -> BenchmarkIRTResult:
    return BenchmarkIRTResult(
        benchmark_id=benchmark_id,
        irt_item_params=_empty_item_params_frame(),
        dropped_pathological_items=_empty_item_params_frame(),
        irt_fit_report={
            "benchmark_id": benchmark_id,
            "irt_backend": "girth",
            "model": "2pl",
            "skipped": True,
            "skipped_reason": skipped_reason,
            "warnings": [],
            "convergence": {
                "status": None,
                "backend_exposes_flag": False,
                "status_available": False,
                "warning_code": None,
            },
            "counts": {
                "preselect_item_count": len(preselect_items),
                "retained_item_count": 0,
                "pathology_warning_count": 0,
                "pathology_excluded_count": 0,
            },
            "pathology": {
                "warning_item_ids": [],
                "excluded_item_ids": [],
                "retained_item_ids": [],
                "excluded_items": [],
            },
            "artifacts": {
                "plots_written": False,
                "plots_reason": "not_implemented_in_t09",
                "dropped_pathological_items_written": False,
            },
            "fit_metrics": {
                "runtime_seconds": 0.0,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        ability_estimates=_empty_ability_frame(),
    )


def _write_irt_artifacts(
    result: IRTResult,
    *,
    run_root: Path,
) -> dict[str, Any]:
    stage_dir = run_root / "artifacts" / "05_irt"
    per_benchmark_paths: dict[str, dict[str, Path]] = {}
    for benchmark_id, benchmark_result in sorted(result.benchmarks.items()):
        benchmark_dir = stage_dir / "per_benchmark" / benchmark_id
        per_benchmark_paths[benchmark_id] = {
            "irt_item_params": write_parquet(
                benchmark_result.irt_item_params,
                benchmark_dir / "irt_item_params.parquet",
            ),
            "dropped_pathological_items": write_parquet(
                benchmark_result.dropped_pathological_items,
                benchmark_dir / "dropped_pathological_items.parquet",
            ),
            "irt_fit_report": write_json(
                benchmark_result.irt_fit_report,
                benchmark_dir / "irt_fit_report.json",
            ),
            "ability_estimates": write_parquet(
                benchmark_result.ability_estimates,
                benchmark_dir / "ability_estimates.parquet",
            ),
        }
    return {"per_benchmark": per_benchmark_paths}


def _empty_item_params_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            ITEM_ID: pd.Series(dtype="string"),
            "irt_backend": pd.Series(dtype="string"),
            "discrimination": pd.Series(dtype="Float64"),
            "difficulty": pd.Series(dtype="Float64"),
            "pathology_warning": pd.Series(dtype=bool),
            "pathology_warning_reasons": pd.Series(dtype=object),
            "pathology_excluded": pd.Series(dtype=bool),
            "pathology_excluded_reasons": pd.Series(dtype=object),
        }
    )


def _empty_ability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            "ability_eap": pd.Series(dtype="Float64"),
        }
    )


def _resolve_run_root(
    bundle: Bundle,
    *,
    out_dir: str | Path | None,
    run_id: str | None,
) -> tuple[Path | None, Path | None]:
    if out_dir is not None:
        resolved_run_id = run_id or bundle.run_id or _default_run_id()
        run_root = Path(out_dir) / resolved_run_id
        return run_root, run_root / "manifest.json"
    if bundle.manifest_path is not None:
        return bundle.manifest_path.parent, bundle.manifest_path
    return None, None


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
