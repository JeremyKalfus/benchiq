"""Bundle loading and stage-00 canonicalization for BenchIQ."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd

from benchiq.config import BenchIQConfig
from benchiq.io.write import write_stage0_bundle
from benchiq.logging import hash_file_sha256
from benchiq.schema.checks import (
    SchemaValidationError,
    ValidationReport,
    coerce_items_table,
    coerce_models_table,
    coerce_responses_long,
)
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID, TableName


@dataclass(slots=True, frozen=True)
class BundleSource:
    """Source metadata for a canonical bundle table."""

    table_name: TableName
    path: str | None
    file_format: Literal["csv", "parquet", "derived"]
    sha256: str | None
    derived: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_name": self.table_name,
            "path": self.path,
            "file_format": self.file_format,
            "sha256": self.sha256,
            "derived": self.derived,
        }


@dataclass(slots=True)
class Bundle:
    """Canonical in-memory bundle for stage-00 outputs."""

    responses_long: pd.DataFrame
    items: pd.DataFrame
    models: pd.DataFrame
    config: BenchIQConfig
    report: ValidationReport
    canonicalization_report: dict[str, Any]
    sources: dict[str, BundleSource]
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    manifest_path: Path | None = None
    run_id: str | None = None

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "responses_long": self.responses_long,
            "items": self.items,
            "models": self.models,
        }


def load_bundle(
    responses_path: str | Path,
    items_path: str | Path | None = None,
    models_path: str | Path | None = None,
    *,
    config: BenchIQConfig | Mapping[str, Any] | None = None,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> Bundle:
    """Load CSV/parquet tables, canonicalize them, and optionally write stage-00 artifacts."""

    if run_id is not None and out_dir is None:
        raise ValueError("run_id requires out_dir so stage-00 artifacts have a destination")

    resolved_config = BenchIQConfig.model_validate({} if config is None else config)
    sources: dict[str, BundleSource] = {}

    raw_responses, sources["responses_long"] = _read_table(
        responses_path, table_name="responses_long"
    )
    responses_long, report = coerce_responses_long(
        raw_responses,
        duplicate_policy=resolved_config.duplicate_policy,
    )
    if responses_long is None:
        raise SchemaValidationError("responses_long failed schema validation", report=report)

    items, items_source = _load_or_derive_items(items_path, responses_long)
    sources["items"] = items_source
    items, items_report = coerce_items_table(items)

    models, models_source = _load_or_derive_models(models_path, responses_long)
    sources["models"] = models_source
    models, models_report = coerce_models_table(models)

    report.extend(items_report)
    report.extend(models_report)
    if items is None or models is None:
        raise SchemaValidationError("bundle failed schema validation", report=report)

    bundle = Bundle(
        responses_long=responses_long,
        items=items,
        models=models,
        config=resolved_config,
        report=report,
        canonicalization_report=_build_canonicalization_report(report, sources, resolved_config),
        sources=sources,
    )

    if out_dir is not None:
        resolved_run_id = run_id or _default_run_id()
        artifact_paths, manifest_path = write_stage0_bundle(bundle, out_dir, run_id=resolved_run_id)
        bundle.artifact_paths = artifact_paths
        bundle.manifest_path = manifest_path
        bundle.run_id = resolved_run_id

    return bundle


def _read_table(path: str | Path, *, table_name: TableName) -> tuple[pd.DataFrame, BundleSource]:
    resolved_path = Path(path)
    suffix = resolved_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(resolved_path)
        file_format: Literal["csv", "parquet", "derived"] = "csv"
    elif suffix == ".parquet":
        frame = pd.read_parquet(resolved_path)
        file_format = "parquet"
    else:
        raise ValueError(f"{table_name} must be loaded from a .csv or .parquet file")

    return frame, BundleSource(
        table_name=table_name,
        path=str(resolved_path.resolve()),
        file_format=file_format,
        sha256=hash_file_sha256(resolved_path),
    )


def _load_or_derive_items(
    items_path: str | Path | None,
    responses_long: pd.DataFrame,
) -> tuple[pd.DataFrame, BundleSource]:
    if items_path is not None:
        return _read_table(items_path, table_name="items")

    items = responses_long[[BENCHMARK_ID, ITEM_ID]].drop_duplicates().reset_index(drop=True)
    return items, BundleSource(
        table_name="items",
        path=None,
        file_format="derived",
        sha256=None,
        derived=True,
    )


def _load_or_derive_models(
    models_path: str | Path | None,
    responses_long: pd.DataFrame,
) -> tuple[pd.DataFrame, BundleSource]:
    if models_path is not None:
        return _read_table(models_path, table_name="models")

    models = responses_long[[MODEL_ID]].drop_duplicates().reset_index(drop=True)
    return models, BundleSource(
        table_name="models",
        path=None,
        file_format="derived",
        sha256=None,
        derived=True,
    )


def _build_canonicalization_report(
    report: ValidationReport,
    sources: Mapping[str, BundleSource],
    config: BenchIQConfig,
) -> dict[str, Any]:
    return {
        "duplicate_policy": config.duplicate_policy,
        "sources": {table_name: source.to_dict() for table_name, source in sorted(sources.items())},
        "validation": report.to_dict(),
    }


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
