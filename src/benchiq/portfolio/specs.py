"""Typed specs for BenchIQ's internal public-benchmark portfolio harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

SourceRole = Literal["optimize", "validate"]
MaterializationStatus = Literal["materialized", "skipped", "failed"]


@dataclass(slots=True, frozen=True)
class BinaryMetricPolicy:
    """Selection policy for deterministic binary-compatible metrics."""

    preferred_fragments: tuple[str, ...]
    rejected_fragments: tuple[str, ...] = ()
    allow_numeric_binary: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class SnapshotSpec:
    """One pinned source snapshot or release."""

    snapshot_id: str
    label: str
    release: str
    source_locator: str
    role: SourceRole
    is_dynamic: bool = False
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True, frozen=True)
class BenchmarkSourceSpec:
    """One named benchmark source plus its pinned snapshots."""

    source_id: str
    label: str
    adapter_id: str
    role: SourceRole
    snapshots: tuple[SnapshotSpec, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "label": self.label,
            "adapter_id": self.adapter_id,
            "role": self.role,
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "notes": list(self.notes),
        }


@dataclass(slots=True, frozen=True)
class PortfolioDatasetDef:
    """One materialized dataset plus the config used for standing passes."""

    dataset_id: str
    label: str
    source_id: str
    snapshot_id: str
    role: SourceRole
    responses_path: str
    items_path: str
    models_path: str
    base_config: Mapping[str, Any]
    base_stage_options: Mapping[str, Mapping[str, Any]]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "label": self.label,
            "source_id": self.source_id,
            "snapshot_id": self.snapshot_id,
            "role": self.role,
            "responses_path": self.responses_path,
            "items_path": self.items_path,
            "models_path": self.models_path,
            "base_config": dict(self.base_config),
            "base_stage_options": {
                stage_name: dict(options)
                for stage_name, options in self.base_stage_options.items()
            },
            "notes": list(self.notes),
        }


@dataclass(slots=True, frozen=True)
class MaterializationResult:
    """One source-snapshot materialization outcome."""

    source_id: str
    snapshot_id: str
    role: SourceRole
    status: MaterializationStatus
    manifest_path: str
    dataset: PortfolioDatasetDef | None = None
    bundle_dir: str | None = None
    skip_reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_id": self.source_id,
            "snapshot_id": self.snapshot_id,
            "role": self.role,
            "status": self.status,
            "manifest_path": self.manifest_path,
            "bundle_dir": self.bundle_dir,
            "skip_reason": self.skip_reason,
            "details": dict(self.details),
        }
        if self.dataset is not None:
            payload["dataset"] = self.dataset.to_dict()
        return payload
