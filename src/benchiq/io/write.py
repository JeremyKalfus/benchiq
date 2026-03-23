"""Stage-00 artifact writing helpers for BenchIQ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchiq.logging import build_manifest, update_manifest


def write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a dataframe to parquet with parent directory creation."""

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(resolved_path, index=False)
    return resolved_path


def write_json(payload: Any, path: str | Path) -> Path:
    """Write a json payload with stable formatting."""

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved_path


def write_stage0_bundle(
    bundle: Any,
    out_dir: str | Path,
    *,
    run_id: str,
) -> tuple[dict[str, Path], Path]:
    """Write the canonical stage-00 artifacts and manifest."""

    run_root = Path(out_dir) / run_id
    stage_dir = run_root / "artifacts" / "00_canonical"
    artifact_paths = {
        "responses_long": write_parquet(
            bundle.responses_long,
            stage_dir / "responses_long.parquet",
        ),
        "items": write_parquet(bundle.items, stage_dir / "items.parquet"),
        "models": write_parquet(bundle.models, stage_dir / "models.parquet"),
        "canonicalization_report": write_json(
            bundle.canonicalization_report,
            stage_dir / "canonicalization_report.json",
        ),
    }
    manifest_path = run_root / "manifest.json"
    manifest_payload = build_manifest(
        run_id=run_id,
        config=bundle.config,
        sources=bundle.sources,
        artifact_paths=artifact_paths,
    )
    update_manifest(manifest_path, manifest_payload)
    return artifact_paths, manifest_path
