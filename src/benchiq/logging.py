"""Manifest and hashing helpers for BenchIQ."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from benchiq.config import BenchIQConfig

DEPENDENCY_PACKAGES = (
    "arviz",
    "click",
    "girth",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "pymc",
    "pyarrow",
    "pydantic",
    "pygam",
    "scikit-learn",
    "scipy",
)


def hash_file_sha256(path: str | Path) -> str:
    """Return a deterministic sha256 for a file on disk."""

    resolved_path = Path(path)
    hasher = sha256()
    with resolved_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_manifest(
    *,
    run_id: str,
    config: BenchIQConfig,
    sources: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Build the stage manifest payload."""

    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "package_version": _package_version(),
        "dependency_versions": _dependency_versions(),
        "seeds": {"random_seed": config.random_seed},
        "git_hash": _git_hash(),
        "input_hashes": {
            table_name: source.sha256 for table_name, source in sorted(sources.items())
        },
        "input_sources": {
            table_name: source.to_dict() for table_name, source in sorted(sources.items())
        },
        "resolved_config": config.model_dump(mode="json"),
        "artifacts": {
            "00_canonical": {
                table_name: str(path) for table_name, path in sorted(artifact_paths.items())
            },
        },
    }


def update_manifest(manifest_path: str | Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    """Merge updates into manifest.json and write the result."""

    resolved_path = Path(manifest_path)
    existing: dict[str, Any] = {}
    if resolved_path.exists():
        existing = json.loads(resolved_path.read_text(encoding="utf-8"))
    merged = _merge_dicts(existing, dict(updates))
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return merged


def _dependency_versions() -> dict[str, str]:
    dependency_versions: dict[str, str] = {}
    for package_name in DEPENDENCY_PACKAGES:
        try:
            dependency_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            dependency_versions[package_name] = "unavailable"
    return dependency_versions


def _package_version() -> str:
    try:
        return version("benchiq")
    except PackageNotFoundError:
        return "0.1.0a0"


def _git_hash() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
