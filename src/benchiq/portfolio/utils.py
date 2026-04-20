"""Utility helpers for the internal portfolio harness."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen


def stable_hash(text: str) -> str:
    """Return a stable sha256 hex digest."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_sample(values: Iterable[str], *, max_count: int, salt: str) -> list[str]:
    """Return a deterministic pseudo-random sample without mutating the input order."""

    unique_values = sorted({str(value) for value in values})
    decorated = [
        (stable_hash(f"{salt}::{value}"), value)
        for value in unique_values
    ]
    decorated.sort()
    return [value for _, value in decorated[:max_count]]


def normalize_slug(text: str) -> str:
    """Normalize free-form text into a deterministic id-friendly slug."""

    lowered = text.strip().lower()
    lowered = lowered.replace("/", "_")
    lowered = lowered.replace(" ", "_")
    lowered = lowered.replace("-", "_")
    lowered = re.sub(r"[^a-z0-9_:.]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered)
    return lowered.strip("_")


def prefixed_benchmark_id(source_id: str, snapshot_id: str, native_benchmark_id: str) -> str:
    """Build a collision-resistant benchmark id."""

    return "__".join(
        (
            normalize_slug(source_id),
            normalize_slug(snapshot_id),
            normalize_slug(native_benchmark_id),
        )
    )


def prefixed_item_id(
    source_id: str,
    snapshot_id: str,
    native_benchmark_id: str,
    native_item_id: str,
) -> str:
    """Build a collision-resistant item id."""

    benchmark_slug = normalize_slug(native_benchmark_id)
    item_slug = normalize_slug(native_item_id)
    return "__".join(
        (
            normalize_slug(source_id),
            normalize_slug(snapshot_id),
            benchmark_slug,
            item_slug,
        )
    )


def infer_model_family(model_id: str) -> str:
    """Infer a lightweight family string from a model id when possible."""

    normalized = str(model_id).strip()
    if "/" in normalized:
        return normalized.split("/", 1)[0]
    if "__" in normalized:
        return normalized.split("__", 1)[0]
    if ":" in normalized:
        return normalized.split(":", 1)[0]
    if "-" in normalized:
        return normalized.split("-", 1)[0]
    return normalized


def http_get_json(url: str) -> Any:
    """Fetch a json response using the stdlib only."""

    request = Request(
        url,
        headers={
            "User-Agent": "BenchIQ-portfolio-harness/0.1",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def http_status(url: str) -> int:
    """Fetch only the response status for a url."""

    request = Request(
        url,
        headers={"User-Agent": "BenchIQ-portfolio-harness/0.1"},
    )
    with urlopen(request, timeout=60) as response:
        return int(response.status)


def gcs_api_path(bucket_path: str) -> str:
    """Encode a bucket-relative object path for the public GCS API."""

    return quote(bucket_path, safe="")


def coerce_binary_metric(value: Any) -> int | None:
    """Convert a candidate metric into a strict binary value when possible."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        if value in (0, 1):
            return int(value)
    return None


def ensure_relative_path(path: str | Path, *, base: Path) -> str:
    """Return a relative path string when possible, else an absolute path string."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(base.resolve()))
    except ValueError:
        return str(resolved)
