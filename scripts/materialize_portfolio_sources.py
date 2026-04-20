#!/usr/bin/env python3
"""Materialize the narrowed public portfolio sources for BenchIQ."""

from __future__ import annotations

from pathlib import Path

from benchiq.portfolio import materialize_catalog, narrowed_public_portfolio_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "out" / "portfolio_sources"


def main() -> None:
    results = materialize_catalog(
        source_specs=narrowed_public_portfolio_catalog(),
        out_dir=OUT_DIR,
    )
    status_lines = [
        f"{result.source_id}::{result.snapshot_id} -> {result.status}"
        for result in results
    ]
    print("\n".join(status_lines))
    print(OUT_DIR / "index_manifest.json")


if __name__ == "__main__":
    main()
