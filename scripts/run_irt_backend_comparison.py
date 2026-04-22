#!/usr/bin/env python3
"""Run the saved IRT backend comparison harness."""

from __future__ import annotations

from benchiq.irt.backend_comparison import compare_irt_backends


def main() -> None:
    result = compare_irt_backends()
    print(result.artifact_paths["summary_md"])


if __name__ == "__main__":
    main()
