#!/usr/bin/env python3
"""Run the optional R-baseline IRT parity comparison."""

from __future__ import annotations

import argparse

from benchiq.irt import run_r_baseline_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BenchIQ's 2PL IRT implementation to an optional R mirt baseline on "
            "identical simulated dichotomous data."
        )
    )
    parser.add_argument("--out", default="reports/irt_r_baseline")
    args = parser.parse_args()

    result = run_r_baseline_comparison(out_dir=args.out)
    print(result.summary_path)


if __name__ == "__main__":
    main()
