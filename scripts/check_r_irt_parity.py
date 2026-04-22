#!/usr/bin/env python3
"""Run the local R parity gate and fail on skip or threshold breach."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchiq.irt import run_r_baseline_comparison


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the BenchIQ vs R mirt IRT parity comparison and exit nonzero if the "
            "comparison is skipped or if the saved gate thresholds are not met."
        )
    )
    parser.add_argument("--out", default="reports/irt_r_baseline")
    parser.add_argument("--backend", default="girth")
    parser.add_argument("--theta-pearson-min", type=float, default=0.95)
    parser.add_argument("--theta-spearman-min", type=float, default=0.95)
    parser.add_argument("--icc-mean-rmse-max", type=float, default=0.08)
    args = parser.parse_args()

    result = run_r_baseline_comparison(
        out_dir=args.out,
        backend=args.backend,
        gate_thresholds={
            "theta_pearson_min": args.theta_pearson_min,
            "theta_spearman_min": args.theta_spearman_min,
            "icc_mean_rmse_max": args.icc_mean_rmse_max,
        },
    )
    json_path = Path(args.out) / "irt_r_baseline_summary.json"
    print(f"summary: {result.summary_path}")
    print(f"report_json: {json_path}")
    print(f"status: {result.report['status']}")
    print(f"backend: {result.report['simulation']['backend']}")
    print(f"gate_passed: {result.report['gate']['passed']}")

    if result.report["status"] == "skipped":
        print(
            "parity gate failed because the optional R baseline is unavailable: "
            + f"{result.report['skip_reason']}",
            file=sys.stderr,
        )
        print(
            "run `Rscript scripts/setup_r_irt_parity.R` to install the required R package first.",
            file=sys.stderr,
        )
        return 2

    if not result.report["gate"]["passed"]:
        print("parity gate failed:", file=sys.stderr)
        for failure in result.report["gate"]["failures"]:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("parity gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
