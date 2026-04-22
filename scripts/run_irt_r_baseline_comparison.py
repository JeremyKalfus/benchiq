#!/usr/bin/env python3
"""Run the optional R-baseline IRT parity comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchiq.irt import run_r_baseline_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BenchIQ's 2PL IRT implementation to an optional R mirt baseline on "
            "identical simulated dichotomous data."
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
        print(f"skip_reason: {result.report['skip_reason']}")
    else:
        metrics = result.report["metrics"]
        print(f"theta_pearson: {metrics['theta']['pearson']:.6f}")
        print(f"theta_spearman: {metrics['theta']['spearman']:.6f}")
        print(f"icc_mean_rmse: {metrics['icc']['mean_rmse']:.6f}")
        print(f"icc_max_rmse: {metrics['icc']['max_rmse']:.6f}")


if __name__ == "__main__":
    main()
