#!/usr/bin/env python3
"""Run the reconstruction-head comparison on the compact regression fixture."""

from __future__ import annotations

import json
from pathlib import Path

import benchiq
from benchiq.reconstruct import run_reconstruction_head_experiments


def main() -> None:
    config_payload = json.loads(Path("tests/data/tiny_example/config.json").read_text())
    reports_dir = Path("reports/experiments/reconstruction_heads")
    workdir = reports_dir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    run_result = benchiq.run(
        "tests/data/compact_validation/responses_long.csv",
        config=config_payload["config"],
        out_dir=workdir,
        run_id="feature_run",
        stage_options=config_payload["stage_options"],
        stop_after="08_features",
    )
    experiment_result = run_reconstruction_head_experiments(
        run_result.stage_results["08_features"],
        methods=("gam", "elastic_net", "xgboost"),
        seeds=(7, 11, 19),
        lam_grid=(0.1, 1.0),
        cv_folds=3,
        n_splines=5,
        out_dir=reports_dir,
    )

    metadata = {
        "source_responses": "tests/data/compact_validation/responses_long.csv",
        "source_config": "tests/data/tiny_example/config.json",
        "feature_run_root": str(run_result.run_root),
        "report_dir": str(reports_dir),
        "methods": ["gam", "elastic_net", "xgboost"],
        "seeds": [7, 11, 19],
    }
    (reports_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(experiment_result.artifact_paths["summary_md"])


if __name__ == "__main__":
    main()
