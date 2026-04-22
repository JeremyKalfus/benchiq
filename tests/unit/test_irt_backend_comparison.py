import json
from pathlib import Path

import pandas as pd

from benchiq.irt.backend_comparison import write_backend_comparison_artifacts


def test_write_backend_comparison_artifacts_writes_report_family(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports" / "irt_backend_comparison"
    run_index = pd.DataFrame(
        [
            {
                "run_signature": "compact_validation_fixture__girth__seed_7",
                "run_id": "compact_validation_fixture__girth__seed_7",
                "dataset_id": "compact_validation_fixture",
                "dataset_label": "compact validation fixture",
                "backend": "girth",
                "seed": 7,
                "status": "ok",
                "run_root": "/tmp/run",
                "metrics_path": "/tmp/run/reports/metrics.json",
                "run_runtime_seconds": 1.5,
                "stage05_runtime_seconds": 0.4,
                "stage09_runtime_seconds": 0.3,
                "warning_count": 0,
                "error_type": None,
                "error_message": None,
            }
        ]
    )
    per_run_metrics = pd.DataFrame(
        [
            {
                "run_signature": "compact_validation_fixture__girth__seed_7",
                "dataset_id": "compact_validation_fixture",
                "dataset_label": "compact validation fixture",
                "backend": "girth",
                "seed": 7,
                "benchmark_id": "b1",
                "marginal_test_rmse": 0.21,
                "joint_test_rmse": None,
                "best_available_test_rmse": 0.21,
                "best_available_model_type": "marginal",
                "joint_available": False,
            }
        ]
    )
    dataset_summary = pd.DataFrame(
        [
            {
                "backend": "girth",
                "dataset_id": "compact_validation_fixture",
                "dataset_label": "compact validation fixture",
                "run_count": 1,
                "successful_run_count": 1,
                "failed_run_count": 0,
                "failure_rate": 0.0,
                "informative_benchmark_rows": 1,
                "best_available_test_rmse_mean": 0.21,
                "best_available_test_rmse_std": None,
                "joint_available_rate": 0.0,
                "run_runtime_mean_seconds": 1.5,
                "stage05_runtime_mean_seconds": 0.4,
                "seed_rmse_std": None,
                "informative_dataset": True,
            }
        ]
    )
    backend_summary = pd.DataFrame(
        [
            {
                "backend": "girth",
                "run_count": 1,
                "failure_rate": 0.0,
                "informative_bundle_count": 1,
                "equal_weight_informative_rmse_mean": 0.21,
                "seed_rmse_std": None,
                "large_bundle_runtime_mean_seconds": 1.5,
                "large_bundle_stage05_runtime_mean_seconds": 0.4,
                "mean_successful_run_runtime_seconds": 1.5,
                "successful_seed_runtime_std": None,
            }
        ]
    )
    deployment_summary = pd.DataFrame(
        [
            {
                "dataset_id": "large_release_default_subset",
                "dataset_label": "large release-default subset",
                "backend": "girth",
                "seed": 7,
                "status": "ok",
                "prediction_available_rate": 1.0,
                "deployment_rmse": 1.2,
                "calibration_root": "/tmp/calibration",
                "prediction_root": "/tmp/predict",
                "error_type": None,
                "error_message": None,
            }
        ]
    )
    parity_summary = pd.DataFrame(
        [
            {
                "backend": "girth",
                "status": "ok",
                "gate_passed": True,
                "skip_reason": None,
                "theta_pearson": 0.99,
                "theta_spearman": 0.99,
                "icc_mean_rmse": 0.03,
                "summary_path": "/tmp/summary.md",
                "report_json_path": "/tmp/report.json",
                "failure_count": 0,
            }
        ]
    )
    report = {
        "generated_at": "2026-04-21T00:00:00+00:00",
        "winner": {
            "backend": "girth",
            "eligible": True,
            "reason": "fixture winner",
        },
        "candidates": [
            {
                "backend": "girth",
                "equal_weight_informative_rmse_mean": 0.21,
                "seed_rmse_std": None,
                "failure_rate": 0.0,
                "large_bundle_runtime_mean_seconds": 1.5,
                "disqualifications": [],
            }
        ],
    }

    artifact_paths = write_backend_comparison_artifacts(
        out_dir=out_dir,
        run_index=run_index,
        per_run_metrics=per_run_metrics,
        dataset_summary=dataset_summary,
        backend_summary=backend_summary,
        deployment_summary=deployment_summary,
        parity_summary=parity_summary,
        report=report,
    )

    assert artifact_paths["summary_md"].exists()
    assert artifact_paths["report_json"].exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "backend_summary.csv").exists()
    assert (out_dir / "deployment_summary.csv").exists()
    assert (out_dir / "parity_summary.csv").exists()
    summary_text = artifact_paths["summary_md"].read_text(encoding="utf-8")
    assert "winner: `girth`" in summary_text
    saved_report = json.loads(artifact_paths["report_json"].read_text(encoding="utf-8"))
    assert saved_report["winner"]["backend"] == "girth"
