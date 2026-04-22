from pathlib import Path

import pytest

from benchiq.irt import run_r_baseline_comparison


def test_run_r_baseline_comparison_runs_or_skips_cleanly(tmp_path: Path) -> None:
    result = run_r_baseline_comparison(out_dir=tmp_path / "irt_r_baseline")

    assert result.summary_path.exists()
    assert result.table_path.exists()
    assert "environment" in result.report
    assert "gate" in result.report
    if result.report["status"] == "skipped":
        assert "skip_reason" in result.report
        assert result.report["gate"]["passed"] is False
        assert result.report["gate"]["checks"][0]["metric"] == "status"
        pytest.skip(result.report["skip_reason"])

    assert result.report["gate"]["passed"] is True
    metrics = result.report["metrics"]
    assert metrics["theta"]["spearman"] >= 0.95
    assert metrics["theta"]["pearson"] >= 0.95
    assert metrics["icc"]["mean_rmse"] <= 0.08
