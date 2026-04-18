from pathlib import Path

import pytest

from benchiq.irt import run_r_baseline_comparison


def test_run_r_baseline_comparison_runs_or_skips_cleanly(tmp_path: Path) -> None:
    result = run_r_baseline_comparison(out_dir=tmp_path / "irt_r_baseline")

    assert result.summary_path.exists()
    assert result.table_path.exists()
    if result.report["status"] == "skipped":
        assert "skip_reason" in result.report
        pytest.skip(result.report["skip_reason"])

    metrics = result.report["metrics"]
    assert metrics["theta"]["spearman"] >= 0.95
    assert metrics["theta"]["pearson"] >= 0.95
    assert metrics["icc"]["mean_rmse"] <= 0.08
