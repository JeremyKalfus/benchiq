import json
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from benchiq.cli.main import main


def test_metabench_validation_cli_writes_reproducible_report(tmp_path: Path) -> None:
    runner = CliRunner()

    first_out = tmp_path / "out-a"
    first = runner.invoke(main, ["metabench", "run", "--out", str(first_out)])
    assert first.exit_code == 0, first.output

    first_root = first_out / "metabench-validation"
    first_report_path = first_root / "reports" / "metabench_validation_report.json"
    assert first_report_path.exists()
    first_report = json.loads(first_report_path.read_text(encoding="utf-8"))
    assert first_report["passed"] is True
    assert first_report["expected"]["available"] is True
    assert first_report["artifact_checks"]["passed"] is True
    assert first_report["evaluation"]["passed"] is True
    assert first_report["evaluation"]["checked_metric_count"] > 0
    assert first_report["mode"]["profile"] == "reduced"
    assert first_report["fixture"]["name"] == "bundled_reduced_fixture"
    assert (first_root / "artifacts" / "03_splits" / "split_report.json").exists()
    assert (
        first_root
        / "artifacts"
        / "04_subsample"
        / "per_benchmark"
        / "b1"
        / "preselect_items.parquet"
    ).exists()
    assert (first_root / "artifacts" / "09_reconstruct" / "reconstruction_report.json").exists()

    second_out = tmp_path / "out-b"
    second = runner.invoke(main, ["metabench", "run", "--out", str(second_out)])
    assert second.exit_code == 0, second.output

    second_root = second_out / "metabench-validation"
    second_report = json.loads(
        (second_root / "reports" / "metabench_validation_report.json").read_text(encoding="utf-8")
    )
    assert second_report["metrics"] == first_report["metrics"]

    first_subset = pd.read_parquet(
        first_root / "artifacts" / "06_select" / "per_benchmark" / "b1" / "subset_final.parquet"
    )
    second_subset = pd.read_parquet(
        second_root / "artifacts" / "06_select" / "per_benchmark" / "b1" / "subset_final.parquet"
    )
    assert first_subset[["item_id"]].equals(second_subset[["item_id"]])
