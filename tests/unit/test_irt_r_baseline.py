import numpy as np
import pandas as pd

from benchiq.irt import align_r_baseline_to_benchiq
from benchiq.irt import r_baseline as r_baseline_module


def test_align_r_baseline_to_benchiq_recovers_known_linear_transform() -> None:
    benchiq_item_params = pd.DataFrame(
        {
            "item_id": pd.Series(["i1", "i2", "i3"], dtype="string"),
            "discrimination": pd.Series([0.8, 1.1, 1.6], dtype="Float64"),
            "difficulty": pd.Series([-1.0, 0.0, 1.2], dtype="Float64"),
        }
    )
    benchiq_theta = pd.DataFrame(
        {
            "model_id": pd.Series(["m1", "m2", "m3", "m4"], dtype="string"),
            "theta_hat": pd.Series([-1.2, -0.1, 0.6, 1.3], dtype="Float64"),
            "theta_se": pd.Series([0.2, 0.2, 0.2, 0.2], dtype="Float64"),
        }
    )

    scale = 1.7
    intercept = -0.4
    r_item_params = pd.DataFrame(
        {
            "item_id": pd.Series(["i1", "i2", "i3"], dtype="string"),
            "discrimination": pd.Series(
                benchiq_item_params["discrimination"].astype(float) / scale,
                dtype="Float64",
            ),
            "difficulty": pd.Series(
                benchiq_item_params["difficulty"].astype(float) * scale + intercept,
                dtype="Float64",
            ),
        }
    )
    r_theta = pd.DataFrame(
        {
            "model_id": pd.Series(["m1", "m2", "m3", "m4"], dtype="string"),
            "theta_hat": pd.Series(
                benchiq_theta["theta_hat"].astype(float) * scale + intercept,
                dtype="Float64",
            ),
            "theta_se": pd.Series([0.34, 0.34, 0.34, 0.34], dtype="Float64"),
        }
    )

    aligned = align_r_baseline_to_benchiq(
        benchiq_item_params=benchiq_item_params,
        benchiq_theta=benchiq_theta,
        r_item_params=r_item_params,
        r_theta=r_theta,
    )

    aligned_items = aligned["item_params_aligned"].sort_values("item_id").reset_index(drop=True)
    aligned_theta = aligned["theta_aligned"].sort_values("model_id").reset_index(drop=True)
    np.testing.assert_allclose(
        aligned_items["discrimination"].astype(float).to_numpy(),
        benchiq_item_params.sort_values("item_id")["discrimination"].astype(float).to_numpy(),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        aligned_items["difficulty"].astype(float).to_numpy(),
        benchiq_item_params.sort_values("item_id")["difficulty"].astype(float).to_numpy(),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        aligned_theta["theta_hat"].astype(float).to_numpy(),
        benchiq_theta.sort_values("model_id")["theta_hat"].astype(float).to_numpy(),
        atol=1e-8,
    )


def test_align_r_baseline_to_benchiq_rejects_empty_model_overlap() -> None:
    benchiq_item_params = pd.DataFrame(
        {
            "item_id": pd.Series(["i1"], dtype="string"),
            "discrimination": pd.Series([1.0], dtype="Float64"),
            "difficulty": pd.Series([0.0], dtype="Float64"),
        }
    )
    benchiq_theta = pd.DataFrame(
        {
            "model_id": pd.Series(["m1"], dtype="string"),
            "theta_hat": pd.Series([0.1], dtype="Float64"),
            "theta_se": pd.Series([0.2], dtype="Float64"),
        }
    )
    r_item_params = pd.DataFrame(
        {
            "item_id": pd.Series(["i1"], dtype="string"),
            "discrimination": pd.Series([1.0], dtype="Float64"),
            "difficulty": pd.Series([0.0], dtype="Float64"),
        }
    )
    r_theta = pd.DataFrame(
        {
            "model_id": pd.Series(["r1"], dtype="string"),
            "theta_hat": pd.Series([0.2], dtype="Float64"),
            "theta_se": pd.Series([0.3], dtype="Float64"),
        }
    )

    try:
        align_r_baseline_to_benchiq(
            benchiq_item_params=benchiq_item_params,
            benchiq_theta=benchiq_theta,
            r_item_params=r_item_params,
            r_theta=r_theta,
        )
    except ValueError as exc:
        assert "no shared model_id values" in str(exc)
    else:
        raise AssertionError("expected align_r_baseline_to_benchiq to reject empty model overlap")


def test_build_gate_report_rejects_skipped_status() -> None:
    gate = r_baseline_module._build_gate_report(
        report={
            "status": "skipped",
            "skip_reason": "R package `mirt` is not installed in this environment",
        },
        gate_thresholds=r_baseline_module.DEFAULT_PARITY_GATE_THRESHOLDS,
    )

    assert gate["passed"] is False
    assert gate["failure_count"] == 1
    assert gate["checks"][0]["metric"] == "status"
    assert gate["checks"][0]["passed"] is False
    assert "mirt" in gate["failures"][0]


def test_run_r_baseline_comparison_skip_report_captures_environment_and_gate(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        r_baseline_module,
        "_collect_environment_metadata",
        lambda: {
            "python": {
                "version": "3.14.4",
                "implementation": "CPython",
                "executable": "/tmp/python",
                "platform": "test-platform",
            },
            "packages": {
                "arviz": "0.23.4",
                "pymc": "5.28.4",
                "numpy": "2.3.4",
                "pandas": "2.3.3",
                "scipy": "1.16.3",
                "girth": "0.8.0",
            },
            "r": {
                "available": True,
                "rscript_path": "/tmp/Rscript",
                "version": "R version 4.5.0",
                "version_error": None,
                "mirt_installed": False,
                "mirt_version": None,
                "package_check_error": None,
            },
        },
    )

    result = r_baseline_module.run_r_baseline_comparison(out_dir=tmp_path / "irt_r_baseline")

    assert result.report["status"] == "skipped"
    assert result.report["skip_reason"] == "R package `mirt` is not installed in this environment"
    assert result.report["environment"]["python"]["platform"] == "test-platform"
    assert result.report["environment"]["r"]["rscript_path"] == "/tmp/Rscript"
    assert result.report["gate"]["passed"] is False
    assert result.report["gate"]["checks"][0]["metric"] == "status"
    assert result.report["gate"]["thresholds"]["theta_pearson_min"] == 0.95
    summary_json = (tmp_path / "irt_r_baseline" / "irt_r_baseline_summary.json").read_text(
        encoding="utf-8"
    )
    assert "test-platform" in summary_json
    assert "theta_pearson_min" in summary_json
