#!/usr/bin/env python3
"""Generate a calibration/deployment walkthrough bundle on the compact fixture."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import benchiq


def main() -> None:
    config_payload = json.loads(Path("tests/data/tiny_example/config.json").read_text())
    reports_dir = Path("reports/calibration_deployment")
    workdir = reports_dir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    calibration_result = benchiq.calibrate(
        "tests/data/metabench_validation/responses_long.csv",
        config=config_payload["config"],
        out_dir=workdir,
        run_id="calibration_run",
        stage_options=config_payload["stage_options"],
    )
    bundle = calibration_result.run_result.stage_results["00_bundle"]
    reduced_responses = _reduced_test_responses(calibration_result)
    reduced_path = reports_dir / "reduced_test_responses.csv"
    reduced_path.parent.mkdir(parents=True, exist_ok=True)
    reduced_responses.to_csv(reduced_path, index=False)

    prediction_result = benchiq.predict(
        calibration_result.calibration_root,
        reduced_path,
        out_dir=workdir,
        run_id="prediction_run",
    )
    comparison = _comparison_frame(calibration_result, prediction_result)
    comparison.to_parquet(reports_dir / "comparison.parquet", index=False)
    comparison.to_csv(reports_dir / "comparison.csv", index=False)

    report = {
        "source_responses": "tests/data/metabench_validation/responses_long.csv",
        "source_config": "tests/data/tiny_example/config.json",
        "calibration_run_root": str(calibration_result.run_result.run_root),
        "calibration_bundle_root": str(calibration_result.calibration_root),
        "prediction_run_root": str(prediction_result.run_root),
        "reduced_response_path": str(reduced_path),
        "comparison_rows": int(len(comparison.index)),
        "max_abs_prediction_delta": float(comparison["abs_delta"].max()),
        "mean_abs_prediction_delta": float(comparison["abs_delta"].mean()),
    }
    (reports_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _scatter_plot(
        comparison,
        out_path=reports_dir / "deployment_vs_calibration.png",
    )
    (reports_dir / "summary.md").write_text(
        _summary_markdown(report=report),
        encoding="utf-8",
    )
    metadata = {
        "bundle_row_count": int(len(bundle.responses_long.index)),
        "benchmark_count": int(bundle.responses_long["benchmark_id"].nunique()),
        "model_count": int(bundle.responses_long["model_id"].nunique()),
    }
    (reports_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(reports_dir / "summary.md")


def _reduced_test_responses(calibration_result) -> pd.DataFrame:
    bundle = calibration_result.run_result.stage_results["00_bundle"]
    split_result = calibration_result.run_result.stage_results["03_splits"]
    select_result = calibration_result.run_result.stage_results["06_select"]

    test_model_ids = {
        model_id
        for split_frame in split_result.per_benchmark_splits.values()
        for model_id in split_frame.loc[
            split_frame["split"] == "test",
            "model_id",
        ]
        .astype("string")
        .tolist()
    }
    selected_item_ids_by_benchmark = {
        benchmark_id: set(
            benchmark_result.subset_final["item_id"].dropna().astype("string").tolist()
        )
        for benchmark_id, benchmark_result in select_result.benchmarks.items()
    }

    reduced = bundle.responses_long.loc[
        bundle.responses_long["model_id"].isin(test_model_ids)
    ].copy()
    keep_mask = reduced.apply(
        lambda row: row["item_id"]
        in selected_item_ids_by_benchmark.get(row["benchmark_id"], set()),
        axis=1,
    )
    return reduced.loc[keep_mask].reset_index(drop=True)


def _comparison_frame(calibration_result, prediction_result) -> pd.DataFrame:
    reconstruction_result = calibration_result.run_result.stage_results["09_reconstruct"]
    reference_rows: list[dict[str, object]] = []
    for benchmark_id, benchmark_result in sorted(reconstruction_result.benchmarks.items()):
        test_rows = benchmark_result.predictions.loc[
            benchmark_result.predictions["split"] == "test"
        ].copy()
        for model_id in test_rows["model_id"].dropna().astype("string").unique().tolist():
            model_rows = test_rows.loc[test_rows["model_id"] == model_id].copy()
            joint_rows = model_rows.loc[model_rows["model_type"] == "joint"].copy()
            chosen_rows = (
                joint_rows
                if not joint_rows.empty
                else model_rows.loc[model_rows["model_type"] == "marginal"].copy()
            )
            if chosen_rows.empty:
                continue
            row = chosen_rows.iloc[0]
            reference_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "model_id": model_id,
                    "reference_model_type": row["model_type"],
                    "reference_predicted_score": row["predicted_score"],
                }
            )

    reference = pd.DataFrame(reference_rows).astype(
        {
            "benchmark_id": "string",
            "model_id": "string",
            "reference_model_type": "string",
            "reference_predicted_score": "Float64",
        }
    )
    predicted = (
        prediction_result.predictions_best_available.loc[
            :,
            [
                "benchmark_id",
                "model_id",
                "selected_model_type",
                "predicted_score",
            ],
        ]
        .rename(
            columns={
                "selected_model_type": "deployment_model_type",
                "predicted_score": "deployment_predicted_score",
            }
        )
        .copy()
    )
    merged = reference.merge(predicted, on=["benchmark_id", "model_id"], how="inner")
    merged["delta"] = merged["deployment_predicted_score"].astype(float) - merged[
        "reference_predicted_score"
    ].astype(float)
    merged["abs_delta"] = merged["delta"].abs()
    return merged.astype(
        {
            "benchmark_id": "string",
            "model_id": "string",
            "reference_model_type": "string",
            "reference_predicted_score": "Float64",
            "deployment_model_type": "string",
            "deployment_predicted_score": "Float64",
            "delta": "Float64",
            "abs_delta": "Float64",
        }
    )


def _scatter_plot(comparison: pd.DataFrame, *, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        comparison["reference_predicted_score"].astype(float),
        comparison["deployment_predicted_score"].astype(float),
        alpha=0.8,
    )
    min_value = min(
        comparison["reference_predicted_score"].astype(float).min(),
        comparison["deployment_predicted_score"].astype(float).min(),
    )
    max_value = max(
        comparison["reference_predicted_score"].astype(float).max(),
        comparison["deployment_predicted_score"].astype(float).max(),
    )
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    ax.set_xlabel("calibration-stage prediction")
    ax.set_ylabel("deployment prediction")
    ax.set_title("deployment vs calibration predictions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summary_markdown(*, report: dict[str, object]) -> str:
    return (
        "# calibration / deployment walkthrough\n\n"
        f"- calibration_run_root: `{report['calibration_run_root']}`\n"
        f"- calibration_bundle_root: `{report['calibration_bundle_root']}`\n"
        f"- prediction_run_root: `{report['prediction_run_root']}`\n"
        f"- reduced_response_path: `{report['reduced_response_path']}`\n"
        f"- comparison_rows: `{report['comparison_rows']}`\n"
        f"- max_abs_prediction_delta: `{report['max_abs_prediction_delta']}`\n"
        f"- mean_abs_prediction_delta: `{report['mean_abs_prediction_delta']}`\n"
    )


if __name__ == "__main__":
    main()
