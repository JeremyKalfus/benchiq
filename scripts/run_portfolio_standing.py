#!/usr/bin/env python3
"""Run the narrowed public portfolio standing pass for BenchIQ."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from click.testing import CliRunner

from benchiq.calibration import calibrate as run_calibration
from benchiq.cli.main import main as cli_main
from benchiq.deployment import predict as run_prediction
from benchiq.io import load_bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.portfolio import (
    build_equal_weight_ranking,
    build_leave_one_out_ranking,
    materialize_catalog,
    narrowed_public_portfolio_catalog,
)
from benchiq.preprocess.optimization import (
    PreprocessingOptimizationDataset,
    PreprocessingOptimizationPlan,
    PreprocessingOptimizationProfile,
    execute_preprocessing_experiment_plans,
    summarize_preprocessing_experiments,
)
from benchiq.runner import run as run_pipeline
from benchiq.schema.tables import MODEL_ID
from benchiq.split.splitters import GLOBAL_SPLIT

REPO_ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_OUT_DIR = REPO_ROOT / "out" / "portfolio_sources"
REPORTS_DIR = REPO_ROOT / "reports" / "portfolio_standing"
WORKDIR = REPORTS_DIR / "workdir"
OPTIMIZATION_DIR = REPORTS_DIR / "optimization_matrix"

EXPECTED_WINNER = "reconstruction_first__deterministic_info"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)

    materialization_results = materialize_catalog(
        source_specs=narrowed_public_portfolio_catalog(),
        out_dir=PORTFOLIO_OUT_DIR,
    )
    materialization_frame = pd.DataFrame.from_records(
        [result.to_dict() for result in materialization_results]
    )
    write_json(
        {"results": materialization_frame.to_dict(orient="records")},
        REPORTS_DIR / "materialization_results.json",
    )

    workflow_rows = [
        _run_source_workflow(result)
        for result in materialization_results
    ]
    workflow_frame = pd.DataFrame.from_records(workflow_rows)
    workflow_csv = REPORTS_DIR / "workflow_status.csv"
    workflow_frame.to_csv(workflow_csv, index=False)
    write_json(
        {"rows": workflow_frame.to_dict(orient="records")},
        REPORTS_DIR / "workflow_status.json",
    )

    optimization_results = [
        result
        for result in materialization_results
        if result.status == "materialized"
        and result.dataset is not None
        and result.role == "optimize"
    ]
    validation_results = [
        result
        for result in materialization_results
        if result.role == "validate"
    ]

    optimization_summary = pd.DataFrame()
    aggregate_ranking = pd.DataFrame()
    leave_one_out = pd.DataFrame()
    recommendation = {
        "status": "no_materialized_optimization_sources",
        "winner_strategy_id": None,
        "reconstruction_first_stands": False,
    }
    if optimization_results:
        optimization_summary = _run_optimization_matrix(optimization_results)
        aggregate_ranking = build_equal_weight_ranking(optimization_summary)
        leave_one_out = build_leave_one_out_ranking(optimization_summary)
        if not aggregate_ranking.empty:
            winner = aggregate_ranking.iloc[0]
            recommendation = {
                "status": "ok",
                "winner_strategy_id": str(winner["strategy_id"]),
                "winner_rmse_mean": float(winner["equal_weight_rmse_mean"]),
                "reconstruction_first_stands": str(winner["strategy_id"]) == EXPECTED_WINNER,
            }

    if not optimization_summary.empty:
        optimization_summary.to_csv(REPORTS_DIR / "optimization_summary.csv", index=False)
        optimization_summary.to_parquet(REPORTS_DIR / "optimization_summary.parquet", index=False)
    if not aggregate_ranking.empty:
        aggregate_ranking.to_csv(REPORTS_DIR / "aggregate_ranking.csv", index=False)
    if not leave_one_out.empty:
        leave_one_out.to_csv(REPORTS_DIR / "leave_one_out.csv", index=False)

    summary_payload = {
        "materialization_results": materialization_frame.to_dict(orient="records"),
        "workflow_status": workflow_frame.to_dict(orient="records"),
        "aggregate_ranking": aggregate_ranking.to_dict(orient="records"),
        "leave_one_out": leave_one_out.to_dict(orient="records"),
        "recommendation": recommendation,
        "validation_sources": [result.to_dict() for result in validation_results],
    }
    write_json(summary_payload, REPORTS_DIR / "standing_report.json")
    (REPORTS_DIR / "summary.md").write_text(
        _summary_markdown(
            materialization_frame=materialization_frame,
            workflow_frame=workflow_frame,
            aggregate_ranking=aggregate_ranking,
            leave_one_out=leave_one_out,
            recommendation=recommendation,
            validation_results=validation_results,
        ),
        encoding="utf-8",
    )
    print(REPORTS_DIR / "summary.md")


def _run_source_workflow(result: Any) -> dict[str, Any]:
    row = {
        "source_id": result.source_id,
        "snapshot_id": result.snapshot_id,
        "role": result.role,
        "materialization_status": result.status,
        "validate_status": "not_run",
        "calibrate_status": "not_run",
        "predict_status": "not_run",
        "run_status": "not_run",
        "workflow_note": result.skip_reason,
    }
    if result.status != "materialized" or result.dataset is None:
        return row

    dataset = result.dataset
    workflow_root = WORKDIR / dataset.dataset_id
    workflow_root.mkdir(parents=True, exist_ok=True)
    config_path = write_json(
        {
            "config": dict(dataset.base_config),
            "stage_options": {
                stage_name: dict(options)
                for stage_name, options in dataset.base_stage_options.items()
            },
        },
        workflow_root / "workflow_config.json",
    )
    cli_runner = CliRunner()
    validate_result = cli_runner.invoke(
        cli_main,
        [
            "validate",
            "--responses",
            dataset.responses_path,
            "--items",
            dataset.items_path,
            "--models",
            dataset.models_path,
            "--config",
            str(config_path),
            "--out",
            str(workflow_root),
        ],
    )
    row["validate_status"] = "ok" if validate_result.exit_code == 0 else "failed"
    row["validate_output"] = validate_result.output.strip()

    if validate_result.exit_code != 0:
        row["workflow_note"] = "validate failed"
        return row

    try:
        calibration_result = run_calibration(
            dataset.responses_path,
            config=dataset.base_config,
            out_dir=workflow_root,
            items_path=dataset.items_path,
            models_path=dataset.models_path,
            run_id="calibration",
            stage_options=dataset.base_stage_options,
        )
        row["calibrate_status"] = "ok"
        row["calibration_root"] = str(calibration_result.calibration_root)
    except Exception as exc:  # pragma: no cover - covered by end-to-end script verification
        row["calibrate_status"] = "failed"
        row["workflow_note"] = str(exc)
        return row

    try:
        split_frame = calibration_result.run_result.stage_results["03_splits"].splits_models
        test_models = (
            split_frame.loc[split_frame[GLOBAL_SPLIT].astype("string") == "test", MODEL_ID]
            .astype("string")
            .tolist()
        )
        if not test_models:
            row["predict_status"] = "skipped"
            row["workflow_note"] = "no test models available for same-source deployment check"
        else:
            calibration_bundle = calibration_result.run_result.stage_results["00_bundle"]
            predict_inputs = calibration_bundle.responses_long.loc[
                calibration_bundle.responses_long[MODEL_ID].astype("string").isin(test_models)
            ].copy()
            prediction_input_path = write_parquet(
                predict_inputs,
                workflow_root / "prediction_inputs" / "responses_long.parquet",
            )
            prediction_result = run_prediction(
                calibration_result.calibration_root,
                prediction_input_path,
                out_dir=workflow_root,
                run_id="prediction",
            )
            prediction_counts = prediction_result.prediction_report["counts"]
            row["predict_status"] = "ok"
            row["prediction_root"] = str(prediction_result.run_root)
            row["prediction_available_rate"] = (
                float(
                    prediction_counts["best_available_non_null_predictions"]
                    / prediction_counts["best_available_rows"]
                )
                if prediction_counts["best_available_rows"]
                else None
            )
    except Exception as exc:  # pragma: no cover - covered by end-to-end script verification
        row["predict_status"] = "failed"
        row["workflow_note"] = str(exc)

    try:
        run_result = run_pipeline(
            dataset.responses_path,
            config=dataset.base_config,
            out_dir=workflow_root,
            items_path=dataset.items_path,
            models_path=dataset.models_path,
            run_id="full-run",
            stage_options=dataset.base_stage_options,
        )
        row["run_status"] = "ok"
        row["run_root"] = str(run_result.run_root)
        summary = run_result.summary()
        row["warning_count"] = int(summary["warning_count"])
        marginal_rmse = summary["metrics"].get("marginal_test_rmse_by_benchmark", {})
        finite_rmse = [
            float(value)
            for value in marginal_rmse.values()
            if value is not None
        ]
        row["marginal_rmse_mean"] = (
            float(sum(finite_rmse) / len(finite_rmse))
            if finite_rmse
            else None
        )
    except Exception as exc:  # pragma: no cover - covered by end-to-end script verification
        row["run_status"] = "failed"
        row["workflow_note"] = str(exc)
    return row


def _run_optimization_matrix(materialized_results: list[Any]) -> pd.DataFrame:
    bundles = {}
    datasets = []
    for result in materialized_results:
        dataset = result.dataset
        assert dataset is not None
        bundles[dataset.dataset_id] = load_bundle(
            dataset.responses_path,
            dataset.items_path,
            dataset.models_path,
            config=dataset.base_config,
        )
        datasets.append(
            PreprocessingOptimizationDataset(
                dataset_id=dataset.dataset_id,
                label=dataset.label,
                source_path=dataset.responses_path,
                base_config=dataset.base_config,
                base_stage_options=dataset.base_stage_options,
                notes=tuple(dataset.notes),
            )
        )

    profiles = [
        PreprocessingOptimizationProfile(
            profile_id="psychometric_default",
            family="psychometric_default",
            description="spec-aligned psychometric defaults",
            config_overrides={
                "drop_low_tail_models_quantile": 0.001,
                "min_item_sd": 0.01,
                "max_item_mean": 0.95,
                "min_abs_point_biserial": 0.05,
                "min_item_coverage": 0.80,
            },
            is_baseline=True,
        ),
        PreprocessingOptimizationProfile(
            profile_id="reconstruction_first",
            family="reconstruction_first",
            description="current reconstruction-first defaults",
            config_overrides={
                "drop_low_tail_models_quantile": 0.002,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.0,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="reconstruction_first_relaxed",
            family="reconstruction_first_relaxed",
            description="reconstruction-first without the low-tail trim",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.0,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="minimal_cleaning",
            family="minimal_cleaning",
            description="minimal cleaning challenger",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 1.0,
                "min_abs_point_biserial": 0.0,
                "min_models_per_item": 1,
                "min_item_coverage": 0.70,
            },
        ),
    ]
    plans: list[PreprocessingOptimizationPlan] = []
    for dataset in datasets:
        plans.append(
            PreprocessingOptimizationPlan(
                search_stage="portfolio_optimize",
                dataset_id=dataset.dataset_id,
                profile_ids=("psychometric_default", "reconstruction_first"),
                preselection_methods=("random_cv", "deterministic_info"),
                seeds=(7,),
            )
        )
        plans.append(
            PreprocessingOptimizationPlan(
                search_stage="portfolio_optimize",
                dataset_id=dataset.dataset_id,
                profile_ids=("reconstruction_first_relaxed", "minimal_cleaning"),
                preselection_methods=("deterministic_info",),
                seeds=(7,),
            )
        )

    raw = execute_preprocessing_experiment_plans(
        bundles=bundles,
        datasets=datasets,
        profiles=profiles,
        plans=plans,
        out_dir=OPTIMIZATION_DIR / "workdir",
    )
    result = summarize_preprocessing_experiments(raw, out_dir=OPTIMIZATION_DIR)
    summary = result.summary.copy()
    summary["strategy_id"] = (
        summary["profile_id"].astype("string")
        + "__"
        + summary["preselection_method"].astype("string")
    )
    summary["source_id"] = summary["dataset_id"].map(
        {dataset.dataset_id: dataset.dataset_id.split("__", 1)[0] for dataset in datasets}
    )
    return summary


def _summary_markdown(
    *,
    materialization_frame: pd.DataFrame,
    workflow_frame: pd.DataFrame,
    aggregate_ranking: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    recommendation: dict[str, Any],
    validation_results: list[Any],
) -> str:
    lines = ["# portfolio standing", ""]

    lines.append("## materialization")
    if materialization_frame.empty:
        lines.append("- no sources were processed")
    else:
        for row in materialization_frame.to_dict(orient="records"):
            lines.append(
                f"- `{row['source_id']}` / `{row['snapshot_id']}`: `{row['status']}`"
                + (
                    f" - {row.get('skip_reason')}"
                    if row.get("skip_reason")
                    else ""
                )
            )
    lines.append("")

    lines.append("## workflow status")
    if workflow_frame.empty:
        lines.append("- no workflows were run")
    else:
        for row in workflow_frame.to_dict(orient="records"):
            lines.append(
                f"- `{row['source_id']}`: validate={row['validate_status']}, "
                f"calibrate={row['calibrate_status']}, predict={row['predict_status']}, "
                f"run={row['run_status']}"
            )
    lines.append("")

    lines.append("## aggregate ranking")
    if aggregate_ranking.empty:
        lines.append("- no informative optimization ranking was produced")
    else:
        for row in aggregate_ranking.head(6).to_dict(orient="records"):
            lines.append(
                f"- `{row['strategy_id']}`: rmse={row['equal_weight_rmse_mean']:.4f}, "
                f"mae={row['equal_weight_mae_mean']:.4f}, "
                f"pearson={row['equal_weight_pearson_mean']:.4f}, "
                f"spearman={row['equal_weight_spearman_mean']:.4f}"
            )
    lines.append("")

    lines.append("## leave-one-out")
    if leave_one_out.empty:
        lines.append("- leave-one-source-out summaries were not produced")
    else:
        winners = (
            leave_one_out.loc[leave_one_out["rank"] == 1, ["left_out_dataset_id", "strategy_id"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        for row in winners:
            lines.append(
                f"- leaving out `{row['left_out_dataset_id']}` -> winner `{row['strategy_id']}`"
            )
    lines.append("")

    lines.append("## validation-only sources")
    if not validation_results:
        lines.append("- no validation-only source records were present")
    else:
        for result in validation_results:
            lines.append(
                f"- `{result.source_id}`: `{result.status}`"
                + (f" - {result.skip_reason}" if result.skip_reason else "")
            )
    lines.append("")

    lines.append("## recommendation")
    if recommendation["status"] != "ok":
        lines.append(
            "- no optimization winner recommendation was produced because no informative aggregate "
            "ranking was available"
        )
    else:
        lines.append(
            f"- winner: `{recommendation['winner_strategy_id']}` "
            f"(equal-weight rmse `{recommendation['winner_rmse_mean']:.4f}`)"
        )
        if recommendation["reconstruction_first_stands"]:
            lines.append("- `reconstruction_first` still stands as the recommended profile")
        else:
            lines.append("- `reconstruction_first` was displaced on this narrowed portfolio")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
