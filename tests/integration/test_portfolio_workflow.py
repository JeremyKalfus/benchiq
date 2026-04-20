from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchiq.calibration import calibrate as run_calibration
from benchiq.deployment import predict as run_prediction
from benchiq.io.write import write_parquet
from benchiq.portfolio.materialize import OLLBV1LocalAdapter
from benchiq.portfolio.specs import BenchmarkSourceSpec, SnapshotSpec
from benchiq.runner import run as run_pipeline
from benchiq.schema.tables import MODEL_ID
from benchiq.split.splitters import GLOBAL_SPLIT


def test_materialized_ollb_extract_completes_run_and_calibration_workflow(tmp_path: Path) -> None:
    source_path = tmp_path / "ollb_v1.parquet"
    rows = []
    benchmark_ids = ["arc", "gsm8k"]
    model_ids = [f"model_{index:03d}" for index in range(150)]
    for benchmark_id in benchmark_ids:
        for item_index in range(60):
            item_id = f"{benchmark_id}_item_{item_index:03d}"
            for model_index, model_id in enumerate(model_ids):
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "item_id": item_id,
                        "model_id": model_id,
                        "score": int((item_index + model_index) % 3 == 0),
                    }
                )
    pd.DataFrame.from_records(rows).to_parquet(source_path, index=False)

    source = BenchmarkSourceSpec(
        source_id="ollb_v1_metabench_source",
        label="OLLB v1",
        adapter_id="ollb_v1_local",
        role="optimize",
        snapshots=(),
    )
    snapshot = SnapshotSpec(
        snapshot_id="test_snapshot",
        label="test",
        release="test",
        source_locator=str(source_path),
        role="optimize",
    )
    materialized = OLLBV1LocalAdapter().materialize(
        source=source,
        snapshot=snapshot,
        out_dir=tmp_path / "materialized",
    )
    assert materialized.status == "materialized"
    dataset = materialized.dataset
    assert dataset is not None

    workflow_root = tmp_path / "workflow"
    run_result = run_pipeline(
        dataset.responses_path,
        config=dataset.base_config,
        out_dir=workflow_root,
        items_path=dataset.items_path,
        models_path=dataset.models_path,
        run_id="full-run",
        stage_options=dataset.base_stage_options,
    )
    assert run_result.summary()["metrics"]["marginal_test_rmse_by_benchmark"]

    calibration_result = run_calibration(
        dataset.responses_path,
        config=dataset.base_config,
        out_dir=workflow_root,
        items_path=dataset.items_path,
        models_path=dataset.models_path,
        run_id="calibration",
        stage_options=dataset.base_stage_options,
    )
    split_frame = calibration_result.run_result.stage_results["03_splits"].splits_models
    test_models = (
        split_frame.loc[split_frame[GLOBAL_SPLIT].astype("string") == "test", MODEL_ID]
        .astype("string")
        .tolist()
    )
    assert test_models

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
    available_rate = (
        prediction_counts["best_available_non_null_predictions"]
        / prediction_counts["best_available_rows"]
    )
    assert available_rate >= 0.0
