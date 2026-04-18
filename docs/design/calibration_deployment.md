# calibration and deployment

BenchIQ now has an explicit split between calibration and deployment.

Public surface today:

- python API: `benchiq.calibrate(...)`, `benchiq.predict(...)`, `benchiq.deploy(...)`,
  `benchiq.load_calibration_bundle(...)`
- CLI: `benchiq calibrate ...` and `benchiq predict ...`
- backward-compatible full pipeline: `benchiq run ...`

## calibration

Calibration is the fit step.

Given a benchmark bundle with many model/checkpoint runs and item-level responses, BenchIQ:

- preprocesses the bundle
- performs model-level splitting
- preselects items
- fits benchmark-local 2PL IRT
- selects final retained items
- estimates theta metadata
- fits reconstruction heads
- saves a reusable `calibration_bundle/`

The calibration bundle is written inside the calibration run root and contains the fitted artifacts
needed for later reuse without retraining.

Important contents:

- `calibration_bundle/manifest.json`
- `calibration_bundle/config_resolved.json`
- `calibration_bundle/reconstruction_summary.parquet`
- `calibration_bundle/per_benchmark/<benchmark_id>/subset_final.parquet`
- `calibration_bundle/per_benchmark/<benchmark_id>/irt_item_params.parquet`
- `calibration_bundle/per_benchmark/<benchmark_id>/theta_scoring_metadata.json`
- `calibration_bundle/per_benchmark/<benchmark_id>/linear_predictor_coefficients.parquet`
- `calibration_bundle/per_benchmark/<benchmark_id>/reconstruction/<model_type>/gam_model.pkl`

## deployment

Deployment is the scoring step.

Given a saved calibration bundle and a new reduced response set, BenchIQ:

- loads the saved fitted bundle
- validates that required fitted artifacts exist
- estimates theta and theta uncertainty from the saved item parameters
- rebuilds the deployment-time feature tables
- predicts full benchmark scores using the saved reconstruction heads

Deployment does **not** refit IRT, redo item selection, or retrain GAMs.

## cli

Preferred product commands:

```bash
benchiq calibrate --responses responses_long.csv --config benchiq.json --out out --run-id fit-001
benchiq predict --bundle out/fit-001 --responses reduced_responses_long.csv --out out --run-id pred-001
```

`predict --bundle` accepts either the calibration run root, the nested `calibration_bundle/`
directory, or the bundle `manifest.json`.

Backward-compatible full pipeline command:

```bash
benchiq run --responses responses_long.csv --config benchiq.json --out out --run-id full-001
```

Use `run` when you want the historical end-to-end run root, including downstream redundancy
analysis. Use `calibrate` and `predict` when the goal is "fit once, score later."
