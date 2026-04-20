# BenchIQ CLI

BenchIQ v0.1 ships an artifact-first CLI with five stable public entrypoints:

- `benchiq validate`
- `benchiq calibrate`
- `benchiq predict`
- `benchiq run`
- `benchiq metabench run`

All commands require an explicit output directory.

The preferred reusable workflow is:

- `benchiq calibrate` to fit and publish a reusable `calibration_bundle/`
- `benchiq predict` to score new reduced responses later without retraining

The default product config now follows `reconstruction_first`, backed by the saved multi-bundle
generalization pass in
[`reports/generalization_optimization/summary.md`](../reports/generalization_optimization/summary.md).
The broader real-data preprocessing follow-up in
[`reports/preprocessing_variation_followup/summary.md`](../reports/preprocessing_variation_followup/summary.md)
further tightened that runtime default with a light `drop_low_tail_models_quantile=0.002` trim.
The spec-aligned psychometric baseline remains available explicitly as `psychometric_default`.

`benchiq run` remains the stable full end-to-end path when you want one inspectable local run root
that also includes the downstream redundancy analysis.

Supported invocation forms after install:

```bash
benchiq ...
python -m benchiq.cli ...
```

## Install

```bash
python -m pip install -e '.[dev]'
```

Core `validate` / `calibrate` / `predict` / `run` workflows do not depend on XGBoost. XGBoost is
kept as an optional experiment dependency for the reconstruction-head comparison harness, and is
also exposed through the `.[experiments]` extra.

## `benchiq validate`

Use this to canonicalize inputs, run schema validation, and write validation artifacts without executing the full pipeline.

Example:

```bash
benchiq validate \
  --responses tests/data/tiny_example/responses_long.csv \
  --config tests/data/tiny_example/config.json \
  --out out/tiny_example_docs
```

Writes to:

- `out/tiny_example_docs/validate/`

Important outputs:

- `manifest.json`
- `config_resolved.json`
- `artifacts/00_canonical/responses_long.parquet`
- `artifacts/00_canonical/items.parquet`
- `artifacts/00_canonical/models.parquet`
- `artifacts/01_preprocess/...`
- `reports/validation_report.json`
- `reports/validation_summary.md`

Failure behavior:

- exits non-zero on hard validation failure
- still writes failure artifacts under `OUT/validate/`
- prints an actionable summary to stderr

## `benchiq run`

Use this to execute the full deterministic pipeline on a generic benchmark bundle.

`benchiq run` remains useful for a full local inspection pass, but the preferred reusable product
split is:

- `benchiq calibrate` to fit and publish a bundle
- `benchiq predict` to score new reduced responses later

The stage-10 redundancy outputs from `run` are secondary analysis artifacts rather than the main
product path.

Example:

```bash
benchiq run \
  --responses tests/data/tiny_example/responses_long.csv \
  --config tests/data/tiny_example/config.json \
  --out out/tiny_example_docs \
  --run-id tiny-example
```

Writes to:

- `out/tiny_example_docs/tiny-example/`

Important outputs:

- `manifest.json`
- `config_resolved.json`
- `reports/run_summary.md`
- `reports/metrics.json`
- `reports/warnings.md`
- `artifacts/00_canonical/...`
- `artifacts/06_select/per_benchmark/<benchmark_id>/subset_final.parquet`
- `artifacts/07_theta/theta_estimates.parquet`
- `artifacts/08_features/features_marginal.parquet`
- `artifacts/08_features/features_joint.parquet`
- `artifacts/09_reconstruct/reconstruction_summary.parquet`
- `artifacts/10_redundancy/redundancy_report.json`

Console behavior:

- prints the run location
- prints selected-item counts by benchmark
- prints marginal RMSE by benchmark
- exits non-zero on schema failure

## `benchiq calibrate`

Use this to fit the reusable calibration stack and publish a saved `calibration_bundle/`.

The run root stays fully inspectable. The nested `calibration_bundle/` directory is the reusable
handoff artifact that `predict` validates later.

Example:

```bash
benchiq calibrate \
  --responses tests/data/tiny_example/responses_long.csv \
  --config tests/data/tiny_example/config.json \
  --out out/tiny_example_docs \
  --run-id tiny-calibration
```

Writes to:

- `out/tiny_example_docs/tiny-calibration/`

Important outputs:

- `manifest.json`
- `config_resolved.json`
- `calibration_bundle/manifest.json`
- `calibration_bundle/config_resolved.json`
- `calibration_bundle/reconstruction_summary.parquet`
- `calibration_bundle/per_benchmark/<benchmark_id>/subset_final.parquet`
- `calibration_bundle/per_benchmark/<benchmark_id>/irt_item_params.parquet`
- `calibration_bundle/per_benchmark/<benchmark_id>/theta_scoring_metadata.json`
- `calibration_bundle/per_benchmark/<benchmark_id>/linear_predictor_coefficients.parquet`
- `calibration_bundle/per_benchmark/<benchmark_id>/reconstruction/<model_type>/gam_model.pkl`
  when that reconstruction head was successfully fit

Console behavior:

- prints the run location
- prints the calibration bundle location
- prints selected-item counts by benchmark
- prints held-out marginal RMSE by benchmark

## `benchiq predict`

Use this to load a saved calibration bundle and score new reduced responses without retraining.

`--bundle` accepts any of the following:

- a calibration run root that contains `calibration_bundle/`
- the `calibration_bundle/` directory itself
- the bundle `manifest.json`

Example:

```bash
benchiq predict \
  --bundle out/tiny_example_docs/tiny-calibration/calibration_bundle \
  --responses tests/data/tiny_example/responses_long.csv \
  --out out/tiny_example_docs \
  --run-id tiny-predict
```

Writes to:

- `out/tiny_example_docs/tiny-predict/`

Important outputs:

- `manifest.json`
- `artifacts/00_canonical/...`
- `artifacts/01_predict/theta_estimates.parquet`
- `artifacts/01_predict/features_marginal.parquet`
- `artifacts/01_predict/features_joint.parquet`
- `artifacts/01_predict/predictions.parquet`
- `artifacts/01_predict/predictions_best_available.parquet`
- `artifacts/01_predict/prediction_report.json`

Behavior:

- fails clearly if the calibration bundle is missing required fitted artifacts
- does not retrain any model
- uses the saved GAM and linear predictor artifacts from calibration time
- requires the selected calibrated items for each benchmark to be present in the response file
- ignores extra non-selected items and extra non-calibrated benchmarks with explicit warnings

## `benchiq metabench run`

Use this only for methodological validation against the metabench reference harness.

Reduced bundled fixture:

```bash
benchiq metabench run --out out/metabench_docs_example
```

Manual full profile:

```bash
benchiq metabench run --profile full --out out/metabench_full_manual
```

Default run root:

- `OUT/metabench-validation/`

Important outputs:

- full normal BenchIQ run directory
- `reports/metabench_validation_report.json`
- `reports/metabench_validation_summary.md`

This mode is strict about validation artifacts and tolerances. It is not the same as generic bundle mode.

## Config Files

`--config` accepts `.json` or `.toml`.

Two supported shapes:

1. plain BenchIQ config object
2. nested CLI config with top-level keys:
   - `config`
   - `stage_options`

The tiny example uses the nested form so the docs can show a short, fast full-pipeline run.

When `--config` is omitted, the CLI uses the default reconstruction-first runtime profile:
light low-tail trimming, relaxed preprocessing thresholds, and `deterministic_info` stage-04
preselection.

Explicit default-profile nested config example:

```json
{
  "config": {
    "drop_low_tail_models_quantile": 0.002,
    "min_item_sd": 0.0,
    "max_item_mean": 0.99,
    "min_abs_point_biserial": 0.0,
    "min_item_coverage": 0.7,
    "random_seed": 7
  },
  "stage_options": {
    "04_subsample": {
      "method": "deterministic_info"
    }
  }
}
```

This matches the first-class default profile exposed in Python as
`benchiq.build_reconstruction_first_profile(...)`. It matches the actual runtime defaults and is
useful when you want to pin that profile explicitly in a saved config file.

## Artifact Layout

Common run-root structure:

```text
RUN_ROOT/
  manifest.json
  config_resolved.json
  calibration_bundle/            # present for calibrate runs
  artifacts/
    00_canonical/
    01_predict/                  # present for predict runs
    01_preprocess/
    02_scores/
    03_splits/
    04_subsample/
    05_irt/
    06_select/
    07_theta/
    08_features/
    09_reconstruct/
    10_redundancy/
  reports/
```

Interpretation:

- `manifest.json` links stages, sources, hashes, and report paths
- each stage directory contains inspectable parquet/json artifacts
- `reports/` holds top-level summaries and validation output
- plot files are written under the stage that produced them
- skip reasons and warnings are written as stage reports rather than hidden in logs
- `calibration_bundle/` and `artifacts/01_predict/` are the primary reusable calibration /
  deployment artifacts
- `artifacts/10_redundancy/` is a secondary analysis path that stays attached to the historical
  full `run` workflow

Linear predictor artifacts live under:

- `artifacts/08_features/per_benchmark/<benchmark_id>/model_outputs.parquet`
- `artifacts/08_features/per_benchmark/<benchmark_id>/coefficients.parquet`
- `artifacts/08_features/per_benchmark/<benchmark_id>/linear_predictor_report.json`

## Quick Inspection Commands

```bash
python -m json.tool out/tiny_example_docs/tiny-example/manifest.json | head -40
python -m json.tool out/tiny_example_docs/tiny-example/reports/metrics.json | head -80
ls out/tiny_example_docs/tiny-example/artifacts/09_reconstruct/per_benchmark/b1
ls out/metabench_docs_example/metabench-validation/reports
```
