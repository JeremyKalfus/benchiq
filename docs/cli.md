# BenchIQ CLI

BenchIQ v0.1 ships an artifact-first CLI with three entrypoints:

- `benchiq validate`
- `benchiq run`
- `benchiq metabench run`

All commands require an explicit output directory.

Supported invocation forms after install:

```bash
benchiq ...
python -m benchiq.cli ...
```

## Install

```bash
python -m pip install -e '.[dev]'
```

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

## Artifact Layout

Common run-root structure:

```text
RUN_ROOT/
  manifest.json
  config_resolved.json
  artifacts/
    00_canonical/
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
