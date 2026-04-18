# Contributing to BenchIQ

BenchIQ is still intentionally narrow. Contributions should preserve the v0.1 contract and keep the repo inspectable for outside verifiers.

## First Read

Before changing code, read:

- [`AGENTS.md`](../AGENTS.md)
- [`PLANS.md`](../PLANS.md)
- [`docs/specs/benchiq_v0_1_spec.md`](specs/benchiq_v0_1_spec.md)

Helpful supporting docs:

- [`README.md`](../README.md)
- [`docs/design/v0_1_scope.md`](design/v0_1_scope.md)
- [`docs/design/schema.md`](design/schema.md)
- [`docs/design/metabench_validation.md`](design/metabench_validation.md)
- [`docs/cli.md`](cli.md)

## Setup

```bash
python -m pip install -e '.[dev]'
```

Supported CLI entrypoints after install:

```bash
benchiq ...
python -m benchiq.cli ...
```

Core `validate` / `calibrate` / `predict` / `run` workflows do not require XGBoost. XGBoost is
kept as an experiment dependency for the reconstruction-head comparison harness and is available in
the contributor install above.

## Canonical Commands

```bash
ruff check .
ruff format --check .
pytest
python -m build
python -m compileall src tests scripts
```

## Run the Tiny Example

Validate:

```bash
benchiq validate \
  --responses tests/data/tiny_example/responses_long.csv \
  --config tests/data/tiny_example/config.json \
  --out out/tiny_example_docs
```

Full run:

```bash
benchiq run \
  --responses tests/data/tiny_example/responses_long.csv \
  --config tests/data/tiny_example/config.json \
  --out out/tiny_example_docs \
  --run-id tiny-example
```

Calibrate once:

```bash
benchiq calibrate \
  --responses tests/data/tiny_example/responses_long.csv \
  --config tests/data/tiny_example/config.json \
  --out out/tiny_example_docs \
  --run-id tiny-calibration
```

Predict later from the saved bundle:

```bash
benchiq predict \
  --bundle out/tiny_example_docs/tiny-calibration/calibration_bundle \
  --responses tests/data/tiny_example/responses_long.csv \
  --out out/tiny_example_docs \
  --run-id tiny-predict
```

Inspect:

```bash
python -m json.tool out/tiny_example_docs/tiny-example/manifest.json | head -40
python -m json.tool out/tiny_example_docs/tiny-example/reports/metrics.json | head -80
```

## Run the metabench Validation Harness

Reduced regression fixture:

```bash
benchiq metabench run --out out/metabench_docs_example
```

Manual full profile reference:

- [`docs/design/metabench_validation_full_profile.json`](design/metabench_validation_full_profile.json)

Frozen real-data reviewer bundle:

```bash
.venv/bin/python scripts/run_metabench_real_data_comparison.py \
  --out out/metabench_real_validation \
  --run-id metabench-real-zenodo-12819251-parity
```

Outputs of interest:

- [`reports/metabench_real_data_comparison.md`](../reports/metabench_real_data_comparison.md)
- [`reports/metabench_real_data_comparison.csv`](../reports/metabench_real_data_comparison.csv)
- [`reports/metabench_real_data_notes.md`](../reports/metabench_real_data_notes.md)

## Contribution Boundaries

Good contributions for v0.1:

- bug fixes
- test coverage
- inspectable artifact/report improvements
- docs and reproducibility improvements
- validation harness maintenance

Avoid widening scope without an approved ticket:

- new modeling families
- dashboard/platform work
- metabench-only product framing
- silent fallback behavior
- test-set-driven selection logic

## Current Known Limitations

- binary item scores only
- Python-first path implements the methodology end to end but does not yet achieve acceptance-grade metabench parity
- the strongest public real-data parity comparison still relies on a validation-only reviewer harness around frozen public release artifacts
- BenchIQ is intended for many-model bundles, not single-model one-off evaluation
