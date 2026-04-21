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

## Run the Saved Product Evaluations

Primary saved experiment and optimization surfaces:

- [`reports/preprocessing_optimization/summary.md`](../reports/preprocessing_optimization/summary.md)
- [`reports/generalization_optimization/summary.md`](../reports/generalization_optimization/summary.md)
- [`reports/preprocessing_variation_followup/summary.md`](../reports/preprocessing_variation_followup/summary.md)
- [`reports/deployment_validation/summary.md`](../reports/deployment_validation/summary.md)
- [`reports/portfolio_standing/summary.md`](../reports/portfolio_standing/summary.md)
- [`reports/portfolio_optimization_cycles/best_so_far.md`](../reports/portfolio_optimization_cycles/best_so_far.md)

Rebuild them with:

```bash
.venv/bin/python scripts/run_preprocessing_optimization.py
.venv/bin/python scripts/run_generalization_optimization.py
.venv/bin/python scripts/run_preprocessing_variation_followup.py
.venv/bin/python scripts/run_calibration_deployment_walkthrough.py
.venv/bin/python scripts/run_portfolio_standing.py
```

`reports/portfolio_standing/` is the original narrowed public-portfolio baseline snapshot.
`reports/portfolio_optimization_cycles/` is the later iterative improvement family, and
`best_so_far.md` is the current summary entrypoint for that track.

## Contribution Boundaries

Good contributions for v0.1:

- bug fixes
- test coverage
- inspectable artifact/report improvements
- docs and reproducibility improvements
- product-evaluation maintenance

Avoid widening scope without an approved ticket:

- new modeling families
- dashboard/platform work
- silent fallback behavior
- test-set-driven selection logic

## Current Known Limitations

- binary item scores only
- BenchIQ is intended for many-model bundles, not single-model one-off evaluation
