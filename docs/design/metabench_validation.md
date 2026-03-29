# metabench Validation Mode

BenchIQ ships a strict metabench-validation harness. This exists to validate the methodology against a public reference case. It does not redefine BenchIQ as a metabench-only product.

## Two Validation Paths

### 1. Reduced bundled fixture

This is the regression path used in tests and quick local checks.

Command:

```bash
benchiq metabench run --out out/metabench_docs_example
```

Default inputs:

- fixture: [`tests/data/metabench_validation/responses_long.csv`](../../tests/data/metabench_validation/responses_long.csv)
- expected metrics: [`tests/regression/expected/metabench_metrics.json`](../../tests/regression/expected/metabench_metrics.json)
- default run id: `metabench-validation`

This path is strict about:

- artifact existence
- preprocessing structure
- split structure
- subsampling structure
- feature presence
- reconstruction tolerances

### 2. Full manual profile

This is the manual profile for local real-data validation work.

Profile reference:

- [`docs/design/metabench_validation_full_profile.json`](metabench_validation_full_profile.json)

The full profile documents:

- frozen public snapshot source and hashes
- published kept-item counts used in the parity-focused comparison
- validation-only notes about the real-data harness
- acceptance thresholds for the reviewer comparison

The built-in CLI profile can be invoked with:

```bash
benchiq metabench run --profile full --out out/metabench_full_manual
```

The current real-data reviewer pass uses a dedicated reproducible script instead of a pure CLI rerun because the frozen public release path requires parity-specific export and comparison logic around public `.rds` artifacts.

## Real-Data Reviewer Bundle

Frozen public snapshot:

- primary source: Zenodo paper data
- DOI: `10.5281/zenodo.12819251`
- record id: `12819251`
- archive SHA256: `24de1a3f387ee3787c163981e8ea6bb441f625e4f173158cc6a4316d06f8283e`

Current reviewer bundle:

- comparison table: [`reports/metabench_real_data_comparison.csv`](../../reports/metabench_real_data_comparison.csv)
- comparison summary: [`reports/metabench_real_data_comparison.md`](../../reports/metabench_real_data_comparison.md)
- detailed notes: [`reports/metabench_real_data_notes.md`](../../reports/metabench_real_data_notes.md)
- rerun script: [`scripts/run_metabench_real_data_comparison.py`](../../scripts/run_metabench_real_data_comparison.py)

Current status, stated plainly:

> BenchIQ has real-data evidence of methodological alignment on a frozen public metabench snapshot, but the current Python-first path does not yet achieve acceptance-grade metabench parity under the current tolerance rule.

## What `benchiq metabench run` Writes

The command writes a normal BenchIQ run directory plus validation reports.

Important outputs:

- `manifest.json`
- `artifacts/00_canonical/...`
- `artifacts/09_reconstruct/...`
- `artifacts/10_redundancy/...`
- `reports/metabench_validation_report.json`
- `reports/metabench_validation_summary.md`

The validation report includes:

- profile and fixture metadata
- expected metrics source
- artifact checks
- metric checks and tolerances
- pass/fail verdict
- warnings
- run-root location

## Why the Real-Data Comparison Is Not Yet a Pass

The current Python-first path still differs from the published R stack in several places:

- `girth` instead of `mirt`
- `pyGAM` instead of `mgcv`
- the frozen public release does not expose the original final selected item identities directly
- the parity harness therefore still has to reconstruct final selections inside BenchIQ

Those notes are documented in [`reports/metabench_real_data_notes.md`](../../reports/metabench_real_data_notes.md). BenchIQ does not claim exact bit-for-bit parity.

## Reproducible Commands

Reduced fixture regression:

```bash
benchiq metabench run --out out/metabench_docs_example
```

Frozen real-data comparison bundle:

```bash
.venv/bin/python scripts/run_metabench_real_data_comparison.py \
  --out out/metabench_real_validation \
  --run-id metabench-real-zenodo-12819251-parity
```

This script regenerates:

- `reports/metabench_real_data_comparison.md`
- `reports/metabench_real_data_comparison.csv`
- `reports/metabench_real_data_notes.md`
