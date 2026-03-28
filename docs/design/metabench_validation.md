# metabench validation mode

BenchIQ is a generic benchmark-bundle distillation tool. Metabench is the
methodological validation harness for v0.1, not the product identity.

## command surface

Use the validation harness through:

```bash
benchiq metabench run --out /path/to/out
```

That default command uses the bundled reduced fixture at
`tests/data/metabench_validation/responses_long.csv` and the pinned tolerance
file at `tests/regression/expected/metabench_metrics.json`.

## reduced fixture vs full snapshot

The repository does not ship the full metabench snapshot in CI because of size
and runtime cost. v0.1 therefore provides two validation profiles:

- `reduced`:
  - the default bundled fixture for CI and contributor smoke validation
  - keeps metabench-style preprocessing thresholds
  - keeps the global grand-mean-stratified split logic and 5-fold subsampling
  - scales sample-size-dependent settings down so the fixture remains runnable
- `full`:
  - intended for a manually supplied local snapshot
  - keeps the larger metabench-style budgets such as `k_preselect=350`
  - is documented for local use, not exercised in CI

Run the full profile with a local snapshot like this:

```bash
benchiq metabench run \
  --profile full \
  --responses /path/to/metabench/responses_long.csv \
  --items /path/to/metabench/items.csv \
  --models /path/to/metabench/models.csv \
  --out /path/to/out
```

## what the validation report checks

The metabench validation report is written to
`reports/metabench_validation_report.json` inside the run directory. It records:

- fixture source and profile
- resolved strict validation preset
- required artifact existence checks
- toleranced regression checks for:
  - preprocessing retained-item counts
  - global test split enablement and counts
  - subsampling structure (`k_preselect` outcome and cv row counts)
  - reconstruction RMSE by benchmark
  - presence of metabench-style reconstruction features:
    `theta_b`, `theta_se_b`, `sub_b`, `lin_b`, `grand_sub`, and `grand_lin`
- run warnings and final pass/fail status

The reduced fixture expects comparable behavior and metric scale, not exact R
parity against metabench’s original code.
