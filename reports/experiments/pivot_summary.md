# post-pivot experiment summary

This note summarizes the compact saved experiment bundle produced for the product-performance pivot.

## reconstruction heads

Source bundle:

- [`reports/experiments/reconstruction_heads/summary.md`](./reconstruction_heads/summary.md)

Observed result on the compact validation fixture:

- GAM remained the best head for both marginal and joint reconstruction
- Elastic Net was cheaper but worse on held-out RMSE
- XGBoost did not beat GAM on this fixture and was clearly worse on marginal reconstruction

This XGBoost result comes from the optional reconstruction-head comparison harness. Core
`calibrate` / `predict` / `run` workflows do not depend on XGBoost.

## preselection methods

Source bundle:

- [`reports/selection_comparison/summary.md`](../selection_comparison/summary.md)

Observed result on the compact validation fixture:

- `deterministic_info` slightly beat `random_cv` on mean best-available held-out RMSE
- `deterministic_info` had perfect cross-seed item stability on this fixture
- `deterministic_info` was much faster than the random-CV baseline here

## preprocessing optimization

Source bundle:

- [`reports/preprocessing_optimization/summary.md`](../preprocessing_optimization/summary.md)

Observed result across the compact search bundle and the real-data confirmation bundle:

- the compact fixture did not discriminate among preprocessing profiles; every profile tied on
  held-out RMSE and retained-item counts there
- the real-data confirmation bundle did discriminate strongly: the psychometric defaults dropped
  enough items to skip half the benchmarks at `k_preselect = 350`
- `reconstruction_relaxed` matched `minimal_cleaning` on held-out RMSE, but won the tie on runtime
  and cleaner guardrails
- the saved recommendation is to keep the spec-aligned defaults as the documented baseline while
  offering the reconstruction-first relaxed profile as the evidence-backed override

## irt parity

Source bundle:

- [`reports/irt_r_baseline/irt_r_baseline_summary.md`](../irt_r_baseline/irt_r_baseline_summary.md)

Observed result in this environment:

- the parity harness ran and wrote its outputs
- the actual R comparison was skipped because `mirt` is not installed locally

## calibration / deployment

Source bundle:

- [`reports/calibration_deployment/summary.md`](../calibration_deployment/summary.md)

Observed result on held-out reduced responses:

- deployment reproduced calibration-time best-available predictions exactly on the saved walkthrough
- max absolute prediction delta was `0.0`

## honest takeaway

The compact evidence bundle supports the product pivot:

- the calibration / deployment split is real and reusable
- the deterministic preselection baseline is promising enough to keep investigating
- the psychometric preprocessing defaults are not the best reconstruction-first choice on the
  saved real-data bundle
- GAM still deserves to stay the default reconstruction head for now
- R-baseline IRT parity infrastructure exists, but the actual parity verdict still depends on an
  environment with `mirt`
