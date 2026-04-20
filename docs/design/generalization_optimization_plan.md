# generalization optimization plan

## purpose

This pass asks a narrower and more product-relevant question than the earlier parity work:

- we are not optimizing for external benchmark parity
- we are optimizing for held-out score-reconstruction quality, reproducibility, and deployment
  usefulness

The current open question is:

- does the current winner generalize well enough across multiple bundle regimes to become a
  promoted product path instead of a one-bundle recommendation note?

Status after completion on 2026-04-18:

- decision: `B`
- outcome: promote the winner to a first-class recommended profile, but do not replace the locked
  psychometric defaults
- report bundle:
  [`reports/generalization_optimization/summary.md`](/Users/jeremykalfus/CodingProjects/BenchIQ/reports/generalization_optimization/summary.md)

Follow-up on 2026-04-19:

- the repo runtime default was later promoted to `reconstruction_first`
- the psychometric baseline remains available explicitly as `psychometric_default`
- a later preprocessing follow-up then tightened the live `reconstruction_first` runtime default
  with a light `drop_low_tail_models_quantile=0.002` trim while keeping
  `deterministic_info`; see
  [`reports/preprocessing_variation_followup/summary.md`](/Users/jeremykalfus/CodingProjects/BenchIQ/reports/preprocessing_variation_followup/summary.md)

## current winner

The current best evidence-backed stack coming into this pass is:

- preprocessing profile: `reconstruction_relaxed`
- preselection method: `deterministic_info`

That winner came from:

- the earlier compact validation fixture
- one stronger real-data confirmation bundle
- compact auxiliary head checks

This new pass is explicitly about validating that winner more broadly before promotion.

Historical note:

- this section describes the winner coming into the generalization pass
- the actual current runtime default is now the later follow-up variant under
  `reconstruction_first`, which adds the light `0.002` low-tail trim on top of the relaxed
  preprocessing settings

## candidate comparisons

The comparison set should stay small and decision-oriented.

Core comparisons:

- current default path: current psychometric-style preprocessing plus current baseline selection path
- current winner: `reconstruction_relaxed` plus `deterministic_info`
- winner with the legacy selection path: `reconstruction_relaxed` plus `random_cv`
- default preprocessing with the less-stochastic selection path: current psychometric defaults plus
  `deterministic_info`

Plausible challenger:

- `minimal_cleaning` plus `deterministic_info`

Additional model-head checks stay targeted:

- keep GAM as the reconstruction baseline
- keep Elastic Net and XGBoost as the existing comparators when useful
- evaluate whether LightGBM is worth adding now
- evaluate whether TabPFN is worth adding now

## bundles to test

This pass uses more than one bundle family and treats them as answering different questions.

Required bundle set:

- compact validation fixture
  - source: `tests/data/compact_validation/responses_long.csv`
  - role: fast fairness check and continuity with the earlier saved bundle
- stronger real-data confirmation bundle
  - source: `out/release_bundle_source/release_default_subset_responses_long.parquet`
  - role: highest-signal real-data confirmation path
- additional generated non-compact bundle family
  - role: stress generalization beyond the one saved real-data bundle
  - include a dense-overlap regime
  - include a sparse-overlap or skip-heavy regime when practical

The generated family should be saved as inspectable artifacts under the new report root so the
evidence is reproducible.

## fairness rules

All core winner-vs-baseline comparisons keep these fixed unless a report section explicitly says
otherwise:

- same split policy
- same train / validation / test fractions
- same reconstruction-head policy
- same size-dependent stage budgets within a dataset
- same evaluation seeds where stochasticity remains
- same artifact and summary conventions

The only intended moving parts in the core matrix are:

- preprocessing profile
- stage-04 preselection method

## metrics

Primary metric:

- held-out RMSE

Secondary comparison metrics:

- MAE
- Pearson
- Spearman
- seed RMSE spread
- stage-06 final-set stability
- runtime
- benchmark coverage
- joint-availability rate
- refusal and skip rates
- deployment readiness

Deployment readiness means:

- calibrate once
- score future reduced responses without retraining
- preserve deterministic outputs where claimed
- avoid hidden prediction holes or unstable artifacts

## promotion criteria

The winner should only be promoted if the evidence clears all of these bars:

- it beats the current default path on the stronger real-data bundle
- it is also competitive on at least one additional non-compact bundle regime, not just the single
  saved real-data confirmation bundle
- it does not win only by refusing or skipping harder benchmarks
- it preserves or improves deployment usefulness
- when quality is close, it earns promotion through cleaner reproducibility, stability, simplicity,
  or deployment behavior

The explicit decision space is:

- A. keep current defaults and document the winner only as an optional profile
- B. promote the winner to a first-class recommended profile
- C. promote the winner into the actual default config
- D. do not promote it because it fails to generalize

## artifacts

This pass should write its outputs to:

- `reports/generalization_optimization/`
- `reports/deployment_validation/`

The final report bundle should include machine-readable tables, markdown summaries, and appendix-safe
plots so the promotion decision is traceable to saved evidence.
