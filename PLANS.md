# PLANS.md

## source of truth

- product and methodology: [docs/specs/benchiq_v0_1_spec.md](/Users/jeremykalfus/CodingProjects/BenchIQ/docs/specs/benchiq_v0_1_spec.md)
- repo working rules: [AGENTS.md](/Users/jeremykalfus/CodingProjects/BenchIQ/AGENTS.md)

## execution rules

- implement only after explicit user approval of this plan.
- work ticket-by-ticket in order unless dependencies are updated here first.
- keep changes small and test after each ticket.
- if a ticket fails, stop and report the blocker instead of guessing.
- keep metabench as the validation harness, not the product identity.
- keep every major stage inspectable and disk-backed.

## current repo state

- T01 is complete: package scaffolding, tooling, and canonical commands are in place.
- T02 is complete: config models, schema constants, validation helpers, and the public `benchiq.validate(...)` entrypoint are in place with unit coverage.
- T03 is complete: `load_bundle(...)`, stage-00 canonicalization/artifact writing, manifest helpers, and structured duplicate/non-binary load failures are in place with unit coverage.
- T04 is complete: preprocessing/filtering now has size-aware low-tail trimming, fidelity coverage for point-biserial/default thresholds, and stage-01 artifacts.
- T05 is complete: stage-02 full-score tables, overlap-aware grand scores, and score reporting artifacts are in place with unit coverage.
- T06 is complete: global test splitting, benchmark-local train/val splitting, explicit fallback diagnostics, and stage-03 split artifacts are in place with unit coverage.
- T07 is complete: the reusable `pygam` wrapper, RMSE-based cross-validation harness, and model/cv artifact writers are in place with unit coverage.
- T08 is complete: cross-validated random subsampling now writes per-benchmark preselect subsets, fold-level cv results, and checkpointable progress artifacts with explicit failed-iteration accounting.
- T09 is complete: the benchmark-local `girth` 2PL adapter, retained/dropped item-parameter artifacts, explicit backend-limitation warnings, and stage-05 diagnostic plots are in place with corrected acceptance coverage.
- T10 is complete: Fisher-information grids, final item selection artifacts, selection reports, and expected test-information plots are in place with unit coverage.
- T11 is complete: MAP/EAP theta estimation, theta standard errors, theta reports, and stage-07 theta artifacts are in place with unit coverage.
- T12 is complete: benchmark-specific no-intercept linear predictors, reduced subscores, coefficients, diagnostics, and deterministic ridge-fallback artifacts are in place without starting T13 feature-table assembly.
- T13 is complete: marginal and joint feature tables, explicit joint skip reporting, and stage-08 feature artifacts are in place with unit coverage.
- T14 is complete: marginal and overlap-gated joint GAM reconstruction, prediction/residual artifacts, summary metrics, and explicit joint-skip reporting are in place with unit coverage.
- T15 is complete: benchmark-level theta/score correlations, overlap-gated factor analysis, cross-only compressibility metrics, and stage-10 redundancy artifacts are in place with unit coverage.
- T16 is complete: the deterministic `BenchIQRunner`, stage-level manifest records, partial rerun support, and synthetic full-pipeline integration coverage are in place.
- T17 is complete: the artifact-first `benchiq validate` and `benchiq run` CLI commands, explicit validation failure exits, and CLI integration smoke coverage are in place.
- T18 is complete: `benchiq metabench run`, the reduced metabench validation fixture, toleranced regression expectations, and validation-mode docs are in place.
- Post-T18 real-data validation exists on the frozen primary Zenodo snapshot, but the first reviewer pass did not meet the parity acceptance rule.
- The current pre-T19 work is intentionally limited to a parity-repair pass on that same snapshot: apply the paper's published kept-item counts, add a dedicated grand-mean validation GAM, and regenerate the reviewer bundle honestly.

## completed tickets

- T01 project scaffold and toolchain
- T02 config models, schema constants, and validation report types
- T03 bundle loading, canonicalization, and artifact writing
- T04 preprocessing statistics and metabench-style filters
- T05 full-score tables and overlap-aware grand scores
- T06 model-level splitters and split diagnostics
- T07 GAM backend wrapper and cross-validation harness
- T08 cross-validated subsampling to fixed `k_preselect`
- T09 unidimensional 2PL IRT adapter with `girth`
- T10 Fisher information grids and final item selection
- T11 theta estimation and theta standard errors
- T12 benchmark linear predictors and reduced subscores
- T13 marginal and joint feature table builder
- T14 GAM-based score reconstruction
- T15 benchmark-level redundancy and compressibility analysis
- T16 runner orchestration and stage manifests
- T17 artifact-first cli
- T18 metabench validation mode and regression harness

## ticket order

### T01 project scaffold and toolchain

Depends on: none

Scope:
- create `pyproject.toml`, package skeleton under `src/benchiq/`, test skeleton, and basic docs
- add runtime deps for numpy, pandas, scipy, scikit-learn, pyarrow, pydantic, joblib, matplotlib, click or typer, `pygam`, and `girth`
- add dev tooling for `pytest`, `ruff`, and `build`
- make the canonical commands in `AGENTS.md` real

Acceptance criteria:
- `python -m pip install -e '.[dev]'` succeeds
- `python -c "import benchiq"` succeeds
- `pytest` runs and reports the initial smoke tests
- `ruff check .` runs clean on the scaffold

Blockers:
- dependency conflicts across `pygam`, `girth`, numpy, and scipy
- packaging metadata problems on the target python version

### T02 config models, schema constants, and validation report types

Depends on: T01

Scope:
- implement `BenchIQConfig` and small result/report dataclasses
- define canonical table column names, dtypes, and schema helpers
- codify hard-fail vs warning checks at the schema layer

Acceptance criteria:
- config validation works on minimal valid settings
- schema helpers can validate the required tables and return structured errors
- unit tests cover missing columns, dtype handling, and duplicate policies

Blockers:
- pandas nullable dtype edge cases for binary scores
- config defaults that drift from the spec

### T03 bundle loading, canonicalization, and artifact writing

Depends on: T02

Scope:
- implement `load_bundle(...)`
- support csv and parquet inputs
- canonicalize ids, enforce primary key uniqueness, validate binary scores, and write stage-00 artifacts
- implement manifest helpers, file hashing, parquet/json writers, and canonicalization reports

Acceptance criteria:
- toy fixtures load through the public api
- canonicalization writes `artifacts/00_canonical/*` and `manifest.json`
- duplicate and non-binary input failures are explicit and test-covered

Blockers:
- pyarrow/parquet compatibility issues
- unclear duplicate-resolution edge cases in messy user data

### T04 preprocessing statistics and metabench-style filters

Depends on: T03

Scope:
- compute benchmark-wise item mean, standard deviation, point-biserial discrimination, and coverage stats
- implement low-tail model trimming, low-variance, near-ceiling, near-zero-discrimination, item-coverage, and model-coverage filters
- emit per-benchmark preprocess reports and summary artifacts

Acceptance criteria:
- constructed fixtures produce the expected retained and dropped items
- refusal conditions trigger when post-filter items or models are too small
- per-benchmark preprocess artifacts are written to disk

Blockers:
- correct point-biserial behavior with missing responses
- filtering cascades that remove too much data unexpectedly

### T05 full-score tables and overlap-aware grand scores

Depends on: T04

Scope:
- compute `scores_full` as percent-correct targets per model and benchmark
- compute `scores_grand` only when complete bundle overlap exists
- record coverage-based missingness and overlap warnings

Acceptance criteria:
- known fixtures produce correct percent scores
- grand scores only appear for models meeting overlap requirements
- score artifacts and score report are written

Blockers:
- ambiguity around how partial coverage should be represented in downstream joins

### T06 model-level splitters and split diagnostics

Depends on: T05

Scope:
- implement global test splitting stratified on grand mean score when feasible
- implement per-benchmark train/val splitting within the remaining pool
- write split tables and diagnostics with no item-level leakage

Acceptance criteria:
- splits are model-level only, disjoint, and roughly match configured proportions
- fallback to benchmark-local splitting is explicit when global overlap is too small
- split artifacts and diagnostics are written

Blockers:
- insufficient unique score bins for stable stratification

### T07 GAM backend wrapper and cross-validation harness

Depends on: T06

Scope:
- implement a stable `pygam` wrapper for fit, predict, serialization, and cv scoring
- support rmse-driven smoothing selection on train folds
- provide reusable utilities for later subsampling and reconstruction stages

Acceptance criteria:
- synthetic nonlinear regression test passes with finite rmse
- cv selects a finite hyperparameter set and beats a naive baseline
- gam metadata is serializable into artifacts

Blockers:
- `pygam` api/version drift
- serialization stability across python versions

### T08 cross-validated subsampling to fixed `k_preselect`

Depends on: T07

Scope:
- implement metabench-style random item subsampling with k-fold cv across models
- use reduced subscore as the predictor and minimax validation rmse as the selection rule
- write iteration-level results, checkpointable progress, and chosen preselect subsets

Acceptance criteria:
- selected subsets have exactly `k_preselect` items when feasible
- `cv_results` has the expected row count and metrics
- failed iterations due to missingness are counted and reported explicitly

Blockers:
- runtime blowups on large bundles
- too many sampled subsets becoming invalid after coverage checks

### T09 unidimensional 2PL IRT adapter with `girth`

Depends on: T08

Scope:
- implement the core `girth` backend adapter and unified item-parameter output schema
- `girth` is the first-pass core backend for v0.1, but if it fails metabench validation materially, an optional parity backend may be introduced later without changing the product identity.
- fit 2pl models on benchmark-specific preselected item sets
- artifact backend convergence-status unavailability explicitly when `girth` does not expose it
- capture pathological-parameter diagnostics and dropped-item artifacts
- write stage-05 item-parameter diagnostic plots to disk

Acceptance criteria:
- synthetic irt fixtures recover item parameters with sensible rank correlation
- retained-only `irt_item_params.parquet` is written with a standardized item-parameter schema
- backend convergence-status unavailability is artifacted as a structured warning when the backend does not expose it
- dropped pathological items are written to an explicit artifact with counts/reasons in the fit report
- stage-05 item-parameter diagnostic plot artifacts are written to disk

Blockers:
- `girth` missing-data handling quirks
- non-convergence on sparse or poorly filtered benchmarks

### T10 Fisher information grids and final item selection

Depends on: T09

Scope:
- implement analytic 2pl probability and fisher-information functions
- build theta grids and per-item information tables
- select final benchmark subsets across the theta range to `k_final`

Acceptance criteria:
- information curves are non-negative and peak near item difficulty in sanity tests
- selection returns no duplicates and respects `k_final` when feasible
- final subset, info-grid, and selection-report artifacts are written
- expected test-information plots are written

Blockers:
- uniformly low-information item pools that cannot satisfy the requested budget

### T11 theta estimation and theta standard errors

Depends on: T10

Scope:
- implement map theta estimation and optional eap grid estimation
- compute theta standard errors from test information
- handle all-correct, all-wrong, and missing-heavy reduced response patterns

Acceptance criteria:
- theta outputs are finite on synthetic fixtures
- higher reduced scores imply higher theta on average
- `theta_estimates.parquet` is written for train, val, and test models
- `theta_report.json` and theta distribution plots are written

Blockers:
- numerical underflow in likelihood calculations
- grid bounds that saturate too many response patterns

### T12 benchmark linear predictors and reduced subscores

Depends on: T11

Scope:
- fit no-intercept benchmark-specific linear predictors from reduced item responses
- compute reduced subscores and store coefficients, training diagnostics, and deterministic fallback-to-ridge metadata
- write per-benchmark stage-08 model-output artifacts for later `sub_b` / `lin_b` feature assembly
  without starting T13 feature-table joins

Acceptance criteria:
- per-benchmark stage-08 outputs expose reduced subscores as `sub_b` and linear predictions as
  `lin_b`
- coefficients, training diagnostics, and fallback-to-ridge metadata are written to disk
- synthetic linear fixtures show sensible coefficient recovery
- rank-deficient cases switch to deterministic ridge with explicit logging

Blockers:
- highly collinear reduced items causing unstable ordinary least squares fits

### T13 marginal and joint feature table builder

Depends on: T12

Scope:
- assemble marginal feature tables with theta, theta se, subscore, linear predictor, targets, and splits
- assemble joint feature tables with bundle-wide theta features plus grand summaries when overlap permits
- ensure train-only fitting decisions are not leaked into held-out rows

Acceptance criteria:
- marginal feature tables are always written
- joint feature tables are written only when overlap requirements are met, otherwise a skip reason is artifacted
- tests assert no leakage across train, val, and test paths

Blockers:
- overlap bookkeeping mistakes across benchmarks

### T14 GAM-based score reconstruction

Depends on: T13

Scope:
- fit marginal reconstruction gams per benchmark
- fit joint reconstruction gams when overlap is sufficient
- write prediction tables, residuals, summary metrics, and calibration plots

Acceptance criteria:
- marginal reconstruction beats a train-mean baseline on synthetic fixtures
- joint reconstruction improves over marginal on correlated synthetic bundles when overlap exists
- reconstruction summary artifacts are written for every configured benchmark

Blockers:
- unstable fits with too many spline terms or too much missingness

### T15 benchmark-level redundancy and compressibility analysis

Depends on: T14

Scope:
- compute theta and score correlations with overlap counts
- run benchmark-level factor analysis when overlap is sufficient
- compute cross-only predictability metrics relative to marginal baselines

Acceptance criteria:
- redundancy artifacts and reports are written
- synthetic correlated bundles show monotonic redundancy behavior as shared latent structure increases
- insufficient-overlap cases skip cleanly with explicit reasons

Blockers:
- factor-analysis backend availability
- overlap too small for stable benchmark-level analysis

### T16 runner orchestration and stage manifests

Depends on: T15

Scope:
- implement `BenchIQRunner`, `RunResult`, and the deterministic stage dag
- support stage-level manifests, partial reruns, and stable artifact paths
- aggregate warnings, metrics, and skip reasons into a top-level summary

Acceptance criteria:
- one runner call executes the full pipeline on the synthetic integration fixture
- same seed and same inputs reproduce selected items and summary metrics within tolerance
- `RunResult` exposes artifact loading and summary helpers

Blockers:
- non-determinism introduced by parallel execution or third-party estimators

### T17 artifact-first cli

Depends on: T16

Scope:
- implement `benchiq validate`, `benchiq run`, and shared cli plumbing
- require explicit output locations
- make validation failures exit non-zero with actionable messages

Acceptance criteria:
- cli smoke tests pass on tiny fixtures
- `benchiq validate` writes validation artifacts
- `benchiq run` writes a complete run directory and prints high-level metrics

Blockers:
- packaging entrypoint issues across local environments

### T18 metabench validation mode and regression harness

Depends on: T17

Scope:
- implement `benchiq metabench run`
- add a pinned metabench validation fixture or documented fetch path
- codify toleranced regression expectations for preprocessing, splitting, subsampling structure, and reconstruction outputs

Acceptance criteria:
- metabench validation mode runs reproducibly on the chosen fixture
- regression tests assert artifact existence and metric tolerances without demanding exact r parity
- docs explain how metabench validation differs from generic bundle mode

Blockers:
- dataset licensing, size, or availability constraints
- performance limits for full metabench runs in ci

### Post-T18 real-data metabench validation pass

Depends on: T18

Scope:
- freeze one public metabench paper snapshot with exact hashes
- add one full-profile manual validation config separate from the reduced ci fixture
- run the strongest feasible frozen-snapshot comparison path, using the public release-default subset view when the raw snapshot is too large for an in-session full python-only pass
- write a reviewer-ready comparison bundle with explicit pass/fail criteria and caveats

Acceptance criteria:
- the exact public source, snapshot, and hashes are recorded on disk
- one reproducible command or script reruns the same frozen-source comparison
- any fallback from the raw full snapshot to a public release-artifact path is explicit and justified
- comparison outputs include benchmark rmse deltas, kept item counts, tolerance-band checks, and an honest overall verdict

Blockers:
- local runtime or memory limits on the full public snapshot
- the frozen paper snapshot exposes public `*.rds` release artifacts more tractably than the full ~153M-row raw csv bundle for this focused reviewer pass
- methodological gaps versus the original r stack that BenchIQ does not yet implement

### T19 docs, examples, and reproducibility pass

Depends on: T18

Scope:
- finish `README.md`, `docs/design/schema.md`, `docs/design/v0_1_scope.md`, `docs/design/metabench_validation.md`, and `docs/cli.md`
- document artifact layouts, failure modes, config examples, and contributor workflow
- add a tiny synthetic example bundle and an end-to-end example command sequence

Acceptance criteria:
- a new contributor can install the package, run the synthetic example, and understand the emitted artifacts
- docs clearly separate v0.1 scope from non-goals
- contributor guidance matches `AGENTS.md` and the implemented commands

Blockers:
- doc drift if implementation details changed during earlier tickets

## stage-level guardrails

- do not broaden the product into a general psychometrics framework.
- do not add dashboards, guis, cat flows, multimodal support, or registry features.
- do not silently skip joint or redundancy stages; either run them or write the skip reason.
- after every ticket: summarize changes, list files touched, report exact test results, and state whether acceptance criteria passed.
