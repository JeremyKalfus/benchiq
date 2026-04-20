# BenchIQ v0.1 Current Spec

This document is the current product and methodology source of truth for the
BenchIQ v0.1 checkout in this repository.

BenchIQ is an artifact-first Python tool for benchmark-bundle distillation,
calibration, deployment-time scoring, score reconstruction, and secondary
benchmark-overlap analysis from item-level model response data.

BenchIQ operates on user-provided benchmark bundles. It is not a hosted
platform, not a benchmark registry, and not a general psychometrics toolkit.

## 1. Product Contract

BenchIQ v0.1 is built around four public workflows:

- `validate`: canonicalize inputs, run schema and preprocessing checks, and
  write validation artifacts without fitting the full stack.
- `calibrate`: fit the reusable reduction, IRT, linear-feature, and
  reconstruction stack once and export a reusable `calibration_bundle/`.
- `predict`: load a saved calibration bundle and score new reduced-response
  inputs without retraining.
- `run`: execute the full local pipeline end to end, including the secondary
  redundancy analysis stage.

The main product objective is to recover held-out full-benchmark percent scores
from a smaller retained item subset while keeping every major stage inspectable
on disk.

The preferred reusable product path is:

1. `validate`
2. `calibrate`
3. `predict`

`run` remains the best single-run local analysis path when the user wants one
inspectable run root with the downstream redundancy outputs included.

## 2. In Scope

BenchIQ v0.1 includes:

- user-chosen benchmark bundles, not a fixed benchmark suite
- binary item-response ingestion in canonical long format
- optional derivation of `items` and `models` tables when omitted
- benchmark-wise preprocessing and refusal logic
- model-level train, validation, and test splits
- stage-04 item preselection with deterministic and random-CV methods
- benchmark-specific unidimensional 2PL IRT
- Fisher-information final item selection
- theta estimation with uncertainty
- reduced-score, linear-predictor, marginal, and joint feature construction
- GAM-based score reconstruction to full-benchmark percent scores
- reusable calibration-bundle export
- deployment-time prediction without retraining
- benchmark-level redundancy and compressibility analysis as a secondary stage
- inspectable artifacts, manifests, warnings, refusal reasons, and skip reasons

## 3. Explicit Non-Goals

BenchIQ v0.1 does not include:

- dashboards, hosted services, or GUI product surfaces
- automatic benchmark collection or leaderboard scraping
- arbitrary IRT-family expansion or a general survey-psychometrics framework
- CAT, item banking, or item generation
- item-level multidimensional latent models
- multimodal benchmark support
- silent fallback behavior

## 4. Public Surface

### Python API

The package root exposes the current high-level workflows directly:

- `load_bundle(...)`
- `validate(...)`
- `calibrate(...)`
- `predict(...)`
- `deploy(...)` as an alias for `predict(...)`
- `run(...)`
- `load_calibration_bundle(...)`

The package root also exposes the profile helpers:

- `build_reconstruction_first_profile(...)`
- `build_psychometric_default_profile(...)`
- `load_profile(...)`

### CLI

The supported CLI entrypoints are:

- `benchiq validate`
- `benchiq calibrate`
- `benchiq predict`
- `benchiq run`

After installation, both of the following invocation forms are supported:

- `benchiq ...`
- `python -m benchiq.cli ...`

### Workflow Meanings

`validate`
: writes canonical stage-00 artifacts plus schema and preprocessing validation
  outputs under `OUT/validate/`

`calibrate`
: executes stages `00` through `09`, then exports a reusable
  `calibration_bundle/`

`predict`
: loads a saved calibration bundle, scores new reduced-response inputs, writes
  `artifacts/01_predict/`, and never retrains fitted models

`run`
: executes the full stage DAG through redundancy analysis and writes the
  complete local run root

## 5. Canonical Inputs

BenchIQ uses one canonical internal source-of-truth table:

- `responses_long`

### Required Table: `responses_long`

Required columns:

- `model_id`
- `benchmark_id`
- `item_id`
- `score`

Rules:

- `score` must be binary in `{0, 1}` with nullable missing values allowed
- `(model_id, benchmark_id, item_id)` must be unique after canonicalization
- input files must be `.csv` or `.parquet`
- duplicate handling is explicit and controlled by `duplicate_policy`
- the default duplicate policy is `error`

### Optional Table: `items`

If provided, `items` must include:

- `benchmark_id`
- `item_id`

If omitted, BenchIQ derives it from `responses_long`.

### Optional Table: `models`

If provided, `models` must include:

- `model_id`

If omitted, BenchIQ derives it from `responses_long`.

## 6. Output Contract

Every run root is artifact-first and manifest-backed.

Common run-root outputs:

- `manifest.json`
- `config_resolved.json`
- `artifacts/`
- `reports/`

### Validation Output

`validate` writes:

- `OUT/validate/manifest.json`
- `OUT/validate/config_resolved.json`
- `OUT/validate/artifacts/00_canonical/...`
- `OUT/validate/artifacts/01_preprocess/...`
- `OUT/validate/reports/validation_report.json`
- `OUT/validate/reports/validation_summary.md`

### Calibration Output

`calibrate` writes a normal run root plus:

- `RUN_ROOT/calibration_bundle/manifest.json`
- `RUN_ROOT/calibration_bundle/config_resolved.json`
- `RUN_ROOT/calibration_bundle/reconstruction_summary.parquet`
- `RUN_ROOT/calibration_bundle/reconstruction_report.json`
- `RUN_ROOT/calibration_bundle/per_benchmark/<benchmark_id>/...`

Each per-benchmark calibration directory contains:

- `subset_final.parquet`
- `selection_report.json`
- `irt_item_params.parquet`
- `irt_fit_report.json`
- `theta_scoring_metadata.json`
- `linear_predictor_coefficients.parquet`
- `linear_predictor_report.json`
- `reconstruction_report.json`
- fitted GAM artifacts under `reconstruction/marginal/` and, when available,
  `reconstruction/joint/`

### Prediction Output

`predict` writes:

- `RUN_ROOT/artifacts/01_predict/theta_estimates.parquet`
- `RUN_ROOT/artifacts/01_predict/features_marginal.parquet`
- `RUN_ROOT/artifacts/01_predict/features_joint.parquet`
- `RUN_ROOT/artifacts/01_predict/predictions.parquet`
- `RUN_ROOT/artifacts/01_predict/predictions_best_available.parquet`
- `RUN_ROOT/artifacts/01_predict/prediction_report.json`

### Full Run Output

`run` writes the full stage DAG through:

- `artifacts/10_redundancy/...`

The redundancy stage is a secondary analysis stage rather than the main product
handoff.

## 7. Runtime Defaults

BenchIQConfig runtime defaults are:

- `duplicate_policy = "error"`
- `allow_low_n = False`
- `drop_low_tail_models_quantile = 0.002`
- `min_item_sd = 0.0`
- `max_item_mean = 0.99`
- `min_abs_point_biserial = 0.0`
- `min_models_per_benchmark = 100`
- `warn_models_per_benchmark = 200`
- `min_items_after_filtering = 50`
- `min_models_per_item = 50`
- `min_item_coverage = 0.7`
- `min_overlap_models_for_joint = 75`
- `min_overlap_models_for_redundancy = 75`
- `p_test = 0.10`
- `p_val = 0.10`
- `n_strata_bins = 10`
- `random_seed = 0`

Runner stage defaults are:

- stage `04_subsample`:
  - `method = "deterministic_info"`
  - `k_preselect = None`
  - `n_iter = 2000`
  - `cv_folds = 5`
  - `checkpoint_interval = 25`
- stage `06_select`:
  - `k_final = 10`
- stage `07_theta`:
  - `theta_method = "MAP"`

The current runtime default behavior is the reconstruction-first stack. The
package helper `build_reconstruction_first_profile(...)` exposes that behavior
explicitly.

BenchIQ also keeps one stricter explicit baseline helper,
`build_psychometric_default_profile(...)`, for deliberate baseline comparisons.
It is not the runtime default.

## 8. Stage Specification

BenchIQ stages are deterministic, disk-backed, and model-level.

### Stage 00: Bundle Loading And Canonicalization

Inputs:

- `responses_long`
- optional `items`
- optional `models`

Behavior:

- loads `.csv` or `.parquet`
- canonicalizes to the internal schema
- validates required columns and duplicate policy
- derives `items` and `models` when omitted
- writes stage-00 artifacts and the canonicalization report

### Stage 01: Preprocessing

Behavior:

- computes per-benchmark item statistics
- optionally trims the lowest-scoring models by
  `drop_low_tail_models_quantile`
- filters items by low variance, near-ceiling mean, low point-biserial, and
  insufficient coverage
- filters models by retained-item coverage
- records warnings and refusal reasons benchmark by benchmark

BenchIQ refuses a benchmark after preprocessing when the retained data falls
below the configured minimum model or item counts.

### Stage 02: Score Tables

Behavior:

- computes full benchmark scores on a percent scale
- computes overlap-aware grand summaries when the bundle supports them
- writes per-model, per-benchmark target scores for later splitting and
  reconstruction

### Stage 03: Model-Level Splits

Behavior:

- performs train, validation, and test splitting at the model level only
- never performs item-level train/test splits
- stratifies on grand mean score when overlap permits
- falls back to benchmark-local score information when necessary and reports the
  fallback explicitly

### Stage 04: Preselection

Supported methods:

- `deterministic_info`
- `random_cv`

Current default:

- `deterministic_info`

The deterministic-information ranking rule is:

`abs(point_biserial)^2 * mean * (1 - mean) * item_coverage`

When `k_preselect` is not provided, the effective default is:

- `min(candidate_item_count, floor(pool_model_count / 4))`

The stage writes candidate rankings, selected preselection subsets, CV summaries,
and explicit skip reasons when the benchmark cannot support the stage.

### Stage 05: Benchmark-Local IRT

Behavior:

- fits unidimensional 2PL models benchmark by benchmark
- uses the benchmark-local preselected items
- records retained and dropped items, fit metadata, and backend details

BenchIQ v0.1 uses the `girth` backend as the core Python IRT path.

### Stage 06: Final Item Selection

Behavior:

- computes Fisher information on the fitted benchmark-local IRT model
- selects `k_final` retained items per benchmark
- writes `subset_final.parquet` and selection diagnostics

### Stage 07: Theta Estimation

Behavior:

- estimates benchmark-local latent ability on the retained item set
- records theta standard errors and coverage diagnostics
- writes a reusable theta scoring metadata file during calibration export

### Stage 08: Linear Features And Feature Tables

Behavior:

- fits a no-intercept linear predictor from retained items to full score
- computes reduced subscores
- assembles marginal feature tables for each benchmark
- assembles joint feature tables when overlap permits

BenchIQ keeps missing-coverage behavior explicit here. It does not silently
invent linear features for models that do not satisfy the required retained-item
coverage.

### Stage 09: Reconstruction

Behavior:

- fits marginal GAM reconstruction heads benchmark by benchmark
- fits joint GAM reconstruction heads when overlap and features permit
- cross-validates reconstruction heads
- writes predictions, residuals, per-split metrics, plots, and model artifacts

Joint reconstruction is optional benchmark by benchmark. When overlap or feature
availability is insufficient, BenchIQ records an explicit skip reason and still
keeps the marginal path available when possible.

### Stage 10: Redundancy

Behavior:

- computes benchmark-level correlations
- computes factor-style summaries
- computes compressibility and cross-benchmark predictability diagnostics

This stage is secondary analysis. It is included in `run`, but not required for
the `calibrate` / `predict` handoff path.

## 9. Prediction Rules

`predict(...)` and `benchiq predict` must obey all of the following:

- load and validate a saved BenchIQ calibration-bundle manifest
- reuse saved IRT, linear, and GAM artifacts
- never retrain model parameters
- require at least one benchmark from the prediction input to exist in the
  calibration bundle
- ignore extra non-calibrated benchmarks with explicit warnings
- ignore extra non-selected items with explicit warnings
- emit both per-model-type predictions and a best-available prediction table

Prediction errors must be explicit when:

- the bundle manifest is not a BenchIQ calibration bundle
- required saved artifacts are missing
- the input contains no calibrated benchmarks
- the input contains no models

## 10. Behavior Rules

BenchIQ v0.1 follows these hard rules:

- `responses_long` is the canonical internal source of truth
- all major stages are inspectable and disk-backed
- model-level splits only
- warnings, refusal reasons, skip reasons, and missing-feature reasons are
  first-class outputs
- no silent fallback behavior
- no silent retraining during prediction
- the score target is always normalized full-benchmark percent score

## 11. Repo Layout

The stable v0.1 layout for this checkout is:

```text
AGENTS.md
PLANS.md
README.md
docs/
  cli.md
  design/
  specs/
    benchiq_v0_1_spec.md
src/benchiq/
  __init__.py
  calibration.py
  deployment.py
  config.py
  runner.py
  profiles.py
  io/
  schema/
  preprocess/
  split/
  subsample/
  irt/
  select/
  reconstruct/
  redundancy/
  cli/
  viz/
tests/
  unit/
  integration/
  regression/
  data/
reports/
```

## 12. Verification Expectations

Changes to the public surface are only done when all of the following stay true:

- the documented workflows are `validate`, `calibrate`, `predict`, and `run`
- the package-root imports match the documented Python API
- calibration and deployment are verified as separate product workflows
- docs, package exports, and CLI help stay aligned
- saved warnings and refusal reasons remain inspectable on disk

This spec is intentionally shorter than earlier planning-heavy documents. Ticket
history, experiment rationale, and old planning notes belong in `PLANS.md`,
`docs/design/`, and the saved report bundles under `reports/`, not here.
