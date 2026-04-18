# post-metabench-feedback pivot plan

## purpose

This document captures the approved pivot after the first real-data metabench reviewer pass.

BenchIQ is staying a general benchmark-bundle distillation tool. The work below does **not**
redefine the product as "metabench in python." metabench remains the reference case, the
validation harness, and the strongest methodological comparison point.

This note mixes two perspectives on purpose:

- the implementation-status bullets describe what has landed
- the audit sections below describe the repo state that motivated the pivot work

If the audit text disagrees with the current public README or CLI docs, treat it as historical
motivation rather than the current public surface.

## implementation status

Current status after the pivot implementation pass:

- calibration / deployment split is implemented and tested
- public package/docs positioning now presents `calibrate` / `predict` as first-class stable
  workflows alongside `validate`, `run`, and `metabench run`
- optional R-baseline parity harness is implemented and writes skipped reports cleanly when `mirt`
  is unavailable
- reconstruction-head comparison artifacts are saved under `reports/experiments/`
- a deterministic stage-04 preselection alternative and comparison bundle are saved under
  `reports/selection_comparison/`
- calibration / deployment walkthrough artifacts are saved under `reports/calibration_deployment/`
- factor analysis remains available, but docs now present it as secondary analysis

## current state audit

### what already implements calibration logic

The repo already has an end-to-end fit path.

- bundle ingest and canonicalization live in `src/benchiq/io/load.py`
- preprocessing and full-score targets live in `src/benchiq/preprocess/filters.py` and
  `src/benchiq/preprocess/scores.py`
- model-level split logic lives in `src/benchiq/split/splitters.py`
- cross-validated random preselection lives in `src/benchiq/subsample/random_cv.py`
- benchmark-local 2PL fitting lives in `src/benchiq/irt/fit.py` and
  `src/benchiq/irt/backends/girth_backend.py`
- Fisher-information final selection lives in `src/benchiq/select/information_filter.py`
- theta estimation lives in `src/benchiq/irt/theta.py`
- reconstruction feature building and head fitting live in
  `src/benchiq/reconstruct/linear_predictor.py`,
  `src/benchiq/reconstruct/features.py`, and
  `src/benchiq/reconstruct/reconstruction.py`
- the full orchestration path lives in `src/benchiq/runner.py`

In practice, the current `benchiq run` command already performs a calibration-like workflow, but
it does so only as one monolithic run root.

### what already implements prediction / reconstruction logic

The repo already computes full-benchmark predictions from reduced subsets.

- `src/benchiq/reconstruct/reconstruction.py` fits marginal and joint GAMs and writes
  `predictions.parquet`, `reconstruction_summary.parquet`, serialized GAM pickles, cv tables, and
  plots
- `src/benchiq/reconstruct/gam.py` already provides a reusable serializable `FittedGAM` wrapper
- `src/benchiq/reconstruct/linear_predictor.py` already fits reusable linear predictor
  coefficients per benchmark
- `src/benchiq/irt/theta.py` already estimates theta and standard errors from saved item
  parameters plus reduced responses

The missing piece is not raw predictive capability. The missing piece is a first-class
artifact/API boundary that says: "fit once here, score later there."

### where the current pipeline is too coupled

The main coupling points are:

1. `src/benchiq/runner.py`
   It treats calibration, evaluation, and redundancy as one stage DAG under one `run` command.

2. `src/benchiq/reconstruct/reconstruction.py`
   It only supports fit-and-evaluate on stage-08 feature tables that already contain full-score
   targets and split labels. It does not expose a deployment-time scoring path that consumes only a
   fitted bundle plus new reduced responses.

3. `src/benchiq/reconstruct/features.py`
   It builds training/evaluation feature tables around `score_full_b` and split-aware model rows.
   There is no deployment-oriented feature builder that says "given selected items, theta metadata,
   and stored predictor coefficients, build features for a new checkpoint without retraining."

4. `src/benchiq/io/write.py` and `src/benchiq/logging.py`
   The manifest system is solid for run directories, but there is no dedicated fitted-bundle
   manifest that declares required deployment artifacts and explicitly forbids silent retraining.

5. CLI surface in `src/benchiq/cli/main.py`
   The CLI currently exposes `validate`, `run`, and `metabench run`, but not the product-level
   split between calibration and deployment.

### where factor analysis is wired into the mainline path

Factor analysis is currently still part of the default story and default run path.

- `src/benchiq/runner.py` stage order always ends with `10_redundancy`
- `src/benchiq/redundancy/compress.py` always runs `run_factor_analysis(...)` inside the stage-10
  path
- `README.md`, `docs/cli.md`, and runner summaries still mention factor-analysis outputs as part
  of the default headline artifact layout
- integration coverage in `tests/integration/test_runner.py` currently asserts that
  `factor_analysis` appears in summary metrics

The code already keeps factor analysis benchmark-level and optional-by-data, but it is still too
prominent in the product path and messaging.

### where cross-validated random subsampling is the main stochastic bottleneck

`src/benchiq/subsample/random_cv.py` is the largest intentional stochastic component.

- item subsets are sampled randomly per iteration
- the selection rule is based on minimax validation RMSE over many random candidate subsets
- checkpointed artifacts are good, but the method still depends on `n_iter`, seed choice, and the
  random sample path

Other stochastic pieces exist, but they are secondary:

- `cross_validate_gam(...)` uses shuffled K-folds
- factor analysis uses a fixed random state, but that stage is not the core bottleneck

For the pivot, the random CV preselection step is the first place where we should test a more
deterministic alternative.

### what experiment / report infrastructure already exists

The repo already has a decent artifact and reporting base to build on.

- artifact-first run directories with manifests and plots already exist across stages
- strict reduced-fixture regression exists in `tests/regression/test_metabench_validation.py`
- a paper-reviewer style real-data comparison script already exists in
  `scripts/run_metabench_real_data_comparison.py`
- frozen markdown and csv reviewer artifacts already exist under `reports/`
- integration coverage already checks end-to-end CLI and runner behavior

What is missing is a stable experiment namespace for new head-to-head comparisons outside the
metabench-specific reviewer script.

## target state

BenchIQ should pivot to two first-class product paths plus secondary experiment/report tooling.

### calibration

Given many model/checkpoint runs with item-level responses across a benchmark bundle, BenchIQ
should:

- preprocess, split, preselect, fit IRT, select final items, estimate theta metadata, and fit
  reconstruction heads
- save a reusable fitted artifact bundle with explicit manifests
- save all config, seeds, selected items, item params, predictor coefficients, GAM artifacts, and
  compatibility metadata needed for reuse

### deployment

Given a saved fitted artifact bundle and a new reduced response set for a new model/checkpoint,
BenchIQ should:

- load the fitted bundle
- validate that required artifacts are present and compatible
- estimate theta and uncertainty from the saved IRT parameters and selected items
- build deployment-time reconstruction features
- emit predicted full benchmark scores without retraining

### experiments

BenchIQ should also grow a stable experiment/report layer for:

- R-baseline IRT parity on simulated data
- reconstruction-head comparison
- selection-strategy comparison
- paper-ready result tables and plots

### redundancy / factor analysis

Redundancy analysis stays in the repo, but it becomes secondary.

- it should not be the lead product story
- factor analysis should move behind an explicit optional analysis path or at least a clearly
  secondary CLI path

## ordered tickets

### T19 calibration / deployment split

Scope:

- add a dedicated calibration entrypoint that fits and saves a reusable artifact bundle
- add a dedicated deployment entrypoint that loads a saved fitted bundle and scores new reduced
  responses without retraining
- introduce explicit fitted-bundle manifests and compatibility checks
- keep the existing end-to-end `run` path working

Main implementation notes:

- reuse existing stage outputs instead of inventing a second modeling stack
- store selected-item subsets, item parameters, theta metadata, linear predictor coefficients,
  serialized reconstruction heads, and deployment config in a stable bundle layout
- add integration coverage for "fit once, predict later"

Acceptance:

- deployment fails loudly if required fitted artifacts are missing
- deployment never calls any fit path
- the existing `run` command still works
- new `calibrate` and `predict` docs and CLI help are clear

### T20 R-baseline IRT simulation parity

Scope:

- add a simulation harness comparing BenchIQ IRT outputs to an R baseline, preferably `mirt`
- keep R optional and test-only
- add a design note stating what "identical simulated data" means under IRT identifiability

Acceptance:

- tests skip clearly when R or `mirt` is unavailable
- when available, tests compare aligned parameterizations, theta ordering, and predicted
  probabilities within justified tolerances
- a saved markdown/csv parity bundle exists under a stable reports path

### T21 reconstruction-head comparison

Scope:

- keep the current GAM path
- add Elastic Net as the regularized linear baseline
- add XGBoost as the tree-based baseline
- build a reusable experiment harness that compares all heads on the same fixed data and splits

Why XGBoost instead of LightGBM:

- XGBoost has simpler cross-platform packaging in typical python-only environments
- it is sufficient for the "strong tree baseline" role without widening scope further

Acceptance:

- outputs include RMSE, MAE, Pearson, Spearman, runtime, and seed stability where applicable
- comparison tables are written as parquet/csv plus markdown summary and plots
- the winning recommendation is stated plainly and backed by saved outputs

### T22 less-stochastic preselection alternative

Scope:

- keep random CV preselection intact
- add at least one less-stochastic alternative, likely deterministic
  information-guided filtering plus greedy refinement
- compare the new method against the current random path on quality, seed variance, runtime, and
  selection stability

Acceptance:

- both methods preserve artifact traces for each benchmark
- the comparison report states honestly whether the new method wins
- no existing baseline path is silently removed

### T23 factor-analysis demotion

Scope:

- make redundancy analysis clearly secondary in docs, CLI help, and summary messaging
- move factor analysis behind an explicit optional analysis path if practical
- stop presenting factor analysis as the main reason to use BenchIQ

Acceptance:

- README and CLI docs lead with distillation, calibration, deployment, and reconstruction
- redundancy stays available
- factor analysis no longer appears as a headline required success metric for the core path

### T24 paper-ready report bundle

Scope:

- standardize experiment output locations under `reports/`
- save machine-readable tables, markdown summaries, and plots for each comparison family
- include an honest calibration/deployment walkthrough

Acceptance:

- reports include current baseline vs new heads, current preselection vs alternative selection,
  R-baseline IRT parity, and a calibration/deployment example bundle
- every claim in the docs can be traced to a stored artifact

## proposed artifact additions

### calibration artifacts

Proposed stable root:

- `artifacts/calibration_bundle/`

Proposed contents:

- `manifest.json`
- `config_resolved.json`
- `bundle_summary.json`
- `per_benchmark/<benchmark_id>/subset_final.parquet`
- `per_benchmark/<benchmark_id>/irt_item_params.parquet`
- `per_benchmark/<benchmark_id>/theta_scoring_metadata.json`
- `per_benchmark/<benchmark_id>/linear_predictor_coefficients.parquet`
- `per_benchmark/<benchmark_id>/reconstruction/<head_name>/...`

### deployment artifacts

Proposed stable root:

- `artifacts/predict/`

Proposed contents:

- `manifest.json`
- `predicted_scores.parquet`
- `theta_estimates.parquet`
- `prediction_report.json`

### experiment artifacts

Proposed stable roots:

- `reports/experiments/reconstruction_heads/`
- `reports/selection_comparison/`
- `reports/irt_r_baseline/`
- `reports/calibration_deployment/`

## risks

### serialization compatibility

`pyGAM` models are already serialized with pickle. That is helpful, but we need an explicit
metadata gate around python/package versions so deployment failures are clear instead of silent.

### deployment feature drift

The deployment path must compute exactly the same feature definitions used during calibration.
That means feature schemas, column names, missingness rules, and scoring metadata must be stored in
the fitted bundle rather than re-inferred loosely.

### optional R integration complexity

The parity harness should not bleed into the main runtime path. The safest design is:

- python remains the default runtime
- R is invoked only by tests/scripts
- missing `Rscript` or missing R packages cause clear skips, not hidden fallbacks

### experiment sprawl

Head comparison and selection comparison can grow quickly. We should keep the harness narrow:

- fixed benchmark bundle fixture(s)
- fixed metrics
- fixed report locations
- no open-ended model zoo

### factor-analysis demotion without breaking users

The main run currently writes redundancy artifacts by default. We should preserve compatibility for
the existing run path while making the docs and optional CLI surface reflect the new emphasis.

## phase acceptance criteria

### phase 0

- this plan exists in the repo
- `PLANS.md` points to it and reflects the new ticket order

### phase 1

- BenchIQ exposes a first-class calibration API/CLI
- BenchIQ exposes a first-class deployment API/CLI
- fit-once / predict-later integration tests pass

### phase 2

- an optional R-baseline parity harness exists
- parity tests are honest about identifiability and tolerance

### phase 3

- GAM, Elastic Net, and XGBoost are compared on the same data
- saved tables/plots show the winner and the tradeoffs

### phase 4

- at least one less-stochastic preselection alternative is implemented
- quality, variance, runtime, and stability comparisons are saved

### phase 5

- factor analysis is no longer the lead product story
- redundancy remains available as a secondary analysis path

### phase 6

- reports are saved under stable repo paths
- the repo contains paper-ready tables, plots, and markdown notes for every tested change

## implementation order for this pass

The first implementation pass should proceed in this order:

1. write this plan and update `PLANS.md`
2. ship calibration / deployment split without breaking `benchiq run`
3. add the R-baseline parity harness
4. add reconstruction-head experiments
5. add less-stochastic preselection experiments
6. demote factor analysis and refresh docs
7. regenerate the experiment/report bundle
