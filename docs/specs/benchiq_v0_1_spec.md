# BenchIQ v0.1 Implementation Plan

## Executive definition

BenchIQ is an open-source Python library for **LLM benchmark distillation and overlap analysis** that ingests **item-level scored responses** (model-by-item, across a user-chosen bundle of benchmarks) and runs a **metabench-style workflow** to produce **reduced benchmark subsets**, **benchmark-specific latent abilities with uncertainty**, **reconstructed full-benchmark normalized scores**, and **benchmark-level redundancy/compressibility diagnostics**, while keeping “metabench” as the methodological reference and validation case—not the product identity. citeturn0search0turn0search4turn7view5turn30view3turn20view2

## Exact scope and non-goals

BenchIQ v0.1 is narrowly scoped to the **benchmark-distillation pipeline** and its immediate benchmark-level overlap analysis:

**In-scope (v0.1)**
- Accept a **user-selected benchmark bundle** and **item-level binary scores** per model per item; canonicalize to one long-format table. citeturn7view5turn30view3turn9search0turn28view0  
- Benchmark-wise preprocessing filters as first-class, with metabench-derived defaults:
  - low item variance filter,
  - near-ceiling (too-easy) filter,
  - near-zero point-biserial (low discrimination) filter,
  - insufficient coverage filters (item and model coverage). citeturn7view5turn8view0  
- Model-level (subject-level) splitting into train/validation/test (no item-level splits), including a **global test split** stratified on a **grand mean score** when feasible. citeturn30view3turn20view2  
- Cross-validated subsampling to a **fixed preselection size per benchmark** (metabench-style random subsampling with k-fold CV across models). citeturn32view5turn32view9turn20view0  
- Benchmark-specific **unidimensional 2PL IRT** on preselected items, with:
  - item parameter estimates,
  - Fisher information computation on a theta grid,
  - information-based item selection across the ability range to a user-controlled budget,
  - theta estimates and theta standard errors per model. citeturn19view0turn15view1turn20view1turn28view0  
- Score reconstruction targeting **normalized full-benchmark scores (percent scale)** via **GAMs** with cross-validation, including:
  - marginal reconstruction (benchmark-local),
  - joint reconstruction (bundle-wide) when overlap permits,
  - metabench-style feature families: theta, theta SE, reduced subscore, benchmark-specific linear predictor, grand subscore/linear predictor. citeturn30view0turn20view2turn23view2turn23view9  
- Benchmark-level redundancy/compressibility analysis (no item-level multidimensional models):
  - correlation structure across benchmark thetas,
  - low-dimensional latent structure (EFA/FA or PCA at benchmark level),
  - cross-benchmark prediction lift/penalty metrics (“can benchmark A be predicted from others?”). citeturn23view7turn15view0turn26view0  

**Explicit non-goals (v0.1)**
- Not a general psychometrics framework (no arbitrary IRT families, CAT/adaptive testing, item banking, automated item generation, or broad survey psychometrics abstractions). citeturn20view1turn28view0  
- No dashboards, GUIs, hosted services, benchmark registries, or multimodal extensions.  
- No benchmark-level leaderboard scraping; users supply the data.  
- No item-level multidimensional factor models, and no claims of measuring “true intelligence”—only pragmatic latent structure used for score reconstruction and overlap diagnostics. citeturn20view1turn20view2turn20view4  
- No strict R parity as product identity; R parity is **optional validation-only** (e.g., comparing against metabench’s R scripts). citeturn0search4turn5view0turn23view0  

## Statistical assumptions and minimum data requirements

BenchIQ v0.1 inherits the practical assumptions discussed in metabench’s justification for applying IRT to LLM benchmark response matrices: multiple items intended to measure a common trait (within benchmark), multiple “test takers” (models) per item, dichotomous scoring, and conditional independence of item responses given latent ability. citeturn20view1turn15view1turn28view0  

### What bundles BenchIQ can handle well (v0.1)
BenchIQ is designed for benchmark bundles where:
- Each benchmark has **enough models** and **enough items** post-filtering to fit a unidimensional 2PL and compute stable information curves. citeturn20view1turn19view0turn28view0  
- The bundle has a **non-trivial overlap set of models** evaluated on multiple benchmarks, enabling joint reconstruction and benchmark-level overlap analysis. citeturn30view3turn20view2turn23view7  

### Minimum data conditions (defaults; configurable)
BenchIQ should **refuse to run** a given benchmark’s distillation if any of the following holds after canonicalization + basic filtering:
- **Binary scoring requirement (v0.1):** `score ∈ {0,1}` (nullable allowed). If non-binary is detected, refuse with an actionable error (“v0.1 requires dichotomous item scores; pre-score your rubric into 0/1”). This aligns with 2PL assumptions and metabench’s predominant multiple-choice framing. citeturn20view1turn28view0  
- **Too few models with sufficient coverage (per benchmark):**
  - `n_models_benchmark < 100` → refuse.  
  - `100 ≤ n_models_benchmark < 200` → warn strongly; run only if user sets `allow_low_n=True`.  
  Rationale: IRT + CV splitting becomes unstable with very small N. (Design choice; thresholds are conservative defaults.) citeturn20view1turn30view3  
- **Too few items after preprocessing:** `n_items_filtered < 50` → refuse; cannot meaningfully distill and still estimate information across theta. citeturn7view5turn19view0  
- **Insufficient matrix density:**
  - Item coverage filter default: each retained item must have responses from at least `min_models_per_item = max(50, 0.2 * n_models_benchmark)` models.  
  - Model coverage filter default: each retained model must have answered at least `min_item_coverage = 0.8` of retained items.  
  If these constraints prune the benchmark below the N thresholds, refuse that benchmark. (This is BenchIQ’s explicit v0.1 definition of “insufficient coverage.”) citeturn7view5turn28view0  

BenchIQ should **refuse joint (cross-benchmark) stages** if:
- The overlap set (models with valid per-benchmark derived features across all selected benchmarks) is `< 75` by default. In that case:
  - run per-benchmark marginal reconstruction only,
  - run pairwise correlation diagnostics where overlap exists,
  - emit a warning and a clear summary of the missingness bottleneck. citeturn30view3turn20view2turn23view7  

### When BenchIQ should warn but proceed
BenchIQ should proceed with warnings (and prominently record them in artifacts) when:
- **Item-to-model ratio too high** after preselection (`n_preselect_items > n_train_models/4`), mirroring metabench’s item-to-subject sanity check. BenchIQ should auto-cap the default preselection size to `floor(n_train_models/4)` unless the user explicitly overrides. citeturn7view5turn5view0  
- Benchmarks exhibit **near-degenerate score distributions** after preprocessing (e.g., very high mean accuracy remains, or extremely low variance in total scores), which undermines discrimination and reconstruction. citeturn7view5turn20view1  

## End-to-end workflow specification

BenchIQ v0.1 is a deterministic, staged pipeline; each stage writes inspectable artifacts and a manifest that records config + hashes for reproducibility.

**Stage flow (high level)**
1. Ingest & canonicalize user data into a single long-format response table (+ optional model/item metadata tables).  
2. Benchmark-wise preprocessing and filtering (variance/easy/discrimination/coverage; optional low-tail model outliers). Defaults mirror metabench’s preprocessing script. citeturn7view5turn8view0  
3. Compute normalized full-benchmark scores (target = percent score) and produce score summaries. citeturn30view0turn23view2  
4. Split models into train/validation/test (global test split stratified on “grand mean” when available; benchmark-local validation splits within training). Model-level only. citeturn30view3turn32view5  
5. Cross-validated subsampling to preselect exactly `k_preselect[b]` items per benchmark using metabench-like random search with k-fold CV on models. citeturn32view5turn32view9turn20view0  
6. Fit benchmark-specific unidimensional 2PL IRT on the preselected items (train models only). Use a stable Python IRT backend (see dependencies section). citeturn15view1turn28view0  
7. Compute Fisher information curves for each item across a theta grid; select `k_final[b]` items across the ability range (quantile/bin coverage) using information filtering. citeturn19view0turn17view0turn20view1  
8. Estimate benchmark-specific theta and theta SE per model on the reduced subset (train/val/test). citeturn15view1turn20view1  
9. Build reconstruction feature tables (per benchmark and bundle-wide), including reduced subscores and metabench-style linear predictors and grand summaries. citeturn23view2turn23view9turn20view2  
10. Reconstruct full benchmark scores via GAMs with cross-validation (marginal and joint); evaluate reconstruction error on held-out test models. citeturn20view2turn32view0turn24search15  
11. Benchmark-level redundancy/compressibility analysis from thetas and reconstruction behavior; emit conservative overlap metrics and factor structure. citeturn23view7turn26view0turn9search2  

## Canonical internal data schema

BenchIQ v0.1 uses **one canonical long-format response table** as source-of-truth; all other tables are derived or are metadata joins.

### Canonical tables (required)

**`responses_long` (source of truth; required)**
- `model_id` (string; required): unique identifier for the evaluated model (e.g., “org/model@commit”).  
- `benchmark_id` (string; required): stable benchmark identifier within the bundle.  
- `item_id` (string; required): stable item identifier within benchmark.  
- `score` (nullable int/bool; required): dichotomous item score in `{0,1}`; missing allowed. citeturn20view1turn28view0  
- `split` (string; derived later): `{train,val,test}` at the model level.
- `weight` (float; optional): reserved for future weighting; v0.1 ignores if present but preserves through I/O.

**Primary key constraint:** `(model_id, benchmark_id, item_id)` must be unique after canonicalization; duplicates must be resolved deterministically (e.g., last-write-wins with an explicit warning artifact).

**`items` (required minimal; can be tiny)**
- `benchmark_id` (string; required)  
- `item_id` (string; required)  
- `content_hash` (string; optional but recommended): hash of prompt+choices+scoring rubric if available; used only for duplicate detection across imports.  

**`models` (required minimal; can be tiny)**
- `model_id` (string; required)  
- `model_family` (string; optional): used only for stratification diagnostics and reporting.

### Optional metadata tables (v0.1; pass-through only)
**`benchmarks`**
- `benchmark_id` (string; required)
- `display_name` (string; optional)
- `n_options_default` (int; optional): used only if user wants chance-corrected difficulty during preprocessing; metabench assumed 4-choice for their difficulty correction logic. citeturn8view0turn7view5  

### Derived tables (BenchIQ emits; not user inputs)
- `scores_full` (per model × benchmark): percent score target for reconstruction. citeturn30view0turn23view2  
- `splits_models` (per model): global split assignment and stratification bins. citeturn30view3  
- `subset_preselect` and `subset_final` (per benchmark): selected item lists and selection diagnostics. citeturn32view2turn19view0  
- `irt_item_params` (per benchmark × item): discrimination/difficulty and fit metadata. citeturn28view0turn15view1  
- `theta_estimates` (per model × benchmark): theta_hat, theta_se, estimation method. citeturn15view1turn20view1  
- `recon_features_marginal` / `recon_features_joint` (per model × benchmark): full feature matrices. citeturn20view2turn23view9  
- `recon_predictions` (per model × benchmark): predicted score, residuals, RMSE summaries. citeturn20view2turn32view0turn24search15  
- `redundancy_metrics` (benchmark × benchmark and per-benchmark): overlap diagnostics. citeturn23view7turn26view0  

## Package layout / repo tree

A v0.1 repo structure that supports reproducibility, inspectable artifacts, and a tight API:

```
benchiq/
  pyproject.toml
  README.md
  LICENSE
  AGENTS.md                  # Codex-specific working agreement (required)
  docs/
    design/
      v0_1_scope.md
      schema.md
      metabench_validation.md
    cli.md
  src/benchiq/
    __init__.py
    config.py                # pydantic config schema + defaults
    logging.py               # structured logs + run manifest utilities
    io/
      load.py                # read csv/parquet, validate schema, canonicalize
      write.py               # artifact writers + manifest updates
    schema/
      tables.py              # column constants, dtypes, validators
      checks.py              # hard checks + warning checks
    preprocess/
      filters.py             # variance/easy/discrimination/coverage filters
      stats.py               # item stats, point-biserial, summaries
    split/
      splitters.py           # model-level split logic (global + per-benchmark)
    subsample/
      random_cv.py           # cross-validated subsampling to k_preselect
    irt/
      backends/
        girth_backend.py     # 2PL fit using girth (core)
        pyirt_backend.py     # optional experimental backend
      fit.py                 # unify backend outputs -> item params
      info.py                # Fisher info functions + grids
      theta.py               # MAP/EAP theta, theta SE from info
    select/
      information_filter.py  # quantile/bin-based selection to k_final
    reconstruct/
      linear_predictor.py    # weighted-average / no-intercept linear model
      features.py            # build marginal/joint feature matrices
      gam.py                 # GAM fit + CV + predict (pyGAM core)
      metrics.py             # RMSE/MAE/R2 + calibration plots
    redundancy/
      corr.py                # correlation matrices
      factor.py              # benchmark-level FA (optional factor_analyzer)
      compress.py            # cross-benchmark predictability metrics
    cli/
      main.py                # entrypoint
      commands_run.py
      commands_validate.py
      commands_metabench.py  # validation-mode helper (optional download)
    viz/
      plots.py               # matplotlib plots written to disk
  tests/
    unit/
    integration/
    regression/
    data/                    # tiny synthetic fixtures only (no large corpora)
```

This layout is aligned with Codex best-practice guidance to provide durable agent instructions (`AGENTS.md`) and explicit “done when” acceptance tests, rather than ad hoc prompting. citeturn33view0turn33view3turn33view2  

## Public Python API

BenchIQ v0.1 should expose a small API that maps to the workflow and artifacts, not a sprawling psychometrics toolkit.

### Core objects
- `BenchIQConfig`: validated configuration (bundle definition, budgets, thresholds, backends, seeds, output paths).
- `BenchIQRunner`: orchestrates staged execution, writes artifacts, returns a `RunResult` handle.

### Minimal user-facing functions

**Load + validate**
- `benchiq.load_bundle(responses_path, items_path=None, models_path=None) -> Bundle`
- `benchiq.validate(bundle, config) -> ValidationReport`

**Run**
- `benchiq.run(bundle, config, out_dir) -> RunResult`

**Inspect results**
- `RunResult.load_artifact(name) -> pd.DataFrame | dict`
- `RunResult.paths() -> dict[str, Path]`
- `RunResult.summary() -> dict` (high-level metrics and warnings)

### Tiny interface sketch (not implementation)

```python
from benchiq import BenchIQConfig, load_bundle, validate, run

bundle = load_bundle("responses.parquet", items_path="items.parquet", models_path="models.parquet")
config = BenchIQConfig.from_yaml("benchiq.yml")

report = validate(bundle, config)
result = run(bundle, config, out_dir="runs/2026-03-30_demo")

print(result.summary()["reconstruction"]["rmse_test_by_benchmark"])
```

BenchIQ’s API should keep “benchmarks are LLM benchmarks” implicit: it assumes (model, item) grids and leaderboard-scale evaluation matrices, not human survey instruments. citeturn20view1turn30view3  

## CLI surface

CLI must be small, reproducible, and artifact-first.

**`benchiq validate`**
- Inputs: `--responses`, optional `--items`, `--models`, `--config`, `--out`
- Behavior: canonicalize + run all schema/stat sanity checks; emit a validation bundle to `out/validate/` including warnings and refusal reasons.

**`benchiq run`**
- Inputs: same as validate plus `--run-id` optional.
- Behavior: executes full pipeline end-to-end, writing artifacts under `out/<run-id>/` and a `manifest.json` that records config, versions, seeds, and hashes.

**`benchiq metabench run`** (validation-mode helper; optional but recommended)
- Behavior:
  - prepare the metabench validation dataset (either by downloading a pinned snapshot or by requiring the user to place files locally),
  - run with metabench-like defaults (preprocess thresholds, split logic, preselect size, selection settings),
  - write a “validation report” comparing BenchIQ outputs to expected reference metrics (stored in repo as toleranced assertions).
- This command exists to maintain the “methodological backbone” and publishable evaluation story without turning BenchIQ into a metabench-only reproduction. citeturn0search4turn5view0turn30view3turn33view0  

## Stage-by-stage algorithms

Below, each stage is specified with: input, output, method, key parameters, and failure modes/diagnostics. (All stages write artifacts.)

### Ingestion / canonicalization
**Input**
- User-provided `responses_long` (CSV/Parquet) or wide matrices convertible to long.

**Output**
- `artifacts/00_canonical/responses_long.parquet`
- `artifacts/00_canonical/items.parquet`
- `artifacts/00_canonical/models.parquet`
- `artifacts/00_canonical/canonicalization_report.json`

**Method**
- Enforce required columns and types.
- Enforce PK uniqueness; if duplicates exist, resolve deterministically and record counts + affected keys.
- Normalize IDs to strings; trim whitespace.
- Validate `score` ∈ {0,1} or null.

**Key parameters**
- `duplicate_policy`: `error | last_write_wins | first_write_wins` (default `error`).
- `score_policy`: `binary_only` (hard requirement v0.1). citeturn20view1turn28view0  

**Failure modes / diagnostics**
- Non-binary scores → hard fail.
- Excessive duplicates (>0.1% rows) → warn even if policy resolves.

### Preprocessing / filtering
**Input**
- canonical artifacts from stage 0.

**Output**
- `artifacts/01_preprocess/per_benchmark/<benchmark_id>/filtered_items.parquet`
- `artifacts/01_preprocess/per_benchmark/<benchmark_id>/filtered_models.parquet`
- `artifacts/01_preprocess/per_benchmark/<benchmark_id>/item_stats.parquet`
- `artifacts/01_preprocess/per_benchmark/<benchmark_id>/preprocess_report.json`
- `artifacts/01_preprocess/summary.parquet`

**Method**
BenchIQ mirrors metabench’s preprocessing logic as default behavior:
- Optional **tail outlier model removal**: drop models in the lowest 0.1% of full-score distribution (per benchmark) as “tail outliers.” citeturn7view5turn8view0  
- Compute per-item:
  - `sd(score)`,
  - `mean(score)` (difficulty proxy; metabench used a chance-corrected form but the operational “too easy” threshold corresponds to raw mean accuracy > 0.95). citeturn7view5turn8view0  
  - point-biserial discrimination: correlation between item score and total score (part-whole correlation), as implemented in metabench’s script. citeturn7view3turn8view0  
- Apply filters (defaults):
  - `sd <= 0.01` → drop (low variance). citeturn7view0  
  - `mean >= 0.95` → drop (near-ceiling). (BenchIQ default expresses directly in raw mean; metabench’s code computes an equivalent threshold via a guessing coefficient.) citeturn7view5turn8view0  
  - `abs(point_biserial) < 0.05` → drop (near-zero discrimination). Metabench used 0.05 generally and 0.02 for Winogrande; BenchIQ defaults to 0.05 unless overridden per benchmark. citeturn7view5turn8view0  
  - Coverage filters (BenchIQ definition; v0.1):
    - drop items with < `min_models_per_item` responses,
    - drop models with < `min_item_coverage` coverage of retained items.

**Key parameters**
- `drop_low_tail_models_quantile` (default `0.001` to match metabench script). citeturn7view5  
- `min_item_sd` (default `0.01`). citeturn7view0  
- `max_item_mean` (default `0.95`). citeturn7view5turn8view0  
- `min_abs_point_biserial` (default `0.05`, per-benchmark override). citeturn7view5turn8view0  
- `min_models_per_item`, `min_item_coverage` (BenchIQ v0.1 defaults; documented as coverage guardrails).

**Failure modes / diagnostics**
- If filtering leaves too few items/models → refuse that benchmark and record refusal in summary.
- Emit plots to disk (histograms of item mean/sd/discrimination) mirroring metabench’s diagnostic plots, but optional in CLI for speed. citeturn7view3turn7view0  

### Score normalization
**Input**
- preprocessed per-benchmark filtered views.

**Output**
- `artifacts/02_scores/scores_full.parquet` (model × benchmark)
- `artifacts/02_scores/scores_grand.parquet` (model-level grand mean across benchmarks when overlap permits)
- `artifacts/02_scores/score_report.json`

**Method**
- For each model×benchmark: compute full benchmark score as percent correct:
  - `score_full = 100 * mean(item_score)` over the benchmark’s filtered item set, using only non-missing items; require coverage ≥ `min_item_coverage` (or set missing). citeturn30view0turn23view2  
- Grand mean score across benchmarks:
  - computed only for models with valid scores on all benchmarks in the configured bundle, mirroring metabench’s “mean across benchmarks” use in splitting. citeturn30view3  

**Key parameters**
- `score_scale = "percent"` (fixed v0.1 target).
- `min_item_coverage` (shared with preprocessing).

**Failure modes / diagnostics**
- If grand overlap too small, skip grand score computation; joint stages will be disabled with warnings.

### Train/validation/test splitting by model
**Input**
- `scores_full` and optionally `scores_grand`.

**Output**
- `artifacts/03_splits/splits_models.parquet`
- `artifacts/03_splits/per_benchmark/<benchmark_id>/splits_models.parquet`
- `artifacts/03_splits/split_report.json`

**Method**
BenchIQ implements metabench-like split logic:
- **Global test split (preferred):** if enough models have scores for all benchmarks in the bundle:
  - stratify on `grand_mean_score`,
  - sample `p_test = 0.10` into test set (metabench used 10% via caret’s stratified partition). citeturn30view3  
- **Local validation split per benchmark (within remaining train+val pool):**
  - for each benchmark, split train vs val (default `p_val = 0.10`) stratified on that benchmark’s full score. (BenchIQ uses binned scores + `StratifiedShuffleSplit`.) citeturn32view5turn20view0  

**Key parameters**
- `p_test` (default 0.10), `p_val` (default 0.10). citeturn30view3turn32view5  
- `n_strata_bins` for score binning (default 10; must adapt if N small).
- `random_seed` (recorded in manifest).

**Failure modes / diagnostics**
- If too few overlap models for global split, fall back to per-benchmark splitting (still model-level) and mark joint stages disabled.
- Emit split balance diagnostics: score histograms per split, min/max, and Kolmogorov-style checks (lightweight).

### Cross-validated subsampling
**Input**
- per-benchmark train/val sets: filtered response matrices (models × items) and full scores.

**Output**
- `artifacts/04_subsample/per_benchmark/<benchmark_id>/preselect_items.parquet`
- `artifacts/04_subsample/per_benchmark/<benchmark_id>/cv_results.parquet`
- `artifacts/04_subsample/per_benchmark/<benchmark_id>/subsample_report.json`

**Method (metabench-style random CV subsampling)**
BenchIQ follows metabench’s random subsampling design:
- Choose target preselection size `k_preselect[b]`.
- On benchmark train+val pool (excluding global test), create **k-fold CV splits across models** (metabench used 5 folds). citeturn32view5turn32view9  
- Repeat for `n_iter` random seeds:
  - sample `k_preselect` items uniformly without replacement,
  - for each fold:
    - build reduced subscore = mean of selected item scores (percent),
    - fit a one-dimensional GAM `full_score ~ s(reduced_score)` on fold-train models,
    - compute RMSE on fold-val models,
    - also compute RMSE on the held-out global test set for tracking (metabench did this). citeturn32view0turn32view2  
- Selection criterion (v0.1 default = metabench-like “minimax validation”):
  - For each random seed, compute `max_rmse_val` across folds,
  - select the seed minimizing `max_rmse_val`. citeturn32view2turn32view3  

**Key parameters**
- `k_preselect` per benchmark (user-configurable; default cap based on N_train_models/4). citeturn7view5  
- `n_folds` (default 5). citeturn32view5  
- `n_iter` (default 2000 generic, 10000 for metabench validation parity; record runtime and allow early-stop if plateau). citeturn32view9  
- `gam_backend = "pygam"`; fit using adaptive splines if supported, otherwise standard spline terms (see dependency choice). citeturn24search15turn24search7turn20view2  

**Failure modes / diagnostics**
- If benchmark is extremely large and `n_iter` is high, runtime can be significant; BenchIQ must (a) log progress, (b) allow reducing `n_iter`, and (c) write intermediate results every K iterations (crash-safe). Metabench used parallel processing; BenchIQ should implement parallelization via `joblib` or `multiprocessing` but keep it optional for v0.1. citeturn32view9turn33view0  
- If selected subset yields too small effective coverage (many missing), discard and resample (count as failed iteration) with explicit counters.

### Benchmark-specific IRT fitting
**Input**
- preselected items per benchmark; benchmark train models (responses matrix, possibly missing).

**Output**
- `artifacts/05_irt/per_benchmark/<benchmark_id>/irt_item_params.parquet`
- `artifacts/05_irt/per_benchmark/<benchmark_id>/irt_fit_report.json`

**Method**
- Fit a **unidimensional 2PL** model on train models’ responses to preselected items. metabench used mirt with EM for 1D models; BenchIQ will use a Python backend but keep the model class fixed to 2PL in v0.1. citeturn15view1turn19view0turn20view1  
- Core backend choice (recommended for v0.1):
  - Use `girth`’s `twopl_mml` (marginal maximum likelihood) to estimate item discrimination and difficulty, with missing data tagged via `tag_missing_data`. citeturn28view0turn10view0  
- Optional experimental backend:
  - `py-irt` (PyTorch/Pyro) for large-scale fits and potential GPU acceleration, but not core due to heavier dependencies and tighter Python version constraints in metadata. citeturn10view3turn10view1turn9search7  

**Key parameters**
- `irt_backend = "girth"` (default).
- `theta_prior = Normal(0,1)` for downstream theta estimation (consistent with metabench’s Gaussian latent density setting in mirt). citeturn15view1turn20view1  
- `max_iter`, `tol` forwarded to backend where available.

**Failure modes / diagnostics**
- Non-convergence / pathological parameters (e.g., extreme discrimination, infinite difficulty):
  - clamp or drop items based on explicit rules (e.g., `a_j` outside [0.1, 5.0] triggers warning; outside [0.05, 10.0] triggers exclusion), and record counts.
- Emit item parameter distribution plots (difficulty vs discrimination scatter) similar to metabench’s exploratory diagnostics, written to disk. citeturn7view3turn28view0  

### Fisher-information item selection
**Input**
- fitted 2PL item parameters and the preselected item list; benchmark train theta distribution proxy.

**Output**
- `artifacts/06_select/per_benchmark/<benchmark_id>/subset_final.parquet`
- `artifacts/06_select/per_benchmark/<benchmark_id>/info_grid.parquet` (theta grid × item info)
- `artifacts/06_select/per_benchmark/<benchmark_id>/selection_report.json`
- `artifacts/06_select/per_benchmark/<benchmark_id>/plots/` (test information curves)

**Method**
BenchIQ implements metabench’s information filtering concept:
- Build a theta grid:
  - default: empirical theta estimates from a preliminary theta pass (or, if not available yet, use a grid spanning the item difficulty range),
  - alternative: uniform grid over `[min_theta, max_theta]`. citeturn17view1turn16view5turn20view1  
- Compute per-item Fisher information across theta:
  - For 2PL logistic, `I_j(θ) = a_j^2 * p_j(θ) * (1 - p_j(θ))`. (BenchIQ uses this analytic form rather than calling R’s `iteminfo`, but it matches the same concept.) citeturn19view0turn20view1  
- Partition theta range into `n_bins` (default = `k_final[b]` or a capped value), using quantiles of theta grid (metabench used quantiles). citeturn16view5turn17view0  
- Selection loop:
  - For each bin, choose the remaining item with the maximum information within that bin, subject to an information threshold `info_min` (metabench used a `threshold` hyperparameter when selecting items). citeturn17view0turn17view2  
  - Remove chosen item from candidate pool, continue until `k_final` items selected or no items meet threshold.
- BenchIQ v0.1 **does not implement Bayesian hyperparameter optimization** over selection settings (metabench’s `reduce.R` included this); instead, v0.1 exposes `info_min`, `n_bins`, and `k_final` directly and logs sensitivity diagnostics. citeturn16view8turn17view2  

**Key parameters**
- `k_final[b]` per benchmark (required user-configurable budget).
- `theta_grid_type = empirical | uniform`.
- `n_bins` (default = `min(k_final, 250)`).
- `info_min` (default 0.0; recommend user tune if subsets are too large or too small). citeturn17view2turn16view8  

**Failure modes / diagnostics**
- If selection yields < 20 items, warn that the reduced benchmark may be too small for stable reconstruction; proceed but flag.
- Emit “expected test information” curves for full vs reduced sets (metabench plotted these). citeturn19view1turn18view0  

### Ability estimation
**Input**
- reduced item set, item parameters, and response patterns for all splits.

**Output**
- `artifacts/07_theta/theta_estimates.parquet` (model × benchmark × split)
- `artifacts/07_theta/theta_report.json`

**Method**
- Estimate theta per model per benchmark using:
  - `method = MAP` (default) and optionally `EAP`. Metabench used MAP and EAPsum as scoring options via mirt’s `fscores`. BenchIQ v0.1 implements MAP and EAP on a fixed grid; EAPsum is not separately implemented unless a clear operational definition is added later. citeturn15view0turn17view2turn20view1  
- Standard error:
  - `theta_se = 1 / sqrt( I_test(theta_hat) )` where `I_test` is the sum of item information values at `theta_hat`. This is consistent with the Fisher information focus of the pipeline. citeturn19view0turn20view1  

**Key parameters**
- `theta_method = "MAP" | "EAP"`.
- `theta_grid = np.linspace(-4, 4, 161)` (default), or adaptive based on item difficulties.
- `prior = Normal(0,1)` fixed v0.1. citeturn15view1turn20view1  

**Failure modes / diagnostics**
- All-correct or all-wrong response patterns on the reduced set can push MAP to extremes; clamp to grid bounds and report saturation counts.
- Emit theta distribution plots per benchmark and split.

### Reconstruction feature building
**Input**
- `theta_estimates`, reduced item responses, and full scores.

**Output**
- `artifacts/08_features/features_marginal.parquet`
- `artifacts/08_features/features_joint.parquet` (if joint enabled)
- `artifacts/08_features/feature_report.json`

**Method**
BenchIQ implements the reconstruction predictor families described in metabench’s Appendix predictors section:
- Per-benchmark features (for benchmark `b`):
  - `theta_b`, `theta_se_b`, reduced subscore `sub_b` (= percent correct on reduced items). citeturn20view2turn23view2turn15view0  
- Benchmark-specific linear predictor (`lin_b`):
  - Fit a **linear model without intercept** predicting the full score from the reduced-item response vector (metabench did this via `train.lm`, yielding `.l` predictions and `.s` subscores). BenchIQ implements the same concept with OLS or ridge-regularized OLS to prevent instability when items are correlated. citeturn23view2turn23view5turn20view2  
- Grand summary features (if joint enabled and overlap sufficient):
  - `grand_sub = mean_b(sub_b)` and `grand_lin = mean_b(lin_b)` across benchmarks in the bundle, mirroring metabench’s `grand.s` and `grand.l`. citeturn23view9turn22view3turn20view2  

**Key parameters**
- `linear_predictor_model = "ols_no_intercept"` (default) with optional `ridge_alpha` smoothing.
- Imputation for missing reduced-item responses in the linear predictor stage:
  - default: treat missing as 0 (incorrect) only if user asserts missing means not-attempted; otherwise require complete reduced-item coverage for models used in linear predictor training. (BenchIQ should default to the conservative choice: require completeness for the training fit, and only score models with sufficient coverage.)

**Failure modes / diagnostics**
- If reduced-item matrix is rank-deficient and OLS is unstable:
  - automatically switch to ridge with a logged alpha sweep (still deterministic) and record chosen alpha.

### GAM-based score reconstruction
**Input**
- feature matrices and full scores; split assignments.

**Output**
- `artifacts/09_reconstruct/per_benchmark/<benchmark_id>/gam_model.json` (serialized where possible)
- `artifacts/09_reconstruct/per_benchmark/<benchmark_id>/predictions.parquet`
- `artifacts/09_reconstruct/reconstruction_summary.parquet`
- plots: calibration, residual histograms, predicted vs actual.

**Method**
BenchIQ uses a true GAM implementation in Python:
- **Core GAM backend: `pyGAM`**, which is explicitly designed for GAMs and has current documentation and active releases; it supports installation via pip and a scikit-learn-like API. citeturn24search15turn24search7turn24search30  
- BenchIQ avoids using statsmodels’ `GLMGam` as the core path because statsmodels labels `GLMGam` as *experimental* and warns that some options/results may be incorrect. It can be offered as an optional backend for users who prefer statsmodels, but not the default. citeturn24search1turn24search0  

**Models**
- Marginal reconstruction per benchmark `b`:
  - Fit `score_full_b ~ s(theta_b) + s(theta_se_b) + s(sub_b) + s(lin_b)` with adaptive spline basis where supported (metabench used adaptive splines `bs="ad"` and specifically chose adaptive splines for fidelity over non-uniform predictor distributions). citeturn20view2turn23view0turn32view0  
- Joint reconstruction per benchmark `b` (only if overlap sufficient):
  - Fit `score_full_b ~ Σ_{b'} s(theta_{b'}) + s(sub_b) + s(grand_sub) + s(lin_b) + s(grand_lin)` (exact formula configurable, but feature families match metabench’s described joint use of all latent abilities plus specific and grand summaries, and their additional linear predictor terms). citeturn20view2turn23view9turn22view3  
- Grand mean reconstruction (optional output but first-class in workflow):
  - Fit `grand_score ~ Σ_{b} s(theta_b) + s(grand_sub) + s(grand_lin)`; metabench notes that for mean-score joint models some terms are discarded. BenchIQ mirrors this by using only bundle-wide summaries. citeturn20view2turn23view9  

**Cross-validation**
- For each GAM, choose smoothing parameters via:
  - k-fold CV on training models (default 5-fold) minimizing RMSE on validation folds,
  - then evaluate once on held-out test models (global test split). This matches the broader metabench practice of CV-on-train and report-on-test. citeturn32view5turn30view3turn20view2  

**Key parameters**
- `gam_terms` per model type (marginal/joint) as a config object, not hard-coded strings.
- `gam_n_splines` (default 20 per term), `lam_grid` (log-spaced), `cv_folds`.
- `metric = RMSE` (default), plus MAE and Spearman correlation for leaderboard relevance.

**Failure modes / diagnostics**
- If joint feature matrix has many missing values because overlap is low, disable joint reconstruction and report overlap stats.
- If a GAM fit fails to converge for some lam settings, skip those and record.

### Benchmark-level factor / redundancy / compressibility analysis
**Input**
- theta estimates (preferably from reduced subsets), full scores, reconstruction results, overlap sets.

**Output**
- `artifacts/10_redundancy/corr_theta.parquet`
- `artifacts/10_redundancy/corr_scores.parquet`
- `artifacts/10_redundancy/factor_loadings.parquet`
- `artifacts/10_redundancy/compressibility.parquet`
- `artifacts/10_redundancy/redundancy_report.json`
- plots: correlation heatmaps, scree-like summaries, factor loading charts.

**Method**
Keep analysis conservative and benchmark-level:
- Compute correlation matrices across benchmarks:
  - `corr(theta_b, theta_b')` and `corr(score_b, score_b')` on the overlap model set; default correlation = Spearman (robust to monotone transforms common in leaderboard scales). citeturn23view7turn23view0  
- Factor analysis (benchmark-level, not item-level):
  - Preferred optional backend: `factor_analyzer`, which explicitly supports EFA with MINRES and rotations, and notes it is partially ported from R’s `psych`—close to metabench’s R implementation. BenchIQ can use it when installed. citeturn26view0turn15view0  
  - Fallback: scikit-learn `FactorAnalysis` (ML) for environments without factor_analyzer. citeturn9search2  
- Compressibility metrics:
  - For each benchmark `b`, quantify how well it is predictable from others:
    - Fit a joint GAM predicting `score_full_b` using features from all other benchmarks (exclude `b`’s own features) and compare RMSE to the marginal GAM for `b`.
    - Define:
      - `redundancy_rmse_gain = rmse_marginal - rmse_cross_only`
      - `redundancy_ratio = rmse_cross_only / rmse_marginal`
    Lower cross-only RMSE indicates higher redundancy/compressibility. This stays “practical and conservative” because it’s an operational, out-of-sample predictability measure rather than a strong latent-theory claim. citeturn20view2turn30view3turn32view5  

**Key parameters**
- `min_overlap_models_for_redundancy` (default 75).
- `n_factors_to_try = [1,2,3]` (small, benchmark-level only).
- `cv_folds` shared with reconstruction stage.

**Failure modes / diagnostics**
- If overlap is insufficient, produce only pairwise correlations with per-pair overlap counts and avoid factor analysis.

## Artifact outputs

BenchIQ must write inspectable artifacts at every major stage. Default on-disk format:
- Parquet for tables (`.parquet`) and JSON for manifests/reports (`.json`), plus `.png` plots.

**Output directory structure**
```
out/<run_id>/
  manifest.json
  config_resolved.json
  logs.txt
  artifacts/
    00_canonical/...
    01_preprocess/...
    02_scores/...
    03_splits/...
    04_subsample/...
    05_irt/...
    06_select/...
    07_theta/...
    08_features/...
    09_reconstruct/...
    10_redundancy/...
  reports/
    run_summary.md
    warnings.md
    metrics.json
  plots/
    per_benchmark/<benchmark_id>/...
    bundle/...
```

**Concrete file list (minimum)**
- `manifest.json`: run_id, timestamp, package version, dependency versions, seeds, git hash (if available), input file hashes, resolved config.  
- `config_resolved.json`: fully expanded config (after defaults and per-benchmark overrides).  
- `artifacts/02_scores/scores_full.parquet`: reconstruction targets (percent scores). citeturn30view0turn23view2  
- `artifacts/03_splits/splits_models.parquet`: model-level split assignment. citeturn30view3  
- For each benchmark:
  - preprocessing report + item stats,
  - `preselect_items.parquet` + `cv_results.parquet` from subsampling, citeturn32view2turn32view5  
  - `irt_item_params.parquet`, citeturn28view0turn15view1  
  - `subset_final.parquet` + information-grid diagnostics, citeturn19view0turn17view0  
  - `theta_estimates.parquet`, citeturn15view0  
  - `predictions.parquet` for marginal and joint recon, with residuals and per-split metrics. citeturn20view2turn32view0  
- Bundle-level:
  - `redundancy_report.json`, correlation matrices, factor loadings, compressibility table. citeturn23view7turn26view0  
- `reports/run_summary.md`: a human-readable summary of what ran, what was skipped, and headline metrics.

## Validation and testing strategy

BenchIQ needs tests that verify both engineering correctness and statistical sanity, without over-claiming universality.

### Unit tests (fast, deterministic)
- Schema validation:
  - rejects non-binary scores,
  - enforces PK uniqueness,
  - preserves dtypes and IDs.
- Preprocessing stats:
  - point-biserial computation matches the formula used in metabench’s preprocessing script (numerical tolerance). citeturn7view3turn8view0  
- Splitters:
  - splitting is model-level only,
  - global test set has correct size and approximate stratification properties. citeturn30view3  
- Fisher information:
  - information curve outputs are non-negative and peak near item difficulty as expected for 2PL logistic (sanity). citeturn19view0turn20view1  
- Theta estimation:
  - MAP optimizer is stable and monotone in simple cases (single-item toy).

### Synthetic statistical tests (small but meaningful)
- Simulated 2PL benchmarks:
  - generate item params + theta, sample responses,
  - ensure BenchIQ recovers monotone ranking of model abilities and reasonable reconstruction of full scores from reduced subsets under known generative structure. citeturn20view4turn28view0  
- Synthetic multi-benchmark overlap:
  - generate two or three correlated theta dimensions and create benchmarks that load on them (similar to metabench’s synergy simulation idea),
  - verify redundancy metrics move in the expected direction as correlation increases. citeturn20view3turn20view4  

### Integration tests (end-to-end)
- Run the entire pipeline on a tiny synthetic bundle (e.g., 3 benchmarks × 200 models × 200 items), verify:
  - all expected artifacts exist,
  - reconstructed RMSE is finite and improves from random baseline as n_iter increases,
  - selection produces exactly `k_final[b]` items when feasible.

### Regression tests (metabench as primary validation case)
BenchIQ’s v0.1 acceptance suite should include a **metabench-validation mode** that runs the pipeline on a pinned metabench dataset snapshot and checks toleranced expectations:
- Preprocessing removes items according to the metabench-derived default thresholds (sd<=0.01, mean>=0.95, abs(point-biserial)<0.05) and tail outliers at 0.1% quantile, within exact match for that snapshot. citeturn7view5turn8view0  
- The split logic matches metabench’s approach: global test split stratified on grand mean score with ~10% test proportion. citeturn30view3  
- Cross-validated subsampling uses 5-fold CV and selects subsets by minimax validation RMSE (same criterion). citeturn32view5turn32view2  
- Reconstruction features include benchmark-specific linear predictors and grand summaries, matching metabench’s `train.lm`/`grand.l`/`grand.s` structure. citeturn23view2turn23view9  

BenchIQ should not assert exact numeric parity with metabench’s R outputs in CI (different backends and optimization), but it **can** assert:
- directions and magnitudes are comparable (e.g., RMSE within a tolerance band and monotone relationships preserved),
- the pipeline completes and produces a consistent artifact set.

### Acceptance criteria (v0.1 “done when”)
- `benchiq run` completes on:
  1) synthetic integration fixture, and  
  2) metabench validation snapshot (or a reduced metabench subset fixture if size is too large for CI). citeturn33view0turn5view0  
- All stages write artifacts; manifest lists versions and seeds.
- Joint reconstruction + redundancy stages either run or are explicitly skipped with a recorded reason (no silent skipping).
- Public API and CLI are stable enough to support a “methods + tooling” paper’s reproducibility checklist.

## Publishability plan and paper claim

### Publishability plan (tooling/evaluation paper)
The publishable contribution is: **a reusable, artifact-first implementation of a benchmark-bundle distillation workflow** for LLM evaluation matrices, plus conservative overlap diagnostics. The paper should position BenchIQ as:
- a workflow that users apply to *their own* bundles,
- methodologically grounded by metabench, validated on metabench, and stress-tested on synthetic bundles (for controlled redundancy experiments). citeturn0search0turn20view2turn20view4turn33view0  

**Experiments/results to include**
- **Metabench replication (core):**
  - show reconstruction RMSE on held-out test models for marginal vs joint GAMs,
  - show distillation curves: RMSE vs subset size (preselect and final),
  - show benchmark theta correlation matrix and benchmark-level factor structure. citeturn20view0turn23view7turn32view2turn20view2  
- **Synthetic overlap study (supports redundancy claims):**
  - vary inter-benchmark latent correlation and cross-loading; show redundancy metrics behave sensibly (mirrors metabench’s synergy simulation motivation). citeturn20view3turn20view4  
- **“Arbitrary bundle” demonstration (non-universal, honest):**
  - include at least one additional bundle format (even if small) to demonstrate that BenchIQ’s schema + pipeline generalize operationally (not claiming validity across all benchmarks). This can be a user-provided bundle in supplementary material or a small public bundle if available; the paper should state explicitly that general validity is not proven and BenchIQ provides diagnostics/warnings instead. citeturn20view1turn33view0  

### Smallest honest paper claim
BenchIQ v0.1 can claim (defensibly):
- “BenchIQ implements an open-source, benchmark-bundle-agnostic pipeline for distilling LLM benchmarks and analyzing benchmark-level overlap using unidimensional IRT, Fisher-information-based item selection, and GAM-based reconstruction; it reproduces the qualitative findings and reconstruction behavior reported in metabench on the metabench dataset and provides artifact-first diagnostics for applying the workflow to new benchmark bundles.” citeturn0search0turn0search4turn20view2turn23view9turn33view0  

## Ordered Codex implementation roadmap

This roadmap is intentionally “ticketized” so a coding agent can execute it with minimal ambiguity. It also embeds Codex-operational guidance: use subagents for parallelizable work (schema vs IRT vs GAM vs CLI), keep tasks small, self-test frequently, and stop if failures can’t be fixed without scope creep—consistent with Codex best-practice and subagent guidance. citeturn33view1turn33view0turn33view2turn33view3  

### Ticket 0 — Repo scaffolding + agent contract
**Goal**  
Create the repository skeleton and the “Codex working agreement” files that enforce micro-steps and frequent self-tests.

**Files/modules**
- `pyproject.toml`
- `README.md` (v0.1 scope)
- `AGENTS.md`
- `docs/design/v0_1_scope.md`

**Implementation tasks**
- Choose Python version: set `python>=3.10,<3.15` to align with `pyGAM`’s declared requirement. citeturn24search30  
- Add dependencies (initial, pinned with compatible ranges):
  - required: numpy, pandas, scipy, scikit-learn, pyarrow, pydantic, joblib, matplotlib, click/typer
  - core modeling: `pygam`, `girth`
  - optional extras: `factor_analyzer`, `statsmodels`, `py-irt`
- Write `AGENTS.md` with:
  - “Goal / Context / Constraints / Done when” prompt template,
  - strict rule: every ticket must add/adjust tests and run them,
  - explicit stop condition if tests fail and can’t be fixed without changing scope,
  - explicit suggestion: spawn subagents for independent modules. citeturn33view0turn33view1turn33view2turn33view3  

**Tests to add**
- None yet (scaffolding).

**Acceptance criteria**
- `pip install -e .` works.
- `python -c "import benchiq"` works.
- `AGENTS.md` exists and is referenced in README.

**Likely failure modes/blockers**
- Dependency conflicts (especially around SciPy/Numpy versions); resolve by adjusting version ranges.

---

### Ticket 1 — Canonical schema + validators
**Goal**  
Implement canonical table schemas and strict validation for `responses_long`, `items`, `models`.

**Files/modules**
- `src/benchiq/schema/tables.py`
- `src/benchiq/schema/checks.py`
- `src/benchiq/config.py` (minimal stub)

**Implementation tasks**
- Define required columns and dtypes.
- Implement PK uniqueness check and duplicate policy.
- Implement binary score check.
- Create a `ValidationReport` dataclass (counts, warnings, errors).

**Tests to add**
- Unit tests covering:
  - missing columns → fail,
  - non-binary score → fail,
  - duplicates under default policy → fail.

**Acceptance criteria**
- `benchiq.validate(...)` on toy data returns expected failures.

**Likely failure modes/blockers**
- Pandas nullable integer dtypes edge cases; keep scores as `Int8` with `pd.NA`.

---

### Ticket 2 — I/O: load_bundle + artifact writing
**Goal**  
Load CSV/Parquet inputs, canonicalize, and write stage artifacts + manifest.

**Files/modules**
- `src/benchiq/io/load.py`
- `src/benchiq/io/write.py`
- `src/benchiq/logging.py`

**Implementation tasks**
- Implement `load_bundle()` reading Parquet/CSV.
- Implement deterministic hashing of inputs (sha256).
- Implement artifact writer utility:
  - `write_parquet(df, path)`,
  - `write_json(obj, path)`.
- Implement `manifest.json` schema and incremental update.

**Tests to add**
- Unit test: round-trip load→write→load preserves row counts and PK uniqueness.
- Unit test: manifest includes dependency versions.

**Acceptance criteria**
- Running load+write on toy fixture produces `out/<run_id>/artifacts/00_canonical/*`.

**Likely failure modes/blockers**
- Parquet engine issues; require pyarrow.

---

### Ticket 3 — Preprocessing stats: item SD/mean/point-biserial + filters
**Goal**  
Compute metabench-style item statistics and apply first-class preprocessing filters.

**Files/modules**
- `src/benchiq/preprocess/stats.py`
- `src/benchiq/preprocess/filters.py`

**Implementation tasks**
- Implement per-benchmark pivot to matrix (models × items) for computation, but preserve long-table as truth.
- Implement point-biserial per metabench formula (correlation of item with total score). citeturn7view3turn8view0  
- Implement filters with defaults mirroring metabench’s preprocessing script thresholds:
  - `sd<=0.01`, `mean>=0.95`, `abs(disc)<0.05`, drop low-tail 0.1% models (optional toggle). citeturn7view5turn8view0  
- Add “insufficient coverage” filters (BenchIQ-defined): item coverage and model coverage thresholds.

**Tests to add**
- Unit tests on small constructed matrices where the filtered items are known.
- Test that thresholds match expected metabench values.

**Acceptance criteria**
- For toy benchmark, preprocessing emits item_stats and filtered lists to disk.

**Likely failure modes/blockers**
- Handling missing responses; ensure stats use available-case logic and record n per item.

---

### Ticket 4 — Score computation (percent) + grand score overlap logic
**Goal**  
Compute full benchmark percent scores and grand mean scores.

**Files/modules**
- `src/benchiq/preprocess/stats.py` (or new `scores.py`)
- `src/benchiq/reconstruct/metrics.py` (shared utilities)

**Implementation tasks**
- Compute `scores_full` per model×benchmark as `100*mean(score)`.
- Enforce coverage thresholds; otherwise set missing.
- Compute `scores_grand` only for models with complete bundle coverage.

**Tests to add**
- Unit tests for correctness with known means and coverage behavior.

**Acceptance criteria**
- `scores_full.parquet` and `scores_grand.parquet` are written and match expected.

**Likely failure modes/blockers**
- Small bundles: no overlap; ensure graceful skip of grand.

---

### Ticket 5 — Model-level splitting (global + per-benchmark)
**Goal**  
Implement metabench-like global test split stratified on grand mean score, plus per-benchmark train/val splits.

**Files/modules**
- `src/benchiq/split/splitters.py`

**Implementation tasks**
- Implement score binning for stratification (quantile bins).
- If grand score available with sufficient N:
  - split `p_test=0.1` into global test set, mirroring metabench’s approach. citeturn30view3  
- Otherwise:
  - create per-benchmark splits only (still by model).
- Emit split diagnostics.

**Tests to add**
- Verify no item-level splitting occurs.
- Verify splits are disjoint and sizes within tolerance.

**Acceptance criteria**
- `splits_models.parquet` written; per-benchmark splits written.

**Likely failure modes/blockers**
- If too few unique bins for stratification, fallback to random split with warning.

---

### Ticket 6 — GAM backend wrapper (pyGAM) + CV harness
**Goal**  
Provide a stable, testable GAM abstraction and cross-validation utilities.

**Files/modules**
- `src/benchiq/reconstruct/gam.py`

**Implementation tasks**
- Implement a wrapper that fits a `pyGAM` regression model and predicts. citeturn24search15turn24search30  
- Implement CV selection of smoothing parameter(s) using k-fold splits (RMSE objective).
- Implement serialization strategy:
  - if direct pickling is used, record python/package versions in manifest,
  - also export a JSON summary of model hyperparameters.

**Tests to add**
- Fit on simple synthetic nonlinear function; RMSE below threshold.
- CV chooses a finite parameter set and improves over naive baseline.

**Acceptance criteria**
- `pyGAM` fit+predict works in isolation and in CV loop.

**Likely failure modes/blockers**
- pyGAM API changes; pin minimum version and add smoke tests based on documented installation. citeturn24search15turn24search30  

---

### Ticket 7 — Cross-validated subsampling to k_preselect
**Goal**  
Implement metabench-style random CV subsampling with minimax validation selection.

**Files/modules**
- `src/benchiq/subsample/random_cv.py`

**Implementation tasks**
- Implement k-fold CV over models.
- For each iteration:
  - sample item subset of size `k_preselect`,
  - compute reduced subscore per model,
  - fit GAM `full_score ~ s(reduced)` on fold-train,
  - compute fold-val RMSE and (optionally) global-test RMSE tracking. citeturn32view0turn32view5turn32view2  
- Aggregate results per seed:
  - compute max RMSE across folds, choose minimax best. citeturn32view2turn32view3  
- Write full CV table, chosen subset, and summary.

**Tests to add**
- Integration-style unit test on small synthetic benchmark:
  - verifies chosen subset has correct size,
  - verifies results table shape matches (`n_iter * n_folds` rows).

**Acceptance criteria**
- For a toy dataset, the function runs and writes artifacts; `k_preselect` respected.

**Likely failure modes/blockers**
- Runtime blowups; implement optional parallelization via joblib and an `n_iter` cap.

---

### Ticket 8 — IRT backend: girth 2PL fit adapter
**Goal**  
Fit 2PL using girth and standardize outputs to BenchIQ item-parameter table format.

**Files/modules**
- `src/benchiq/irt/backends/girth_backend.py`
- `src/benchiq/irt/fit.py`

**Implementation tasks**
- Convert responses to girth format; tag missing data using `tag_missing_data` as documented. citeturn28view0turn27search15  
- Call `twopl_mml` and capture `Discrimination`, `Difficulty`. citeturn28view0turn10view0  
- Standardize:
  - ensure discrimination is positive (or record sign conventions),
  - store fit diagnostics and runtime.

**Tests to add**
- Synthetic IRT data test: recovered parameters correlate with true params.

**Acceptance criteria**
- `irt_item_params.parquet` written with expected columns.

**Likely failure modes/blockers**
- Missing data tagging edge cases; follow girth README guidance and add explicit checks. citeturn28view0  

---

### Ticket 9 — Fisher information grid + information-based item selection to k_final
**Goal**  
Compute per-item information across theta grid and select items across ability range.

**Files/modules**
- `src/benchiq/irt/info.py`
- `src/benchiq/select/information_filter.py`

**Implementation tasks**
- Implement analytic 2PL `p(θ)` and `I(θ)`; verify non-negativity.
- Implement theta grid strategies:
  - uniform,
  - difficulty-range-based.
- Implement quantile/bin selection loop:
  - one item per bin by max info, remove selected item from pool, until `k_final`. citeturn17view0turn16view5turn19view0  
- Emit expected test information curves plots.

**Tests to add**
- Unit test: selection returns ≤ k_final and no duplicates.
- Unit test: with contrived info surfaces, picks expected items.

**Acceptance criteria**
- `subset_final.parquet` contains the final item list and selection diagnostics.

**Likely failure modes/blockers**
- If info is uniformly low and thresholding yields too few items, warn and relax threshold only if user configured; otherwise output smaller subset and flag.

---

### Ticket 10 — Theta estimation (MAP/EAP) + theta SE
**Goal**  
Compute theta and theta SE per model per benchmark on reduced items.

**Files/modules**
- `src/benchiq/irt/theta.py`

**Implementation tasks**
- Implement MAP via 1D optimization (bounded) with Normal(0,1) prior. citeturn15view1turn20view1  
- Implement EAP via grid posterior integration (optional).
- Compute theta SE from test info at theta_hat.
- Handle edge cases: all-correct/all-wrong, missing-heavy patterns.

**Tests to add**
- Known monotonicity checks (higher raw scores → higher theta on average).
- Stability test: theta finite and SE finite.

**Acceptance criteria**
- `theta_estimates.parquet` produced for train/val/test splits.

**Likely failure modes/blockers**
- Numerical underflow in likelihood; use log-sum-exp and safe logs.

---

### Ticket 11 — Linear predictor feature (no-intercept weighted-average model)
**Goal**  
Implement benchmark-specific linear predictor and reduced subscore generation consistent with metabench’s `train.lm` concept.

**Files/modules**
- `src/benchiq/reconstruct/linear_predictor.py`

**Implementation tasks**
- For each benchmark:
  - build reduced item response matrix `X` (train models),
  - target `y` = full benchmark percent score,
  - fit OLS without intercept (optionally ridge),
  - produce predictions `lin_b` for train and test,
  - compute `sub_b` as percent correct on reduced items. citeturn23view2turn23view5turn20view2  
- Store coefficients, training RMSE, and notes.

**Tests to add**
- Small synthetic linear case: recovered coefficients close to true weights.

**Acceptance criteria**
- Feature table contains `lin_b` and `sub_b` and matches expected shapes.

**Likely failure modes/blockers**
- Rank deficiency; automatically switch to ridge and record.

---

### Ticket 12 — Feature matrix builder (marginal + joint)
**Goal**  
Build feature matrices for GAM reconstruction (theta, theta SE, subscore, linpred, and grand summaries).

**Files/modules**
- `src/benchiq/reconstruct/features.py`

**Implementation tasks**
- Create `features_marginal` per benchmark.
- If overlap sufficient, create `features_joint` including:
  - all benchmark thetas,
  - grand_sub, grand_lin (means across benchmarks). citeturn23view9turn20view2  
- Join split labels and targets.

**Tests to add**
- Assert no leakage: test models are never used to fit transformations that are learned (e.g., ridge alpha selection) unless explicitly permitted (v0.1 should keep it train-only).

**Acceptance criteria**
- `features_marginal.parquet` always written; `features_joint.parquet` written or skipped with explicit reason.

**Likely failure modes/blockers**
- Overlap computation mistakes; include overlap count tests.

---

### Ticket 13 — GAM reconstruction: marginal and joint, cross-validated
**Goal**  
Fit GAMs, generate predictions, compute RMSE/MAE/corr metrics, and write reconstruction artifacts.

**Files/modules**
- `src/benchiq/reconstruct/gam.py` (extend)
- `src/benchiq/reconstruct/metrics.py`
- `src/benchiq/viz/plots.py`

**Implementation tasks**
- For each benchmark:
  - fit marginal GAM on train, tune on val (CV),
  - evaluate on test, store predictions and residuals,
  - if joint enabled, fit joint GAM and evaluate similarly. citeturn20view2turn32view5turn30view3  
- Produce calibration plots (pred vs actual) and residual plots.

**Tests to add**
- Integration test: on synthetic bundle, marginal RMSE < baseline RMSE (baseline = predicting train mean).
- If joint enabled in synthetic correlated case, joint RMSE improves over marginal.

**Acceptance criteria**
- `reconstruction_summary.parquet` written with per-benchmark metrics.

**Likely failure modes/blockers**
- pyGAM fit instability; reduce spline counts and adjust lam grid; keep deterministic.

---

### Ticket 14 — Redundancy / compressibility module (benchmark level only)
**Goal**  
Implement correlation matrices, optional factor analysis, and cross-benchmark predictability metrics.

**Files/modules**
- `src/benchiq/redundancy/corr.py`
- `src/benchiq/redundancy/factor.py`
- `src/benchiq/redundancy/compress.py`

**Implementation tasks**
- Compute theta correlation matrix with overlap counts.
- Factor analysis:
  - if `factor_analyzer` installed, run MINRES + oblimin for 1–3 factors; otherwise sklearn FactorAnalysis (no rotation). citeturn26view0turn9search2  
- Compressibility:
  - for each benchmark, fit “cross-only” predictor excluding the benchmark’s own features and compute RMSE vs marginal baseline. citeturn20view2turn32view5  
- Write all artifacts and plots.

**Tests to add**
- Synthetic correlated bundle: redundancy metrics behave monotonically with correlation increases.

**Acceptance criteria**
- `compressibility.parquet` and `redundancy_report.json` written.

**Likely failure modes/blockers**
- Small overlap sets; ensure skip with explicit reason.

---

### Ticket 15 — CLI implementation
**Goal**  
Implement `benchiq validate`, `benchiq run`, and `benchiq metabench run`.

**Files/modules**
- `src/benchiq/cli/main.py`
- `src/benchiq/cli/commands_*.py`
- `docs/cli.md`

**Implementation tasks**
- Wire CLI to the runner and config loading.
- Ensure CLI is reproducible:
  - requires an explicit output dir,
  - prints run_id and key metrics,
  - exits non-zero on hard validation failures.

**Tests to add**
- CLI smoke tests using `pytest` and `subprocess` on tiny fixtures.

**Acceptance criteria**
- `benchiq validate ...` and `benchiq run ...` work end-to-end on synthetic fixture.

**Likely failure modes/blockers**
- Windows path quirks; use pathlib everywhere.

---

### Ticket 16 — BenchIQRunner orchestration + stage manifests
**Goal**  
Implement the orchestrator that runs all stages, writes stage-level manifests, and supports partial reruns (deterministic).

**Files/modules**
- `src/benchiq/runner.py` (new)
- `src/benchiq/__init__.py` (exports)

**Implementation tasks**
- Define stage DAG and ensure stage outputs are present before proceeding.
- Implement `--start-at` / `--stop-after` for debugging (optional but useful).
- Record per-stage runtime/memory notes.

**Tests to add**
- Integration test: rerun with same config produces identical selected item IDs and identical metrics (within floating tolerances).

**Acceptance criteria**
- Runner returns `RunResult` with paths and summary.

**Likely failure modes/blockers**
- Non-determinism from parallelism; default to deterministic unless user enables parallel.

---

### Ticket 17 — Metabench validation mode (data + toleranced assertions)
**Goal**  
Provide a validation-mode harness that demonstrates methodological correctness without making BenchIQ a metabench-only tool.

**Files/modules**
- `src/benchiq/cli/commands_metabench.py`
- `docs/design/metabench_validation.md`
- `tests/regression/test_metabench_validation.py`
- `tests/regression/expected/metabench_metrics.json` (toleranced expectations)

**Implementation tasks**
- Define “strict metabench-validation mode” config:
  - preprocessing defaults fixed to metabench thresholds,
  - split logic matches global grand-mean stratification,
  - subsampling defaults `k_preselect=350`, `n_folds=5`, `n_iter` reduced for CI but full for manual runs,
  - selection defaults approximate metabench’s use of ~250 quantiles and lambda recommendation (documented). citeturn5view0turn16view4turn30view3turn32view5  
- Add a “generic arbitrary-benchmark mode” config path that does not assume metabench benchmark IDs or sizes.

**Tests to add**
- Regression test that runs a small metabench snapshot (or prepackaged tiny subset) and checks that:
  - artifacts exist,
  - key metrics are within tolerance.

**Acceptance criteria**
- A reader can run `benchiq metabench run` and get a reproducible report.

**Likely failure modes/blockers**
- Dataset availability/licensing and size; if full dataset can’t ship, ship a tiny derived fixture and document how to fetch the full snapshot.

---

### Ticket 18 — Documentation + “how to prompt Codex” guidance for contributors
**Goal**  
Make the repo self-explanatory for contributors and explicitly teach how to drive Codex effectively on this project.

**Files/modules**
- `README.md` (expand)
- `docs/design/schema.md`
- `docs/design/v0_1_scope.md` (finalize)
- `docs/contributing.md`

**Implementation tasks**
- Add a “Codex execution protocol” section:
  - spawn subagents for (schema, IRT, GAM, CLI, tests),
  - require micro-steps and self-tests,
  - include a prompt template aligning with Codex’s “Goal/Context/Constraints/Done when” best practice guidance. citeturn33view0turn33view1turn33view2turn33view3  
- Provide example prompts for each ticket type (bugfix, new stage, refactor, test coverage).

**Tests to add**
- None (docs).

**Acceptance criteria**
- New contributor can run the synthetic demo and understand artifacts.

**Likely failure modes/blockers**
- Docs drift; enforce in PR checklist.

---

**Minimum Publishable BenchIQ v0.1 checklist**
- [ ] One canonical `responses_long` table is the internal source of truth; all derived matrices are reproducible views.  
- [ ] Splits are **by model only**; no item-level split leakage; global test split supported when grand overlap exists. citeturn30view3turn32view5  
- [ ] Benchmark-wise preprocessing implements (and defaults to) metabench-style variance / ceiling / point-biserial filters + explicit coverage filters; all decisions logged. citeturn7view5turn8view0  
- [ ] Cross-validated subsampling to fixed `k_preselect[b]` works and writes CV artifacts (minimax selection supported). citeturn32view2turn32view5  
- [ ] 2PL IRT fitting backend works (core: girth) and produces item parameters. citeturn28view0turn10view0  
- [ ] Fisher-information selection produces `k_final[b]` reduced subsets across the theta range and writes information diagnostics. citeturn19view0turn17view0  
- [ ] Theta and theta SE estimates are emitted per benchmark per model for all splits. citeturn20view1turn15view1  
- [ ] Reconstruction features include theta/theta SE/subscore + benchmark linear predictor + grand summaries; written to disk. citeturn23view2turn23view9turn20view2  
- [ ] GAM reconstruction (core: pyGAM) runs marginal and (when overlap permits) joint; cross-validated; writes predictions and RMSE on held-out test models. citeturn24search15turn20view2turn30view3  
- [ ] Benchmark-level redundancy/compressibility outputs include correlation matrices and a conservative cross-predictability metric; no item-level multidimensional claims. citeturn23view7turn26view0turn20view2  
- [ ] Strict metabench-validation mode exists and is the primary regression test harness; generic mode works on arbitrary benchmark bundles with explicit warnings/refusals. citeturn0search4turn5view0turn33view0  
- [ ] Every major stage writes inspectable artifacts + manifest; CLI runs are reproducible. citeturn33view0turn33view3  

**1-sentence product claim:** BenchIQ is an open-source, artifact-first Python workflow that distills user-chosen LLM benchmark bundles into compact subsets and benchmark-specific latent abilities, reconstructs full normalized benchmark scores with cross-validated GAMs, and reports conservative benchmark-level redundancy/compressibility—validated against metabench as a methodological reference case. citeturn0search0turn20view2turn33view0