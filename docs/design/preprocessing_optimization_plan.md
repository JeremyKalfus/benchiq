# preprocessing optimization plan

## purpose

This pass optimizes BenchIQ preprocessing for held-out score-reconstruction quality and
reproducibility, not for metabench parity.

Companion artifacts from the completed pass:

- [`reports/preprocessing_optimization/summary.md`](/Users/jeremykalfus/CodingProjects/BenchIQ/reports/preprocessing_optimization/summary.md)
- [`reports/preprocessing_optimization/best_config.json`](/Users/jeremykalfus/CodingProjects/BenchIQ/reports/preprocessing_optimization/best_config.json)
- [`scripts/run_preprocessing_optimization.py`](/Users/jeremykalfus/CodingProjects/BenchIQ/scripts/run_preprocessing_optimization.py)

The compact saved comparison harness and the real-data release subset are both useful here, but
they answer different questions:

- the compact harness is the fast search space for broad profile exploration
- the real-data release subset is the higher-signal confirmation path for whether a preprocessing
  winner still holds up outside the tiny fixture

## current default rules

There are two current baselines that matter for this pass.

### product default config

`BenchIQConfig()` currently means:

- `drop_low_tail_models_quantile = 0.001`
- `min_item_sd = 0.01`
- `max_item_mean = 0.95`
- `min_abs_point_biserial = 0.05`
- `min_models_per_benchmark = 100`
- `warn_models_per_benchmark = 200`
- `min_items_after_filtering = 50`
- `min_models_per_item = 50`
- `min_item_coverage = 0.8`
- `min_overlap_models_for_joint = 75`
- `min_overlap_models_for_redundancy = 75`

Stage-01 item coverage uses the spec-aligned effective floor:

- `effective_min_models_per_item = max(min_models_per_item, ceil(0.2 * n_models_benchmark))`

This means `min_models_per_item` is not a raw threshold in practice unless it exceeds the
20%-of-models floor.

### compact saved comparison setup

The existing compact saved comparison reports under `reports/selection_comparison/` and
`reports/experiments/reconstruction_heads/` do not use the full product defaults. They use the
fixture-sized setup from `tests/data/tiny_example/config.json`:

- `allow_low_n = true`
- `max_item_mean = 0.99`
- `min_abs_point_biserial = 0.0`
- `min_models_per_benchmark = 15`
- `warn_models_per_benchmark = 20`
- `min_items_after_filtering = 5`
- `min_models_per_item = 10`
- `min_overlap_models_for_joint = 15`
- `min_overlap_models_for_redundancy = 15`
- `random_seed = 7`

The compact comparison stage options are currently:

- stage 04: `k_preselect = 4`, `n_iter = 4`, `cv_folds = 3`, `checkpoint_interval = 2`,
  `lam_grid = [0.1, 1.0]`
- stage 06: `k_final = 3`, `theta_grid_size = 101`
- stage 07: `theta_grid_size = 81`
- stage 09: `lam_grid = [0.1, 1.0]`, `cv_folds = 3`, `n_splines = 5`

For the saved selection-comparison script, random-CV was widened slightly to `n_iter = 12`.

## optimization objective

Primary objective:

- minimize held-out reconstruction RMSE

Secondary objectives:

- minimize seed spread
- improve subset stability
- reduce runtime when quality is tied or nearly tied
- prefer cleaner retained-item counts when quality is tied or nearly tied
- avoid preprocessing profiles that increase warning/refusal rates or trigger skipped downstream
  stages

Operational scoring rule for this pass:

- compare benchmark-level held-out test metrics
- define `best_available_*` as joint reconstruction when joint is available, otherwise marginal
- rank profiles first by mean `best_available_test_rmse`
- treat profiles within a very small RMSE band as ties and then break ties with seed stability,
  selection stability, runtime, and retained-item efficiency

## hypotheses

- the current psychometric-style filters are likely too strict for BenchIQ’s reconstruction-first
  objective, especially the ceiling and discrimination thresholds
- turning off low-tail trimming may help held-out reconstruction on smaller bundles where dropping
  even one or two models is expensive
- relaxed or minimal-cleaning profiles may improve downstream GAM reconstruction by preserving more
  informative but imperfect items
- fully minimal cleaning may over-admit noisy items and hurt stability, so the winner is more
  likely to be a relaxed middle ground than a no-filter extreme
- `deterministic_info` is a plausible multiplier on preprocessing gains because it reduces stage-04
  seed noise
- the strongest preprocessing winner should remain at least directionally good under Elastic Net,
  not only under GAM

## evaluation protocol

### fairness rules

All profile comparisons keep the following fixed unless a comparison explicitly says otherwise:

- same response data
- same split policy
- same stage budgets (`k_preselect`, `k_final`, theta grid sizes)
- same reconstruction head for the primary comparison
- same preselection method for within-method comparisons
- same output/report conventions

### datasets

This pass uses two dataset scales:

- compact search path:
  - responses: `tests/data/metabench_validation/responses_long.csv`
  - baseline config family anchored to the current saved comparison setup
- larger confirmation path:
  - responses: `out/metabench_real_source/release_default_subset_responses_long.parquet`
  - six-benchmark release-default public subset with 6,832 models and 2,100 benchmark-item rows
  - stage budgets anchored to the full metabench-style size regime, but interpreted here only as a
    practical large-bundle BenchIQ confirmation path

### staged search strategy

The search is intentionally staged instead of running a full cross-product on the 11.7M-row real
table.

1. Run a broad compact sweep across profile families, both preselection methods, and multiple
   seeds.
2. Shortlist the most promising compact profiles plus the relevant baselines.
3. Confirm that shortlist on the larger real-data subset.
4. Run a smaller second-head check with Elastic Net on the main baseline vs winner comparison.

### preprocessing families to test

The harness must support:

- low-tail trimming on vs off
- psychometric-default thresholds vs relaxed thresholds
- discrimination filter on vs off
- ceiling filter strength variants
- minimal cleaning vs strict cleaning vs reconstruction-first relaxed cleaning
- direct threshold sweeps over:
  - `drop_low_tail_models_quantile`
  - `min_item_sd`
  - `max_item_mean`
  - `min_abs_point_biserial`
  - `min_models_per_item`
  - `min_item_coverage`

Because the current code applies `max(min_models_per_item, ceil(0.2 * n_models_benchmark))`, the
experiment outputs must record the effective stage-01 item-coverage threshold, not only the raw
requested config value.

### metrics to save

For every run and benchmark:

- RMSE
- MAE
- Pearson
- Spearman
- runtime
- retained items after preprocessing
- retained models after preprocessing
- final selected-item count
- warning count
- refusal count / skipped-stage indicators

Across seeds and profiles:

- seed spread for RMSE and MAE
- stage-06 final-set stability
- stage-04 preselect-set stability when available

## what counts as better

A preprocessing profile counts as better only if it satisfies all of the following:

- it lowers mean held-out `best_available_test_rmse` relative to the relevant baseline
- it does not win only by causing benchmark refusals or hiding hard cases behind skipped stages
- it is at least competitive on the larger confirmation path, not only on the compact fixture
- if the RMSE difference is tiny, it must win on at least one of:
  - lower seed spread
  - higher subset stability
  - lower runtime
  - fewer retained items with comparable reconstruction quality

If no profile clears that bar cleanly, the correct outcome is:

- keep the default story honest
- document the Pareto frontier
- recommend a reconstruction-first profile only as an optional override
