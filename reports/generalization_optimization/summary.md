# generalization optimization

## decision

- explicit decision: `B`
- reason: winner beat the default on every informative non-compact bundle, improved best-available benchmark coverage on the real bundle, and kept deployment deterministic; the sparse stress bundle was non-informative for rmse rather than negative
- baseline strategy: `baseline_current__random_cv`
- generalized winner: `reconstruction_relaxed__deterministic_info`
- challenger tracked in this pass: `minimal_cleaning__deterministic_info`

## bundle set

- `compact_validation_fixture`: rows=1260, models=60, benchmarks=3, items=21
- `large_release_default_subset`: rows=11725350, models=6832, benchmarks=6, items=1608
- `synthetic_dense_overlap`: rows=187564, models=320, benchmarks=5, items=600
- `synthetic_sparse_overlap`: rows=114661, models=650, benchmarks=6, items=630

## experiment matrix

- compact strategies: [('baseline_current', 'random_cv'), ('baseline_current', 'deterministic_info'), ('reconstruction_relaxed', 'random_cv'), ('reconstruction_relaxed', 'deterministic_info')]
- non-compact strategies: [('baseline_current', 'random_cv'), ('baseline_current', 'deterministic_info'), ('reconstruction_relaxed', 'random_cv'), ('reconstruction_relaxed', 'deterministic_info'), ('minimal_cleaning', 'deterministic_info')]
- seeds: [7, 11, 19]
- planned runs: 57
- completed run rows: 57

## core summary rows

| search_stage | dataset_id | strategy_id | best_available_test_rmse_mean | best_available_test_mae_mean | best_available_test_pearson_mean | best_available_test_spearman_mean | seed_rmse_std | final_selection_stability_mean | run_runtime_mean_seconds | best_available_benchmark_rate | benchmark_coverage_rate | joint_available_rate | refusal_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact_validation | compact_validation_fixture | reconstruction_relaxed__deterministic_info | 13.9418 | 11.2321 | 0.8558 | 0.8454 | 1.6494 | 0.8889 | 2.0342 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| compact_validation | compact_validation_fixture | baseline_current__random_cv | 13.9467 | 11.8768 | 0.8610 | 0.8521 | 2.0382 | 0.4222 | 2.6444 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| real_generalization | large_release_default_subset | minimal_cleaning__deterministic_info | 0.8955 | 0.6019 | 0.9981 | 0.9974 | 0.0220 | 0.9428 | 272.6858 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| real_generalization | large_release_default_subset | reconstruction_relaxed__deterministic_info | 0.8955 | 0.6019 | 0.9981 | 0.9974 | 0.0220 | 0.9428 | 264.5636 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| real_generalization | large_release_default_subset | baseline_current__random_cv | 1.0558 | 0.7699 | 0.9981 | 0.9966 | 0.0280 | 0.9707 | 192.6530 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| synthetic_generalization | synthetic_dense_overlap | minimal_cleaning__deterministic_info | 4.8981 | 3.8361 | 0.9462 | 0.9184 | 0.4829 | 0.8087 | 9.6493 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| synthetic_generalization | synthetic_dense_overlap | reconstruction_relaxed__deterministic_info | 4.8981 | 3.8361 | 0.9462 | 0.9184 | 0.4829 | 0.8087 | 9.6230 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| synthetic_generalization | synthetic_dense_overlap | baseline_current__random_cv | 5.5068 | 4.4523 | 0.9337 | 0.8990 | 0.4719 | 0.3818 | 10.5308 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| synthetic_generalization | synthetic_sparse_overlap | baseline_current__random_cv |  |  |  |  | 0.0000 | 0.4308 | 9.8589 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| synthetic_generalization | synthetic_sparse_overlap | minimal_cleaning__deterministic_info |  |  |  |  | 0.0000 | 0.8625 | 8.5051 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| synthetic_generalization | synthetic_sparse_overlap | reconstruction_relaxed__deterministic_info |  |  |  |  | 0.0000 | 0.8625 | 8.4931 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

## head checks

| case_id | model_type | method | rmse_mean | mae_mean | runtime_mean_seconds | seed_rmse_std |
| --- | --- | --- | --- | --- | --- | --- |
| real_winner | joint | gam | 1.0023 | 0.6858 | 0.4750 | 0.0002 |
| real_winner | joint | xgboost | 1.0101 | 0.7084 | 0.3309 | 0.0050 |
| real_winner | joint | elastic_net | 1.0433 | 0.7309 | 0.6507 | 0.0000 |
| real_winner | marginal | gam | 1.0773 | 0.7491 | 0.1785 | 0.0000 |
| real_winner | marginal | elastic_net | 1.1056 | 0.7770 | 0.1569 | 0.0000 |
| real_winner | marginal | xgboost | 1.1095 | 0.7842 | 0.1922 | 0.0000 |

## deployment validation

- deployment rmse: 0.9238
- deployment mae: 0.5959
- deployment pearson: 0.9992
- deployment spearman: 0.9992
- prediction available rate: 1.0000
- max abs delta vs calibration: 0.000000
- max abs delta repeat prediction: 0.000000

## method notes

- LightGBM: not run in this pass because XGBoost already covered the tree-comparator slot and the winner-feature tables are low-dimensional
- TabPFN: not run in this pass because it would add a heavier experimental dependency and a less deployment-friendly model family without strong expected payoff

## artifact index

- `summary.csv` and `summary.parquet`: aggregated strategy-by-bundle metrics
- `per_run_metrics.parquet`: per-run, per-benchmark metrics
- `best_profile.json`: explicit promotion decision and rationale
- `plots/`: appendix-ready figures
- `head_summary.csv`: targeted GAM / Elastic Net / XGBoost comparison on the fixed winner profile
- `reports/deployment_validation/`: deployment metrics, comparison tables, and plots

