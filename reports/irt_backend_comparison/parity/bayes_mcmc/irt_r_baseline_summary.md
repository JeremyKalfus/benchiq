# irt r-baseline comparison

- generated_at: `2026-04-21T23:27:57.241114+00:00`
- status: `ok`
- table: `/Users/jeremykalfus/CodingProjects/BenchIQ/reports/irt_backend_comparison/parity/bayes_mcmc/irt_r_baseline_item_comparison.csv`
- model_count: `300`
- item_count: `6`
- random_seed: `11`
- theta_method: `EAP`
- backend: `bayes_mcmc`
- backend_options: `{'draws': 1, 'tune': 1, 'chains': 1, 'cores': 1, 'target_accept': 0.9}`

## environment

- python version: `3.14.4`
- python implementation: `CPython`
- python executable: `/Users/jeremykalfus/CodingProjects/BenchIQ/.venv/bin/python`
- python platform: `macOS-15.2-arm64-arm-64bit-Mach-O`
- arviz version: `0.23.4`
- pymc version: `5.28.4`
- numpy version: `2.4.4`
- pandas version: `2.3.3`
- scipy version: `1.16.3`
- girth version: `0.8.0`
- rscript available: `True`
- rscript path: `/opt/homebrew/bin/Rscript`
- r version: `R version 4.5.2 (2025-10-31)`
- r version_error: `None`
- mirt installed: `True`
- mirt version: `1.46.1`
- mirt check_error: `None`

## alignment

- sign_applied: `1.0`
- theta_slope: `0.9600400287924934`
- theta_intercept: `0.01982603581183938`

## metrics

- theta pearson: `0.9788901291195302`
- theta spearman: `0.96197255746966`
- discrimination mae: `1.0873943048017616`
- difficulty mae: `0.1741856081398092`
- mean icc rmse: `0.08005333792266486`
- max icc rmse: `0.13791904467115726`

## gate

- passed: `False`
- failure_count: `1`
- theta_pearson_min: `0.95`
- theta_spearman_min: `0.95`
- icc_mean_rmse_max: `0.08`

### checks

- status: `ok` == `ok` -> `True`
- theta.pearson: `0.9788901291195302` >= `0.95` -> `True`
- theta.spearman: `0.96197255746966` >= `0.95` -> `True`
- icc.mean_rmse: `0.08005333792266486` <= `0.08` -> `False`

### failures

- icc.mean_rmse was 0.080053, required <= 0.080000
