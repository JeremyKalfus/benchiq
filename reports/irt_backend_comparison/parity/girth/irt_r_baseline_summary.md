# irt r-baseline comparison

- generated_at: `2026-04-21T22:57:40.863010+00:00`
- status: `ok`
- table: `/Users/jeremykalfus/CodingProjects/BenchIQ/reports/irt_backend_comparison/parity/girth/irt_r_baseline_item_comparison.csv`
- model_count: `300`
- item_count: `6`
- random_seed: `11`
- theta_method: `EAP`
- backend: `girth`
- backend_options: `{'max_iteration': 60}`

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
- theta_slope: `1.0000816224271316`
- theta_intercept: `0.0019307601171090528`

## metrics

- theta pearson: `0.9999997152940514`
- theta spearman: `0.9999999999999999`
- discrimination mae: `0.0022081624825220545`
- difficulty mae: `0.0016424761525569986`
- mean icc rmse: `0.0003833838871593282`
- max icc rmse: `0.0007705634290327179`

## gate

- passed: `True`
- failure_count: `0`
- theta_pearson_min: `0.95`
- theta_spearman_min: `0.95`
- icc_mean_rmse_max: `0.08`

### checks

- status: `ok` == `ok` -> `True`
- theta.pearson: `0.9999997152940514` >= `0.95` -> `True`
- theta.spearman: `0.9999999999999999` >= `0.95` -> `True`
- icc.mean_rmse: `0.0003833838871593282` <= `0.08` -> `True`
