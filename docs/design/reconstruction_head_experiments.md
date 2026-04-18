# reconstruction head experiments

BenchIQ keeps GAMs as the current product reconstruction head, but the pivot work adds explicit
comparison experiments against two alternatives:

- Elastic Net as the regularized linear baseline
- XGBoost as the tree baseline

This comparison harness is part of the repo's supported experiment stack, not the core
calibration/deployment runtime path. The XGBoost dependency lives in the optional
`.[experiments]` extra and is also included in `.[dev]` for contributor installs.

## why xgboost

This repo uses XGBoost instead of LightGBM for the comparison harness because it has straightforward
python-only wheel availability in the environments BenchIQ already targets.

That keeps the tree comparison strong without widening the repo into broader build-tool or native
dependency work.

## comparison rules

The experiment harness compares all heads on:

- the same stage-08 feature tables
- the same train / val / test splits
- the same reduced subsets produced upstream

For each method it records:

- held-out RMSE
- MAE
- Pearson
- Spearman
- runtime
- seed spread across repeated runs

The outputs live under `reports/experiments/reconstruction_heads/`.
