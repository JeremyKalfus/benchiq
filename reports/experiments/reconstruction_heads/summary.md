# reconstruction head comparison

## winners

- `joint` winner: `gam` (rmse_mean=13.1662, runtime_mean_seconds=0.0493)
- `marginal` winner: `gam` (rmse_mean=13.8702, runtime_mean_seconds=0.0292)

## summary

- `joint` `gam`: rmse_mean=13.1662, mae_mean=12.7773, pearson_mean=0.9246, spearman_mean=0.8676, runtime_mean_seconds=0.0493, seed_rmse_std=0.0000
- `joint` `xgboost`: rmse_mean=13.3196, mae_mean=12.0103, pearson_mean=0.9057, spearman_mean=0.9549, runtime_mean_seconds=0.0608, seed_rmse_std=0.0000
- `joint` `elastic_net`: rmse_mean=13.6599, mae_mean=13.2054, pearson_mean=0.9184, spearman_mean=0.8676, runtime_mean_seconds=0.0234, seed_rmse_std=0.0203
- `marginal` `gam`: rmse_mean=13.8702, mae_mean=11.0342, pearson_mean=0.9304, spearman_mean=0.9404, runtime_mean_seconds=0.0292, seed_rmse_std=0.2483
- `marginal` `elastic_net`: rmse_mean=14.6013, mae_mean=11.7821, pearson_mean=0.8970, spearman_mean=0.8508, runtime_mean_seconds=0.0178, seed_rmse_std=0.1299
- `marginal` `xgboost`: rmse_mean=17.2444, mae_mean=14.8686, pearson_mean=0.7946, spearman_mean=0.7016, runtime_mean_seconds=0.0537, seed_rmse_std=0.1736
