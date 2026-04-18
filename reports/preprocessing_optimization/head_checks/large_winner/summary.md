# reconstruction head comparison

## winners

- `joint` winner: `gam` (rmse_mean=1.0023, runtime_mean_seconds=0.4670)
- `marginal` winner: `gam` (rmse_mean=1.0773, runtime_mean_seconds=0.1729)

## summary

- `joint` `gam`: rmse_mean=1.0023, mae_mean=0.6858, pearson_mean=0.9976, spearman_mean=0.9968, runtime_mean_seconds=0.4670, seed_rmse_std=0.0002
- `joint` `elastic_net`: rmse_mean=1.0433, mae_mean=0.7309, pearson_mean=0.9974, spearman_mean=0.9966, runtime_mean_seconds=0.6374, seed_rmse_std=0.0000
- `marginal` `gam`: rmse_mean=1.0773, mae_mean=0.7491, pearson_mean=0.9973, spearman_mean=0.9963, runtime_mean_seconds=0.1729, seed_rmse_std=0.0000
- `marginal` `elastic_net`: rmse_mean=1.1056, mae_mean=0.7770, pearson_mean=0.9971, spearman_mean=0.9962, runtime_mean_seconds=0.1537, seed_rmse_std=0.0000
