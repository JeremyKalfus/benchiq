# reconstruction head comparison

## winners

- `joint` winner: `elastic_net` (rmse_mean=16.7690, runtime_mean_seconds=0.0240)
- `marginal` winner: `elastic_net` (rmse_mean=18.1735, runtime_mean_seconds=0.0182)

## summary

- `joint` `elastic_net`: rmse_mean=16.7690, mae_mean=13.7143, pearson_mean=0.8250, spearman_mean=0.8986, runtime_mean_seconds=0.0240, seed_rmse_std=0.7732
- `joint` `gam`: rmse_mean=18.4342, mae_mean=14.5434, pearson_mean=0.7865, spearman_mean=0.8986, runtime_mean_seconds=0.0529, seed_rmse_std=0.0000
- `marginal` `elastic_net`: rmse_mean=18.1735, mae_mean=15.6192, pearson_mean=0.7565, spearman_mean=0.7701, runtime_mean_seconds=0.0182, seed_rmse_std=0.2988
- `marginal` `gam`: rmse_mean=18.3068, mae_mean=15.6136, pearson_mean=0.7520, spearman_mean=0.7701, runtime_mean_seconds=0.0348, seed_rmse_std=0.0000
