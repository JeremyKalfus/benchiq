# irt backend comparison

- generated_at: `2026-04-21T23:27:57.255384+00:00`
- winner: `girth`
- winner_eligible: `True`
- winner_reason: `winner selected by held-out equal-weight rmse, then seed rmse stability, failure rate, and runtime`

## candidates

- `bayes_mcmc`: equal_weight_informative_rmse_mean=NA, seed_rmse_std=NA, failure_rate=1.0000, large_bundle_runtime_mean_seconds=NA, disqualifications=['failed calibration or prediction on `compact_validation_fixture` while the other backend succeeded', 'failed calibration or prediction on `large_release_default_subset` while the other backend succeeded', 'failed calibration or prediction on `synthetic_dense_overlap` while the other backend succeeded', 'failed calibration or prediction on the large release-default subset', 'failed pipeline runs on `compact_validation_fixture` while the other backend completed', 'failed pipeline runs on `large_release_default_subset` while the other backend completed', 'failed pipeline runs on `synthetic_dense_overlap` while the other backend completed', 'failed pipeline runs on `synthetic_sparse_overlap` while the other backend completed', 'failed the aligned R parity gate']
- `girth`: equal_weight_informative_rmse_mean=6.6326, seed_rmse_std=0.5501, failure_rate=0.0000, large_bundle_runtime_mean_seconds=572.8493, disqualifications=[]

## dataset summary

- `compact_validation_fixture` / `bayes_mcmc`: rmse_mean=NA, seed_rmse_std=NA, failure_rate=1.0000, informative_dataset=False
- `compact_validation_fixture` / `girth`: rmse_mean=13.9418, seed_rmse_std=1.6494, failure_rate=0.0000, informative_dataset=True
- `large_release_default_subset` / `bayes_mcmc`: rmse_mean=NA, seed_rmse_std=NA, failure_rate=1.0000, informative_dataset=False
- `large_release_default_subset` / `girth`: rmse_mean=0.8786, seed_rmse_std=0.0199, failure_rate=0.0000, informative_dataset=True
- `synthetic_dense_overlap` / `bayes_mcmc`: rmse_mean=NA, seed_rmse_std=NA, failure_rate=1.0000, informative_dataset=False
- `synthetic_dense_overlap` / `girth`: rmse_mean=5.0772, seed_rmse_std=0.3669, failure_rate=0.0000, informative_dataset=True
- `synthetic_sparse_overlap` / `bayes_mcmc`: rmse_mean=NA, seed_rmse_std=NA, failure_rate=1.0000, informative_dataset=False
- `synthetic_sparse_overlap` / `girth`: rmse_mean=NA, seed_rmse_std=NA, failure_rate=0.0000, informative_dataset=False

## parity

- `girth`: status=ok, gate_passed=True, theta_pearson=1.0000, theta_spearman=1.0000, icc_mean_rmse=0.0004
- `bayes_mcmc`: status=ok, gate_passed=False, theta_pearson=0.9789, theta_spearman=0.9620, icc_mean_rmse=0.0801

## deployment

- `compact_validation_fixture` / `girth`: status=ok, prediction_available_rate=1.0000, deployment_rmse=14.2796
- `large_release_default_subset` / `girth`: status=ok, prediction_available_rate=1.0000, deployment_rmse=0.9395
- `synthetic_dense_overlap` / `girth`: status=ok, prediction_available_rate=1.0000, deployment_rmse=5.5249
- `synthetic_sparse_overlap` / `girth`: status=failed, prediction_available_rate=NA, deployment_rmse=NA
- `compact_validation_fixture` / `bayes_mcmc`: status=skipped, prediction_available_rate=NA, deployment_rmse=NA
- `large_release_default_subset` / `bayes_mcmc`: status=skipped, prediction_available_rate=NA, deployment_rmse=NA
- `synthetic_dense_overlap` / `bayes_mcmc`: status=skipped, prediction_available_rate=NA, deployment_rmse=NA
- `synthetic_sparse_overlap` / `bayes_mcmc`: status=skipped, prediction_available_rate=NA, deployment_rmse=NA
