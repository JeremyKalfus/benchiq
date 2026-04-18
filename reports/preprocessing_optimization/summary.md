# preprocessing optimization

## recommendation

- default decision: keep the current psychometric default story, but add a recommended reconstruction-first profile
- recommended profile: `reconstruction_relaxed` (`reconstruction_first_relaxed`)
- recommended preselection method for the reconstruction-first path: `deterministic_info`
- default rationale: the locked v0.1 spec still defines psychometric-style defaults, so this pass documents the stronger reconstruction-first profile instead of silently changing `BenchIQConfig()`

## search plan

- compact broad search profiles: 11
- compact broad search methods: `random_cv`, `deterministic_info`
- compact broad search seeds: [7, 11, 19]
- large confirmation shortlist: ['baseline_current', 'reconstruction_relaxed', 'minimal_cleaning', 'no_low_tail']
- large confirmation method: `deterministic_info`
- large method check method: `random_cv` on the baseline and confirmed winner

## compact top rows

- `baseline_current`: rmse_mean=13.9418, seed_rmse_std=1.6494, final_selection_stability_mean=0.8888888888888888, runtime_mean_seconds=1.9114
- `psychometric_default`: rmse_mean=13.9418, seed_rmse_std=1.6494, final_selection_stability_mean=0.8888888888888888, runtime_mean_seconds=1.9580
- `no_low_tail`: rmse_mean=13.9418, seed_rmse_std=1.6494, final_selection_stability_mean=0.8888888888888888, runtime_mean_seconds=2.0176
- `ceiling_strict`: rmse_mean=13.9418, seed_rmse_std=1.6494, final_selection_stability_mean=0.8888888888888888, runtime_mean_seconds=2.0414
- `ceiling_off`: rmse_mean=13.9418, seed_rmse_std=1.6494, final_selection_stability_mean=0.8888888888888888, runtime_mean_seconds=2.0515

## large confirmation top rows

- `reconstruction_relaxed`: rmse_mean=0.8955, seed_rmse_std=0.0220, final_selection_stability_mean=0.9428384381902152, runtime_mean_seconds=270.3517
- `minimal_cleaning`: rmse_mean=0.8955, seed_rmse_std=0.0220, final_selection_stability_mean=0.9428384381902152, runtime_mean_seconds=275.9078
- `no_low_tail`: rmse_mean=1.0172, seed_rmse_std=0.0235, final_selection_stability_mean=0.9707400332284165, runtime_mean_seconds=174.8927
- `baseline_current`: rmse_mean=1.0558, seed_rmse_std=0.0280, final_selection_stability_mean=0.9707036069012446, runtime_mean_seconds=181.3124

## head checks

- compact_baseline / joint / elastic_net: rmse_mean=16.7690, mae_mean=13.7143, runtime_mean_seconds=0.0240
- compact_baseline / joint / gam: rmse_mean=18.4342, mae_mean=14.5434, runtime_mean_seconds=0.0529
- compact_baseline / marginal / elastic_net: rmse_mean=18.1735, mae_mean=15.6192, runtime_mean_seconds=0.0182
- compact_baseline / marginal / gam: rmse_mean=18.3068, mae_mean=15.6136, runtime_mean_seconds=0.0348
- compact_winner / joint / elastic_net: rmse_mean=16.7690, mae_mean=13.7143, runtime_mean_seconds=0.0242
- compact_winner / joint / gam: rmse_mean=18.4342, mae_mean=14.5434, runtime_mean_seconds=0.0577
- compact_winner / marginal / elastic_net: rmse_mean=18.1735, mae_mean=15.6192, runtime_mean_seconds=0.0191
- compact_winner / marginal / gam: rmse_mean=18.3068, mae_mean=15.6136, runtime_mean_seconds=0.0344
- large_baseline / marginal / elastic_net: rmse_mean=nan, mae_mean=nan, runtime_mean_seconds=0.1287
- large_baseline / marginal / gam: rmse_mean=nan, mae_mean=nan, runtime_mean_seconds=0.2121
- large_winner / joint / gam: rmse_mean=1.0023, mae_mean=0.6858, runtime_mean_seconds=0.4670
- large_winner / joint / elastic_net: rmse_mean=1.0433, mae_mean=0.7309, runtime_mean_seconds=0.6374
- large_winner / marginal / gam: rmse_mean=1.0773, mae_mean=0.7491, runtime_mean_seconds=0.1729
- large_winner / marginal / elastic_net: rmse_mean=1.1056, mae_mean=0.7770, runtime_mean_seconds=0.1537
