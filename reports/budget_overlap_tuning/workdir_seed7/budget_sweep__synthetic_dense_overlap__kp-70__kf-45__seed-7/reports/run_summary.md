# run summary

- run_id: `budget_sweep__synthetic_dense_overlap__kp-70__kf-45__seed-7`
- executed stages: 00_bundle, 01_preprocess, 02_scores, 03_splits, 04_subsample, 05_irt, 06_select, 07_theta, 08_linear, 08_features, 09_reconstruct
- warning count: 17

## skip reasons

- `08_features`: {"joint_features": "joint_feature_overlap_below_threshold"}
- `09_reconstruct`: {"dense_b1": "joint_feature_overlap_below_threshold", "dense_b2": "joint_feature_overlap_below_threshold", "dense_b3": "joint_feature_overlap_below_threshold", "dense_b4": "joint_feature_overlap_below_threshold", "dense_b5": "joint_feature_overlap_below_threshold"}

## headline metrics

- selected items by benchmark: {"dense_b1": 45, "dense_b2": 45, "dense_b3": 45, "dense_b4": 45, "dense_b5": 45}
- marginal test rmse by benchmark: {"dense_b1": 3.024082503557523, "dense_b2": 4.020094263054955, "dense_b3": 4.3520704920623245, "dense_b4": 5.497887862337627, "dense_b5": 5.937431209047222}
- joint test rmse by benchmark: {"dense_b1": null, "dense_b2": null, "dense_b3": null, "dense_b4": null, "dense_b5": null}
