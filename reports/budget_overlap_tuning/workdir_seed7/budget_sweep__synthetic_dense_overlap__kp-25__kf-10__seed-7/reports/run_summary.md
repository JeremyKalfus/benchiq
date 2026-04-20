# run summary

- run_id: `budget_sweep__synthetic_dense_overlap__kp-25__kf-10__seed-7`
- executed stages: 00_bundle, 01_preprocess, 02_scores, 03_splits, 04_subsample, 05_irt, 06_select, 07_theta, 08_linear, 08_features, 09_reconstruct
- warning count: 22

## skip reasons

- `08_features`: {"joint_features": "joint_feature_overlap_below_threshold"}
- `09_reconstruct`: {"dense_b1": "joint_feature_overlap_below_threshold", "dense_b2": "joint_feature_overlap_below_threshold", "dense_b3": "joint_feature_overlap_below_threshold", "dense_b4": "joint_feature_overlap_below_threshold", "dense_b5": "joint_feature_overlap_below_threshold"}

## headline metrics

- selected items by benchmark: {"dense_b1": 10, "dense_b2": 10, "dense_b3": 10, "dense_b4": 10, "dense_b5": 10}
- marginal test rmse by benchmark: {"dense_b1": 7.685303181153688, "dense_b2": 8.827811406560878, "dense_b3": 9.60249750923712, "dense_b4": 9.281949041294665, "dense_b5": 7.843721756447624}
- joint test rmse by benchmark: {"dense_b1": null, "dense_b2": null, "dense_b3": null, "dense_b4": null, "dense_b5": null}
