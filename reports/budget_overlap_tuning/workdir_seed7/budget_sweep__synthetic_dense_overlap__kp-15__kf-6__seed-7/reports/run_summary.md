# run summary

- run_id: `budget_sweep__synthetic_dense_overlap__kp-15__kf-6__seed-7`
- executed stages: 00_bundle, 01_preprocess, 02_scores, 03_splits, 04_subsample, 05_irt, 06_select, 07_theta, 08_linear, 08_features, 09_reconstruct
- warning count: 22

## skip reasons

- `08_features`: {"joint_features": "joint_feature_overlap_below_threshold"}
- `09_reconstruct`: {"dense_b1": "joint_feature_overlap_below_threshold", "dense_b2": "joint_feature_overlap_below_threshold", "dense_b3": "joint_feature_overlap_below_threshold", "dense_b4": "joint_feature_overlap_below_threshold", "dense_b5": "joint_feature_overlap_below_threshold"}

## headline metrics

- selected items by benchmark: {"dense_b1": 6, "dense_b2": 6, "dense_b3": 6, "dense_b4": 6, "dense_b5": 6}
- marginal test rmse by benchmark: {"dense_b1": 9.43528234151773, "dense_b2": 11.385368714588978, "dense_b3": 11.053946629769062, "dense_b4": 9.851499016039647, "dense_b5": 9.25454986426354}
- joint test rmse by benchmark: {"dense_b1": null, "dense_b2": null, "dense_b3": null, "dense_b4": null, "dense_b5": null}
