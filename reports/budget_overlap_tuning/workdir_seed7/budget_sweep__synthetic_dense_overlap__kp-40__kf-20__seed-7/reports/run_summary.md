# run summary

- run_id: `budget_sweep__synthetic_dense_overlap__kp-40__kf-20__seed-7`
- executed stages: 00_bundle, 01_preprocess, 02_scores, 03_splits, 04_subsample, 05_irt, 06_select, 07_theta, 08_linear, 08_features, 09_reconstruct
- warning count: 17

## skip reasons

- `08_features`: {"joint_features": "joint_feature_overlap_below_threshold"}
- `09_reconstruct`: {"dense_b1": "joint_feature_overlap_below_threshold", "dense_b2": "joint_feature_overlap_below_threshold", "dense_b3": "joint_feature_overlap_below_threshold", "dense_b4": "joint_feature_overlap_below_threshold", "dense_b5": "joint_feature_overlap_below_threshold"}

## headline metrics

- selected items by benchmark: {"dense_b1": 20, "dense_b2": 20, "dense_b3": 20, "dense_b4": 20, "dense_b5": 20}
- marginal test rmse by benchmark: {"dense_b1": 5.437947793480156, "dense_b2": 6.31133526887825, "dense_b3": 7.36488912198793, "dense_b4": 7.150277465632351, "dense_b5": 6.533344749472478}
- joint test rmse by benchmark: {"dense_b1": null, "dense_b2": null, "dense_b3": null, "dense_b4": null, "dense_b5": null}
