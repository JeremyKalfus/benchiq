# run summary

- run_id: `budget_sweep__synthetic_sparse_overlap__kp-25__kf-10__seed-7`
- executed stages: 00_bundle, 01_preprocess, 02_scores, 03_splits, 04_subsample, 05_irt, 06_select, 07_theta, 08_linear, 08_features, 09_reconstruct
- warning count: 47

## skip reasons

- `02_scores`: {"grand_scores": "overlap_below_joint_threshold"}
- `08_features`: {"joint_features": "joint_feature_overlap_below_threshold"}
- `09_reconstruct`: {"sparse_b1": "joint_feature_overlap_below_threshold", "sparse_b2": "joint_feature_overlap_below_threshold", "sparse_b3": "joint_feature_overlap_below_threshold", "sparse_b4": "joint_feature_overlap_below_threshold", "sparse_b5": "joint_feature_overlap_below_threshold", "sparse_b6": "joint_feature_overlap_below_threshold"}

## headline metrics

- selected items by benchmark: {"sparse_b1": 10, "sparse_b2": 10, "sparse_b3": 10, "sparse_b4": 10, "sparse_b5": 10, "sparse_b6": 10}
- marginal test rmse by benchmark: {"sparse_b1": null, "sparse_b2": null, "sparse_b3": null, "sparse_b4": null, "sparse_b5": null, "sparse_b6": null}
- joint test rmse by benchmark: {"sparse_b1": null, "sparse_b2": null, "sparse_b3": null, "sparse_b4": null, "sparse_b5": null, "sparse_b6": null}
