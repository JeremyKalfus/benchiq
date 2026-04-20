# run summary

- run_id: `real_exploration_seed7__large_release_default_subset__relaxed_pb_005__deterministic_info__seed-7`
- executed stages: 00_bundle, 01_preprocess, 02_scores, 03_splits, 04_subsample, 05_irt, 06_select, 07_theta, 08_linear, 08_features, 09_reconstruct
- warning count: 22

## skip reasons

- `04_subsample`: {"arc": "k_preselect_exceeds_candidate_items", "hellaswag": "k_preselect_exceeds_candidate_items", "winogrande": "k_preselect_exceeds_candidate_items"}
- `05_irt`: {"arc": "k_preselect_exceeds_candidate_items", "hellaswag": "k_preselect_exceeds_candidate_items", "winogrande": "k_preselect_exceeds_candidate_items"}
- `06_select`: {"arc": "benchmark_skipped_in_irt", "hellaswag": "benchmark_skipped_in_irt", "winogrande": "benchmark_skipped_in_irt"}
- `08_features`: {"joint_features": "joint_feature_values_missing"}
- `09_reconstruct`: {"arc": "joint_feature_values_missing", "gsm8k": "joint_feature_values_missing", "hellaswag": "joint_feature_values_missing", "mmlu": "joint_feature_values_missing", "truthfulqa": "joint_feature_values_missing", "winogrande": "joint_feature_values_missing"}

## headline metrics

- selected items by benchmark: {"arc": 0, "gsm8k": 250, "hellaswag": 0, "mmlu": 250, "truthfulqa": 250, "winogrande": 0}
- marginal test rmse by benchmark: {"arc": null, "gsm8k": 1.2611652690272683, "hellaswag": null, "mmlu": 0.8146944277236093, "truthfulqa": 0.9504702256886021, "winogrande": null}
- joint test rmse by benchmark: {"arc": null, "gsm8k": null, "hellaswag": null, "mmlu": null, "truthfulqa": null, "winogrande": null}
