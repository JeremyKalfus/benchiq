# preprocessing variation follow-up

## setup

- dataset: `large_release_default_subset`
- source_path: `out/release_bundle_source/release_default_subset_responses_long.parquet`
- preselection_method: `deterministic_info`
- exploration seeds: [7]
- confirmation seeds: [7, 11, 19]
- profile count explored: 11
- note: this follow-up focuses on low-tail trimming, light point-biserial floors, and ceiling cuts below 0.95 because those are the stage-01 knobs that still move the real bundle

## exploration top rows

- `relaxed_low_tail_001`: rmse_mean=0.8805, retained_items_mean=350.00, retained_models_mean=5578.17, runtime_mean_seconds=288.53
- `relaxed_low_tail_0005`: rmse_mean=0.8963, retained_items_mean=350.00, retained_models_mean=5581.17, runtime_mean_seconds=293.02
- `relaxed_low_tail_0005_pb_002`: rmse_mean=0.8963, retained_items_mean=350.00, retained_models_mean=5581.17, runtime_mean_seconds=350.04
- `relaxed_low_tail_002`: rmse_mean=0.9014, retained_items_mean=350.00, retained_models_mean=5572.67, runtime_mean_seconds=291.20
- `relaxed_pb_002`: rmse_mean=0.9015, retained_items_mean=350.00, retained_models_mean=5583.50, runtime_mean_seconds=331.43
- `reconstruction_relaxed`: rmse_mean=0.9015, retained_items_mean=350.00, retained_models_mean=5583.50, runtime_mean_seconds=347.92
- `relaxed_ceiling_092`: rmse_mean=0.9855, retained_items_mean=330.67, retained_models_mean=5583.50, runtime_mean_seconds=230.96
- `relaxed_low_tail_001_pb_005`: rmse_mean=1.0066, retained_items_mean=348.50, retained_models_mean=5578.17, runtime_mean_seconds=240.14

## confirmation

- `relaxed_low_tail_002`: rmse_mean=0.8786, seed_rmse_std=0.0199, retained_items_mean=350.00, retained_models_mean=5572.67, final_selection_stability_mean=0.9411
- `relaxed_low_tail_001`: rmse_mean=0.8853, seed_rmse_std=0.0044, retained_items_mean=350.00, retained_models_mean=5578.17, final_selection_stability_mean=0.9488
- `relaxed_low_tail_0005_pb_002`: rmse_mean=0.8900, seed_rmse_std=0.0066, retained_items_mean=350.00, retained_models_mean=5581.17, final_selection_stability_mean=0.9403
- `relaxed_low_tail_0005`: rmse_mean=0.8900, seed_rmse_std=0.0066, retained_items_mean=350.00, retained_models_mean=5581.17, final_selection_stability_mean=0.9403
- `reconstruction_relaxed`: rmse_mean=0.8955, seed_rmse_std=0.0220, retained_items_mean=350.00, retained_models_mean=5583.50, final_selection_stability_mean=0.9428

## benchmark deltas vs current

- `truthfulqa`: challenger_minus_current_rmse=-0.0521
- `hellaswag`: challenger_minus_current_rmse=-0.0462
- `winogrande`: challenger_minus_current_rmse=-0.0052
- `mmlu`: challenger_minus_current_rmse=-0.0035
- `arc`: challenger_minus_current_rmse=-0.0030
- `gsm8k`: challenger_minus_current_rmse=0.0089

## decision

- relaxed_low_tail_002 lowered mean held-out best-available RMSE from 0.8955 to 0.8786 on the multi-seed confirmation pass.
