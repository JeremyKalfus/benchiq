# best so far

- cycle: `cycle_010_ollb44_44_low_val_chooser`
- winner: `psychometric_default__deterministic_info`
- equal-weight rmse: `2.9344`
- informative source count: `3`

## per-source rmse
- `helm_objective__capabilities_v1_0_0`: rmse `3.7785`, mae `3.3180`
- `ollb_v1_metabench_source__release_default_subset_20260405`: rmse `1.7499`, mae `1.2123`
- `openeval__hf_94f8112_20260314`: rmse `3.2749`, mae `2.6850`

## leave-one-out winners
- leaving out `helm_objective__capabilities_v1_0_0` -> `psychometric_default__deterministic_info` at rmse `2.5124`
- leaving out `ollb_v1_metabench_source__release_default_subset_20260405` -> `psychometric_default__deterministic_info` at rmse `3.5267`
- leaving out `openeval__hf_94f8112_20260314` -> `psychometric_default__deterministic_info` at rmse `2.7642`

## recommendation
- current recommended profile/method combination: `psychometric_default__deterministic_info`
