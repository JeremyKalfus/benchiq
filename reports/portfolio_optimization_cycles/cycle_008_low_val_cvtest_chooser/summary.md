# cycle 008 - low-support cv-test chooser

## hypothesis
Benchmarks with fewer than four validation rows per model should not pick between marginal and joint reconstruction heads from that tiny holdout alone; using the existing fold-CV test summary in those low-support cases should improve HELM without disturbing higher-support sources like OpenEval.

## keep or drop
keep

## headline
old best equal-weight rmse: 3.3978 (psychometric_default__deterministic_info)
new best equal-weight rmse: 3.1589 (psychometric_default__deterministic_info)
delta: -0.2389
informative source count: 3

## code changes
- src/benchiq/reconstruct/reconstruction.py: keep validation-based preferred-model selection by default, but when both model types have fewer than 4 validation rows, switch the chooser to the existing fold-CV mean test RMSE tied to each model's selected lambda.
- tests/unit/test_reconstruction.py: add coverage for the low-support CV chooser override and the threshold guard that preserves validation-based selection at 4 rows or more.

## source deltas
- helm_objective__capabilities_v1_0_0 / psychometric_default__random_cv: rmse 8.8303 -> 7.2234 (delta -1.6069); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__random_cv: rmse 6.5345 -> 5.3958 (delta -1.1387); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__deterministic_info: rmse 4.4952 -> 3.7785 (delta -0.7168); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / minimal_cleaning__deterministic_info: rmse 3.9816 -> 3.9245 (delta -0.0571); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__deterministic_info: rmse 3.9816 -> 3.9245 (delta -0.0571); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first_relaxed__deterministic_info: rmse 3.9816 -> 3.9245 (delta -0.0571); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- ollb_v1_metabench_source__release_default_subset_20260405 / minimal_cleaning__deterministic_info: rmse 2.8342 -> 2.8342 (delta +0.0000); preselect items 40.00 -> 40.00; final items 40.00 -> 40.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__deterministic_info: rmse 2.4233 -> 2.4233 (delta +0.0000); preselect items 40.00 -> 40.00; final items 40.00 -> 40.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__random_cv: rmse 2.3156 -> 2.3156 (delta +0.0000); preselect items 40.00 -> 40.00; final items 40.00 -> 40.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__deterministic_info: rmse 2.8342 -> 2.8342 (delta +0.0000); preselect items 40.00 -> 40.00; final items 40.00 -> 40.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__random_cv: rmse 2.5562 -> 2.5562 (delta +0.0000); preselect items 40.00 -> 40.00; final items 40.00 -> 40.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first_relaxed__deterministic_info: rmse 2.8342 -> 2.8342 (delta +0.0000); preselect items 40.00 -> 40.00; final items 40.00 -> 40.00
- openeval__hf_94f8112_20260314 / minimal_cleaning__deterministic_info: rmse 3.8255 -> 3.8255 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / psychometric_default__deterministic_info: rmse 3.2749 -> 3.2749 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / psychometric_default__random_cv: rmse 3.4958 -> 3.4958 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / reconstruction_first__deterministic_info: rmse 3.8255 -> 3.8255 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / reconstruction_first__random_cv: rmse 6.1760 -> 6.1760 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / reconstruction_first_relaxed__deterministic_info: rmse 3.8255 -> 3.8255 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00

## workflow checks
- OLLB same-source run marginal rmse mean: 2.9072
- OpenEval same-source run marginal rmse mean: 10.3811
- HELM same-source run marginal rmse mean: 3.9816
- materialized optimize sources stayed at 3 and validation-only skips remained explicit.

## tests run
- `.venv/bin/ruff check src/benchiq/reconstruct/reconstruction.py tests/unit/test_reconstruction.py`
- `.venv/bin/pytest tests/unit/test_reconstruction.py tests/integration/test_calibration_deployment.py -q`
- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
