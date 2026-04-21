# cycle 002 - validation preferred model

## hypothesis
benchmarks should choose between marginal and joint reconstruction using validation performance, not a blanket joint-first fallback, so best-available portfolio metrics reflect the model type that generalizes better per benchmark.

## keep or drop
keep

## headline
old best equal-weight rmse: 6.9390 (minimal_cleaning__deterministic_info)
new best equal-weight rmse: 6.1347 (reconstruction_first__random_cv)
delta: -0.8043
informative source count: 3

## code changes
- src/benchiq/reconstruct/reconstruction.py: compute per-benchmark preferred reconstruction model types from validation RMSE and keep them in reconstruction outputs.
- src/benchiq/calibration.py: persist preferred model types into the calibration bundle manifest.
- src/benchiq/deployment.py: prefer the saved model type during best-available prediction selection with explicit fallback if the preferred model is unavailable.
- src/benchiq/preprocess/optimization.py: use the preferred model type when summarizing best-available optimization metrics.
- src/benchiq/reconstruct/reconstruction.py: preserve preferred-model metadata in written reconstruction reports so the choice stays inspectable on disk.
- tests/unit/test_reconstruction.py, tests/unit/test_deployment.py, tests/integration/test_calibration_deployment.py: cover validation-based preference selection and artifact persistence.

## source deltas
- helm_objective__capabilities_v1_0_0 / reconstruction_first__random_cv: rmse 10.9554 -> 8.5881 (delta -2.3672); joint rate 1.00 -> 1.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__random_cv: rmse 12.7659 -> 9.3347 (delta -3.4312); joint rate 1.00 -> 1.00
- helm_objective__capabilities_v1_0_0 / minimal_cleaning__deterministic_info: rmse 9.8653 -> 10.4729 (delta 0.6076); joint rate 1.00 -> 1.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__deterministic_info: rmse 9.8653 -> 10.4729 (delta 0.6076); joint rate 1.00 -> 1.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first_relaxed__deterministic_info: rmse 9.8653 -> 10.4729 (delta 0.6076); joint rate 1.00 -> 1.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__deterministic_info: rmse 10.4822 -> 11.4118 (delta 0.9296); joint rate 1.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / minimal_cleaning__deterministic_info: rmse 4.6267 -> 4.5735 (delta -0.0533); joint rate 1.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__deterministic_info: rmse 4.6267 -> 4.5735 (delta -0.0533); joint rate 1.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first_relaxed__deterministic_info: rmse 4.6267 -> 4.5735 (delta -0.0533); joint rate 1.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__deterministic_info: rmse 4.3185 -> 4.6325 (delta 0.3140); joint rate 1.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__random_cv: rmse 4.9269 -> 5.1597 (delta 0.2328); joint rate 1.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__random_cv: rmse 5.6308 -> 5.8465 (delta 0.2156); joint rate 1.00 -> 1.00
- openeval__hf_94f8112_20260314 / reconstruction_first__random_cv: rmse 4.9126 -> 3.9694 (delta -0.9432); joint rate 1.00 -> 1.00
- openeval__hf_94f8112_20260314 / psychometric_default__random_cv: rmse 4.5231 -> 4.5231 (delta 0.0000); joint rate 1.00 -> 1.00
- openeval__hf_94f8112_20260314 / minimal_cleaning__deterministic_info: rmse 6.3249 -> 5.9431 (delta -0.3817); joint rate 1.00 -> 1.00
- openeval__hf_94f8112_20260314 / reconstruction_first__deterministic_info: rmse 6.3249 -> 5.9431 (delta -0.3817); joint rate 1.00 -> 1.00
- openeval__hf_94f8112_20260314 / reconstruction_first_relaxed__deterministic_info: rmse 6.3249 -> 5.9431 (delta -0.3817); joint rate 1.00 -> 1.00
- openeval__hf_94f8112_20260314 / psychometric_default__deterministic_info: rmse 7.0360 -> 7.0360 (delta 0.0000); joint rate 1.00 -> 1.00

## workflow checks
- OpenEval same-source predict availability: 1.00 -> 1.00
- materialized optimize sources stayed at 3 and validation-only skips remained explicit.

## tests run
- `.venv/bin/ruff check src/benchiq/reconstruct/reconstruction.py tests/unit/test_reconstruction.py tests/unit/test_deployment.py tests/integration/test_calibration_deployment.py`
- `.venv/bin/pytest tests/unit/test_reconstruction.py tests/unit/test_deployment.py tests/integration/test_calibration_deployment.py -q`
- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
