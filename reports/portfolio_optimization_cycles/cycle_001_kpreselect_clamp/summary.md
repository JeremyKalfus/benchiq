# cycle 001 - k_preselect clamp

## hypothesis
benchmarks with retained candidate pools smaller than requested k_preselect should clamp explicitly instead of being skipped, because the skip cascades through IRT, feature assembly, joint reconstruction, and same-source deployment availability.

## keep or drop
keep

## headline
old best equal-weight rmse: 8.3589 (reconstruction_first__random_cv)
new best equal-weight rmse: 6.9390 (minimal_cleaning__deterministic_info)
delta: -1.4199
informative source count: 3

## code changes
- src/benchiq/subsample/random_cv.py: clamp oversized requested `k_preselect` to the surviving candidate count and write explicit warnings instead of skipping the benchmark.
- tests/unit/test_subsample.py: added deterministic and random-cv coverage for the clamp behavior.

## source deltas
- helm_objective__capabilities_v1_0_0 / minimal_cleaning__deterministic_info: rmse 9.8653 -> 9.8653 (delta 0.0000); joint rate 1.00 -> 1.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__deterministic_info: rmse 9.8653 -> 9.8653 (delta 0.0000); joint rate 1.00 -> 1.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first_relaxed__deterministic_info: rmse 9.8653 -> 9.8653 (delta 0.0000); joint rate 1.00 -> 1.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__deterministic_info: rmse 10.4822 -> 10.4822 (delta 0.0000); joint rate 1.00 -> 1.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__random_cv: rmse 10.9554 -> 10.9554 (delta 0.0000); joint rate 1.00 -> 1.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__random_cv: rmse 12.7659 -> 12.7659 (delta 0.0000); joint rate 1.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__deterministic_info: rmse 5.5477 -> 4.3185 (delta -1.2291); joint rate 0.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / minimal_cleaning__deterministic_info: rmse 4.6267 -> 4.6267 (delta 0.0000); joint rate 1.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__deterministic_info: rmse 4.6267 -> 4.6267 (delta 0.0000); joint rate 1.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first_relaxed__deterministic_info: rmse 4.6267 -> 4.6267 (delta 0.0000); joint rate 1.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__random_cv: rmse 7.0773 -> 4.9269 (delta -2.1504); joint rate 0.00 -> 1.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__random_cv: rmse 5.6308 -> 5.6308 (delta 0.0000); joint rate 1.00 -> 1.00
- openeval__hf_94f8112_20260314 / psychometric_default__random_cv: rmse 7.3610 -> 4.5231 (delta -2.8379); joint rate 0.00 -> 1.00
- openeval__hf_94f8112_20260314 / reconstruction_first__random_cv: rmse 8.4905 -> 4.9126 (delta -3.5779); joint rate 0.00 -> 1.00
- openeval__hf_94f8112_20260314 / minimal_cleaning__deterministic_info: rmse 16.5381 -> 6.3249 (delta -10.2132); joint rate 0.00 -> 1.00
- openeval__hf_94f8112_20260314 / reconstruction_first__deterministic_info: rmse 16.5381 -> 6.3249 (delta -10.2132); joint rate 0.00 -> 1.00
- openeval__hf_94f8112_20260314 / reconstruction_first_relaxed__deterministic_info: rmse 16.5381 -> 6.3249 (delta -10.2132); joint rate 0.00 -> 1.00
- openeval__hf_94f8112_20260314 / psychometric_default__deterministic_info: rmse 17.3349 -> 7.0360 (delta -10.2989); joint rate 0.00 -> 1.00

## workflow checks
- OpenEval same-source predict availability: 0.75 -> 1.00
- materialized optimize sources stayed at 3 and validation-only skips remained explicit.

## tests run
- `.venv/bin/pytest tests/unit/test_subsample.py -q`
- `.venv/bin/ruff check src/benchiq/subsample/random_cv.py tests/unit/test_subsample.py`
- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
