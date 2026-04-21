# cycle 003 - helm k_final 18

## hypothesis
the HELM objective adapter is over-compressing stable binary scenarios at k_final=12; raising the HELM-only final-item budget to 18 should reduce held-out reconstruction error without changing the admitted source set.

## keep or drop
keep

## headline
old best equal-weight rmse: 6.1347 (reconstruction_first__random_cv)
new best equal-weight rmse: 5.9759 (minimal_cleaning__deterministic_info)
delta: -0.1587
informative source count: 3

## code changes
- src/benchiq/portfolio/materialize.py: raise the HELM objective portfolio stage-06 budget from k_final=12 to k_final=18 while keeping k_preselect=28.
- tests/unit/test_portfolio_materialize.py: lock the HELM objective public portfolio stage options to k_preselect=28 and k_final=18.

## source deltas
- helm_objective__capabilities_v1_0_0 / minimal_cleaning__deterministic_info: rmse 10.4729 -> 7.4112 (delta -3.0616); final items 12 -> 18
- helm_objective__capabilities_v1_0_0 / reconstruction_first__deterministic_info: rmse 10.4729 -> 7.4112 (delta -3.0616); final items 12 -> 18
- helm_objective__capabilities_v1_0_0 / reconstruction_first_relaxed__deterministic_info: rmse 10.4729 -> 7.4112 (delta -3.0616); final items 12 -> 18
- helm_objective__capabilities_v1_0_0 / psychometric_default__deterministic_info: rmse 11.4118 -> 11.5589 (delta 0.1471); final items 12 -> 18
- helm_objective__capabilities_v1_0_0 / reconstruction_first__random_cv: rmse 8.5881 -> 12.6568 (delta 4.0687); final items 12 -> 18
- helm_objective__capabilities_v1_0_0 / psychometric_default__random_cv: rmse 9.3347 -> 19.3767 (delta 10.0420); final items 12 -> 18
- ollb_v1_metabench_source__release_default_subset_20260405 / minimal_cleaning__deterministic_info: rmse 4.5735 -> 4.5735 (delta 0.0000); final items 24 -> 24
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__deterministic_info: rmse 4.5735 -> 4.5735 (delta 0.0000); final items 24 -> 24
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first_relaxed__deterministic_info: rmse 4.5735 -> 4.5735 (delta 0.0000); final items 24 -> 24
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__deterministic_info: rmse 4.6325 -> 4.6325 (delta 0.0000); final items 24 -> 24
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__random_cv: rmse 5.1597 -> 5.1597 (delta 0.0000); final items 24 -> 24
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__random_cv: rmse 5.8465 -> 5.8465 (delta 0.0000); final items 24 -> 24
- openeval__hf_94f8112_20260314 / reconstruction_first__random_cv: rmse 3.9694 -> 3.9694 (delta 0.0000); final items 18 -> 18
- openeval__hf_94f8112_20260314 / psychometric_default__random_cv: rmse 4.5231 -> 4.5231 (delta 0.0000); final items 18 -> 18
- openeval__hf_94f8112_20260314 / minimal_cleaning__deterministic_info: rmse 5.9431 -> 5.9431 (delta 0.0000); final items 18 -> 18
- openeval__hf_94f8112_20260314 / reconstruction_first__deterministic_info: rmse 5.9431 -> 5.9431 (delta 0.0000); final items 18 -> 18
- openeval__hf_94f8112_20260314 / reconstruction_first_relaxed__deterministic_info: rmse 5.9431 -> 5.9431 (delta 0.0000); final items 18 -> 18
- openeval__hf_94f8112_20260314 / psychometric_default__deterministic_info: rmse 7.0360 -> 7.0360 (delta 0.0000); final items 18 -> 18

## workflow checks
- HELM same-source run marginal rmse mean: 9.5028 -> 7.4112
- materialized optimize sources stayed at 3 and validation-only skips remained explicit.

## tests run
- `.venv/bin/ruff check src/benchiq/portfolio/materialize.py tests/unit/test_portfolio_materialize.py`
- `.venv/bin/pytest tests/unit/test_portfolio_materialize.py -q`
- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
