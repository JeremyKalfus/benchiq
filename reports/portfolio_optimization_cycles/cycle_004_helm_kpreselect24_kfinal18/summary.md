# cycle 004 - helm k_preselect 24 / k_final 18

## hypothesis
HELM benefits from a smaller deterministic preselection pool once k_final is already raised to 18; cutting HELM-only k_preselect from 28 to 24 should remove weaker candidates before stage 06 and lower held-out reconstruction error.

## keep or drop
keep

## headline
old best equal-weight rmse: 5.9759 (minimal_cleaning__deterministic_info)
new best equal-weight rmse: 5.6718 (minimal_cleaning__deterministic_info)
delta: -0.3041
informative source count: 3

## code changes
- src/benchiq/portfolio/materialize.py: lower the HELM objective portfolio stage-04 budget from k_preselect=28 to k_preselect=24 while keeping k_final=18.
- tests/unit/test_portfolio_materialize.py: lock the HELM objective public portfolio stage options to k_preselect=24 and k_final=18.

## source deltas
- helm_objective__capabilities_v1_0_0 / psychometric_default__deterministic_info: rmse 11.5589 -> 5.9214 (delta -5.6375); preselect items 28.00 -> 24.00
- helm_objective__capabilities_v1_0_0 / minimal_cleaning__deterministic_info: rmse 7.4112 -> 6.4988 (delta -0.9124); preselect items 28.00 -> 24.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__deterministic_info: rmse 7.4112 -> 6.4988 (delta -0.9124); preselect items 28.00 -> 24.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first_relaxed__deterministic_info: rmse 7.4112 -> 6.4988 (delta -0.9124); preselect items 28.00 -> 24.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__random_cv: rmse 19.3767 -> 10.8392 (delta -8.5375); preselect items 28.00 -> 24.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__random_cv: rmse 12.6568 -> 13.5732 (delta 0.9164); preselect items 28.00 -> 24.00
- ollb_v1_metabench_source__release_default_subset_20260405 / minimal_cleaning__deterministic_info: rmse 4.5735 -> 4.5735 (delta 0.0000); preselect items 50.00 -> 50.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__deterministic_info: rmse 4.5735 -> 4.5735 (delta 0.0000); preselect items 50.00 -> 50.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first_relaxed__deterministic_info: rmse 4.5735 -> 4.5735 (delta 0.0000); preselect items 50.00 -> 50.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__deterministic_info: rmse 4.6325 -> 4.6325 (delta 0.0000); preselect items 48.17 -> 48.17
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__random_cv: rmse 5.1597 -> 5.1597 (delta 0.0000); preselect items 48.17 -> 48.17
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__random_cv: rmse 5.8465 -> 5.8465 (delta 0.0000); preselect items 50.00 -> 50.00
- openeval__hf_94f8112_20260314 / reconstruction_first__random_cv: rmse 3.9694 -> 3.9694 (delta 0.0000); preselect items 36.75 -> 36.75
- openeval__hf_94f8112_20260314 / psychometric_default__random_cv: rmse 4.5231 -> 4.5231 (delta 0.0000); preselect items 36.75 -> 36.75
- openeval__hf_94f8112_20260314 / minimal_cleaning__deterministic_info: rmse 5.9431 -> 5.9431 (delta 0.0000); preselect items 36.75 -> 36.75
- openeval__hf_94f8112_20260314 / reconstruction_first__deterministic_info: rmse 5.9431 -> 5.9431 (delta 0.0000); preselect items 36.75 -> 36.75
- openeval__hf_94f8112_20260314 / reconstruction_first_relaxed__deterministic_info: rmse 5.9431 -> 5.9431 (delta 0.0000); preselect items 36.75 -> 36.75
- openeval__hf_94f8112_20260314 / psychometric_default__deterministic_info: rmse 7.0360 -> 7.0360 (delta 0.0000); preselect items 36.75 -> 36.75

## workflow checks
- HELM same-source run marginal rmse mean: 7.4112 -> 6.4988
- materialized optimize sources stayed at 3 and validation-only skips remained explicit.

## tests run
- `.venv/bin/ruff check src/benchiq/portfolio/materialize.py tests/unit/test_portfolio_materialize.py`
- `.venv/bin/pytest tests/unit/test_portfolio_materialize.py -q`
- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
