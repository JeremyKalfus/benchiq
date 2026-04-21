# cycle 005 - helm 24/18 plus openeval 40/26

## hypothesis
Keeping HELM at 24/18 while raising OpenEval to 40/26 should let psychometric_default__deterministic_info capture the stronger OpenEval signal without sacrificing OLLB or HELM coverage, and should lower the portfolio equal-weight RMSE enough to displace the cycle-004 winner.

## keep or drop
keep

## headline
old best equal-weight rmse: 5.6718 (minimal_cleaning__deterministic_info)
new best equal-weight rmse: 4.6096 (psychometric_default__deterministic_info)
delta: -1.0622
informative source count: 3

## code changes
- src/benchiq/portfolio/materialize.py: keep HELM at k_preselect=24 / k_final=18 and raise the OpenEval public portfolio stage-06 budget from k_final=18 to k_final=26 while preserving k_preselect=40.
- tests/unit/test_portfolio_materialize.py: lock the OpenEval public portfolio stage options to k_preselect=40 and k_final=26.

## source deltas
- openeval__hf_94f8112_20260314 / psychometric_default__deterministic_info: rmse 7.0360 -> 3.2749 (delta -3.7612); preselect items 36.75 -> 36.75; final items 18.00 -> 26.00
- openeval__hf_94f8112_20260314 / minimal_cleaning__deterministic_info: rmse 5.9431 -> 3.8255 (delta -2.1176); preselect items 36.75 -> 36.75; final items 18.00 -> 26.00
- openeval__hf_94f8112_20260314 / reconstruction_first__deterministic_info: rmse 5.9431 -> 3.8255 (delta -2.1176); preselect items 36.75 -> 36.75; final items 18.00 -> 26.00
- openeval__hf_94f8112_20260314 / reconstruction_first_relaxed__deterministic_info: rmse 5.9431 -> 3.8255 (delta -2.1176); preselect items 36.75 -> 36.75; final items 18.00 -> 26.00
- openeval__hf_94f8112_20260314 / psychometric_default__random_cv: rmse 4.5231 -> 3.4958 (delta -1.0273); preselect items 36.75 -> 36.75; final items 18.00 -> 26.00
- helm_objective__capabilities_v1_0_0 / minimal_cleaning__deterministic_info: rmse 6.4988 -> 6.4988 (delta +0.0000); preselect items 24.00 -> 24.00; final items 18.00 -> 18.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__deterministic_info: rmse 5.9214 -> 5.9214 (delta +0.0000); preselect items 24.00 -> 24.00; final items 18.00 -> 18.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__random_cv: rmse 10.8392 -> 10.8392 (delta +0.0000); preselect items 24.00 -> 24.00; final items 18.00 -> 18.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__deterministic_info: rmse 6.4988 -> 6.4988 (delta +0.0000); preselect items 24.00 -> 24.00; final items 18.00 -> 18.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__random_cv: rmse 13.5732 -> 13.5732 (delta +0.0000); preselect items 24.00 -> 24.00; final items 18.00 -> 18.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first_relaxed__deterministic_info: rmse 6.4988 -> 6.4988 (delta +0.0000); preselect items 24.00 -> 24.00; final items 18.00 -> 18.00
- ollb_v1_metabench_source__release_default_subset_20260405 / minimal_cleaning__deterministic_info: rmse 4.5735 -> 4.5735 (delta +0.0000); preselect items 50.00 -> 50.00; final items 24.00 -> 24.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__deterministic_info: rmse 4.6325 -> 4.6325 (delta +0.0000); preselect items 48.17 -> 48.17; final items 24.00 -> 24.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__random_cv: rmse 5.1597 -> 5.1597 (delta +0.0000); preselect items 48.17 -> 48.17; final items 24.00 -> 24.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__deterministic_info: rmse 4.5735 -> 4.5735 (delta +0.0000); preselect items 50.00 -> 50.00; final items 24.00 -> 24.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__random_cv: rmse 5.8465 -> 5.8465 (delta +0.0000); preselect items 50.00 -> 50.00; final items 24.00 -> 24.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first_relaxed__deterministic_info: rmse 4.5735 -> 4.5735 (delta +0.0000); preselect items 50.00 -> 50.00; final items 24.00 -> 24.00
- openeval__hf_94f8112_20260314 / reconstruction_first__random_cv: rmse 3.9694 -> 6.1760 (delta +2.2066); preselect items 36.75 -> 36.75; final items 18.00 -> 26.00

## workflow checks
- OpenEval same-source run marginal rmse mean: 10.3811
- HELM same-source run marginal rmse mean: 6.4988
- OLLB same-source run marginal rmse mean: 5.1223
- materialized optimize sources stayed at 3 and validation-only skips remained explicit.

## tests run
- `.venv/bin/ruff check src/benchiq/portfolio/materialize.py tests/unit/test_portfolio_materialize.py`
- `.venv/bin/pytest tests/unit/test_portfolio_materialize.py -q`
- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
