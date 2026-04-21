# cycle 006 - helm 24/22 plus ollb 40/30

## hypothesis
HELM still had excess error at 24/18 and OLLB still looked constrained by a too-small final subset, so moving HELM to 24/22 and OLLB to 40/30 should improve the psychometric-default deterministic standing winner without reducing informative-source coverage or disturbing the OpenEval gain at 40/26.

## keep or drop
keep

## headline
old best equal-weight rmse: 4.6096 (psychometric_default__deterministic_info)
new best equal-weight rmse: 3.6833 (psychometric_default__deterministic_info)
delta: -0.9263
informative source count: 3

## code changes
- src/benchiq/portfolio/materialize.py: raise the HELM public portfolio stage-06 budget from k_final=18 to k_final=22 while keeping k_preselect=24, and tighten OLLB v1 to k_preselect=40 while raising k_final from 24 to 30.
- tests/unit/test_portfolio_materialize.py: lock the OLLB public portfolio stage options to k_preselect=40 / k_final=30 and the HELM public portfolio stage options to k_preselect=24 / k_final=22.

## source deltas
- helm_objective__capabilities_v1_0_0 / reconstruction_first__random_cv: rmse 13.5732 -> 6.5345 (delta -7.0387); preselect items 24.00 -> 24.00; final items 18.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / minimal_cleaning__deterministic_info: rmse 6.4988 -> 3.9816 (delta -2.5172); preselect items 24.00 -> 24.00; final items 18.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__deterministic_info: rmse 6.4988 -> 3.9816 (delta -2.5172); preselect items 24.00 -> 24.00; final items 18.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first_relaxed__deterministic_info: rmse 6.4988 -> 3.9816 (delta -2.5172); preselect items 24.00 -> 24.00; final items 18.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__random_cv: rmse 10.8392 -> 8.8303 (delta -2.0089); preselect items 24.00 -> 24.00; final items 18.00 -> 22.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__random_cv: rmse 5.8465 -> 3.8761 (delta -1.9704); preselect items 50.00 -> 40.00; final items 24.00 -> 30.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__random_cv: rmse 5.1597 -> 3.3785 (delta -1.7813); preselect items 48.17 -> 40.00; final items 24.00 -> 30.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__deterministic_info: rmse 5.9214 -> 4.4952 (delta -1.4261); preselect items 24.00 -> 24.00; final items 18.00 -> 22.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__deterministic_info: rmse 4.6325 -> 3.2796 (delta -1.3528); preselect items 48.17 -> 40.00; final items 24.00 -> 30.00
- ollb_v1_metabench_source__release_default_subset_20260405 / minimal_cleaning__deterministic_info: rmse 4.5735 -> 3.7648 (delta -0.8087); preselect items 50.00 -> 40.00; final items 24.00 -> 30.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__deterministic_info: rmse 4.5735 -> 3.7648 (delta -0.8087); preselect items 50.00 -> 40.00; final items 24.00 -> 30.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first_relaxed__deterministic_info: rmse 4.5735 -> 3.7648 (delta -0.8087); preselect items 50.00 -> 40.00; final items 24.00 -> 30.00
- openeval__hf_94f8112_20260314 / minimal_cleaning__deterministic_info: rmse 3.8255 -> 3.8255 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / psychometric_default__deterministic_info: rmse 3.2749 -> 3.2749 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / psychometric_default__random_cv: rmse 3.4958 -> 3.4958 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / reconstruction_first__deterministic_info: rmse 3.8255 -> 3.8255 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / reconstruction_first__random_cv: rmse 6.1760 -> 6.1760 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / reconstruction_first_relaxed__deterministic_info: rmse 3.8255 -> 3.8255 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00

## workflow checks
- OLLB same-source run marginal rmse mean: 3.8204
- OpenEval same-source run marginal rmse mean: 10.3811
- HELM same-source run marginal rmse mean: 3.9816
- materialized optimize sources stayed at 3 and validation-only skips remained explicit.

## tests run
- `.venv/bin/ruff check src/benchiq/portfolio/materialize.py tests/unit/test_portfolio_materialize.py`
- `.venv/bin/pytest tests/unit/test_portfolio_materialize.py -q`
- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
