# cycle 010 - ollb 44/44 plus low-support chooser

## hypothesis
OLLB still improved when the final subset rose from 40 to 44 at the already-better 44-item preselection budget, so matching the final subset to the retained 44-item set should produce another portfolio gain before source-budget tuning fully saturates.

## keep or drop
keep

## headline
old best equal-weight rmse: 3.0541 (psychometric_default__deterministic_info)
new best equal-weight rmse: 2.9344 (psychometric_default__deterministic_info)
delta: -0.1197
informative source count: 3

## code changes
- src/benchiq/portfolio/materialize.py: keep OLLB at k_preselect=44 and raise the final selection budget from k_final=40 to k_final=44.
- tests/unit/test_portfolio_materialize.py: lock the OLLB public portfolio stage options to k_preselect=44 / k_final=44.

## source deltas
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__random_cv: rmse 2.1364 -> 1.5647 (delta -0.5717); preselect items 43.67 -> 43.67; final items 40.00 -> 43.67
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__deterministic_info: rmse 2.1090 -> 1.7499 (delta -0.3592); preselect items 43.67 -> 43.67; final items 40.00 -> 43.67
- ollb_v1_metabench_source__release_default_subset_20260405 / minimal_cleaning__deterministic_info: rmse 2.4258 -> 2.2517 (delta -0.1740); preselect items 44.00 -> 44.00; final items 40.00 -> 44.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__deterministic_info: rmse 2.4258 -> 2.2517 (delta -0.1740); preselect items 44.00 -> 44.00; final items 40.00 -> 44.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first_relaxed__deterministic_info: rmse 2.4258 -> 2.2517 (delta -0.1740); preselect items 44.00 -> 44.00; final items 40.00 -> 44.00
- ollb_v1_metabench_source__release_default_subset_20260405 / reconstruction_first__random_cv: rmse 2.6804 -> 2.5536 (delta -0.1268); preselect items 44.00 -> 44.00; final items 40.00 -> 44.00
- helm_objective__capabilities_v1_0_0 / minimal_cleaning__deterministic_info: rmse 3.9245 -> 3.9245 (delta +0.0000); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__deterministic_info: rmse 3.7785 -> 3.7785 (delta +0.0000); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / psychometric_default__random_cv: rmse 7.2234 -> 7.2234 (delta +0.0000); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__deterministic_info: rmse 3.9245 -> 3.9245 (delta +0.0000); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first__random_cv: rmse 5.3958 -> 5.3958 (delta +0.0000); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- helm_objective__capabilities_v1_0_0 / reconstruction_first_relaxed__deterministic_info: rmse 3.9245 -> 3.9245 (delta +0.0000); preselect items 24.00 -> 24.00; final items 22.00 -> 22.00
- openeval__hf_94f8112_20260314 / minimal_cleaning__deterministic_info: rmse 3.8255 -> 3.8255 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / psychometric_default__deterministic_info: rmse 3.2749 -> 3.2749 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / psychometric_default__random_cv: rmse 3.4958 -> 3.4958 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / reconstruction_first__deterministic_info: rmse 3.8255 -> 3.8255 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / reconstruction_first__random_cv: rmse 6.1760 -> 6.1760 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00
- openeval__hf_94f8112_20260314 / reconstruction_first_relaxed__deterministic_info: rmse 3.8255 -> 3.8255 (delta +0.0000); preselect items 36.75 -> 36.75; final items 26.00 -> 26.00

## workflow checks
- OLLB same-source run marginal rmse mean: 2.4031
- OpenEval same-source run marginal rmse mean: 10.3811
- HELM same-source run marginal rmse mean: 3.9816
- materialized optimize sources stayed at 3 and validation-only skips remained explicit.

## tests run
- `.venv/bin/ruff check src/benchiq/portfolio/materialize.py tests/unit/test_portfolio_materialize.py`
- `.venv/bin/pytest tests/unit/test_portfolio_materialize.py -q`
- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
