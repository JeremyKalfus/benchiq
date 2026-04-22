# cycle 011 - post irt refresh rerun

## hypothesis
The April 21, 2026 IRT winner-promotion refresh could change the narrowed real-source portfolio winner even if the portfolio source budgets stayed fixed.

## keep or drop
keep

## headline
old best equal-weight rmse: 2.9344 (psychometric_default__deterministic_info)
new best equal-weight rmse: 2.9344 (psychometric_default__deterministic_info)
delta: +0.0000
informative source count: 3

## code changes
- none. reran the existing standing harness against the post-April-21 code with cycle-local output overrides only.

## source deltas
- helm_objective__capabilities_v1_0_0 / psychometric_default__deterministic_info: rmse 3.7785 -> 3.7785 (delta +0.0000); preselect items 24.00; final items 22.00
- ollb_v1_metabench_source__release_default_subset_20260405 / psychometric_default__deterministic_info: rmse 1.7499 -> 1.7499 (delta +0.0000); preselect items 43.67; final items 43.67
- openeval__hf_94f8112_20260314 / psychometric_default__deterministic_info: rmse 3.2749 -> 3.2749 (delta +0.0000); preselect items 36.75; final items 26.00

## workflow checks
- OLLB same-source run marginal rmse mean: 2.4031
- OpenEval same-source run marginal rmse mean: 10.3811
- HELM same-source run marginal rmse mean: 3.9816
- materialized optimize sources stayed at 3, `ollb_v2` remained a skipped optimize source, and validation-only skips remained explicit for `livecodebench` and `belebele`.
- stage-05 backend read from saved per-benchmark `irt_fit_report.json` files in the new cycle outputs: `girth`.

## tests run
- no code changes; no targeted lint/tests were required.
- end-to-end rerun via `.venv/bin/python` importing `scripts.run_portfolio_standing` with cycle-local output-directory overrides.
- post-run backend verification by reading saved `manifest.json -> artifacts.05_irt.per_benchmark.*.irt_fit_report.json` files from the new cycle outputs.
