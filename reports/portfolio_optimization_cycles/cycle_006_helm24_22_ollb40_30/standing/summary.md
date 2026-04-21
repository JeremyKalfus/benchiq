# portfolio standing

## materialization
- `ollb_v1_metabench_source` / `release_default_subset_20260405`: `materialized`
- `ollb_v2` / `results_20250315`: `skipped` - OLLB v2 aggregate results are public, but the required item-level details dataset was not accessible without authentication: open-llm-leaderboard/01-ai__Yi-34B-details
- `openeval` / `hf_94f8112_20260314`: `materialized`
- `helm_objective` / `capabilities_v1_0_0`: `materialized`
- `livecodebench` / `public_release_202604`: `skipped` - official public LiveCodeBench datasets expose benchmark items and execution artifacts, but not a public many-model item-response matrix
- `belebele` / `facebook_20230503`: `skipped` - the official public Belebele dataset exposes benchmark items only; no public many-model item-response bank was configured for BenchIQ ingestion

## workflow status
- `ollb_v1_metabench_source`: validate=ok, calibrate=ok, predict=ok, run=ok
- `ollb_v2`: validate=not_run, calibrate=not_run, predict=not_run, run=not_run
- `openeval`: validate=ok, calibrate=ok, predict=ok, run=ok
- `helm_objective`: validate=ok, calibrate=ok, predict=ok, run=ok
- `livecodebench`: validate=not_run, calibrate=not_run, predict=not_run, run=not_run
- `belebele`: validate=not_run, calibrate=not_run, predict=not_run, run=not_run

## aggregate ranking
- `psychometric_default__deterministic_info`: rmse=3.6833, mae=3.1273, pearson=0.9879, spearman=0.9892
- `minimal_cleaning__deterministic_info`: rmse=3.8573, mae=3.2836, pearson=0.9849, spearman=0.9723
- `reconstruction_first__deterministic_info`: rmse=3.8573, mae=3.2836, pearson=0.9849, spearman=0.9723
- `reconstruction_first_relaxed__deterministic_info`: rmse=3.8573, mae=3.2836, pearson=0.9849, spearman=0.9723
- `psychometric_default__random_cv`: rmse=5.2349, mae=4.2698, pearson=0.9774, spearman=0.9025
- `reconstruction_first__random_cv`: rmse=5.5289, mae=4.1925, pearson=0.9529, spearman=0.9204

## leave-one-out
- leaving out `helm_objective__capabilities_v1_0_0` -> winner `psychometric_default__deterministic_info`
- leaving out `ollb_v1_metabench_source__release_default_subset_20260405` -> winner `psychometric_default__deterministic_info`
- leaving out `openeval__hf_94f8112_20260314` -> winner `minimal_cleaning__deterministic_info`

## validation-only sources
- `livecodebench`: `skipped` - official public LiveCodeBench datasets expose benchmark items and execution artifacts, but not a public many-model item-response matrix
- `belebele`: `skipped` - the official public Belebele dataset exposes benchmark items only; no public many-model item-response bank was configured for BenchIQ ingestion

## recommendation
- winner: `psychometric_default__deterministic_info` (equal-weight rmse `3.6833`)
- `reconstruction_first` was displaced on this narrowed portfolio
