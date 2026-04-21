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
- `minimal_cleaning__deterministic_info`: rmse=6.9390, mae=5.4096, pearson=0.9154, spearman=0.9604
- `reconstruction_first__deterministic_info`: rmse=6.9390, mae=5.4096, pearson=0.9154, spearman=0.9604
- `reconstruction_first_relaxed__deterministic_info`: rmse=6.9390, mae=5.4096, pearson=0.9154, spearman=0.9604
- `reconstruction_first__random_cv`: rmse=7.1663, mae=5.9780, pearson=0.9007, spearman=0.8700
- `psychometric_default__deterministic_info`: rmse=7.2789, mae=5.7973, pearson=0.9100, spearman=0.8744
- `psychometric_default__random_cv`: rmse=7.4053, mae=5.8210, pearson=0.8625, spearman=0.8905

## leave-one-out
- leaving out `helm_objective__capabilities_v1_0_0` -> winner `psychometric_default__random_cv`
- leaving out `ollb_v1_metabench_source__release_default_subset_20260405` -> winner `reconstruction_first__random_cv`
- leaving out `openeval__hf_94f8112_20260314` -> winner `minimal_cleaning__deterministic_info`

## validation-only sources
- `livecodebench`: `skipped` - official public LiveCodeBench datasets expose benchmark items and execution artifacts, but not a public many-model item-response matrix
- `belebele`: `skipped` - the official public Belebele dataset exposes benchmark items only; no public many-model item-response bank was configured for BenchIQ ingestion

## recommendation
- winner: `minimal_cleaning__deterministic_info` (equal-weight rmse `6.9390`)
- `reconstruction_first` was displaced on this narrowed portfolio
