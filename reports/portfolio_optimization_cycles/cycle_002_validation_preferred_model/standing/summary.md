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
- `reconstruction_first__random_cv`: rmse=6.1347, mae=5.1639, pearson=0.9000, spearman=0.8699
- `psychometric_default__random_cv`: rmse=6.3392, mae=5.0994, pearson=0.8458, spearman=0.8898
- `minimal_cleaning__deterministic_info`: rmse=6.9965, mae=5.7754, pearson=0.8788, spearman=0.8779
- `reconstruction_first__deterministic_info`: rmse=6.9965, mae=5.7754, pearson=0.8788, spearman=0.8779
- `reconstruction_first_relaxed__deterministic_info`: rmse=6.9965, mae=5.7754, pearson=0.8788, spearman=0.8779
- `psychometric_default__deterministic_info`: rmse=7.6934, mae=6.4231, pearson=0.8643, spearman=0.7918

## leave-one-out
- leaving out `helm_objective__capabilities_v1_0_0` -> winner `psychometric_default__random_cv`
- leaving out `ollb_v1_metabench_source__release_default_subset_20260405` -> winner `reconstruction_first__random_cv`
- leaving out `openeval__hf_94f8112_20260314` -> winner `reconstruction_first__random_cv`

## validation-only sources
- `livecodebench`: `skipped` - official public LiveCodeBench datasets expose benchmark items and execution artifacts, but not a public many-model item-response matrix
- `belebele`: `skipped` - the official public Belebele dataset exposes benchmark items only; no public many-model item-response bank was configured for BenchIQ ingestion

## recommendation
- winner: `reconstruction_first__random_cv` (equal-weight rmse `6.1347`)
- `reconstruction_first` was displaced on this narrowed portfolio
