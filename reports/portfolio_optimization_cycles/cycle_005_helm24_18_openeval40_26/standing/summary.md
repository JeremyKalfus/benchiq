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
- `psychometric_default__deterministic_info`: rmse=4.6096, mae=3.5820, pearson=0.9521, spearman=0.9794
- `minimal_cleaning__deterministic_info`: rmse=4.9659, mae=4.1814, pearson=0.9391, spearman=0.9655
- `reconstruction_first__deterministic_info`: rmse=4.9659, mae=4.1814, pearson=0.9391, spearman=0.9655
- `reconstruction_first_relaxed__deterministic_info`: rmse=4.9659, mae=4.1814, pearson=0.9391, spearman=0.9655
- `psychometric_default__random_cv`: rmse=6.4982, mae=5.5826, pearson=0.9660, spearman=0.9774
- `reconstruction_first__random_cv`: rmse=8.5319, mae=7.2749, pearson=0.6737, spearman=0.6575

## leave-one-out
- leaving out `helm_objective__capabilities_v1_0_0` -> winner `psychometric_default__deterministic_info`
- leaving out `ollb_v1_metabench_source__release_default_subset_20260405` -> winner `psychometric_default__deterministic_info`
- leaving out `openeval__hf_94f8112_20260314` -> winner `psychometric_default__deterministic_info`

## validation-only sources
- `livecodebench`: `skipped` - official public LiveCodeBench datasets expose benchmark items and execution artifacts, but not a public many-model item-response matrix
- `belebele`: `skipped` - the official public Belebele dataset exposes benchmark items only; no public many-model item-response bank was configured for BenchIQ ingestion

## recommendation
- winner: `psychometric_default__deterministic_info` (equal-weight rmse `4.6096`)
- `reconstruction_first` was displaced on this narrowed portfolio
