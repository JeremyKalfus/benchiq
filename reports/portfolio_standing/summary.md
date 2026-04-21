# portfolio standing

This summary is the original narrowed public-portfolio standing baseline snapshot written by
`scripts/run_portfolio_standing.py`.

For the latest iterative narrowed public-portfolio best-so-far recommendation, see
`reports/portfolio_optimization_cycles/best_so_far.md`.

## materialization
- `ollb_v1_metabench_source` / `release_default_subset_20260405`: `materialized`
- `ollb_v2` / `results_20250315`: `skipped` - OLLB v2 aggregate results are public, but the required item-level details dataset was not accessible without authentication: open-llm-leaderboard/01-ai__Yi-34B-details
- `openeval` / `hf_94f8112_20260314`: `materialized`
- `helm_objective` / `capabilities_v1_0_0`: `materialized`
- `livecodebench` / `public_release_202604`: `skipped` - official public LiveCodeBench datasets expose benchmark items and execution artifacts, but not a public many-model item-response matrix
- `belebele` / `facebook_20230503`: `skipped` - the official public Belebele dataset exposes benchmark items only; no public many-model item-response bank was configured for BenchIQ ingestion

## workflow status
- `belebele`: validate=not_run, calibrate=not_run, predict=not_run, run=not_run
- `helm_objective`: validate=ok, calibrate=ok, predict=ok, run=ok
- `livecodebench`: validate=not_run, calibrate=not_run, predict=not_run, run=not_run
- `ollb_v1_metabench_source`: validate=ok, calibrate=ok, predict=ok, run=ok
- `ollb_v2`: validate=not_run, calibrate=not_run, predict=not_run, run=not_run
- `openeval`: validate=ok, calibrate=ok, predict=ok, run=ok

## aggregate ranking at the initial standing pass
- `reconstruction_first__random_cv`: rmse=8.3589, mae=6.9331, pearson=0.8334, spearman=0.8243
- `psychometric_default__random_cv`: rmse=9.0681, mae=6.6728, pearson=0.8481, spearman=0.8892
- `minimal_cleaning__deterministic_info`: rmse=10.3434, mae=7.6912, pearson=0.7582, spearman=0.7591
- `reconstruction_first__deterministic_info`: rmse=10.3434, mae=7.6912, pearson=0.7582, spearman=0.7591
- `reconstruction_first_relaxed__deterministic_info`: rmse=10.3434, mae=7.6912, pearson=0.7582, spearman=0.7591
- `psychometric_default__deterministic_info`: rmse=11.1216, mae=8.4944, pearson=0.7432, spearman=0.6949

## leave-one-out at the initial standing pass
- leaving out `helm_objective__capabilities_v1_0_0` -> winner `reconstruction_first__random_cv`
- leaving out `ollb_v1_metabench_source__release_default_subset_20260405` -> winner `reconstruction_first__random_cv`
- leaving out `openeval__hf_94f8112_20260314` -> winner `minimal_cleaning__deterministic_info`

## validation-only sources
- `livecodebench`: `skipped` - official public LiveCodeBench datasets expose benchmark items and execution artifacts, but not a public many-model item-response matrix
- `belebele`: `skipped` - the official public Belebele dataset exposes benchmark items only; no public many-model item-response bank was configured for BenchIQ ingestion

## recommendation at the initial standing pass
- winner: `reconstruction_first__random_cv` (equal-weight rmse `8.3589`)
- `reconstruction_first` was displaced on this narrowed portfolio
