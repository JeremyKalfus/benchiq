# metabench real-data notes

## source used

- source used: `metabench - Paper Data`
- primary or secondary release: `primary`
- zenodo doi: `10.5281/zenodo.12819251`
- zenodo record id: `12819251`
- publication date: `2024-07-25`
- frozen archive: `/tmp/benchiq_metabench_real/data.tgz`
- archive md5: `9f2d5d6bbf6cf730494e0c29507850c7`
- archive sha256: `24de1a3f387ee3787c163981e8ea6bb441f625e4f173158cc6a4316d06f8283e`
- exported responses_long: `/tmp/benchiq_metabench_real/release_default_subset_responses_long.parquet`
- exported release metadata: `/tmp/benchiq_metabench_real/release_default_subset_metadata.parquet`
- BenchIQ run root: `out/metabench_real_validation/metabench-real-zenodo-12819251`

## what this run actually validates

- This pass uses the primary Zenodo paper snapshot, but it does not run the full raw 153M-row archive through BenchIQ end to end.
- Instead, it exports the paper release's default `*-sub.rds` selected subsets and fixed train/test scores, then runs BenchIQ's downstream Python reconstruction stack on that real public split/subset.
- This is the strongest honest in-session comparison path because it preserves the public selected items and held-out evaluation split while staying computationally tractable.

## deviations from the original r metabench stack

- BenchIQ is not claiming bit-for-bit parity with the original r pipeline.
- The frozen public data unpacking step in this harness uses `Rscript` to read the published `.rds` release artifacts. The downstream modeling stages remain BenchIQ's Python-first path.
- Current BenchIQ therefore is **not** using only the Python-first path for this validation harness; the import step depends on the public r artifact format.
- BenchIQ uses girth instead of mirt for 2PL fitting and pyGAM instead of mgcv for reconstruction.
- BenchIQ v0.1 does not yet implement the published dedicated grand-mean GAM for the Open LLM Leaderboard mean score.

## snapshot-to-target mismatch to know about

- The user-requested primary targets are preserved below exactly as the acceptance baseline.
- The frozen paper snapshot's default `rmse.test` values inside the public `*-sub.rds` artifacts do not exactly match those targets for every benchmark, especially HellaSwag.
- arc: published target `1.166`, release-default rmse.test `1.375`
- gsm8k: published target `1.555`, release-default rmse.test `1.348`
- hellaswag: published target `0.999`, release-default rmse.test `1.646`
- mmlu: published target `1.430`, release-default rmse.test `1.329`
- truthfulqa: published target `1.104`, release-default rmse.test `1.086`
- winogrande: published target `1.195`, release-default rmse.test `1.280`

## likely explanations for any gap

- pyGAM smoothing selection will not exactly match mgcv's spline path.
- girth's 2PL estimation and pathology handling differ from mirt.
- BenchIQ's mean-score comparison is only a proxy because the dedicated grand-mean GAM is still missing.
- The real-data comparison here starts from the public release-default subset artifacts, not from a full raw reimplementation of every upstream r decision.

## acceptance outcome

- overall_pass: `False`
- strong_pass: `False`
- acceptable_pass: `False`
- verdict_reason: mean absolute benchmark delta 0.277 exceeded the acceptable limit 0.150; BenchIQ v0.1 does not yet implement the published dedicated grand-mean GAM; the value below is a derived proxy from joint benchmark predictions and is reported for context only

## smallest optional parity path if this still misses

- Add an optional parity-only validation mode that can call the original r/mirt/mgcv stack, likely via `rpy2` or explicit Rscript orchestration, without changing BenchIQ's product identity or default Python-first path.

## rerun command

- `.venv/bin/python /Users/jeremykalfus/CodingProjects/BenchIQ/scripts/run_metabench_real_data_comparison.py --out out/metabench_real_validation --run-id metabench-real-zenodo-12819251`
