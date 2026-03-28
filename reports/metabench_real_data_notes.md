# metabench real-data notes

## source used

- source used: `metabench - Paper Data`
- primary or secondary release: `primary`
- zenodo doi: `10.5281/zenodo.12819251`
- zenodo record id: `12819251`
- publication date: `2024-07-25`
- frozen archive: `/Users/jeremykalfus/CodingProjects/BenchIQ/out/metabench_real_source/data.tgz`
- archive md5: `9f2d5d6bbf6cf730494e0c29507850c7`
- archive sha256: `24de1a3f387ee3787c163981e8ea6bb441f625e4f173158cc6a4316d06f8283e`
- exported responses_long: `/Users/jeremykalfus/CodingProjects/BenchIQ/out/metabench_real_source/release_default_subset_responses_long.parquet`
- exported release metadata: `/Users/jeremykalfus/CodingProjects/BenchIQ/out/metabench_real_source/release_default_subset_metadata.parquet`
- BenchIQ run root: `out/metabench_real_validation/metabench-real-zenodo-12819251-parity`

## what this run actually validates

- This pass uses the primary Zenodo paper snapshot, but it does not run the full raw 153M-row archive through BenchIQ end to end.
- Instead, it exports the paper release's default `*-sub.rds` selected subsets and fixed train/test scores, then runs BenchIQ's downstream Python reconstruction stack on that real public split/subset.
- This parity repair keeps the same frozen snapshot, preserves the public fixed split/subset behavior, then applies the paper's published final kept-item counts and a dedicated grand-mean GAM inside the validation harness.

## deviations from the original r metabench stack

- BenchIQ is not claiming bit-for-bit parity with the original r pipeline.
- The frozen public data unpacking step in this harness uses `Rscript` to read the published `.rds` release artifacts. The downstream modeling stages remain BenchIQ's Python-first path.
- Current BenchIQ therefore is **not** using only the Python-first path for this validation harness; the `.rds` import step is parity-specific, but the downstream modeling remains BenchIQ's Python-first path.
- BenchIQ uses girth instead of mirt for 2PL fitting and pyGAM instead of mgcv for reconstruction.
- The dedicated grand-mean GAM added here is parity-specific validation logic in this script. It does not change BenchIQ's generic product identity.

## snapshot-to-target mismatch to know about

- The user-requested primary targets are preserved below exactly as the acceptance baseline.
- The frozen paper snapshot's default `rmse.test` values inside the public `*-sub.rds` artifacts do not exactly match those targets for every benchmark, especially HellaSwag.
- arc: published target `1.166`, release-default rmse.test `1.375`
- gsm8k: published target `1.555`, release-default rmse.test `1.348`
- hellaswag: published target `0.999`, release-default rmse.test `1.646`
- mmlu: published target `1.430`, release-default rmse.test `1.329`
- truthfulqa: published target `1.104`, release-default rmse.test `1.086`
- winogrande: published target `1.195`, release-default rmse.test `1.280`

## likely explanations for any remaining gap

- pyGAM smoothing selection will not exactly match mgcv's spline path.
- girth's 2PL estimation and pathology handling differ from mirt.
- The public snapshot does not expose the original post-IRT/final-selection artifacts directly, so this harness still reconstructs the final selection inside BenchIQ from the public 350-item release subset.
- The real-data comparison still does not replay every upstream r decision from the raw archive; it is the closest tractable like-for-like public path.

## direct answers for this parity repair

- Did matching the published kept-item counts materially reduce the RMSE deltas? `no`
- previous mean absolute benchmark delta: `0.277`
- current mean absolute benchmark delta: `0.547`
- Did adding the dedicated grand-mean path materially change the mean-score comparison? `yes`
- previous mean-score rmse: `0.428`
- current dedicated grand-mean rmse: `1.027`
- Is the remaining gap now small enough to claim acceptance-grade parity under the BenchIQ tolerance rule? `False`
- If not, what is the smallest remaining cause of mismatch? the frozen public snapshot exposes the release-default 350-item subsets but not the released final item identities, so BenchIQ still has to reconstruct the final subset with girth/pyGAM instead of replaying the original mirt/mgcv item path

## acceptance outcome

- overall_pass: `False`
- strong_pass: `False`
- acceptable_pass: `False`
- verdict_reason: mean absolute benchmark delta 0.547 exceeded the acceptable limit 0.150; benchmark rmse deltas and/or the published mean-score parity band were exceeded

## smallest optional parity path if this still misses

- Add the smallest optional parity backend needed to replay the final mirt/mgcv behavior on the frozen snapshot, likely through an `rpy2` or explicit Rscript-backed validation-only mode, without changing BenchIQ's default Python-first product path.

## rerun command

- `.venv/bin/python /Users/jeremykalfus/CodingProjects/BenchIQ/scripts/run_metabench_real_data_comparison.py --out out/metabench_real_validation --run-id metabench-real-zenodo-12819251-parity`
