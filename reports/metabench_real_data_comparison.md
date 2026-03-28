# metabench real-data comparison

- generated_at: `2026-04-05T07:06:48.300617+00:00`
- data_source: `metabench - Paper Data` (10.5281/zenodo.12819251)
- release_tier: `primary`
- archive_path: `/tmp/benchiq_metabench_real/data.tgz`
- responses_path: `/tmp/benchiq_metabench_real/release_default_subset_responses_long.parquet`
- metadata_path: `/tmp/benchiq_metabench_real/release_default_subset_metadata.parquet`
- run_root: `out/metabench_real_validation/metabench-real-zenodo-12819251`
- overall_pass: `False`
- strong_pass: `False`
- acceptable_pass: `False`

| benchmark | published target rmse | public release rmse | BenchIQ marginal rmse | BenchIQ joint rmse | absolute delta | kept items | within ±0.10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| arc | 1.166 | 1.375 | 1.048 | 0.866 | 0.300 | 350 | False |
| gsm8k | 1.555 | 1.348 | 1.190 | 1.207 | 0.348 | 350 | False |
| hellaswag | 0.999 | 1.646 | 0.980 | 0.810 | 0.189 | 350 | False |
| mmlu | 1.430 | 1.329 | 1.083 | 1.024 | 0.406 | 350 | False |
| truthfulqa | 1.104 | 1.086 | 1.009 | 0.969 | 0.135 | 350 | False |
| winogrande | 1.195 | 1.280 | 0.954 | 0.915 | 0.280 | 350 | False |

## mean-score comparison

- published Open LLM LB mean RMSE target: `0.582`
- BenchIQ derived mean-score proxy RMSE: `0.428`
- proxy absolute delta: `0.154`
- proxy overlap models: `464`
- proxy within ±0.05: `False`
- note: BenchIQ v0.1 does not yet implement the published dedicated grand-mean GAM; the value below is a derived proxy from joint benchmark predictions and is reported for context only

## verdict

- mean absolute delta across the six benchmarks: `0.2765262596579254`
- verdict_reason: mean absolute benchmark delta 0.277 exceeded the acceptable limit 0.150; BenchIQ v0.1 does not yet implement the published dedicated grand-mean GAM; the value below is a derived proxy from joint benchmark predictions and is reported for context only
