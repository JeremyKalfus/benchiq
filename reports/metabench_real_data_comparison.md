# metabench real-data comparison

- generated_at: `2026-04-05T16:37:44.543166+00:00`
- data_source: `metabench - Paper Data` (10.5281/zenodo.12819251)
- release_tier: `primary`
- archive_path: `/Users/jeremykalfus/CodingProjects/BenchIQ/out/metabench_real_source/data.tgz`
- responses_path: `/Users/jeremykalfus/CodingProjects/BenchIQ/out/metabench_real_source/release_default_subset_responses_long.parquet`
- metadata_path: `/Users/jeremykalfus/CodingProjects/BenchIQ/out/metabench_real_source/release_default_subset_metadata.parquet`
- run_root: `out/metabench_real_validation/metabench-real-zenodo-12819251-parity`
- overall_pass: `False`
- strong_pass: `False`
- acceptable_pass: `False`

| benchmark | published target rmse | public release rmse | BenchIQ marginal rmse | BenchIQ joint rmse | absolute delta | kept items | within ±0.10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| arc | 1.166 | 1.375 | 2.557 | 1.875 | 0.709 | 100 | False |
| gsm8k | 1.555 | 1.348 | 1.793 | 1.774 | 0.219 | 237 | False |
| hellaswag | 0.999 | 1.646 | 2.676 | 1.805 | 0.806 | 58 | False |
| mmlu | 1.430 | 1.329 | 2.232 | 2.096 | 0.666 | 102 | False |
| truthfulqa | 1.104 | 1.086 | 1.724 | 1.635 | 0.531 | 136 | False |
| winogrande | 1.195 | 1.280 | 1.901 | 1.548 | 0.353 | 106 | False |

## mean-score comparison

- published Open LLM LB mean RMSE target: `0.582`
- BenchIQ dedicated grand-mean RMSE: `1.027`
- absolute delta: `0.445`
- overlap test models: `464`
- within ±0.05: `False`
- note: dedicated grand-mean GAM fit on fixed complete-overlap train/test models

## parity-repair delta vs previous reviewer pass

- previous mean absolute benchmark delta: `0.277`
- current mean absolute benchmark delta: `0.547`
- previous mean-score rmse: `0.428`
- current mean-score rmse: `1.027`

## verdict

- mean absolute delta across the six benchmarks: `0.5474750835463504`
- verdict_reason: mean absolute benchmark delta 0.547 exceeded the acceptable limit 0.150; benchmark rmse deltas and/or the published mean-score parity band were exceeded
