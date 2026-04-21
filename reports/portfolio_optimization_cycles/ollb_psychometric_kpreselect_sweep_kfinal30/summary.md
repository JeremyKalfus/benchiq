# ollb psychometric deterministic k_preselect sweep

## setup
- dataset: `ollb_v1_metabench_source__release_default_subset_20260405`
- profile: `psychometric_default`
- preselection method: `deterministic_info`
- fixed k_final: `30`
- current candidate rmse: `3.2408`

## best tested result
- k_preselect `40` -> rmse `3.2796` (delta `+0.0388`)
- keep decision: `drop`

## ranked results
- k_preselect `40` -> rmse `3.2796`, mae `2.5684`, delta `+0.0388`
- k_preselect `44` -> rmse `3.7689`, mae `2.8730`, delta `+0.5281`
- k_preselect `52` -> rmse `3.9462`, mae `2.9476`, delta `+0.7054`
- k_preselect `48` -> rmse `4.0654`, mae `3.1444`, delta `+0.8247`
- k_preselect `56` -> rmse `4.3486`, mae `3.2097`, delta `+1.1078`
