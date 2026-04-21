# ollb psychometric deterministic k_preselect sweep at k_final 40

## setup
- dataset: `ollb_v1_metabench_source__release_default_subset_20260405`
- profile: `psychometric_default`
- preselection method: `deterministic_info`
- fixed k_final: `40`
- current standing rmse: `2.4233`

## best tested result
- k_preselect `44` -> rmse `2.1090` (delta `-0.3142`)
- keep decision: `candidate`

## ranked results
- k_preselect `44` -> rmse `2.1090`, mae `1.5255`, delta `-0.3142`
- k_preselect `56` -> rmse `2.4086`, mae `1.7407`, delta `-0.0147`
- k_preselect `40` -> rmse `2.4233`, mae `1.8279`, delta `+0.0000`
- k_preselect `48` -> rmse `2.4290`, mae `1.7695`, delta `+0.0058`
- k_preselect `52` -> rmse `2.6536`, mae `1.9655`, delta `+0.2304`
