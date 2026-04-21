# helm psychometric deterministic k_preselect sweep

## setup
- dataset: `helm_objective__capabilities_v1_0_0`
- profile: `psychometric_default`
- preselection method: `deterministic_info`
- fixed k_final: `22`
- current candidate rmse: `4.4952`

## best tested result
- k_preselect `24` -> rmse `4.4952` (delta `+0.0000`)
- keep decision: `drop`

## ranked results
- k_preselect `24` -> rmse `4.4952`, mae `4.1284`, delta `+0.0000`
- k_preselect `36` -> rmse `7.4994`, mae `6.8265`, delta `+3.0042`
- k_preselect `32` -> rmse `8.2660`, mae `6.8603`, delta `+3.7707`
- k_preselect `28` -> rmse `10.0594`, mae `9.1439`, delta `+5.5642`
- k_preselect `26` -> rmse `10.3143`, mae `9.4849`, delta `+5.8191`
- k_preselect `22` -> rmse `10.7801`, mae `9.2107`, delta `+6.2849`
