# helm psychometric deterministic k_final sweep after low-support chooser

## setup
- dataset: `helm_objective__capabilities_v1_0_0`
- profile: `psychometric_default`
- preselection method: `deterministic_info`
- fixed k_preselect: `24`
- current standing rmse: `3.7785`

## best tested result
- k_final `22` -> rmse `3.7785` (delta `+0.0000`)
- keep decision: `drop`

## ranked results
- k_final `22` -> rmse `3.7785`, mae `3.3180`, delta `+0.0000`
- k_final `18` -> rmse `5.9214`, mae `4.7181`, delta `+2.1429`
- k_final `24` -> rmse `5.9529`, mae `4.3912`, delta `+2.1745`
- k_final `20` -> rmse `8.8505`, mae `7.8794`, delta `+5.0720`
