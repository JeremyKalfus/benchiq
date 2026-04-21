# helm psychometric deterministic k_final sweep

## setup
- dataset: `helm_objective__capabilities_v1_0_0`
- profile: `psychometric_default`
- preselection method: `deterministic_info`
- fixed k_preselect: `24`
- current standing rmse: `5.9214`

## best tested result
- k_final `22` -> rmse `4.4952` (delta `-1.4261`)
- keep decision: `candidate`

## ranked results
- k_final `22` -> rmse `4.4952`, mae `4.1284`, delta `-1.4261`
- k_final `18` -> rmse `5.9214`, mae `4.7181`, delta `+0.0000`
- k_final `24` -> rmse `6.2442`, mae `4.9834`, delta `+0.3229`
- k_final `12` -> rmse `6.7047`, mae `6.3062`, delta `+0.7833`
- k_final `14` -> rmse `7.8965`, mae `7.2845`, delta `+1.9752`
- k_final `20` -> rmse `8.9390`, mae `8.1851`, delta `+3.0177`
- k_final `16` -> rmse `34.0568`, mae `24.1270`, delta `+28.1354`
