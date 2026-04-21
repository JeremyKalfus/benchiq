# helm k_preselect sweep

- dataset: `helm_objective__capabilities_v1_0_0`
- method: `deterministic_info`
- k_final fixed at `18`
- k_preselect candidates: `18, 24, 28, 32, 36, 40`

## ranking
- `minimal_cleaning` / k_preselect `24`: rmse `6.4988`, mae `5.9896`, preselect `24.0`
- `reconstruction_first` / k_preselect `24`: rmse `6.4988`, mae `5.9896`, preselect `24.0`
- `reconstruction_first_relaxed` / k_preselect `24`: rmse `6.4988`, mae `5.9896`, preselect `24.0`
- `minimal_cleaning` / k_preselect `28`: rmse `7.4112`, mae `6.2138`, preselect `28.0`
- `reconstruction_first` / k_preselect `28`: rmse `7.4112`, mae `6.2138`, preselect `28.0`
- `reconstruction_first_relaxed` / k_preselect `28`: rmse `7.4112`, mae `6.2138`, preselect `28.0`
- `minimal_cleaning` / k_preselect `36`: rmse `8.6644`, mae `8.1515`, preselect `36.0`
- `reconstruction_first` / k_preselect `36`: rmse `8.6644`, mae `8.1515`, preselect `36.0`
- `reconstruction_first_relaxed` / k_preselect `36`: rmse `8.6644`, mae `8.1515`, preselect `36.0`
- `minimal_cleaning` / k_preselect `18`: rmse `9.8916`, mae `7.7669`, preselect `18.0`
- `reconstruction_first` / k_preselect `18`: rmse `9.8916`, mae `7.7669`, preselect `18.0`
- `reconstruction_first_relaxed` / k_preselect `18`: rmse `9.8916`, mae `7.7669`, preselect `18.0`
- `minimal_cleaning` / k_preselect `32`: rmse `10.1934`, mae `9.1672`, preselect `32.0`
- `reconstruction_first` / k_preselect `32`: rmse `10.1934`, mae `9.1672`, preselect `32.0`
- `reconstruction_first_relaxed` / k_preselect `32`: rmse `10.1934`, mae `9.1672`, preselect `32.0`
- `minimal_cleaning` / k_preselect `40`: rmse `24.3834`, mae `21.1389`, preselect `40.0`
- `reconstruction_first` / k_preselect `40`: rmse `24.3834`, mae `21.1389`, preselect `40.0`
- `reconstruction_first_relaxed` / k_preselect `40`: rmse `24.3834`, mae `21.1389`, preselect `40.0`
