import numpy as np
import pandas as pd

from benchiq.irt import align_r_baseline_to_benchiq


def test_align_r_baseline_to_benchiq_recovers_known_linear_transform() -> None:
    benchiq_item_params = pd.DataFrame(
        {
            "item_id": pd.Series(["i1", "i2", "i3"], dtype="string"),
            "discrimination": pd.Series([0.8, 1.1, 1.6], dtype="Float64"),
            "difficulty": pd.Series([-1.0, 0.0, 1.2], dtype="Float64"),
        }
    )
    benchiq_theta = pd.DataFrame(
        {
            "model_id": pd.Series(["m1", "m2", "m3", "m4"], dtype="string"),
            "theta_hat": pd.Series([-1.2, -0.1, 0.6, 1.3], dtype="Float64"),
            "theta_se": pd.Series([0.2, 0.2, 0.2, 0.2], dtype="Float64"),
        }
    )

    scale = 1.7
    intercept = -0.4
    r_item_params = pd.DataFrame(
        {
            "item_id": pd.Series(["i1", "i2", "i3"], dtype="string"),
            "discrimination": pd.Series(
                benchiq_item_params["discrimination"].astype(float) / scale,
                dtype="Float64",
            ),
            "difficulty": pd.Series(
                benchiq_item_params["difficulty"].astype(float) * scale + intercept,
                dtype="Float64",
            ),
        }
    )
    r_theta = pd.DataFrame(
        {
            "model_id": pd.Series(["m1", "m2", "m3", "m4"], dtype="string"),
            "theta_hat": pd.Series(
                benchiq_theta["theta_hat"].astype(float) * scale + intercept,
                dtype="Float64",
            ),
            "theta_se": pd.Series([0.34, 0.34, 0.34, 0.34], dtype="Float64"),
        }
    )

    aligned = align_r_baseline_to_benchiq(
        benchiq_item_params=benchiq_item_params,
        benchiq_theta=benchiq_theta,
        r_item_params=r_item_params,
        r_theta=r_theta,
    )

    aligned_items = aligned["item_params_aligned"].sort_values("item_id").reset_index(drop=True)
    aligned_theta = aligned["theta_aligned"].sort_values("model_id").reset_index(drop=True)
    np.testing.assert_allclose(
        aligned_items["discrimination"].astype(float).to_numpy(),
        benchiq_item_params.sort_values("item_id")["discrimination"].astype(float).to_numpy(),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        aligned_items["difficulty"].astype(float).to_numpy(),
        benchiq_item_params.sort_values("item_id")["difficulty"].astype(float).to_numpy(),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        aligned_theta["theta_hat"].astype(float).to_numpy(),
        benchiq_theta.sort_values("model_id")["theta_hat"].astype(float).to_numpy(),
        atol=1e-8,
    )
