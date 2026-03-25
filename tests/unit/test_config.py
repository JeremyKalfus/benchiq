import pytest
from pydantic import ValidationError

from benchiq import BenchIQConfig


def test_config_accepts_defaults() -> None:
    config = BenchIQConfig()

    assert config.duplicate_policy == "error"
    assert config.drop_low_tail_models_quantile == pytest.approx(0.001)
    assert config.min_item_sd == pytest.approx(0.01)
    assert config.max_item_mean == pytest.approx(0.95)
    assert config.min_abs_point_biserial == pytest.approx(0.05)
    assert config.min_item_coverage == pytest.approx(0.8)
    assert config.min_overlap_models_for_redundancy == 75
    assert config.p_test == pytest.approx(0.10)
    assert config.p_val == pytest.approx(0.10)
    assert config.n_strata_bins == 10
    assert config.p_test == pytest.approx(0.10)
    assert config.p_val == pytest.approx(0.10)
    assert config.n_strata_bins == 10


def test_config_rejects_invalid_item_coverage() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        BenchIQConfig(min_item_coverage=1.2)


def test_config_rejects_inverted_model_thresholds() -> None:
    with pytest.raises(
        ValidationError,
        match="warn_models_per_benchmark must be greater than or equal to min_models_per_benchmark",
    ):
        BenchIQConfig(min_models_per_benchmark=200, warn_models_per_benchmark=100)
