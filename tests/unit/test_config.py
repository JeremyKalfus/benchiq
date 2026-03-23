import pytest
from pydantic import ValidationError

from benchiq import BenchIQConfig


def test_config_accepts_defaults() -> None:
    config = BenchIQConfig()

    assert config.duplicate_policy == "error"
    assert config.min_item_coverage == pytest.approx(0.8)


def test_config_rejects_invalid_item_coverage() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        BenchIQConfig(min_item_coverage=1.2)


def test_config_rejects_inverted_model_thresholds() -> None:
    with pytest.raises(
        ValidationError,
        match="warn_models_per_benchmark must be greater than or equal to min_models_per_benchmark",
    ):
        BenchIQConfig(min_models_per_benchmark=200, warn_models_per_benchmark=100)
