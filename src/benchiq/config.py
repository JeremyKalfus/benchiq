"""Configuration models for BenchIQ."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchiq.schema.tables import DuplicatePolicy


class BenchIQConfig(BaseModel):
    """Validated runtime configuration for BenchIQ."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    duplicate_policy: DuplicatePolicy = "error"
    allow_low_n: bool = False
    drop_low_tail_models_quantile: float = Field(default=0.001, ge=0.0, lt=1.0)
    min_item_sd: float = Field(default=0.01, ge=0.0)
    max_item_mean: float = Field(default=0.95, ge=0.0, le=1.0)
    min_abs_point_biserial: float = Field(default=0.05, ge=0.0, le=1.0)
    min_models_per_benchmark: int = Field(default=100, ge=1)
    warn_models_per_benchmark: int = Field(default=200, ge=1)
    min_items_after_filtering: int = Field(default=50, ge=1)
    min_models_per_item: int = Field(default=50, ge=1)
    min_item_coverage: float = Field(default=0.8, gt=0.0, le=1.0)
    min_overlap_models_for_joint: int = Field(default=75, ge=1)
    random_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_threshold_ordering(self) -> "BenchIQConfig":
        if self.warn_models_per_benchmark < self.min_models_per_benchmark:
            raise ValueError(
                "warn_models_per_benchmark must be greater than or equal to "
                "min_models_per_benchmark",
            )
        return self
