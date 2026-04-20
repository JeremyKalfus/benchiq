"""Public product profiles for BenchIQ."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from benchiq.config import BenchIQConfig


@dataclass(slots=True, frozen=True)
class BenchIQProfile:
    """Named product profile with config and stage overrides."""

    profile_id: str
    description: str
    config: BenchIQConfig
    stage_options: dict[str, dict[str, Any]]
    notes: tuple[str, ...] = ()

    def stage_options_copy(self) -> dict[str, dict[str, Any]]:
        """Return a deep copy of the stage overrides."""

        return deepcopy(self.stage_options)


def build_psychometric_default_profile(*, random_seed: int = 0) -> BenchIQProfile:
    """Return the spec-aligned psychometric baseline profile."""

    return BenchIQProfile(
        profile_id="psychometric_default",
        description="spec-aligned psychometric baseline with random-cv preselection",
        config=BenchIQConfig(
            drop_low_tail_models_quantile=0.001,
            min_item_sd=0.01,
            max_item_mean=0.95,
            min_abs_point_biserial=0.05,
            min_item_coverage=0.80,
            random_seed=random_seed,
        ),
        stage_options={"04_subsample": {"method": "random_cv"}},
        notes=(
            "retains the locked v0.1 psychometric thresholds as an explicit legacy baseline",
            "leaves size-dependent budgets like k_preselect and k_final to the caller",
        ),
    )


def build_reconstruction_first_profile(*, random_seed: int = 0) -> BenchIQProfile:
    """Return the default reconstruction-first product profile."""

    return BenchIQProfile(
        profile_id="reconstruction_first",
        description=(
            "default reconstruction-first profile built from the "
            "relaxed-low-tail follow-up winner with deterministic-info preselection"
        ),
        config=BenchIQConfig(random_seed=random_seed),
        stage_options={"04_subsample": {"method": "deterministic_info"}},
        notes=(
            "matches the current BenchIQ runtime defaults",
            (
                "adds a light 0.002 low-tail trim on top of the relaxed "
                "reconstruction-first thresholds"
            ),
            "keeps the item-count floors and split policy intact",
            "leaves size-dependent budgets like k_preselect and k_final to the caller",
        ),
    )


def load_profile(profile_id: str, *, random_seed: int = 0) -> BenchIQProfile:
    """Resolve a named public product profile."""

    normalized = profile_id.strip().lower().replace("-", "_")
    if normalized in {
        "reconstruction_first",
        "default",
        "recommended",
        "reconstruction_recommended",
    }:
        return build_reconstruction_first_profile(random_seed=random_seed)
    if normalized in {"psychometric_default", "baseline", "legacy", "spec"}:
        return build_psychometric_default_profile(random_seed=random_seed)
    raise ValueError(f"unsupported BenchIQ profile: {profile_id}")


def product_profiles() -> dict[str, str]:
    """Return the public product profiles and when to use them."""

    return {
        "reconstruction_first": (
            "default reconstruction-first profile with light low-tail trimming, "
            "relaxed preprocessing, and "
            "deterministic-info preselection"
        ),
        "psychometric_default": (
            "spec-aligned psychometric baseline with metabench-style thresholds and "
            "random-cv preselection"
        ),
    }


__all__ = [
    "BenchIQProfile",
    "build_psychometric_default_profile",
    "build_reconstruction_first_profile",
    "load_profile",
    "product_profiles",
]
