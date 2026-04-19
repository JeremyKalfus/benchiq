import benchiq


def test_reconstruction_first_profile_exposes_generalized_winner_defaults() -> None:
    profile = benchiq.build_reconstruction_first_profile(random_seed=11)

    assert profile.profile_id == "reconstruction_first"
    assert profile.config.random_seed == 11
    assert profile.config.drop_low_tail_models_quantile == 0.0
    assert profile.config.min_item_sd == 0.0
    assert profile.config.max_item_mean == 0.99
    assert profile.config.min_abs_point_biserial == 0.0
    assert profile.config.min_item_coverage == 0.70
    assert profile.stage_options_copy()["04_subsample"]["method"] == "deterministic_info"


def test_psychometric_default_profile_stays_spec_aligned() -> None:
    profile = benchiq.build_psychometric_default_profile(random_seed=19)

    assert profile.profile_id == "psychometric_default"
    assert profile.config == benchiq.BenchIQConfig(random_seed=19)
    assert profile.stage_options_copy()["04_subsample"]["method"] == "random_cv"


def test_load_profile_accepts_named_aliases() -> None:
    recommended = benchiq.load_profile("recommended", random_seed=7)
    baseline = benchiq.load_profile("baseline", random_seed=7)

    assert recommended.profile_id == "reconstruction_first"
    assert baseline.profile_id == "psychometric_default"
    assert "reconstruction_first" in benchiq.product_profiles()
