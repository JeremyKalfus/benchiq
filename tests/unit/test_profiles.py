import benchiq


def test_reconstruction_first_profile_exposes_generalized_winner_defaults() -> None:
    profile = benchiq.build_reconstruction_first_profile(random_seed=11)

    assert profile.profile_id == "reconstruction_first"
    assert profile.config.random_seed == 11
    assert profile.config.drop_low_tail_models_quantile == 0.002
    assert profile.config.min_item_sd == 0.0
    assert profile.config.max_item_mean == 0.99
    assert profile.config.min_abs_point_biserial == 0.0
    assert profile.config.min_item_coverage == 0.70
    assert profile.stage_options_copy()["04_subsample"]["method"] == "deterministic_info"
    assert profile.stage_options_copy()["05_irt"]["backend"] == "girth"


def test_psychometric_default_profile_stays_spec_aligned() -> None:
    profile = benchiq.build_psychometric_default_profile(random_seed=19)

    assert profile.profile_id == "psychometric_default"
    assert profile.config.drop_low_tail_models_quantile == 0.001
    assert profile.config.min_item_sd == 0.01
    assert profile.config.max_item_mean == 0.95
    assert profile.config.min_abs_point_biserial == 0.05
    assert profile.config.min_item_coverage == 0.80
    assert profile.stage_options_copy()["04_subsample"]["method"] == "random_cv"
    assert profile.stage_options_copy()["05_irt"]["backend"] == "girth"


def test_load_profile_accepts_named_aliases() -> None:
    default = benchiq.load_profile("default", random_seed=7)
    recommended = benchiq.load_profile("recommended", random_seed=7)
    baseline = benchiq.load_profile("baseline", random_seed=7)

    assert default.profile_id == "reconstruction_first"
    assert recommended.profile_id == "reconstruction_first"
    assert baseline.profile_id == "psychometric_default"
    assert "reconstruction_first" in benchiq.product_profiles()


def test_runtime_defaults_match_reconstruction_first_profile() -> None:
    profile = benchiq.build_reconstruction_first_profile(random_seed=23)
    runner = benchiq.BenchIQRunner()

    assert profile.config == benchiq.BenchIQConfig(random_seed=23)
    assert runner.stage_options["04_subsample"]["method"] == "deterministic_info"
    assert runner.stage_options["05_irt"]["backend"] == "girth"
