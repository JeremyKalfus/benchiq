#!/usr/bin/env python3
"""Run one focused preprocessing optimization pass for BenchIQ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import benchiq
from benchiq.preprocess.optimization import (
    PreprocessingOptimizationDataset,
    PreprocessingOptimizationPlan,
    PreprocessingOptimizationProfile,
    best_summary_row,
    combine_preprocessing_experiment_raw_results,
    execute_preprocessing_experiment_plans,
    resolve_experiment_config,
    resolve_experiment_stage_options,
    summarize_preprocessing_experiments,
    top_summary_rows,
)
from benchiq.reconstruct import run_reconstruction_head_experiments

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports" / "preprocessing_optimization"
COMPACT_SEARCH_DIR = REPORTS_DIR / "compact_search"
LARGE_CONFIRM_DIR = REPORTS_DIR / "large_confirmation"
LARGE_METHOD_DIR = REPORTS_DIR / "large_method_check"
HEAD_CHECKS_DIR = REPORTS_DIR / "head_checks"

COMPACT_DATASET_ID = "compact_validation_fixture"
LARGE_DATASET_ID = "large_release_default_subset"
DEFAULT_SEEDS = (7, 11, 19)
COMPACT_SHORTLIST_SIZE = 4

LARGE_PROFILE_ALIASES = {
    "psychometric_default": "baseline_current",
    "ceiling_strict": "baseline_current",
    "discrimination_strict": "strict_cleaning",
}


def _rename_summary_artifact(artifact_paths: dict[str, Path], filename: str) -> None:
    summary_path = artifact_paths.get("summary_md")
    if summary_path is None or not summary_path.exists():
        return
    target_path = summary_path.with_name(filename)
    if target_path.exists():
        target_path.unlink()
    summary_path.rename(target_path)
    artifact_paths["summary_md"] = target_path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    compact_dataset = build_compact_dataset()
    large_dataset = build_large_dataset()
    compact_profiles = build_compact_profiles()
    large_profiles = build_large_profiles()

    compact_bundles = load_bundles([compact_dataset])
    compact_raw = execute_preprocessing_experiment_plans(
        bundles=compact_bundles,
        datasets=[compact_dataset],
        profiles=compact_profiles,
        plans=[
            PreprocessingOptimizationPlan(
                search_stage="compact_broad",
                dataset_id=COMPACT_DATASET_ID,
                profile_ids=tuple(profile.profile_id for profile in compact_profiles),
                preselection_methods=("random_cv", "deterministic_info"),
                seeds=DEFAULT_SEEDS,
            )
        ],
        out_dir=COMPACT_SEARCH_DIR,
    )
    compact_summary = summarize_preprocessing_experiments(
        compact_raw,
        out_dir=COMPACT_SEARCH_DIR,
    )
    _rename_summary_artifact(compact_summary.artifact_paths, "search_results.md")

    compact_top_rows = top_summary_rows(
        compact_summary.summary,
        dataset_id=COMPACT_DATASET_ID,
        search_stage="compact_broad",
        preselection_method="deterministic_info",
        limit=COMPACT_SHORTLIST_SIZE,
    )
    shortlist_profile_ids = shortlist_for_large_confirmation(compact_top_rows)

    large_bundles = load_bundles([large_dataset])
    large_confirmation_raw = execute_preprocessing_experiment_plans(
        bundles=large_bundles,
        datasets=[large_dataset],
        profiles=large_profiles,
        plans=[
            PreprocessingOptimizationPlan(
                search_stage="large_confirmation",
                dataset_id=LARGE_DATASET_ID,
                profile_ids=tuple(shortlist_profile_ids),
                preselection_methods=("deterministic_info",),
                seeds=DEFAULT_SEEDS,
            )
        ],
        out_dir=LARGE_CONFIRM_DIR,
    )
    large_confirmation_summary = summarize_preprocessing_experiments(
        large_confirmation_raw,
        out_dir=LARGE_CONFIRM_DIR,
    )
    large_winner = best_summary_row(
        large_confirmation_summary.summary,
        dataset_id=LARGE_DATASET_ID,
        search_stage="large_confirmation",
        preselection_method="deterministic_info",
    )
    if large_winner is None:
        raise RuntimeError("large confirmation did not produce a winner")

    large_method_profiles = tuple(
        dict.fromkeys(["baseline_current", str(large_winner["profile_id"])]),
    )
    large_method_raw = execute_preprocessing_experiment_plans(
        bundles=large_bundles,
        datasets=[large_dataset],
        profiles=large_profiles,
        plans=[
            PreprocessingOptimizationPlan(
                search_stage="large_method_check",
                dataset_id=LARGE_DATASET_ID,
                profile_ids=large_method_profiles,
                preselection_methods=("random_cv",),
                seeds=DEFAULT_SEEDS,
            )
        ],
        out_dir=LARGE_METHOD_DIR,
    )
    large_method_summary = summarize_preprocessing_experiments(
        large_method_raw,
        out_dir=LARGE_METHOD_DIR,
    )
    _rename_summary_artifact(large_method_summary.artifact_paths, "method_check.md")

    combined_raw = combine_preprocessing_experiment_raw_results(
        [compact_raw, large_confirmation_raw, large_method_raw],
    )
    final_result = summarize_preprocessing_experiments(
        combined_raw,
        out_dir=REPORTS_DIR,
    )

    compact_winner = best_summary_row(
        final_result.summary,
        dataset_id=COMPACT_DATASET_ID,
        search_stage="compact_broad",
        preselection_method="deterministic_info",
    )
    overall_winner = best_summary_row(
        final_result.summary,
        dataset_id=LARGE_DATASET_ID,
        search_stage="large_confirmation",
        preselection_method="deterministic_info",
    )
    if overall_winner is None:
        raise RuntimeError("overall winner selection failed")

    head_check_summary = run_head_checks(
        compact_dataset=compact_dataset,
        large_dataset=large_dataset,
        compact_profiles=compact_profiles,
        large_profiles=large_profiles,
        compact_bundles=compact_bundles,
        large_bundles=large_bundles,
        compact_winner_profile_id=(
            str(compact_winner["profile_id"]) if compact_winner else "baseline_current"
        ),
        large_winner_profile_id=str(overall_winner["profile_id"]),
    )
    write_head_check_artifacts(head_check_summary)

    recommendation_payload = build_recommendation_payload(
        compact_dataset=compact_dataset,
        large_dataset=large_dataset,
        compact_profiles=compact_profiles,
        large_profiles=large_profiles,
        final_result=final_result,
        compact_winner=compact_winner,
        overall_winner=overall_winner,
        shortlist_profile_ids=shortlist_profile_ids,
        head_check_summary=head_check_summary,
    )
    (REPORTS_DIR / "best_config.json").write_text(
        json.dumps(recommendation_payload["best_config"], indent=2),
        encoding="utf-8",
    )
    (REPORTS_DIR / "summary.md").write_text(
        recommendation_payload["summary_markdown"],
        encoding="utf-8",
    )
    (REPORTS_DIR / "report.json").write_text(
        json.dumps(recommendation_payload["report_json"], indent=2),
        encoding="utf-8",
    )
    print(REPORTS_DIR / "summary.md")


def build_compact_dataset() -> PreprocessingOptimizationDataset:
    config_payload = json.loads(
        (REPO_ROOT / "tests" / "data" / "tiny_example" / "config.json").read_text(encoding="utf-8"),
    )
    stage_options = json.loads(json.dumps(config_payload["stage_options"]))
    stage_options.setdefault("04_subsample", {})
    stage_options["04_subsample"]["n_iter"] = 12
    stage_options["04_subsample"]["checkpoint_interval"] = 4
    return PreprocessingOptimizationDataset(
        dataset_id=COMPACT_DATASET_ID,
        label="compact validation fixture",
        source_path="tests/data/compact_validation/responses_long.csv",
        base_config=config_payload["config"],
        base_stage_options=stage_options,
        notes=("matches the saved compact comparison setup except that random_cv uses n_iter=12",),
    )


def build_large_dataset() -> PreprocessingOptimizationDataset:
    profile = benchiq.build_psychometric_default_profile(random_seed=7)
    stage_options = {
        "04_subsample": {
            "method": "random_cv",
            "k_preselect": 350,
            "n_iter": 24,
            "cv_folds": 5,
            "checkpoint_interval": 8,
            "lam_grid": [0.1, 1.0],
        },
        "05_irt": {"backend_options": None},
        "06_select": {
            "k_final": 250,
            "n_bins": 250,
            "theta_grid_size": 251,
        },
        "07_theta": {"theta_method": "MAP", "theta_grid_size": 251},
        "09_reconstruct": {
            "lam_grid": [0.1, 1.0],
            "cv_folds": 5,
            "n_splines": 10,
        },
        "10_redundancy": {
            "lam_grid": [0.1, 1.0],
            "cv_folds": 5,
            "n_splines": 10,
            "n_factors_to_try": [1, 2, 3],
        },
    }
    return PreprocessingOptimizationDataset(
        dataset_id=LARGE_DATASET_ID,
        label="large release-default real-data subset",
        source_path="out/release_bundle_source/release_default_subset_responses_long.parquet",
        base_config=profile.config.model_dump(mode="json"),
        base_stage_options=stage_options,
        notes=(
            "uses the 6-benchmark public release-default subset from the frozen Zenodo snapshot",
            (
                "keeps full-profile downstream budgets but trims random_cv iterations "
                "to a controlled experiment budget"
            ),
        ),
    )


def build_compact_profiles() -> list[PreprocessingOptimizationProfile]:
    return [
        PreprocessingOptimizationProfile(
            profile_id="baseline_current",
            family="current_saved_setup",
            description="current saved compact comparison setup",
            config_overrides={},
            is_baseline=True,
        ),
        PreprocessingOptimizationProfile(
            profile_id="psychometric_default",
            family="psychometric_default",
            description=(
                "restore the psychometric ceiling and discrimination filters on the compact fixture"
            ),
            config_overrides={
                "max_item_mean": 0.95,
                "min_abs_point_biserial": 0.05,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="no_low_tail",
            family="low_tail_toggle",
            description="disable low-tail model trimming",
            config_overrides={"drop_low_tail_models_quantile": 0.0},
        ),
        PreprocessingOptimizationProfile(
            profile_id="ceiling_strict",
            family="ceiling_variants",
            description=(
                "use the stricter 0.95 near-ceiling threshold without re-enabling discrimination"
            ),
            config_overrides={"max_item_mean": 0.95},
        ),
        PreprocessingOptimizationProfile(
            profile_id="ceiling_off",
            family="ceiling_variants",
            description="turn the near-ceiling filter off",
            config_overrides={"max_item_mean": 1.0},
        ),
        PreprocessingOptimizationProfile(
            profile_id="discrimination_strict",
            family="discrimination_variants",
            description="push the discrimination floor above the spec default",
            config_overrides={"min_abs_point_biserial": 0.10},
        ),
        PreprocessingOptimizationProfile(
            profile_id="coverage_soft",
            family="coverage_variants",
            description="soften model coverage while keeping other filters light",
            config_overrides={"min_item_coverage": 0.70},
        ),
        PreprocessingOptimizationProfile(
            profile_id="coverage_strict",
            family="coverage_variants",
            description="tighten model coverage and item response count",
            config_overrides={
                "min_models_per_item": 18,
                "min_item_coverage": 0.90,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="reconstruction_relaxed",
            family="reconstruction_first_relaxed",
            description="remove low-tail trimming and variance filtering, keep only light coverage",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 1.0,
                "min_abs_point_biserial": 0.0,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="minimal_cleaning",
            family="minimal_cleaning",
            description=(
                "preserve almost everything except the compact fixture's "
                "effective item-coverage floor"
            ),
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 1.0,
                "min_abs_point_biserial": 0.0,
                "min_models_per_item": 1,
                "min_item_coverage": 0.60,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="strict_cleaning",
            family="strict_cleaning",
            description="push all major filters toward a stricter psychometric cleaning profile",
            config_overrides={
                "drop_low_tail_models_quantile": 0.005,
                "min_item_sd": 0.02,
                "max_item_mean": 0.90,
                "min_abs_point_biserial": 0.10,
                "min_models_per_item": 18,
                "min_item_coverage": 0.90,
            },
        ),
    ]


def build_large_profiles() -> list[PreprocessingOptimizationProfile]:
    return [
        PreprocessingOptimizationProfile(
            profile_id="baseline_current",
            family="psychometric_default",
            description="current full-size psychometric default profile",
            config_overrides={},
            is_baseline=True,
        ),
        PreprocessingOptimizationProfile(
            profile_id="no_low_tail",
            family="low_tail_toggle",
            description="disable low-tail model trimming on the large bundle",
            config_overrides={"drop_low_tail_models_quantile": 0.0},
        ),
        PreprocessingOptimizationProfile(
            profile_id="ceiling_off",
            family="ceiling_variants",
            description="turn the near-ceiling filter off on the large bundle",
            config_overrides={"max_item_mean": 1.0},
        ),
        PreprocessingOptimizationProfile(
            profile_id="discrimination_off",
            family="discrimination_variants",
            description="disable the point-biserial discrimination filter",
            config_overrides={"min_abs_point_biserial": 0.0},
        ),
        PreprocessingOptimizationProfile(
            profile_id="coverage_soft",
            family="coverage_variants",
            description="soften model coverage while leaving the spec item-count floor intact",
            config_overrides={"min_item_coverage": 0.70},
        ),
        PreprocessingOptimizationProfile(
            profile_id="coverage_strict",
            family="coverage_variants",
            description=(
                "tighten model coverage and raise the explicit item-count "
                "requirement above the 20% floor"
            ),
            config_overrides={
                "min_models_per_item": 1700,
                "min_item_coverage": 0.90,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="reconstruction_relaxed",
            family="reconstruction_first_relaxed",
            description=(
                "keep the spec item-count floor but relax low-tail, variance, "
                "ceiling, and discrimination filters"
            ),
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.0,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="minimal_cleaning",
            family="minimal_cleaning",
            description=(
                "minimal cleaning under the current spec floor: no low-tail, "
                "no variance, no ceiling, no discrimination, soft model coverage"
            ),
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 1.0,
                "min_abs_point_biserial": 0.0,
                "min_models_per_item": 1,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="strict_cleaning",
            family="strict_cleaning",
            description="strict psychometric cleaning with tighter coverage",
            config_overrides={
                "drop_low_tail_models_quantile": 0.005,
                "min_item_sd": 0.02,
                "max_item_mean": 0.90,
                "min_abs_point_biserial": 0.10,
                "min_models_per_item": 1700,
                "min_item_coverage": 0.90,
            },
        ),
    ]


def load_bundles(
    datasets: list[PreprocessingOptimizationDataset],
) -> dict[str, benchiq.Bundle]:
    bundles: dict[str, benchiq.Bundle] = {}
    for dataset in datasets:
        bundles[dataset.dataset_id] = benchiq.load_bundle(
            REPO_ROOT / dataset.source_path,
            config=dataset.base_config,
        )
    return bundles


def shortlist_for_large_confirmation(compact_top_rows: pd.DataFrame) -> list[str]:
    shortlist = ["baseline_current", "reconstruction_relaxed", "minimal_cleaning"]
    for profile_id in compact_top_rows["profile_id"].astype(str).tolist():
        shortlist.append(LARGE_PROFILE_ALIASES.get(profile_id, profile_id))
    ordered = list(dict.fromkeys(shortlist))
    available = {profile.profile_id for profile in build_large_profiles()}
    return [profile_id for profile_id in ordered if profile_id in available]


def run_head_checks(
    *,
    compact_dataset: PreprocessingOptimizationDataset,
    large_dataset: PreprocessingOptimizationDataset,
    compact_profiles: list[PreprocessingOptimizationProfile],
    large_profiles: list[PreprocessingOptimizationProfile],
    compact_bundles: dict[str, benchiq.Bundle],
    large_bundles: dict[str, benchiq.Bundle],
    compact_winner_profile_id: str,
    large_winner_profile_id: str,
) -> pd.DataFrame:
    cases = [
        (
            "compact_baseline",
            compact_dataset,
            compact_bundles[COMPACT_DATASET_ID],
            compact_profiles,
            "baseline_current",
        ),
        (
            "compact_winner",
            compact_dataset,
            compact_bundles[COMPACT_DATASET_ID],
            compact_profiles,
            compact_winner_profile_id,
        ),
        (
            "large_baseline",
            large_dataset,
            large_bundles[LARGE_DATASET_ID],
            large_profiles,
            "baseline_current",
        ),
        (
            "large_winner",
            large_dataset,
            large_bundles[LARGE_DATASET_ID],
            large_profiles,
            large_winner_profile_id,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for case_id, dataset, bundle, profiles, profile_id in cases:
        profile = next(profile for profile in profiles if profile.profile_id == profile_id)
        config = resolve_experiment_config(dataset=dataset, profile=profile, seed=7)
        stage_options = resolve_experiment_stage_options(
            dataset=dataset,
            preselection_method="deterministic_info",
        )
        run_result = benchiq.run(
            bundle,
            config=config,
            out_dir=HEAD_CHECKS_DIR / "workdir",
            run_id=f"{case_id}__feature_run",
            stage_options=stage_options,
            stop_after="08_features",
        )
        report_dir = HEAD_CHECKS_DIR / case_id
        report_dir.mkdir(parents=True, exist_ok=True)
        experiment_result = run_reconstruction_head_experiments(
            run_result.stage_results["08_features"],
            methods=("gam", "elastic_net"),
            seeds=DEFAULT_SEEDS,
            lam_grid=tuple(stage_options["09_reconstruct"]["lam_grid"]),
            cv_folds=int(stage_options["09_reconstruct"]["cv_folds"]),
            n_splines=int(stage_options["09_reconstruct"]["n_splines"]),
            out_dir=report_dir,
        )
        _rename_summary_artifact(experiment_result.artifact_paths, "head_comparison.md")
        summary = experiment_result.summary.copy()
        summary["case_id"] = case_id
        summary["dataset_id"] = dataset.dataset_id
        summary["dataset_label"] = dataset.label
        summary["profile_id"] = profile.profile_id
        summary["family"] = profile.family
        rows.extend(summary.to_dict(orient="records"))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows)


def write_head_check_artifacts(head_check_summary: pd.DataFrame) -> None:
    HEAD_CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    head_check_summary.to_csv(HEAD_CHECKS_DIR / "head_check_summary.csv", index=False)
    head_check_summary.to_parquet(HEAD_CHECKS_DIR / "head_check_summary.parquet", index=False)
    lines = ["# preprocessing head checks", ""]
    if head_check_summary.empty:
        lines.append("- none")
    else:
        for row in head_check_summary.to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['case_id']} / {row['model_type']} / {row['method']}: "
                f"rmse_mean={_fmt_float(row['rmse_mean'])}, "
                f"mae_mean={_fmt_float(row['mae_mean'])}, "
                f"runtime_mean_seconds={_fmt_float(row['runtime_mean_seconds'])}"
            )
    (HEAD_CHECKS_DIR / "head_check_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_recommendation_payload(
    *,
    compact_dataset: PreprocessingOptimizationDataset,
    large_dataset: PreprocessingOptimizationDataset,
    compact_profiles: list[PreprocessingOptimizationProfile],
    large_profiles: list[PreprocessingOptimizationProfile],
    final_result,
    compact_winner: dict[str, Any] | None,
    overall_winner: dict[str, Any],
    shortlist_profile_ids: list[str],
    head_check_summary: pd.DataFrame,
) -> dict[str, Any]:
    large_profile_map = {profile.profile_id: profile for profile in large_profiles}
    overall_profile = large_profile_map[str(overall_winner["profile_id"])]
    baseline_profile = large_profile_map["baseline_current"]
    best_config = {
        "recommended_action": "keep_defaults_but_recommend_profile",
        "reason": (
            "the locked v0.1 spec still describes psychometric-style defaults, so this pass "
            "recommends an evidence-backed reconstruction-first profile instead of silently "
            "rewriting BenchIQConfig defaults"
        ),
        "winner_dataset_id": LARGE_DATASET_ID,
        "winner_search_stage": "large_confirmation",
        "winner_preselection_method": "deterministic_info",
        "winner_profile_id": overall_profile.profile_id,
        "winner_family": overall_profile.family,
        "evaluation_seeds": list(DEFAULT_SEEDS),
        "profile_overrides": dict(overall_profile.config_overrides),
        "recommended_config": resolve_experiment_config(
            dataset=large_dataset,
            profile=overall_profile,
            seed=7,
        ).model_dump(mode="json"),
        "recommended_stage_options": resolve_experiment_stage_options(
            dataset=large_dataset,
            preselection_method="deterministic_info",
        ),
        "baseline_config": resolve_experiment_config(
            dataset=large_dataset,
            profile=baseline_profile,
            seed=7,
        ).model_dump(mode="json"),
    }

    compact_top_table = top_summary_rows(
        final_result.summary,
        dataset_id=COMPACT_DATASET_ID,
        search_stage="compact_broad",
        preselection_method="deterministic_info",
        limit=5,
    )
    large_top_table = top_summary_rows(
        final_result.summary,
        dataset_id=LARGE_DATASET_ID,
        search_stage="large_confirmation",
        preselection_method="deterministic_info",
        limit=5,
    )

    summary_lines = [
        "# preprocessing optimization",
        "",
        "## recommendation",
        "",
        (
            "- default decision: keep the current psychometric default story, "
            "but add a recommended reconstruction-first profile"
        ),
        f"- recommended profile: `{overall_profile.profile_id}` (`{overall_profile.family}`)",
        (
            "- recommended preselection method for the reconstruction-first path: "
            "`deterministic_info`"
        ),
        (
            "- default rationale: the locked v0.1 spec still defines "
            "psychometric-style defaults, so this pass documents the stronger "
            "reconstruction-first profile instead of silently changing "
            "`BenchIQConfig()`"
        ),
        "",
        "## search plan",
        "",
        f"- compact broad search profiles: {len(compact_profiles)}",
        "- compact broad search methods: `random_cv`, `deterministic_info`",
        f"- compact broad search seeds: {list(DEFAULT_SEEDS)}",
        f"- large confirmation shortlist: {shortlist_profile_ids}",
        "- large confirmation method: `deterministic_info`",
        "- large method check method: `random_cv` on the baseline and confirmed winner",
        "",
        "## compact top rows",
        "",
    ]
    if compact_top_table.empty:
        summary_lines.append("- none")
    else:
        for row in compact_top_table.to_dict(orient="records"):
            summary_lines.append(
                "- "
                f"`{row['profile_id']}`: "
                f"rmse_mean={row['best_available_test_rmse_mean']:.4f}, "
                f"seed_rmse_std={row['seed_rmse_std']:.4f}, "
                f"final_selection_stability_mean={row.get('final_selection_stability_mean')}, "
                f"runtime_mean_seconds={row['run_runtime_mean_seconds']:.4f}"
            )
    summary_lines.extend(
        [
            "",
            "## large confirmation top rows",
            "",
        ]
    )
    if large_top_table.empty:
        summary_lines.append("- none")
    else:
        for row in large_top_table.to_dict(orient="records"):
            summary_lines.append(
                "- "
                f"`{row['profile_id']}`: "
                f"rmse_mean={row['best_available_test_rmse_mean']:.4f}, "
                f"seed_rmse_std={row['seed_rmse_std']:.4f}, "
                f"final_selection_stability_mean={row.get('final_selection_stability_mean')}, "
                f"runtime_mean_seconds={row['run_runtime_mean_seconds']:.4f}"
            )
    summary_lines.extend(
        [
            "",
            "## head checks",
            "",
        ]
    )
    if head_check_summary.empty:
        summary_lines.append("- none")
    else:
        for row in head_check_summary.to_dict(orient="records"):
            summary_lines.append(
                "- "
                f"{row['case_id']} / {row['model_type']} / {row['method']}: "
                f"rmse_mean={_fmt_float(row['rmse_mean'])}, "
                f"mae_mean={_fmt_float(row['mae_mean'])}, "
                f"runtime_mean_seconds={_fmt_float(row['runtime_mean_seconds'])}"
            )
    summary_lines.append("")

    report_json = dict(final_result.report)
    report_json["overall_recommendation"] = {
        "compact_winner": compact_winner,
        "large_winner": overall_winner,
        "action": "keep_defaults_but_recommend_profile",
        "recommended_profile_id": overall_profile.profile_id,
        "recommended_family": overall_profile.family,
        "recommended_preselection_method": "deterministic_info",
        "shortlist_profile_ids": shortlist_profile_ids,
        "head_checks": head_check_summary.to_dict(orient="records"),
        "best_config_path": str(REPORTS_DIR / "best_config.json"),
    }
    return {
        "best_config": best_config,
        "summary_markdown": "\n".join(summary_lines),
        "report_json": report_json,
    }


def _fmt_float(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
