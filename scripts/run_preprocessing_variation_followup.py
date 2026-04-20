#!/usr/bin/env python3
"""Run a wider real-data preprocessing follow-up around the promoted profile."""

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
    summarize_preprocessing_experiments,
    top_summary_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports" / "preprocessing_variation_followup"
EXPLORATION_DIR = REPORTS_DIR / "real_exploration"
CONFIRMATION_DIR = REPORTS_DIR / "real_confirmation"
COMBINED_DIR = REPORTS_DIR / "all_runs"

DATASET_ID = "large_release_default_subset"
BASELINE_PROFILE_ID = "reconstruction_relaxed"
PRESELECTION_METHOD = "deterministic_info"
EXPLORATION_SEEDS = (7,)
CONFIRMATION_SEEDS = (7, 11, 19)
CONFIRMATION_PROFILE_LIMIT = 4


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

    dataset = build_dataset()
    profiles = build_profiles()
    bundle = benchiq.load_bundle(REPO_ROOT / dataset.source_path, config=dataset.base_config)
    bundles = {dataset.dataset_id: bundle}

    exploration_raw = execute_preprocessing_experiment_plans(
        bundles=bundles,
        datasets=[dataset],
        profiles=profiles,
        plans=[
            PreprocessingOptimizationPlan(
                search_stage="real_exploration_seed7",
                dataset_id=dataset.dataset_id,
                profile_ids=tuple(profile.profile_id for profile in profiles),
                preselection_methods=(PRESELECTION_METHOD,),
                seeds=EXPLORATION_SEEDS,
            )
        ],
        out_dir=EXPLORATION_DIR,
    )
    exploration_result = summarize_preprocessing_experiments(
        exploration_raw,
        out_dir=EXPLORATION_DIR,
    )
    _rename_summary_artifact(exploration_result.artifact_paths, "exploration_results.md")
    confirmation_profile_ids = shortlist_confirmation_profiles(exploration_result.summary)

    confirmation_raw = execute_preprocessing_experiment_plans(
        bundles=bundles,
        datasets=[dataset],
        profiles=profiles,
        plans=[
            PreprocessingOptimizationPlan(
                search_stage="real_confirmation_multi_seed",
                dataset_id=dataset.dataset_id,
                profile_ids=tuple(confirmation_profile_ids),
                preselection_methods=(PRESELECTION_METHOD,),
                seeds=CONFIRMATION_SEEDS,
            )
        ],
        out_dir=CONFIRMATION_DIR,
    )
    confirmation_result = summarize_preprocessing_experiments(
        confirmation_raw,
        out_dir=CONFIRMATION_DIR,
    )

    combined_result = summarize_preprocessing_experiments(
        combine_preprocessing_experiment_raw_results([exploration_raw, confirmation_raw]),
        out_dir=COMBINED_DIR,
    )
    _rename_summary_artifact(combined_result.artifact_paths, "combined_results.md")

    baseline_row = _require_summary_row(
        confirmation_result.summary,
        profile_id=BASELINE_PROFILE_ID,
    )
    winner_row = _require_best_row(confirmation_result.summary)
    benchmark_delta_rows = _benchmark_delta_rows(
        confirmation_result.per_run_metrics,
        baseline_profile_id=BASELINE_PROFILE_ID,
        winner_profile_id=str(winner_row["profile_id"]),
    )

    decision_payload = build_decision_payload(
        baseline_row=baseline_row,
        winner_row=winner_row,
        confirmation_profile_ids=confirmation_profile_ids,
        benchmark_delta_rows=benchmark_delta_rows,
    )
    summary_markdown = build_summary_markdown(
        dataset=dataset,
        profiles=profiles,
        exploration_summary=exploration_result.summary,
        confirmation_summary=confirmation_result.summary,
        decision_payload=decision_payload,
        benchmark_delta_rows=benchmark_delta_rows,
    )

    (REPORTS_DIR / "decision.json").write_text(
        json.dumps(decision_payload, indent=2),
        encoding="utf-8",
    )
    (REPORTS_DIR / "summary.md").write_text(summary_markdown, encoding="utf-8")
    (REPORTS_DIR / "report.json").write_text(
        json.dumps(
            {
                "dataset": dataset_to_json(dataset),
                "profiles": [profile_to_json(profile) for profile in profiles],
                "exploration_top_rows": _json_safe_records(
                    top_summary_rows(
                        exploration_result.summary,
                        dataset_id=dataset.dataset_id,
                        search_stage="real_exploration_seed7",
                        preselection_method=PRESELECTION_METHOD,
                        limit=8,
                    )
                ),
                "confirmation_rows": _json_safe_records(
                    confirmation_result.summary.sort_values(
                        [
                            "best_available_test_rmse_mean",
                            "seed_rmse_std",
                            "final_selection_instability",
                            "run_runtime_mean_seconds",
                            "retained_items_mean",
                        ],
                        ascending=[True, True, True, True, True],
                    )
                ),
                "decision": decision_payload,
                "benchmark_deltas_vs_baseline": benchmark_delta_rows,
                "artifacts": {
                    "exploration_dir": str(EXPLORATION_DIR),
                    "confirmation_dir": str(CONFIRMATION_DIR),
                    "combined_dir": str(COMBINED_DIR),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(REPORTS_DIR / "summary.md")


def build_dataset() -> PreprocessingOptimizationDataset:
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
        dataset_id=DATASET_ID,
        label="large release-default real-data subset",
        source_path="out/release_bundle_source/release_default_subset_responses_long.parquet",
        base_config=profile.config.model_dump(mode="json"),
        base_stage_options=stage_options,
        notes=(
            "real-data follow-up around the promoted reconstruction-first preprocessing profile",
            (
                "focuses on the stage-01 knobs that still move this bundle: "
                "low-tail trimming, mild discrimination, and stricter ceiling cuts"
            ),
        ),
    )


def build_profiles() -> list[PreprocessingOptimizationProfile]:
    return [
        PreprocessingOptimizationProfile(
            profile_id="reconstruction_relaxed",
            family="promoted_current",
            description="current promoted preprocessing profile",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.0,
                "min_item_coverage": 0.70,
            },
            is_baseline=True,
        ),
        PreprocessingOptimizationProfile(
            profile_id="relaxed_low_tail_0005",
            family="winner_low_tail",
            description="winner plus very light low-tail trimming",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0005,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.0,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="relaxed_low_tail_001",
            family="winner_low_tail",
            description="winner plus spec-sized low-tail trimming",
            config_overrides={
                "drop_low_tail_models_quantile": 0.001,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.0,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="relaxed_low_tail_002",
            family="winner_low_tail",
            description="winner plus a slightly stronger low-tail trim",
            config_overrides={
                "drop_low_tail_models_quantile": 0.002,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.0,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="relaxed_pb_002",
            family="winner_discrimination",
            description="winner plus a very light point-biserial floor",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.02,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="relaxed_pb_005",
            family="winner_discrimination",
            description="winner plus a mild point-biserial floor",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.05,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="relaxed_pb_010",
            family="winner_discrimination",
            description="winner plus a stricter point-biserial floor",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.10,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="relaxed_ceiling_092",
            family="winner_ceiling",
            description="winner plus a 0.92 ceiling cut that actually bites on this bundle",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 0.92,
                "min_abs_point_biserial": 0.0,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="relaxed_ceiling_090",
            family="winner_ceiling",
            description="winner plus a stronger 0.90 ceiling cut",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 0.90,
                "min_abs_point_biserial": 0.0,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="relaxed_low_tail_0005_pb_002",
            family="winner_combo",
            description="winner plus very light low-tail trimming and a soft point-biserial floor",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0005,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.02,
                "min_item_coverage": 0.70,
            },
        ),
        PreprocessingOptimizationProfile(
            profile_id="relaxed_low_tail_001_pb_005",
            family="winner_combo",
            description="winner plus spec low-tail trimming and a mild point-biserial floor",
            config_overrides={
                "drop_low_tail_models_quantile": 0.001,
                "min_item_sd": 0.0,
                "max_item_mean": 0.99,
                "min_abs_point_biserial": 0.05,
                "min_item_coverage": 0.70,
            },
        ),
    ]


def shortlist_confirmation_profiles(summary: pd.DataFrame) -> list[str]:
    top_rows = top_summary_rows(
        summary,
        dataset_id=DATASET_ID,
        search_stage="real_exploration_seed7",
        preselection_method=PRESELECTION_METHOD,
        limit=CONFIRMATION_PROFILE_LIMIT,
    )
    shortlist = [BASELINE_PROFILE_ID]
    shortlist.extend(top_rows["profile_id"].astype(str).tolist())
    return list(dict.fromkeys(shortlist))


def build_decision_payload(
    *,
    baseline_row: dict[str, Any],
    winner_row: dict[str, Any],
    confirmation_profile_ids: list[str],
    benchmark_delta_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_rmse = float(baseline_row["best_available_test_rmse_mean"])
    winner_rmse = float(winner_row["best_available_test_rmse_mean"])
    rmse_delta = winner_rmse - baseline_rmse
    adopt = str(winner_row["profile_id"]) != BASELINE_PROFILE_ID and rmse_delta < 0.0
    decision = "adopt_new_profile" if adopt else "keep_current_profile"
    benchmark_summary = [
        {
            "benchmark_id": row["benchmark_id"],
            "rmse_delta": row["rmse_delta"],
        }
        for row in benchmark_delta_rows
    ]
    if adopt:
        reason = (
            f"{winner_row['profile_id']} lowered mean held-out best-available RMSE from "
            f"{baseline_rmse:.4f} to {winner_rmse:.4f} on the multi-seed confirmation pass."
        )
    elif str(winner_row["profile_id"]) == BASELINE_PROFILE_ID:
        reason = (
            f"none of the broader preprocessing variants beat the current promoted profile; "
            f"`{BASELINE_PROFILE_ID}` stayed best at {baseline_rmse:.4f} mean held-out RMSE."
        )
    else:
        reason = (
            f"the best challenger `{winner_row['profile_id']}` did not beat the current "
            f"promoted profile: {winner_rmse:.4f} vs {baseline_rmse:.4f} mean held-out RMSE."
        )
    return {
        "decision": decision,
        "adopt": adopt,
        "baseline_profile_id": BASELINE_PROFILE_ID,
        "winner_profile_id": str(winner_row["profile_id"]),
        "confirmation_profile_ids": confirmation_profile_ids,
        "baseline_rmse_mean": baseline_rmse,
        "winner_rmse_mean": winner_rmse,
        "rmse_delta_vs_baseline": rmse_delta,
        "baseline_seed_rmse_std": float(baseline_row["seed_rmse_std"]),
        "winner_seed_rmse_std": float(winner_row["seed_rmse_std"]),
        "winner_retained_items_mean": float(winner_row["retained_items_mean"]),
        "winner_retained_models_mean": float(winner_row["retained_models_mean"]),
        "benchmark_rmse_deltas_vs_baseline": benchmark_summary,
        "reason": reason,
    }


def build_summary_markdown(
    *,
    dataset: PreprocessingOptimizationDataset,
    profiles: list[PreprocessingOptimizationProfile],
    exploration_summary: pd.DataFrame,
    confirmation_summary: pd.DataFrame,
    decision_payload: dict[str, Any],
    benchmark_delta_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# preprocessing variation follow-up",
        "",
        "## setup",
        "",
        f"- dataset: `{dataset.dataset_id}`",
        f"- source_path: `{dataset.source_path}`",
        f"- preselection_method: `{PRESELECTION_METHOD}`",
        f"- exploration seeds: {list(EXPLORATION_SEEDS)}",
        f"- confirmation seeds: {list(CONFIRMATION_SEEDS)}",
        f"- profile count explored: {len(profiles)}",
        (
            "- note: this follow-up focuses on low-tail trimming, light "
            "point-biserial floors, and ceiling cuts below 0.95 because those "
            "are the stage-01 knobs that still move the real bundle"
        ),
        "",
        "## exploration top rows",
        "",
    ]
    exploration_top = top_summary_rows(
        exploration_summary,
        dataset_id=DATASET_ID,
        search_stage="real_exploration_seed7",
        preselection_method=PRESELECTION_METHOD,
        limit=8,
    )
    if exploration_top.empty:
        lines.append("- none")
    else:
        for row in exploration_top.to_dict(orient="records"):
            lines.append(
                "- "
                f"`{row['profile_id']}`: rmse_mean={row['best_available_test_rmse_mean']:.4f}, "
                f"retained_items_mean={row['retained_items_mean']:.2f}, "
                f"retained_models_mean={row['retained_models_mean']:.2f}, "
                f"runtime_mean_seconds={row['run_runtime_mean_seconds']:.2f}"
            )
    lines.extend(["", "## confirmation", ""])
    ordered_confirmation = confirmation_summary.sort_values(
        [
            "best_available_test_rmse_mean",
            "seed_rmse_std",
            "final_selection_instability",
            "run_runtime_mean_seconds",
            "retained_items_mean",
        ],
        ascending=[True, True, True, True, True],
    )
    for row in ordered_confirmation.to_dict(orient="records"):
        lines.append(
            "- "
            f"`{row['profile_id']}`: rmse_mean={row['best_available_test_rmse_mean']:.4f}, "
            f"seed_rmse_std={row['seed_rmse_std']:.4f}, "
            f"retained_items_mean={row['retained_items_mean']:.2f}, "
            f"retained_models_mean={row['retained_models_mean']:.2f}, "
            f"final_selection_stability_mean={row['final_selection_stability_mean']:.4f}"
        )
    lines.extend(
        [
            "",
            "## benchmark deltas vs current",
            "",
        ]
    )
    if not benchmark_delta_rows:
        lines.append("- none")
    else:
        for row in benchmark_delta_rows:
            lines.append(
                "- "
                f"`{row['benchmark_id']}`: challenger_minus_current_rmse={row['rmse_delta']:.4f}"
            )
    lines.extend(
        [
            "",
            "## decision",
            "",
            f"- {decision_payload['reason']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _benchmark_delta_rows(
    per_run_metrics: pd.DataFrame,
    *,
    baseline_profile_id: str,
    winner_profile_id: str,
) -> list[dict[str, Any]]:
    if winner_profile_id == baseline_profile_id:
        return []
    frame = per_run_metrics.loc[
        per_run_metrics["profile_id"].isin([baseline_profile_id, winner_profile_id])
    ].copy()
    grouped = (
        frame.groupby(["profile_id", "benchmark_id"], dropna=False)["best_available_test_rmse"]
        .mean()
        .reset_index()
        .pivot(index="benchmark_id", columns="profile_id", values="best_available_test_rmse")
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for row in grouped.to_dict(orient="records"):
        baseline_rmse = float(row[baseline_profile_id])
        winner_rmse = float(row[winner_profile_id])
        rows.append(
            {
                "benchmark_id": str(row["benchmark_id"]),
                "baseline_rmse": baseline_rmse,
                "winner_rmse": winner_rmse,
                "rmse_delta": winner_rmse - baseline_rmse,
            }
        )
    return sorted(rows, key=lambda row: row["rmse_delta"])


def _require_best_row(summary: pd.DataFrame) -> dict[str, Any]:
    row = best_summary_row(
        summary,
        dataset_id=DATASET_ID,
        search_stage="real_confirmation_multi_seed",
        preselection_method=PRESELECTION_METHOD,
    )
    if row is None:
        raise RuntimeError("confirmation sweep did not produce a winner")
    return row


def _require_summary_row(summary: pd.DataFrame, *, profile_id: str) -> dict[str, Any]:
    frame = summary.loc[
        (summary["dataset_id"] == DATASET_ID)
        & (summary["search_stage"] == "real_confirmation_multi_seed")
        & (summary["preselection_method"] == PRESELECTION_METHOD)
        & (summary["profile_id"] == profile_id)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"missing summary row for profile {profile_id}")
    return _json_safe(frame.iloc[0].to_dict())


def dataset_to_json(dataset: PreprocessingOptimizationDataset) -> dict[str, Any]:
    return {
        "dataset_id": dataset.dataset_id,
        "label": dataset.label,
        "source_path": dataset.source_path,
        "base_config": dict(dataset.base_config),
        "base_stage_options": json.loads(json.dumps(dataset.base_stage_options)),
        "notes": list(dataset.notes),
    }


def profile_to_json(profile: PreprocessingOptimizationProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "family": profile.family,
        "description": profile.description,
        "config_overrides": dict(profile.config_overrides),
        "is_baseline": bool(profile.is_baseline),
    }


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in frame.to_dict(orient="records")]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


if __name__ == "__main__":
    main()
