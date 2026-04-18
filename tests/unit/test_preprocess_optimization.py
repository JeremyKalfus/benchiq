import pandas as pd

from benchiq.preprocess.optimization import (
    PreprocessingExperimentRawResult,
    best_summary_row,
    combine_preprocessing_experiment_raw_results,
    summarize_preprocessing_experiments,
)


def test_summarize_preprocessing_experiments_picks_lowest_rmse_winner(tmp_path) -> None:
    raw_result = _raw_result()

    result = summarize_preprocessing_experiments(raw_result, out_dir=tmp_path / "reports")

    winner = best_summary_row(
        result.summary,
        dataset_id="fixture",
        search_stage="compact_broad",
        preselection_method="deterministic_info",
    )

    assert winner is not None
    assert winner["profile_id"] == "winner"
    assert result.artifact_paths["summary_md"].exists()
    assert result.artifact_paths["report_json"].exists()
    assert result.artifact_paths["default_vs_winner_plot"].exists()
    assert result.summary["final_selection_stability_mean"].notna().all()


def test_combine_preprocessing_experiment_raw_results_concatenates_batches() -> None:
    first = _raw_result()
    second = _raw_result(offset=100)

    combined = combine_preprocessing_experiment_raw_results([first, second])

    assert len(combined.run_index.index) == 8
    assert len(combined.per_run_metrics.index) == 16
    assert len(combined.selection_sets.index) == 32


def _raw_result(offset: int = 0) -> PreprocessingExperimentRawResult:
    experiment_matrix = pd.DataFrame(
        [
            {
                "search_stage": "compact_broad",
                "dataset_id": "fixture",
                "dataset_label": "fixture",
                "source_path": "fixture.csv",
                "profile_id": profile_id,
                "family": family,
                "description": profile_id,
                "preselection_method": "deterministic_info",
                "seed": seed + offset,
                "is_baseline": profile_id == "baseline",
                "requested_drop_low_tail_models_quantile": 0.0,
                "requested_min_item_sd": 0.0,
                "requested_max_item_mean": 1.0,
                "requested_min_abs_point_biserial": 0.0,
                "requested_min_models_per_item": 1,
                "requested_min_item_coverage": 0.7,
            }
            for profile_id, family in (("baseline", "baseline"), ("winner", "relaxed"))
            for seed in (7, 11)
        ]
    )
    run_index = pd.DataFrame(
        [
            _run_row(profile_id=profile_id, family=family, seed=seed + offset)
            for profile_id, family in (("baseline", "baseline"), ("winner", "relaxed"))
            for seed in (7, 11)
        ]
    )
    per_run_metrics = pd.DataFrame(
        [
            _metric_row(
                profile_id=profile_id,
                family=family,
                seed=seed + offset,
                benchmark_id=benchmark_id,
                rmse=rmse,
            )
            for profile_id, family, rmse in (
                ("baseline", "baseline", 1.10),
                ("winner", "relaxed", 0.90),
            )
            for seed in (7, 11)
            for benchmark_id in ("b1", "b2")
        ]
    )
    selection_sets = pd.DataFrame(
        [
            _selection_row(
                profile_id=profile_id,
                family=family,
                seed=seed + offset,
                benchmark_id=benchmark_id,
                selection_stage=selection_stage,
                selected_items=selected_items,
            )
            for seed in (7, 11)
            for profile_id, family, selected_items in (
                ("baseline", "baseline", "['i1', 'i2']" if seed == 7 else "['i1', 'i3']"),
                ("winner", "relaxed", "['i1', 'i2']"),
            )
            for benchmark_id in ("b1", "b2")
            for selection_stage in ("preselect", "final")
        ]
    )
    selection_sets["selected_items"] = selection_sets["selected_items"].str.replace("'", '"')
    return PreprocessingExperimentRawResult(
        experiment_matrix=experiment_matrix,
        run_index=run_index,
        per_run_metrics=per_run_metrics,
        selection_sets=selection_sets,
    )


def _run_row(*, profile_id: str, family: str, seed: int) -> dict[str, object]:
    return {
        "run_signature": f"compact_broad::fixture::{profile_id}::deterministic_info::seed-{seed}",
        "run_id": f"{profile_id}-{seed}",
        "search_stage": "compact_broad",
        "dataset_id": "fixture",
        "dataset_label": "fixture",
        "profile_id": profile_id,
        "family": family,
        "description": profile_id,
        "preselection_method": "deterministic_info",
        "seed": seed,
        "is_baseline": profile_id == "baseline",
        "run_root": f"/tmp/{profile_id}-{seed}",
        "config_path": f"/tmp/{profile_id}-{seed}/config.json",
        "metrics_path": f"/tmp/{profile_id}-{seed}/metrics.json",
        "run_runtime_seconds": 1.0,
        "stage01_runtime_seconds": 0.2,
        "stage04_runtime_seconds": 0.3,
        "stage09_runtime_seconds": 0.1,
        "warning_count": 0,
        "drop_low_tail_models_quantile": 0.0,
        "min_item_sd": 0.0,
        "max_item_mean": 1.0,
        "min_abs_point_biserial": 0.0,
        "min_models_per_item": 1,
        "min_item_coverage": 0.7,
    }


def _metric_row(
    *,
    profile_id: str,
    family: str,
    seed: int,
    benchmark_id: str,
    rmse: float,
) -> dict[str, object]:
    row = {
        "run_signature": f"compact_broad::fixture::{profile_id}::deterministic_info::seed-{seed}",
        "run_id": f"{profile_id}-{seed}",
        "search_stage": "compact_broad",
        "dataset_id": "fixture",
        "dataset_label": "fixture",
        "profile_id": profile_id,
        "family": family,
        "description": profile_id,
        "preselection_method": "deterministic_info",
        "seed": seed,
        "is_baseline": profile_id == "baseline",
        "benchmark_id": benchmark_id,
        "refused": False,
        "refusal_reasons": None,
        "benchmark_warning_count": 0,
        "run_warning_count": 0,
        "retained_models": 20,
        "retained_items": 8 if profile_id == "baseline" else 6,
        "selected_items_final": 4,
        "selected_items_preselect": 5,
        "joint_available": True,
        "best_available_model_type": "joint",
        "best_available_test_rmse": rmse,
        "best_available_test_mae": rmse / 2.0,
        "best_available_test_pearson": 0.9,
        "best_available_test_spearman": 0.85,
        "marginal_test_rmse": rmse + 0.1,
        "marginal_test_mae": rmse / 2.0 + 0.1,
        "marginal_test_pearson": 0.88,
        "marginal_test_spearman": 0.83,
        "joint_test_rmse": rmse,
        "joint_test_mae": rmse / 2.0,
        "joint_test_pearson": 0.9,
        "joint_test_spearman": 0.85,
        "run_runtime_seconds": 1.0 if profile_id == "baseline" else 0.8,
        "stage01_runtime_seconds": 0.2,
        "stage04_runtime_seconds": 0.3,
        "stage09_runtime_seconds": 0.1,
    }
    for threshold_name in (
        "drop_low_tail_models_quantile",
        "min_item_sd",
        "max_item_mean",
        "min_abs_point_biserial",
        "min_models_per_item",
        "min_item_coverage",
    ):
        row[f"requested_{threshold_name}"] = 0.0 if threshold_name != "min_item_coverage" else 0.7
        row[f"effective_{threshold_name}"] = row[f"requested_{threshold_name}"]
    row["requested_min_models_per_item"] = 1
    row["effective_min_models_per_item"] = 1
    row["requested_max_item_mean"] = 1.0
    row["effective_max_item_mean"] = 1.0
    return row


def _selection_row(
    *,
    profile_id: str,
    family: str,
    seed: int,
    benchmark_id: str,
    selection_stage: str,
    selected_items: str,
) -> dict[str, object]:
    return {
        "run_signature": f"compact_broad::fixture::{profile_id}::deterministic_info::seed-{seed}",
        "search_stage": "compact_broad",
        "dataset_id": "fixture",
        "dataset_label": "fixture",
        "profile_id": profile_id,
        "family": family,
        "description": profile_id,
        "preselection_method": "deterministic_info",
        "seed": seed,
        "is_baseline": profile_id == "baseline",
        "benchmark_id": benchmark_id,
        "selection_stage": selection_stage,
        "selected_items": selected_items,
        "selected_item_count": 2,
    }
