#!/usr/bin/env python3
"""Run the multi-bundle generalization and deployment validation pass for BenchIQ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import benchiq
from benchiq.preprocess.optimization import (
    THRESHOLD_COLUMNS,
    PreprocessingExperimentRawResult,
    PreprocessingOptimizationDataset,
    PreprocessingOptimizationPlan,
    PreprocessingOptimizationProfile,
    combine_preprocessing_experiment_raw_results,
    execute_preprocessing_experiment_plans,
    resolve_experiment_config,
    resolve_experiment_stage_options,
    summarize_preprocessing_experiments,
)
from benchiq.reconstruct import run_reconstruction_head_experiments

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports" / "generalization_optimization"
PLOTS_DIR = REPORTS_DIR / "plots"
GENERATED_BUNDLES_DIR = REPORTS_DIR / "generated_bundles"
HEAD_CHECKS_DIR = REPORTS_DIR / "head_checks"
DEPLOYMENT_DIR = REPO_ROOT / "reports" / "deployment_validation"

COMPACT_DATASET_ID = "compact_validation_fixture"
REAL_DATASET_ID = "large_release_default_subset"
SYNTHETIC_DENSE_DATASET_ID = "synthetic_dense_overlap"
SYNTHETIC_SPARSE_DATASET_ID = "synthetic_sparse_overlap"

DEFAULT_SEEDS = (7, 11, 19)
CORE_STRATEGIES = (
    ("baseline_current", "random_cv"),
    ("baseline_current", "deterministic_info"),
    ("reconstruction_relaxed", "random_cv"),
    ("reconstruction_relaxed", "deterministic_info"),
    ("minimal_cleaning", "deterministic_info"),
)
COMPACT_STRATEGIES = (
    ("baseline_current", "random_cv"),
    ("baseline_current", "deterministic_info"),
    ("reconstruction_relaxed", "random_cv"),
    ("reconstruction_relaxed", "deterministic_info"),
)
NONCOMPACT_DATASET_IDS = (
    REAL_DATASET_ID,
    SYNTHETIC_DENSE_DATASET_ID,
    SYNTHETIC_SPARSE_DATASET_ID,
)
BASELINE_STRATEGY_ID = "baseline_current__random_cv"
WINNER_STRATEGY_ID = "reconstruction_relaxed__deterministic_info"
CHALLENGER_STRATEGY_ID = "minimal_cleaning__deterministic_info"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets()
    profiles = build_profiles()
    bundles = load_bundles(datasets)

    raw_batches = [
        load_or_execute_preprocessing_experiment_plans(
            bundles=bundles,
            datasets=datasets,
            profiles=profiles,
            plans=build_compact_plans(),
            out_dir=REPORTS_DIR / "workdir" / "compact_validation",
        ),
        load_or_execute_preprocessing_experiment_plans(
            bundles=bundles,
            datasets=datasets,
            profiles=profiles,
            plans=build_synthetic_plans(),
            out_dir=REPORTS_DIR / "workdir" / "synthetic_validation",
        ),
        load_or_execute_preprocessing_experiment_plans(
            bundles=bundles,
            datasets=datasets,
            profiles=profiles,
            plans=build_real_confirmation_plans(),
            out_dir=REPORTS_DIR / "workdir" / "real_confirmation",
        ),
    ]
    combined = combine_preprocessing_experiment_raw_results(raw_batches)
    result = summarize_preprocessing_experiments(combined, out_dir=REPORTS_DIR)
    dataset_metadata = _dataset_metadata_map(datasets, bundles=bundles)
    enriched_summary = _enrich_summary(
        result.summary,
        per_run_metrics=result.per_run_metrics,
        dataset_metadata=dataset_metadata,
    )
    result.summary = enriched_summary
    _rewrite_summary_artifacts(
        summary=enriched_summary,
        summary_csv=REPORTS_DIR / "summary.csv",
        summary_parquet=REPORTS_DIR / "summary.parquet",
    )

    head_summary = load_or_run_head_checks(
        datasets={dataset.dataset_id: dataset for dataset in datasets},
        bundles=bundles,
        profiles={profile.profile_id: profile for profile in profiles},
    )
    deployment_summary = load_or_run_deployment_validation(
        dataset=next(dataset for dataset in datasets if dataset.dataset_id == REAL_DATASET_ID),
        bundle=bundles[REAL_DATASET_ID],
        profile=next(
            profile for profile in profiles if profile.profile_id == "reconstruction_relaxed"
        ),
    )
    decision_payload = build_decision_payload(
        summary=enriched_summary,
        head_summary=head_summary,
        deployment_summary=deployment_summary,
        dataset_metadata=dataset_metadata,
    )

    write_additional_artifacts(
        summary=enriched_summary,
        result=result,
        head_summary=head_summary,
        deployment_summary=deployment_summary,
        decision_payload=decision_payload,
        dataset_metadata=dataset_metadata,
    )
    print(REPORTS_DIR / "summary.md")


def build_datasets() -> list[PreprocessingOptimizationDataset]:
    return [
        build_compact_dataset(),
        build_real_dataset(),
        *build_synthetic_datasets(),
    ]


def build_profiles() -> list[PreprocessingOptimizationProfile]:
    return [
        PreprocessingOptimizationProfile(
            profile_id="baseline_current",
            family="psychometric_default",
            description="current psychometric-style baseline",
            config_overrides={},
            is_baseline=True,
        ),
        PreprocessingOptimizationProfile(
            profile_id="reconstruction_relaxed",
            family="reconstruction_first_relaxed",
            description=(
                "relaxed reconstruction-first preprocessing with deterministic-info support"
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
            description="minimal cleaning challenger with softer item-count floor",
            config_overrides={
                "drop_low_tail_models_quantile": 0.0,
                "min_item_sd": 0.0,
                "max_item_mean": 1.0,
                "min_abs_point_biserial": 0.0,
                "min_models_per_item": 1,
                "min_item_coverage": 0.70,
            },
        ),
    ]


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
        notes=("fast continuity check against the earlier compact bundle",),
    )


def build_real_dataset() -> PreprocessingOptimizationDataset:
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
        dataset_id=REAL_DATASET_ID,
        label="large release-default real-data subset",
        source_path="out/release_bundle_source/release_default_subset_responses_long.parquet",
        base_config=profile.config.model_dump(mode="json"),
        base_stage_options=stage_options,
        notes=(
            "primary real-data confirmation bundle",
            "keeps full-profile budgets while comparing only a small decision-oriented matrix",
        ),
    )


def build_synthetic_datasets() -> list[PreprocessingOptimizationDataset]:
    dense_metadata = _generate_synthetic_dataset(
        dataset_id=SYNTHETIC_DENSE_DATASET_ID,
        label="synthetic dense-overlap bundle",
        benchmark_ids=[f"dense_b{index}" for index in range(1, 6)],
        benchmark_model_ids=_dense_model_map(),
        item_count=120,
        random_seed=314,
        base_missing_rate=0.02,
        hard_missing_rate=0.06,
        easy_item_fraction=0.18,
        noisy_item_fraction=0.08,
        overlap_family="dense",
    )
    sparse_metadata = _generate_synthetic_dataset(
        dataset_id=SYNTHETIC_SPARSE_DATASET_ID,
        label="synthetic sparse-overlap bundle",
        benchmark_ids=[f"sparse_b{index}" for index in range(1, 7)],
        benchmark_model_ids=_sparse_model_map(),
        item_count=105,
        random_seed=2718,
        base_missing_rate=0.08,
        hard_missing_rate=0.18,
        easy_item_fraction=0.16,
        noisy_item_fraction=0.10,
        overlap_family="sparse",
    )
    return [
        PreprocessingOptimizationDataset(
            dataset_id=SYNTHETIC_DENSE_DATASET_ID,
            label="synthetic dense-overlap bundle",
            source_path=dense_metadata["responses_relative_path"],
            base_config={
                "allow_low_n": True,
                "drop_low_tail_models_quantile": 0.001,
                "min_item_sd": 0.01,
                "max_item_mean": 0.95,
                "min_abs_point_biserial": 0.05,
                "min_models_per_benchmark": 180,
                "warn_models_per_benchmark": 220,
                "min_items_after_filtering": 60,
                "min_models_per_item": 60,
                "min_item_coverage": 0.80,
                "min_overlap_models_for_joint": 180,
                "min_overlap_models_for_redundancy": 180,
                "random_seed": 7,
            },
            base_stage_options=_synthetic_stage_options(k_preselect=70, k_final=45),
            notes=(
                "all benchmarks share most models",
                "designed to test whether relaxed preprocessing still wins when "
                "joint reconstruction is easy",
            ),
        ),
        PreprocessingOptimizationDataset(
            dataset_id=SYNTHETIC_SPARSE_DATASET_ID,
            label="synthetic sparse-overlap bundle",
            source_path=sparse_metadata["responses_relative_path"],
            base_config={
                "allow_low_n": True,
                "drop_low_tail_models_quantile": 0.001,
                "min_item_sd": 0.01,
                "max_item_mean": 0.95,
                "min_abs_point_biserial": 0.05,
                "min_models_per_benchmark": 120,
                "warn_models_per_benchmark": 160,
                "min_items_after_filtering": 45,
                "min_models_per_item": 40,
                "min_item_coverage": 0.80,
                "min_overlap_models_for_joint": 90,
                "min_overlap_models_for_redundancy": 90,
                "random_seed": 7,
            },
            base_stage_options=_synthetic_stage_options(k_preselect=55, k_final=35),
            notes=(
                "core overlap is intentionally limited and some responses are missing",
                "designed to stress coverage, refusal behavior, and "
                "marginal-only deployment readiness",
            ),
        ),
    ]


def _synthetic_stage_options(*, k_preselect: int, k_final: int) -> dict[str, dict[str, Any]]:
    return {
        "04_subsample": {
            "k_preselect": k_preselect,
            "n_iter": 12,
            "cv_folds": 5,
            "checkpoint_interval": 4,
            "lam_grid": [0.1, 1.0],
        },
        "06_select": {
            "k_final": k_final,
            "theta_grid_size": 151,
        },
        "07_theta": {
            "theta_grid_size": 151,
        },
        "09_reconstruct": {
            "lam_grid": [0.1, 1.0],
            "cv_folds": 5,
            "n_splines": 8,
        },
        "10_redundancy": {
            "lam_grid": [0.1, 1.0],
            "cv_folds": 5,
            "n_splines": 8,
            "n_factors_to_try": [1, 2, 3],
        },
    }


def _dense_model_map() -> dict[str, list[str]]:
    shared_models = [f"dense_m{index:04d}" for index in range(1, 321)]
    return {
        benchmark_id: list(shared_models)
        for benchmark_id in [f"dense_b{index}" for index in range(1, 6)]
    }


def _sparse_model_map() -> dict[str, list[str]]:
    benchmark_ids = [f"sparse_b{index}" for index in range(1, 7)]
    core_models = [f"sparse_core_{index:03d}" for index in range(1, 81)]
    benchmark_specific = {
        benchmark_id: [f"{benchmark_id}_solo_{index:03d}" for index in range(1, 71)]
        for benchmark_id in benchmark_ids
    }
    pair_models: dict[tuple[str, str], list[str]] = {}
    for left, right in zip(benchmark_ids[:-1], benchmark_ids[1:], strict=True):
        pair_models[(left, right)] = [f"{left}_{right}_pair_{index:03d}" for index in range(1, 31)]
    result: dict[str, list[str]] = {}
    for index, benchmark_id in enumerate(benchmark_ids):
        members = list(core_models)
        members.extend(benchmark_specific[benchmark_id])
        if index > 0:
            members.extend(pair_models[(benchmark_ids[index - 1], benchmark_id)])
        if index < len(benchmark_ids) - 1:
            members.extend(pair_models[(benchmark_id, benchmark_ids[index + 1])])
        result[benchmark_id] = members
    return result


def _generate_synthetic_dataset(
    *,
    dataset_id: str,
    label: str,
    benchmark_ids: list[str],
    benchmark_model_ids: dict[str, list[str]],
    item_count: int,
    random_seed: int,
    base_missing_rate: float,
    hard_missing_rate: float,
    easy_item_fraction: float,
    noisy_item_fraction: float,
    overlap_family: str,
) -> dict[str, Any]:
    dataset_dir = GENERATED_BUNDLES_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    all_model_ids = sorted({model_id for ids in benchmark_model_ids.values() for model_id in ids})
    global_theta = {model_id: float(rng.normal()) for model_id in all_model_ids}
    rows: list[dict[str, Any]] = []
    per_benchmark_rows: list[dict[str, Any]] = []

    easy_items = max(1, int(round(item_count * easy_item_fraction)))
    noisy_items = max(1, int(round(item_count * noisy_item_fraction)))
    informative_items = max(1, item_count - easy_items - noisy_items)

    for benchmark_index, benchmark_id in enumerate(benchmark_ids):
        model_ids = benchmark_model_ids[benchmark_id]
        benchmark_shift = rng.normal(0.0, 0.20)
        benchmark_noise = {model_id: float(rng.normal(0.0, 0.35)) for model_id in model_ids}
        benchmark_theta = {
            model_id: 0.78 * global_theta[model_id]
            + 0.22 * benchmark_noise[model_id]
            + benchmark_shift
            + float(rng.normal(0.0, 0.05))
            for model_id in model_ids
        }
        informative_difficulties = np.linspace(-1.5, 1.5, informative_items) + rng.normal(
            0.0,
            0.18,
            informative_items,
        )
        easy_difficulties = np.linspace(-2.3, -1.4, easy_items) + rng.normal(
            0.0,
            0.10,
            easy_items,
        )
        noisy_difficulties = rng.normal(0.0, 0.7, noisy_items)
        item_specs: list[tuple[str, float, float, float]] = []
        for item_index, difficulty in enumerate(informative_difficulties):
            item_specs.append(
                (
                    f"{benchmark_id}_item_{item_index + 1:03d}",
                    float(difficulty + benchmark_index * 0.05),
                    float(rng.uniform(0.9, 1.5)),
                    base_missing_rate,
                )
            )
        start = len(item_specs)
        for item_index, difficulty in enumerate(easy_difficulties, start=start):
            item_specs.append(
                (
                    f"{benchmark_id}_item_{item_index + 1:03d}",
                    float(difficulty + benchmark_index * 0.02),
                    float(rng.uniform(0.7, 1.2)),
                    base_missing_rate,
                )
            )
        start = len(item_specs)
        for item_index, difficulty in enumerate(noisy_difficulties, start=start):
            item_specs.append(
                (
                    f"{benchmark_id}_item_{item_index + 1:03d}",
                    float(difficulty),
                    float(rng.uniform(0.15, 0.45)),
                    hard_missing_rate,
                )
            )
        item_specs = item_specs[:item_count]

        for item_id, difficulty, discrimination, missing_rate in item_specs:
            theta_values = np.array(
                [benchmark_theta[model_id] for model_id in model_ids], dtype=float
            )
            logits = discrimination * (theta_values - difficulty)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            outcomes = rng.binomial(1, probabilities).astype(int)
            observed_mask = rng.random(len(model_ids)) >= missing_rate
            for model_id, score, observed in zip(
                model_ids,
                outcomes,
                observed_mask,
                strict=True,
            ):
                if not bool(observed):
                    continue
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "item_id": item_id,
                        "model_id": model_id,
                        "score": int(score),
                    }
                )
        per_benchmark_rows.append(
            {
                "benchmark_id": benchmark_id,
                "assigned_model_count": len(model_ids),
                "shared_overlap_count": _shared_overlap_count(
                    benchmark_model_ids=benchmark_model_ids,
                    benchmark_id=benchmark_id,
                ),
            }
        )

    responses = pd.DataFrame.from_records(rows).astype(
        {
            "benchmark_id": "string",
            "item_id": "string",
            "model_id": "string",
            "score": "Int64",
        }
    )
    responses_parquet = dataset_dir / "responses_long.parquet"
    responses_csv = dataset_dir / "responses_long.csv"
    responses.to_parquet(responses_parquet, index=False)
    responses.to_csv(responses_csv, index=False)

    overlap_matrix = _overlap_matrix_frame(benchmark_model_ids)
    overlap_matrix.to_csv(dataset_dir / "overlap_matrix.csv", index=False)
    metadata = {
        "dataset_id": dataset_id,
        "label": label,
        "overlap_family": overlap_family,
        "random_seed": random_seed,
        "benchmark_count": len(benchmark_ids),
        "model_count": len(all_model_ids),
        "row_count": int(len(responses.index)),
        "item_count_total": int(responses["item_id"].nunique()),
        "per_benchmark": per_benchmark_rows,
        "responses_relative_path": str(responses_parquet.relative_to(REPO_ROOT)),
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _shared_overlap_count(
    *,
    benchmark_model_ids: dict[str, list[str]],
    benchmark_id: str,
) -> int:
    candidates = set(benchmark_model_ids[benchmark_id])
    for other_benchmark_id, model_ids in benchmark_model_ids.items():
        if other_benchmark_id == benchmark_id:
            continue
        candidates &= set(model_ids)
    return len(candidates)


def _overlap_matrix_frame(benchmark_model_ids: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    benchmark_ids = sorted(benchmark_model_ids)
    for left in benchmark_ids:
        left_models = set(benchmark_model_ids[left])
        for right in benchmark_ids:
            rows.append(
                {
                    "left_benchmark_id": left,
                    "right_benchmark_id": right,
                    "shared_model_count": len(left_models & set(benchmark_model_ids[right])),
                }
            )
    return pd.DataFrame.from_records(rows)


def build_compact_plans() -> list[PreprocessingOptimizationPlan]:
    return _plans_for_strategy_pairs(
        search_stage="compact_validation",
        dataset_ids=[COMPACT_DATASET_ID],
        strategy_pairs=COMPACT_STRATEGIES,
    )


def build_synthetic_plans() -> list[PreprocessingOptimizationPlan]:
    return _plans_for_strategy_pairs(
        search_stage="synthetic_generalization",
        dataset_ids=[SYNTHETIC_DENSE_DATASET_ID, SYNTHETIC_SPARSE_DATASET_ID],
        strategy_pairs=CORE_STRATEGIES,
    )


def build_real_confirmation_plans() -> list[PreprocessingOptimizationPlan]:
    return _plans_for_strategy_pairs(
        search_stage="real_generalization",
        dataset_ids=[REAL_DATASET_ID],
        strategy_pairs=CORE_STRATEGIES,
    )


def _plans_for_strategy_pairs(
    *,
    search_stage: str,
    dataset_ids: Iterable[str],
    strategy_pairs: Iterable[tuple[str, str]],
) -> list[PreprocessingOptimizationPlan]:
    grouped: dict[str, list[str]] = {}
    for profile_id, method in strategy_pairs:
        grouped.setdefault(method, []).append(profile_id)
    plans: list[PreprocessingOptimizationPlan] = []
    for dataset_id in dataset_ids:
        for method, profile_ids in grouped.items():
            plans.append(
                PreprocessingOptimizationPlan(
                    search_stage=search_stage,
                    dataset_id=dataset_id,
                    profile_ids=tuple(dict.fromkeys(profile_ids)),
                    preselection_methods=(method,),
                    seeds=DEFAULT_SEEDS,
                )
            )
    return plans


def load_or_execute_preprocessing_experiment_plans(
    *,
    bundles: dict[str, benchiq.Bundle],
    datasets: list[PreprocessingOptimizationDataset],
    profiles: list[PreprocessingOptimizationProfile],
    plans: list[PreprocessingOptimizationPlan],
    out_dir: Path,
) -> PreprocessingExperimentRawResult:
    if _completed_run_dirs_exist(plans=plans, out_dir=out_dir):
        return collect_preprocessing_experiment_plans(
            datasets=datasets,
            profiles=profiles,
            plans=plans,
            out_dir=out_dir,
        )
    return execute_preprocessing_experiment_plans(
        bundles=bundles,
        datasets=datasets,
        profiles=profiles,
        plans=plans,
        out_dir=out_dir,
    )


def _completed_run_dirs_exist(
    *,
    plans: list[PreprocessingOptimizationPlan],
    out_dir: Path,
) -> bool:
    workdir = out_dir / "workdir"
    if not workdir.exists():
        return False
    for plan in plans:
        for profile_id in plan.profile_ids:
            for method in plan.preselection_methods:
                for seed in plan.seeds:
                    run_dir = workdir / _planned_run_id(
                        search_stage=plan.search_stage,
                        dataset_id=plan.dataset_id,
                        profile_id=profile_id,
                        preselection_method=method,
                        seed=seed,
                    )
                    if not _run_dir_complete(run_dir):
                        return False
    return True


def _run_dir_complete(run_dir: Path) -> bool:
    required_paths = (
        run_dir / "reports" / "metrics.json",
        run_dir / "config_resolved.json",
        run_dir / "artifacts" / "09_reconstruct" / "reconstruction_summary.parquet",
    )
    return all(path.exists() for path in required_paths)


def collect_preprocessing_experiment_plans(
    *,
    datasets: list[PreprocessingOptimizationDataset],
    profiles: list[PreprocessingOptimizationProfile],
    plans: list[PreprocessingOptimizationPlan],
    out_dir: Path,
) -> PreprocessingExperimentRawResult:
    dataset_map = {dataset.dataset_id: dataset for dataset in datasets}
    profile_map = {profile.profile_id: profile for profile in profiles}
    workdir = out_dir / "workdir"

    matrix_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for plan in plans:
        dataset = dataset_map[plan.dataset_id]
        for profile_id in plan.profile_ids:
            profile = profile_map[profile_id]
            for method in plan.preselection_methods:
                for seed in plan.seeds:
                    matrix_rows.append(
                        _experiment_matrix_row_from_plan(
                            dataset=dataset,
                            profile=profile,
                            plan=plan,
                            preselection_method=method,
                            seed=seed,
                        )
                    )
                    run_id = _planned_run_id(
                        search_stage=plan.search_stage,
                        dataset_id=dataset.dataset_id,
                        profile_id=profile.profile_id,
                        preselection_method=method,
                        seed=seed,
                    )
                    run_dir = workdir / run_id
                    payload = json.loads((run_dir / "reports" / "metrics.json").read_text())
                    config = benchiq.BenchIQConfig.model_validate(
                        json.loads((run_dir / "config_resolved.json").read_text())
                    )
                    run_rows.append(
                        _run_index_row_from_disk(
                            run_dir=run_dir,
                            payload=payload,
                            dataset=dataset,
                            profile=profile,
                            plan=plan,
                            preselection_method=method,
                            seed=seed,
                            config=config,
                        )
                    )
                    metric_rows.extend(
                        _benchmark_metric_rows_from_disk(
                            run_dir=run_dir,
                            payload=payload,
                            config=config,
                            dataset=dataset,
                            profile=profile,
                            plan=plan,
                            preselection_method=method,
                            seed=seed,
                        )
                    )
                    selection_rows.extend(
                        _selection_rows_from_disk(
                            run_dir=run_dir,
                            dataset=dataset,
                            profile=profile,
                            plan=plan,
                            preselection_method=method,
                            seed=seed,
                        )
                    )

    return PreprocessingExperimentRawResult(
        experiment_matrix=_records_frame(matrix_rows),
        run_index=_records_frame(run_rows),
        per_run_metrics=_records_frame(metric_rows),
        selection_sets=_records_frame(selection_rows),
    )


def _experiment_matrix_row_from_plan(
    *,
    dataset: PreprocessingOptimizationDataset,
    profile: PreprocessingOptimizationProfile,
    plan: PreprocessingOptimizationPlan,
    preselection_method: str,
    seed: int,
) -> dict[str, Any]:
    row = {
        "search_stage": plan.search_stage,
        "dataset_id": dataset.dataset_id,
        "dataset_label": dataset.label,
        "source_path": dataset.source_path,
        "profile_id": profile.profile_id,
        "family": profile.family,
        "description": profile.description,
        "preselection_method": preselection_method,
        "seed": int(seed),
        "is_baseline": bool(profile.is_baseline),
    }
    for threshold_name in THRESHOLD_COLUMNS:
        row[f"requested_{threshold_name}"] = profile.config_overrides.get(
            threshold_name,
            dataset.base_config.get(threshold_name),
        )
    return row


def _run_index_row_from_disk(
    *,
    run_dir: Path,
    payload: dict[str, Any],
    dataset: PreprocessingOptimizationDataset,
    profile: PreprocessingOptimizationProfile,
    plan: PreprocessingOptimizationPlan,
    preselection_method: str,
    seed: int,
    config,
) -> dict[str, Any]:
    stage_records = payload["stage_records"]
    return {
        "run_signature": _run_signature(
            search_stage=plan.search_stage,
            dataset_id=dataset.dataset_id,
            profile_id=profile.profile_id,
            preselection_method=preselection_method,
            seed=seed,
        ),
        "run_id": payload["run_id"],
        "search_stage": plan.search_stage,
        "dataset_id": dataset.dataset_id,
        "dataset_label": dataset.label,
        "profile_id": profile.profile_id,
        "family": profile.family,
        "description": profile.description,
        "preselection_method": preselection_method,
        "seed": int(seed),
        "is_baseline": bool(profile.is_baseline),
        "run_root": str(run_dir),
        "config_path": str(run_dir / "config_resolved.json"),
        "metrics_path": str(run_dir / "reports" / "metrics.json"),
        "run_runtime_seconds": float(
            sum(float(record["duration_seconds"]) for record in stage_records.values())
        ),
        "stage01_runtime_seconds": float(stage_records["01_preprocess"]["duration_seconds"]),
        "stage04_runtime_seconds": float(stage_records["04_subsample"]["duration_seconds"]),
        "stage09_runtime_seconds": float(stage_records["09_reconstruct"]["duration_seconds"]),
        "warning_count": int(payload["warning_count"]),
        **{threshold_name: getattr(config, threshold_name) for threshold_name in THRESHOLD_COLUMNS},
    }


def _benchmark_metric_rows_from_disk(
    *,
    run_dir: Path,
    payload: dict[str, Any],
    config,
    dataset: PreprocessingOptimizationDataset,
    profile: PreprocessingOptimizationProfile,
    plan: PreprocessingOptimizationPlan,
    preselection_method: str,
    seed: int,
) -> list[dict[str, Any]]:
    stage_records = payload["stage_records"]
    reconstruction_summary = pd.read_parquet(
        run_dir / "artifacts" / "09_reconstruct" / "reconstruction_summary.parquet"
    )
    preprocess_root = run_dir / "artifacts" / "01_preprocess" / "per_benchmark"
    preselect_root = run_dir / "artifacts" / "04_subsample" / "per_benchmark"
    select_root = run_dir / "artifacts" / "06_select" / "per_benchmark"
    rows: list[dict[str, Any]] = []

    for preprocess_dir in sorted(path for path in preprocess_root.iterdir() if path.is_dir()):
        benchmark_id = preprocess_dir.name
        preprocess_report = json.loads((preprocess_dir / "preprocess_report.json").read_text())
        selection_report_path = select_root / benchmark_id / "selection_report.json"
        selection_report = (
            json.loads(selection_report_path.read_text())
            if selection_report_path.exists()
            else None
        )
        preselect_path = preselect_root / benchmark_id / "preselect_items.parquet"
        preselect_count = (
            int(len(pd.read_parquet(preselect_path).index)) if preselect_path.exists() else 0
        )
        marginal_metrics = _split_metrics_from_summary(
            reconstruction_summary,
            benchmark_id=benchmark_id,
            model_type="marginal",
            split_name="test",
        )
        joint_metrics = _split_metrics_from_summary(
            reconstruction_summary,
            benchmark_id=benchmark_id,
            model_type="joint",
            split_name="test",
        )
        best_available = joint_metrics if joint_metrics["rmse"] is not None else marginal_metrics
        preprocess_counts = preprocess_report["counts"]
        preprocess_thresholds = preprocess_report["thresholds"]
        row = {
            "run_signature": _run_signature(
                search_stage=plan.search_stage,
                dataset_id=dataset.dataset_id,
                profile_id=profile.profile_id,
                preselection_method=preselection_method,
                seed=seed,
            ),
            "run_id": payload["run_id"],
            "search_stage": plan.search_stage,
            "dataset_id": dataset.dataset_id,
            "dataset_label": dataset.label,
            "profile_id": profile.profile_id,
            "family": profile.family,
            "description": profile.description,
            "preselection_method": preselection_method,
            "seed": int(seed),
            "is_baseline": bool(profile.is_baseline),
            "benchmark_id": benchmark_id,
            "refused": bool(preprocess_report["refused"]),
            "refusal_reasons": ";".join(preprocess_report["refusal_reasons"]) or None,
            "benchmark_warning_count": int(
                len(preprocess_report["warnings"])
                + (0 if selection_report is None else len(selection_report["warnings"]))
            ),
            "run_warning_count": int(payload["warning_count"]),
            "retained_models": int(preprocess_counts["retained_models"]),
            "retained_items": int(preprocess_counts["retained_items"]),
            "selected_items_final": (
                0
                if selection_report is None
                else int(selection_report["counts"]["selected_item_count"])
            ),
            "selected_items_preselect": preselect_count,
            "joint_available": bool(joint_metrics["rmse"] is not None),
            "best_available_model_type": best_available["model_type"],
            "best_available_test_rmse": best_available["rmse"],
            "best_available_test_mae": best_available["mae"],
            "best_available_test_pearson": best_available["pearson_r"],
            "best_available_test_spearman": best_available["spearman_r"],
            "marginal_test_rmse": marginal_metrics["rmse"],
            "marginal_test_mae": marginal_metrics["mae"],
            "marginal_test_pearson": marginal_metrics["pearson_r"],
            "marginal_test_spearman": marginal_metrics["spearman_r"],
            "joint_test_rmse": joint_metrics["rmse"],
            "joint_test_mae": joint_metrics["mae"],
            "joint_test_pearson": joint_metrics["pearson_r"],
            "joint_test_spearman": joint_metrics["spearman_r"],
            "run_runtime_seconds": float(
                sum(float(record["duration_seconds"]) for record in stage_records.values())
            ),
            "stage01_runtime_seconds": float(stage_records["01_preprocess"]["duration_seconds"]),
            "stage04_runtime_seconds": float(stage_records["04_subsample"]["duration_seconds"]),
            "stage09_runtime_seconds": float(stage_records["09_reconstruct"]["duration_seconds"]),
        }
        for threshold_name in THRESHOLD_COLUMNS:
            row[f"requested_{threshold_name}"] = getattr(config, threshold_name)
            row[f"effective_{threshold_name}"] = preprocess_thresholds.get(
                threshold_name,
                getattr(config, threshold_name),
            )
        rows.append(row)
    return rows


def _selection_rows_from_disk(
    *,
    run_dir: Path,
    dataset: PreprocessingOptimizationDataset,
    profile: PreprocessingOptimizationProfile,
    plan: PreprocessingOptimizationPlan,
    preselection_method: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_signature = _run_signature(
        search_stage=plan.search_stage,
        dataset_id=dataset.dataset_id,
        profile_id=profile.profile_id,
        preselection_method=preselection_method,
        seed=seed,
    )
    preselect_root = run_dir / "artifacts" / "04_subsample" / "per_benchmark"
    select_root = run_dir / "artifacts" / "06_select" / "per_benchmark"
    for benchmark_dir in sorted(path for path in preselect_root.iterdir() if path.is_dir()):
        benchmark_id = benchmark_dir.name
        preselect_path = benchmark_dir / "preselect_items.parquet"
        if preselect_path.exists():
            preselect_items = (
                pd.read_parquet(preselect_path)["item_id"]
                .dropna()
                .astype("string")
                .sort_values()
                .tolist()
            )
            rows.append(
                {
                    "run_signature": run_signature,
                    "search_stage": plan.search_stage,
                    "dataset_id": dataset.dataset_id,
                    "dataset_label": dataset.label,
                    "profile_id": profile.profile_id,
                    "family": profile.family,
                    "description": profile.description,
                    "preselection_method": preselection_method,
                    "seed": int(seed),
                    "is_baseline": bool(profile.is_baseline),
                    "benchmark_id": benchmark_id,
                    "selection_stage": "preselect",
                    "selected_items": json.dumps(preselect_items),
                    "selected_item_count": len(preselect_items),
                }
            )
        final_path = select_root / benchmark_id / "subset_final.parquet"
        if final_path.exists():
            final_items = (
                pd.read_parquet(final_path)["item_id"]
                .dropna()
                .astype("string")
                .sort_values()
                .tolist()
            )
            rows.append(
                {
                    "run_signature": run_signature,
                    "search_stage": plan.search_stage,
                    "dataset_id": dataset.dataset_id,
                    "dataset_label": dataset.label,
                    "profile_id": profile.profile_id,
                    "family": profile.family,
                    "description": profile.description,
                    "preselection_method": preselection_method,
                    "seed": int(seed),
                    "is_baseline": bool(profile.is_baseline),
                    "benchmark_id": benchmark_id,
                    "selection_stage": "final",
                    "selected_items": json.dumps(final_items),
                    "selected_item_count": len(final_items),
                }
            )
    return rows


def _split_metrics_from_summary(
    reconstruction_summary: pd.DataFrame,
    *,
    benchmark_id: str,
    model_type: str,
    split_name: str,
) -> dict[str, Any]:
    rows = reconstruction_summary.loc[
        (reconstruction_summary["benchmark_id"] == benchmark_id)
        & (reconstruction_summary["model_type"] == model_type)
        & (reconstruction_summary["split"] == split_name)
    ].copy()
    if rows.empty:
        return {
            "model_type": model_type,
            "rmse": None,
            "mae": None,
            "pearson_r": None,
            "spearman_r": None,
        }
    row = rows.iloc[0]
    return {
        "model_type": model_type,
        "rmse": _float_or_none(row["rmse"]),
        "mae": _float_or_none(row["mae"]),
        "pearson_r": _float_or_none(row["pearson_r"]),
        "spearman_r": _float_or_none(row["spearman_r"]),
    }


def _planned_run_id(
    *,
    search_stage: str,
    dataset_id: str,
    profile_id: str,
    preselection_method: str,
    seed: int,
) -> str:
    return f"{search_stage}__{dataset_id}__{profile_id}__{preselection_method}__seed-{seed}"


def _run_signature(
    *,
    search_stage: str,
    dataset_id: str,
    profile_id: str,
    preselection_method: str,
    seed: int,
) -> str:
    return f"{search_stage}::{dataset_id}::{profile_id}::{preselection_method}::seed-{seed}"


def _records_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows).convert_dtypes()


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


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


def _dataset_metadata_map(
    datasets: list[PreprocessingOptimizationDataset],
    *,
    bundles: dict[str, benchiq.Bundle],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        bundle = bundles[dataset.dataset_id]
        responses = bundle.responses_long
        metadata[dataset.dataset_id] = {
            "dataset_id": dataset.dataset_id,
            "dataset_label": dataset.label,
            "source_path": dataset.source_path,
            "benchmark_total": int(responses["benchmark_id"].nunique()),
            "model_total": int(responses["model_id"].nunique()),
            "item_total": int(responses["item_id"].nunique()),
            "row_total": int(len(responses.index)),
            "notes": list(dataset.notes),
        }
    return metadata


def _enrich_summary(
    summary: pd.DataFrame,
    *,
    per_run_metrics: pd.DataFrame,
    dataset_metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    frame = summary.copy()
    if frame.empty:
        return frame
    frame["strategy_id"] = (
        frame["profile_id"].astype("string") + "__" + frame["preselection_method"].astype("string")
    )
    frame["strategy_label"] = frame.apply(_strategy_label, axis=1)
    frame["benchmark_total"] = frame["dataset_id"].map(
        lambda dataset_id: dataset_metadata[str(dataset_id)]["benchmark_total"],
    )
    frame["benchmark_coverage_rate"] = frame["benchmark_count"].astype(float) / frame[
        "benchmark_total"
    ].astype(float)
    frame["refusal_rate"] = frame["refused_benchmark_count"].astype(float) / frame[
        "benchmark_total"
    ].astype(float)
    availability = (
        per_run_metrics.assign(
            best_available_metric_present=per_run_metrics["best_available_test_rmse"].notna()
        )
        .groupby(
            [
                "search_stage",
                "dataset_id",
                "dataset_label",
                "profile_id",
                "family",
                "description",
                "preselection_method",
                "is_baseline",
            ],
            dropna=False,
        )["best_available_metric_present"]
        .mean()
        .rename("best_available_benchmark_rate")
        .reset_index()
    )
    frame = frame.merge(
        availability,
        on=[
            "search_stage",
            "dataset_id",
            "dataset_label",
            "profile_id",
            "family",
            "description",
            "preselection_method",
            "is_baseline",
        ],
        how="left",
    )
    frame["method_rank_within_dataset_stage"] = (
        frame.groupby(["search_stage", "dataset_id"], dropna=False)["best_available_test_rmse_mean"]
        .rank(method="dense")
        .astype("Int64")
    )
    return frame.sort_values(
        ["search_stage", "dataset_id", "best_available_test_rmse_mean", "strategy_id"],
    ).reset_index(drop=True)


def _rewrite_summary_artifacts(
    *,
    summary: pd.DataFrame,
    summary_csv: Path,
    summary_parquet: Path,
) -> None:
    summary.to_csv(summary_csv, index=False)
    summary.to_parquet(summary_parquet, index=False)


def run_head_checks(
    *,
    datasets: dict[str, PreprocessingOptimizationDataset],
    bundles: dict[str, benchiq.Bundle],
    profiles: dict[str, PreprocessingOptimizationProfile],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cases = [
        ("real_winner", REAL_DATASET_ID),
        ("synthetic_dense_winner", SYNTHETIC_DENSE_DATASET_ID),
        ("synthetic_sparse_winner", SYNTHETIC_SPARSE_DATASET_ID),
    ]
    winner_profile = profiles["reconstruction_relaxed"]
    for case_id, dataset_id in cases:
        dataset = datasets[dataset_id]
        config = resolve_experiment_config(dataset=dataset, profile=winner_profile, seed=7)
        stage_options = resolve_experiment_stage_options(
            dataset=dataset,
            preselection_method="deterministic_info",
        )
        feature_run = benchiq.run(
            bundles[dataset_id],
            config=config,
            out_dir=HEAD_CHECKS_DIR / "workdir",
            run_id=f"{case_id}__feature_run",
            stage_options=stage_options,
            stop_after="08_features",
        )
        case_dir = HEAD_CHECKS_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        experiment_result = run_reconstruction_head_experiments(
            feature_run.stage_results["08_features"],
            methods=("gam", "elastic_net", "xgboost"),
            seeds=DEFAULT_SEEDS,
            lam_grid=tuple(stage_options["09_reconstruct"]["lam_grid"]),
            cv_folds=int(stage_options["09_reconstruct"]["cv_folds"]),
            n_splines=int(stage_options["09_reconstruct"]["n_splines"]),
            out_dir=case_dir,
        )
        summary = experiment_result.summary.copy()
        summary["case_id"] = case_id
        summary["dataset_id"] = dataset_id
        summary["dataset_label"] = dataset.label
        summary["profile_id"] = winner_profile.profile_id
        rows.extend(summary.to_dict(orient="records"))
    head_summary = pd.DataFrame.from_records(rows)
    if head_summary.empty:
        return head_summary
    head_summary = head_summary.astype(
        {
            "case_id": "string",
            "dataset_id": "string",
            "dataset_label": "string",
            "profile_id": "string",
            "model_type": "string",
            "method": "string",
        }
    )
    head_summary.to_csv(HEAD_CHECKS_DIR / "head_summary.csv", index=False)
    head_summary.to_parquet(HEAD_CHECKS_DIR / "head_summary.parquet", index=False)
    return head_summary


def load_or_run_head_checks(
    *,
    datasets: dict[str, PreprocessingOptimizationDataset],
    bundles: dict[str, benchiq.Bundle],
    profiles: dict[str, PreprocessingOptimizationProfile],
) -> pd.DataFrame:
    summary_path = HEAD_CHECKS_DIR / "head_summary.parquet"
    if summary_path.exists():
        return pd.read_parquet(summary_path)
    return run_head_checks(datasets=datasets, bundles=bundles, profiles=profiles)


def run_deployment_validation(
    *,
    dataset: PreprocessingOptimizationDataset,
    bundle,
    profile: PreprocessingOptimizationProfile,
) -> dict[str, Any]:
    deployment_workdir = DEPLOYMENT_DIR / "workdir"
    deployment_workdir.mkdir(parents=True, exist_ok=True)
    config = resolve_experiment_config(dataset=dataset, profile=profile, seed=7)
    stage_options = resolve_experiment_stage_options(
        dataset=dataset,
        preselection_method="deterministic_info",
    )
    calibration_result = benchiq.calibrate(
        bundle,
        config=config,
        out_dir=deployment_workdir,
        run_id="real_winner_calibration",
        stage_options=stage_options,
    )
    reduced_responses = _deployment_reduced_test_responses(calibration_result)
    reduced_path = DEPLOYMENT_DIR / "reduced_test_responses.parquet"
    reduced_responses.to_parquet(reduced_path, index=False)

    first_prediction = benchiq.predict(
        calibration_result.calibration_root,
        reduced_path,
        out_dir=deployment_workdir,
        run_id="real_winner_prediction_a",
    )
    second_prediction = benchiq.predict(
        calibration_result.calibration_root,
        reduced_path,
        out_dir=deployment_workdir,
        run_id="real_winner_prediction_b",
    )
    comparison = _deployment_comparison_frame(
        calibration_result, first_prediction, second_prediction
    )
    comparison.to_csv(DEPLOYMENT_DIR / "prediction_comparison.csv", index=False)
    comparison.to_parquet(DEPLOYMENT_DIR / "prediction_comparison.parquet", index=False)

    metrics = {
        "dataset_id": dataset.dataset_id,
        "profile_id": profile.profile_id,
        "preselection_method": "deterministic_info",
        "reduced_response_rows": int(len(reduced_responses.index)),
        "comparison_rows": int(len(comparison.index)),
        "prediction_available_rate": float(comparison["deployment_available"].mean()),
        "deployment_rmse": _rmse(
            comparison["actual_score"].astype(float).to_numpy(),
            comparison["deployment_predicted_score"].astype(float).to_numpy(),
        ),
        "deployment_mae": float(
            np.mean(
                np.abs(
                    comparison["deployment_predicted_score"].astype(float)
                    - comparison["actual_score"].astype(float)
                )
            )
        ),
        "deployment_pearson": _safe_correlation(
            comparison["actual_score"].astype(float).to_numpy(),
            comparison["deployment_predicted_score"].astype(float).to_numpy(),
            method="pearson",
        ),
        "deployment_spearman": _safe_correlation(
            comparison["actual_score"].astype(float).to_numpy(),
            comparison["deployment_predicted_score"].astype(float).to_numpy(),
            method="spearman",
        ),
        "max_abs_delta_vs_calibration": float(comparison["abs_delta_vs_calibration"].max()),
        "mean_abs_delta_vs_calibration": float(comparison["abs_delta_vs_calibration"].mean()),
        "max_abs_delta_repeat_prediction": float(comparison["abs_delta_repeat_prediction"].max()),
        "mean_abs_delta_repeat_prediction": float(comparison["abs_delta_repeat_prediction"].mean()),
        "prediction_warning_count": int(len(first_prediction.prediction_report["warnings"])),
        "selected_joint_rate": float(
            (
                comparison["deployment_model_type"]
                .astype("string")
                .fillna("missing")
                .eq("joint")
                .mean()
            )
        ),
        "calibration_bundle_root": str(calibration_result.calibration_root),
        "prediction_run_root": str(first_prediction.run_root),
        "repeat_prediction_run_root": str(second_prediction.run_root),
    }
    _plot_deployment_quality(
        comparison,
        out_path=DEPLOYMENT_DIR / "deployment_quality.png",
    )
    _plot_deployment_stability(
        comparison,
        out_path=DEPLOYMENT_DIR / "deployment_stability.png",
    )
    (DEPLOYMENT_DIR / "deployment_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([metrics]).to_csv(DEPLOYMENT_DIR / "deployment_metrics.csv", index=False)
    pd.DataFrame([metrics]).to_parquet(DEPLOYMENT_DIR / "deployment_metrics.parquet", index=False)
    return metrics


def load_or_run_deployment_validation(
    *,
    dataset: PreprocessingOptimizationDataset,
    bundle,
    profile: PreprocessingOptimizationProfile,
) -> dict[str, Any]:
    metrics_path = DEPLOYMENT_DIR / "deployment_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return run_deployment_validation(dataset=dataset, bundle=bundle, profile=profile)


def _deployment_reduced_test_responses(calibration_result) -> pd.DataFrame:
    bundle = calibration_result.run_result.stage_results["00_bundle"]
    split_result = calibration_result.run_result.stage_results["03_splits"]
    select_result = calibration_result.run_result.stage_results["06_select"]

    test_model_ids = {
        str(model_id)
        for split_frame in split_result.per_benchmark_splits.values()
        for model_id in split_frame.loc[
            split_frame["split"] == "test",
            "model_id",
        ]
        .astype("string")
        .tolist()
    }
    selected_by_benchmark = {
        benchmark_id: set(
            benchmark_result.subset_final["item_id"].dropna().astype("string").tolist()
        )
        for benchmark_id, benchmark_result in select_result.benchmarks.items()
    }
    reduced = bundle.responses_long.loc[
        bundle.responses_long["model_id"].astype("string").isin(sorted(test_model_ids))
    ].copy()
    keep_mask = reduced.apply(
        lambda row: row["item_id"] in selected_by_benchmark.get(str(row["benchmark_id"]), set()),
        axis=1,
    )
    return reduced.loc[keep_mask].reset_index(drop=True)


def _deployment_comparison_frame(
    calibration_result, first_prediction, second_prediction
) -> pd.DataFrame:
    reference_rows: list[dict[str, Any]] = []
    reconstruction_result = calibration_result.run_result.stage_results["09_reconstruct"]
    for benchmark_id, benchmark_result in sorted(reconstruction_result.benchmarks.items()):
        test_rows = benchmark_result.predictions.loc[
            benchmark_result.predictions["split"] == "test"
        ].copy()
        for model_id in test_rows["model_id"].dropna().astype("string").unique().tolist():
            model_rows = test_rows.loc[test_rows["model_id"] == model_id].copy()
            joint_rows = model_rows.loc[model_rows["model_type"] == "joint"].copy()
            chosen_rows = (
                joint_rows
                if not joint_rows.empty
                else model_rows.loc[model_rows["model_type"] == "marginal"].copy()
            )
            if chosen_rows.empty:
                continue
            row = chosen_rows.iloc[0]
            reference_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "model_id": model_id,
                    "reference_model_type": row["model_type"],
                    "actual_score": row["actual_score"],
                    "reference_predicted_score": row["predicted_score"],
                }
            )
    reference = pd.DataFrame.from_records(reference_rows).astype(
        {
            "benchmark_id": "string",
            "model_id": "string",
            "reference_model_type": "string",
            "actual_score": "Float64",
            "reference_predicted_score": "Float64",
        }
    )
    deployment = (
        first_prediction.predictions_best_available.loc[
            :,
            ["benchmark_id", "model_id", "selected_model_type", "predicted_score"],
        ]
        .rename(
            columns={
                "selected_model_type": "deployment_model_type",
                "predicted_score": "deployment_predicted_score",
            }
        )
        .copy()
    )
    deployment_repeat = (
        second_prediction.predictions_best_available.loc[
            :,
            ["benchmark_id", "model_id", "predicted_score"],
        ]
        .rename(columns={"predicted_score": "repeat_predicted_score"})
        .copy()
    )
    merged = reference.merge(deployment, on=["benchmark_id", "model_id"], how="left")
    merged = merged.merge(deployment_repeat, on=["benchmark_id", "model_id"], how="left")
    merged["deployment_available"] = merged["deployment_predicted_score"].notna()
    merged["delta_vs_calibration"] = merged["deployment_predicted_score"].astype(float) - merged[
        "reference_predicted_score"
    ].astype(float)
    merged["abs_delta_vs_calibration"] = merged["delta_vs_calibration"].abs()
    merged["delta_repeat_prediction"] = merged["repeat_predicted_score"].astype(float) - merged[
        "deployment_predicted_score"
    ].astype(float)
    merged["abs_delta_repeat_prediction"] = merged["delta_repeat_prediction"].abs()
    return merged.astype(
        {
            "benchmark_id": "string",
            "model_id": "string",
            "reference_model_type": "string",
            "actual_score": "Float64",
            "reference_predicted_score": "Float64",
            "deployment_model_type": "string",
            "deployment_predicted_score": "Float64",
            "repeat_predicted_score": "Float64",
            "deployment_available": "boolean",
            "delta_vs_calibration": "Float64",
            "abs_delta_vs_calibration": "Float64",
            "delta_repeat_prediction": "Float64",
            "abs_delta_repeat_prediction": "Float64",
        }
    )


def build_decision_payload(
    *,
    summary: pd.DataFrame,
    head_summary: pd.DataFrame,
    deployment_summary: dict[str, Any],
    dataset_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    core = summary.loc[
        summary["strategy_id"].isin(
            [BASELINE_STRATEGY_ID, WINNER_STRATEGY_ID, CHALLENGER_STRATEGY_ID]
        )
    ].copy()
    comparisons: list[dict[str, Any]] = []
    for dataset_id in NONCOMPACT_DATASET_IDS:
        dataset_rows = core.loc[core["dataset_id"] == dataset_id].copy()
        baseline_row = _row_for_strategy(dataset_rows, BASELINE_STRATEGY_ID)
        winner_row = _row_for_strategy(dataset_rows, WINNER_STRATEGY_ID)
        challenger_row = _row_for_strategy(dataset_rows, CHALLENGER_STRATEGY_ID)
        if baseline_row is None or winner_row is None:
            continue
        baseline_rmse = _float_or_none(baseline_row["best_available_test_rmse_mean"])
        winner_rmse = _float_or_none(winner_row["best_available_test_rmse_mean"])
        challenger_rmse = (
            None
            if challenger_row is None
            else _float_or_none(challenger_row["best_available_test_rmse_mean"])
        )
        informative_for_rmse = baseline_rmse is not None and winner_rmse is not None
        rmse_delta = None if not informative_for_rmse else float(baseline_rmse - winner_rmse)
        availability_delta = float(winner_row["best_available_benchmark_rate"]) - float(
            baseline_row["best_available_benchmark_rate"]
        )
        joint_delta = float(winner_row["joint_available_rate"]) - float(
            baseline_row["joint_available_rate"]
        )
        refusal_delta = float(baseline_row["refusal_rate"]) - float(winner_row["refusal_rate"])
        comparisons.append(
            {
                "dataset_id": dataset_id,
                "dataset_label": dataset_metadata[dataset_id]["dataset_label"],
                "informative_for_rmse": informative_for_rmse,
                "baseline_strategy_id": BASELINE_STRATEGY_ID,
                "winner_strategy_id": WINNER_STRATEGY_ID,
                "challenger_strategy_id": CHALLENGER_STRATEGY_ID,
                "baseline_rmse": baseline_rmse,
                "winner_rmse": winner_rmse,
                "challenger_rmse": challenger_rmse,
                "rmse_delta_vs_baseline": rmse_delta,
                "baseline_best_available_benchmark_rate": float(
                    baseline_row["best_available_benchmark_rate"]
                ),
                "winner_best_available_benchmark_rate": float(
                    winner_row["best_available_benchmark_rate"]
                ),
                "challenger_best_available_benchmark_rate": (
                    None
                    if challenger_row is None
                    else float(challenger_row["best_available_benchmark_rate"])
                ),
                "availability_delta_vs_baseline": availability_delta,
                "joint_delta_vs_baseline": joint_delta,
                "refusal_delta_vs_baseline": refusal_delta,
                "note": (
                    None
                    if informative_for_rmse
                    else (
                        "bundle was non-informative for held-out rmse because every strategy "
                        "produced zero best-available test rows; keep it as a sparse "
                        "coverage/stability stress case only"
                    )
                ),
            }
        )

    real_comparison = next(
        comparison for comparison in comparisons if comparison["dataset_id"] == REAL_DATASET_ID
    )
    informative_comparisons = [
        comparison for comparison in comparisons if comparison["informative_for_rmse"]
    ]
    winner_beats_baseline_on = sum(
        1
        for comparison in informative_comparisons
        if comparison["rmse_delta_vs_baseline"] is not None
        and comparison["rmse_delta_vs_baseline"] > 0.0
    )
    winner_tied_or_better_on = sum(
        1
        for comparison in informative_comparisons
        if comparison["rmse_delta_vs_baseline"] is not None
        and comparison["rmse_delta_vs_baseline"] >= 0.0
    )
    deployment_clean = (
        float(deployment_summary["max_abs_delta_vs_calibration"]) <= 1e-12
        and float(deployment_summary["max_abs_delta_repeat_prediction"]) <= 1e-12
    )

    decision = "D"
    reason = "winner did not clear the promotion bar"
    if (
        real_comparison["rmse_delta_vs_baseline"] is not None
        and real_comparison["rmse_delta_vs_baseline"] > 0.0
        and len(informative_comparisons) >= 2
        and winner_tied_or_better_on == len(informative_comparisons)
        and deployment_clean
        and real_comparison["availability_delta_vs_baseline"] >= 0.0
    ):
        decision = "B"
        reason = (
            "winner beat the default on every informative non-compact bundle, improved "
            "best-available benchmark coverage on the real bundle, and kept deployment "
            "deterministic; the sparse stress bundle was non-informative for rmse rather than "
            "negative"
        )
        if (
            len(informative_comparisons) == len(NONCOMPACT_DATASET_IDS)
            and winner_beats_baseline_on == len(informative_comparisons)
            and all(
                comparison["availability_delta_vs_baseline"] >= 0.0
                for comparison in informative_comparisons
            )
            and all(comparison["joint_delta_vs_baseline"] >= 0.0 for comparison in comparisons)
            and real_comparison["rmse_delta_vs_baseline"] >= 0.05
        ):
            decision = "C"
            reason = (
                "winner beat the default on every non-compact bundle with no coverage or "
                "deployment regressions"
            )

    recommended_profile = benchiq.build_reconstruction_first_profile(random_seed=7)
    lightgbm_note = {
        "tested": False,
        "reason": (
            "not run in this pass because the repo already carries XGBoost as the tree comparator, "
            "the winner-feature tables are low-dimensional and smooth, and adding LightGBM would "
            "widen the dependency surface without a strong expected payoff"
        ),
    }
    tabpfn_note = {
        "tested": False,
        "reason": (
            "not run in this pass because it would add a much heavier experimental "
            "dependency and a less deployment-friendly model family without strong "
            "evidence that it could beat GAM on this feature shape"
        ),
    }
    return {
        "decision": decision,
        "reason": reason,
        "winner_strategy_id": WINNER_STRATEGY_ID,
        "baseline_strategy_id": BASELINE_STRATEGY_ID,
        "challenger_strategy_id": CHALLENGER_STRATEGY_ID,
        "comparisons": comparisons,
        "recommended_product_profile": {
            "profile_id": recommended_profile.profile_id,
            "description": recommended_profile.description,
            "config": recommended_profile.config.model_dump(mode="json"),
            "stage_options": recommended_profile.stage_options_copy(),
            "notes": list(recommended_profile.notes),
        },
        "lightgbm": lightgbm_note,
        "tabpfn": tabpfn_note,
        "deployment_validation": deployment_summary,
        "head_summary_rows": ([] if head_summary.empty else head_summary.to_dict(orient="records")),
    }


def write_additional_artifacts(
    *,
    summary: pd.DataFrame,
    result,
    head_summary: pd.DataFrame,
    deployment_summary: dict[str, Any],
    decision_payload: dict[str, Any],
    dataset_metadata: dict[str, dict[str, Any]],
) -> None:
    (REPORTS_DIR / "datasets.json").write_text(
        json.dumps(dataset_metadata, indent=2),
        encoding="utf-8",
    )
    (REPORTS_DIR / "best_profile.json").write_text(
        json.dumps(_json_safe(decision_payload), indent=2),
        encoding="utf-8",
    )
    if not head_summary.empty:
        head_summary.to_csv(REPORTS_DIR / "head_summary.csv", index=False)
        head_summary.to_parquet(REPORTS_DIR / "head_summary.parquet", index=False)

    _plot_baseline_vs_winner(summary, out_path=PLOTS_DIR / "baseline_vs_winner_rmse.png")
    _plot_seed_stability(summary, out_path=PLOTS_DIR / "seed_spread_stability.png")
    _plot_runtime_vs_rmse(summary, out_path=PLOTS_DIR / "runtime_vs_rmse.png")
    _plot_coverage_vs_joint(summary, out_path=PLOTS_DIR / "coverage_joint_availability.png")
    _plot_core_strategy_comparison(
        summary, out_path=PLOTS_DIR / "default_vs_winner_vs_challenger.png"
    )
    if not head_summary.empty:
        _plot_head_comparison(head_summary, out_path=PLOTS_DIR / "head_comparison.png")

    summary_md = _summary_markdown(
        summary=summary,
        result=result,
        head_summary=head_summary,
        deployment_summary=deployment_summary,
        decision_payload=decision_payload,
        dataset_metadata=dataset_metadata,
    )
    (REPORTS_DIR / "summary.md").write_text(summary_md, encoding="utf-8")
    (DEPLOYMENT_DIR / "summary.md").write_text(
        _deployment_markdown(deployment_summary),
        encoding="utf-8",
    )


def _summary_markdown(
    *,
    summary: pd.DataFrame,
    result,
    head_summary: pd.DataFrame,
    deployment_summary: dict[str, Any],
    decision_payload: dict[str, Any],
    dataset_metadata: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# generalization optimization",
        "",
        "## decision",
        "",
        f"- explicit decision: `{decision_payload['decision']}`",
        f"- reason: {decision_payload['reason']}",
        f"- baseline strategy: `{decision_payload['baseline_strategy_id']}`",
        f"- generalized winner: `{decision_payload['winner_strategy_id']}`",
        f"- challenger tracked in this pass: `{decision_payload['challenger_strategy_id']}`",
        "",
        "## bundle set",
        "",
    ]
    for dataset_id in [
        COMPACT_DATASET_ID,
        REAL_DATASET_ID,
        SYNTHETIC_DENSE_DATASET_ID,
        SYNTHETIC_SPARSE_DATASET_ID,
    ]:
        dataset = dataset_metadata[dataset_id]
        lines.append(
            "- "
            f"`{dataset_id}`: rows={dataset['row_total']}, models={dataset['model_total']}, "
            f"benchmarks={dataset['benchmark_total']}, items={dataset['item_total']}"
        )
    lines.extend(
        [
            "",
            "## experiment matrix",
            "",
            f"- compact strategies: {list(COMPACT_STRATEGIES)}",
            f"- non-compact strategies: {list(CORE_STRATEGIES)}",
            f"- seeds: {list(DEFAULT_SEEDS)}",
            f"- planned runs: {int(len(result.experiment_matrix.index))}",
            f"- completed run rows: {int(len(result.run_index.index))}",
            "",
            "## core summary rows",
            "",
            _markdown_table(
                summary.loc[
                    summary["strategy_id"].isin(
                        [BASELINE_STRATEGY_ID, WINNER_STRATEGY_ID, CHALLENGER_STRATEGY_ID]
                    )
                ].loc[
                    :,
                    [
                        "search_stage",
                        "dataset_id",
                        "strategy_id",
                        "best_available_test_rmse_mean",
                        "best_available_test_mae_mean",
                        "best_available_test_pearson_mean",
                        "best_available_test_spearman_mean",
                        "seed_rmse_std",
                        "final_selection_stability_mean",
                        "run_runtime_mean_seconds",
                        "best_available_benchmark_rate",
                        "benchmark_coverage_rate",
                        "joint_available_rate",
                        "refusal_rate",
                    ],
                ],
            ),
            "",
            "## head checks",
            "",
        ]
    )
    if head_summary.empty:
        lines.append("- none")
    else:
        display_head_summary = head_summary.loc[
            head_summary["rmse_mean"].notna() & (head_summary["runs"].fillna(0).astype(int) > 0)
        ].copy()
        lines.append(
            _markdown_table(
                display_head_summary.loc[
                    :,
                    [
                        "case_id",
                        "model_type",
                        "method",
                        "rmse_mean",
                        "mae_mean",
                        "runtime_mean_seconds",
                        "seed_rmse_std",
                    ],
                ],
            )
        )
    lines.extend(
        [
            "",
            "## deployment validation",
            "",
            f"- deployment rmse: {deployment_summary['deployment_rmse']:.4f}",
            f"- deployment mae: {deployment_summary['deployment_mae']:.4f}",
            f"- deployment pearson: {_fmt_float(deployment_summary['deployment_pearson'])}",
            f"- deployment spearman: {_fmt_float(deployment_summary['deployment_spearman'])}",
            f"- prediction available rate: {deployment_summary['prediction_available_rate']:.4f}",
            (
                "- max abs delta vs calibration: "
                f"{deployment_summary['max_abs_delta_vs_calibration']:.6f}"
            ),
            (
                "- max abs delta repeat prediction: "
                f"{deployment_summary['max_abs_delta_repeat_prediction']:.6f}"
            ),
            "",
            "## method notes",
            "",
            (
                "- LightGBM: not run in this pass because XGBoost already covered "
                "the tree-comparator slot and the winner-feature tables are "
                "low-dimensional"
            ),
            (
                "- TabPFN: not run in this pass because it would add a heavier "
                "experimental dependency and a less deployment-friendly model "
                "family without strong expected payoff"
            ),
            "",
            "## artifact index",
            "",
            "- `summary.csv` and `summary.parquet`: aggregated strategy-by-bundle metrics",
            "- `per_run_metrics.parquet`: per-run, per-benchmark metrics",
            "- `best_profile.json`: explicit promotion decision and rationale",
            "- `plots/`: appendix-ready figures",
            (
                "- `head_summary.csv`: targeted GAM / Elastic Net / XGBoost "
                "comparison on the fixed winner profile"
            ),
            "- `reports/deployment_validation/`: deployment metrics, comparison tables, and plots",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _deployment_markdown(deployment_summary: dict[str, Any]) -> str:
    return (
        "# deployment validation\n\n"
        f"- dataset_id: `{deployment_summary['dataset_id']}`\n"
        f"- profile_id: `{deployment_summary['profile_id']}`\n"
        f"- preselection_method: `{deployment_summary['preselection_method']}`\n"
        f"- deployment_rmse: `{deployment_summary['deployment_rmse']:.6f}`\n"
        f"- deployment_mae: `{deployment_summary['deployment_mae']:.6f}`\n"
        f"- deployment_pearson: `{_fmt_float(deployment_summary['deployment_pearson'])}`\n"
        f"- deployment_spearman: `{_fmt_float(deployment_summary['deployment_spearman'])}`\n"
        f"- prediction_available_rate: `{deployment_summary['prediction_available_rate']:.6f}`\n"
        "- max_abs_delta_vs_calibration: "
        f"`{deployment_summary['max_abs_delta_vs_calibration']:.6f}`\n"
        "- max_abs_delta_repeat_prediction: "
        f"`{deployment_summary['max_abs_delta_repeat_prediction']:.6f}`\n"
        f"- calibration_bundle_root: `{deployment_summary['calibration_bundle_root']}`\n"
        f"- prediction_run_root: `{deployment_summary['prediction_run_root']}`\n"
    )


def _plot_baseline_vs_winner(summary: pd.DataFrame, *, out_path: Path) -> None:
    plot_rows = summary.loc[
        summary["strategy_id"].isin([BASELINE_STRATEGY_ID, WINNER_STRATEGY_ID])
        & summary["dataset_id"].isin(NONCOMPACT_DATASET_IDS)
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dataset_order = list(NONCOMPACT_DATASET_IDS)
    x = np.arange(len(dataset_order), dtype=float)
    width = 0.36
    for offset, strategy_id in [
        (-width / 2.0, BASELINE_STRATEGY_ID),
        (width / 2.0, WINNER_STRATEGY_ID),
    ]:
        values = []
        for dataset_id in dataset_order:
            row = _row_for_strategy(
                plot_rows.loc[plot_rows["dataset_id"] == dataset_id], strategy_id
            )
            values.append(np.nan if row is None else float(row["best_available_test_rmse_mean"]))
        ax.bar(x + offset, values, width=width, label=strategy_id)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_order, rotation=15)
    ax.set_ylabel("held-out rmse")
    ax.set_title("baseline vs winner rmse across non-compact bundles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_seed_stability(summary: pd.DataFrame, *, out_path: Path) -> None:
    plot_rows = summary.loc[
        summary["strategy_id"].isin(
            [BASELINE_STRATEGY_ID, WINNER_STRATEGY_ID, CHALLENGER_STRATEGY_ID]
        )
        & summary["dataset_id"].isin(NONCOMPACT_DATASET_IDS)
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy_id, marker in [
        (BASELINE_STRATEGY_ID, "o"),
        (WINNER_STRATEGY_ID, "s"),
        (CHALLENGER_STRATEGY_ID, "^"),
    ]:
        strategy_rows = plot_rows.loc[plot_rows["strategy_id"] == strategy_id].copy()
        if strategy_rows.empty:
            continue
        ax.scatter(
            strategy_rows["seed_rmse_std"].astype(float),
            strategy_rows["final_selection_stability_mean"].astype(float),
            label=strategy_id,
            marker=marker,
            s=70,
        )
        for _, row in strategy_rows.iterrows():
            ax.annotate(
                str(row["dataset_id"]),
                (float(row["seed_rmse_std"]), float(row["final_selection_stability_mean"])),
            )
    ax.set_xlabel("seed rmse spread")
    ax.set_ylabel("final selection stability")
    ax.set_title("seed spread vs selection stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_runtime_vs_rmse(summary: pd.DataFrame, *, out_path: Path) -> None:
    plot_rows = summary.loc[
        summary["strategy_id"].isin(
            [BASELINE_STRATEGY_ID, WINNER_STRATEGY_ID, CHALLENGER_STRATEGY_ID]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy_id, marker in [
        (BASELINE_STRATEGY_ID, "o"),
        (WINNER_STRATEGY_ID, "s"),
        (CHALLENGER_STRATEGY_ID, "^"),
    ]:
        strategy_rows = plot_rows.loc[plot_rows["strategy_id"] == strategy_id].copy()
        if strategy_rows.empty:
            continue
        ax.scatter(
            strategy_rows["run_runtime_mean_seconds"].astype(float),
            strategy_rows["best_available_test_rmse_mean"].astype(float),
            label=strategy_id,
            marker=marker,
            s=70,
        )
    ax.set_xlabel("runtime mean seconds")
    ax.set_ylabel("held-out rmse")
    ax.set_title("runtime vs rmse")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_coverage_vs_joint(summary: pd.DataFrame, *, out_path: Path) -> None:
    plot_rows = summary.loc[
        summary["strategy_id"].isin(
            [BASELINE_STRATEGY_ID, WINNER_STRATEGY_ID, CHALLENGER_STRATEGY_ID]
        )
        & summary["dataset_id"].isin(NONCOMPACT_DATASET_IDS)
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy_id, marker in [
        (BASELINE_STRATEGY_ID, "o"),
        (WINNER_STRATEGY_ID, "s"),
        (CHALLENGER_STRATEGY_ID, "^"),
    ]:
        strategy_rows = plot_rows.loc[plot_rows["strategy_id"] == strategy_id].copy()
        if strategy_rows.empty:
            continue
        ax.scatter(
            strategy_rows["benchmark_coverage_rate"].astype(float),
            strategy_rows["joint_available_rate"].astype(float),
            label=strategy_id,
            marker=marker,
            s=70,
        )
        for _, row in strategy_rows.iterrows():
            ax.annotate(
                str(row["dataset_id"]),
                (float(row["benchmark_coverage_rate"]), float(row["joint_available_rate"])),
            )
    ax.set_xlabel("benchmark coverage rate")
    ax.set_ylabel("joint availability rate")
    ax.set_title("coverage vs joint availability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_core_strategy_comparison(summary: pd.DataFrame, *, out_path: Path) -> None:
    plot_rows = summary.loc[
        summary["strategy_id"].isin(
            [BASELINE_STRATEGY_ID, WINNER_STRATEGY_ID, CHALLENGER_STRATEGY_ID]
        )
        & summary["dataset_id"].isin(NONCOMPACT_DATASET_IDS)
    ].copy()
    fig, axes = plt.subplots(1, len(NONCOMPACT_DATASET_IDS), figsize=(14, 4), sharey=True)
    axes_list = list(np.atleast_1d(axes))
    for axis, dataset_id in zip(axes_list, NONCOMPACT_DATASET_IDS, strict=True):
        dataset_rows = plot_rows.loc[plot_rows["dataset_id"] == dataset_id].copy()
        strategy_order = [BASELINE_STRATEGY_ID, WINNER_STRATEGY_ID, CHALLENGER_STRATEGY_ID]
        values = []
        labels = []
        for strategy_id in strategy_order:
            row = _row_for_strategy(dataset_rows, strategy_id)
            if row is None:
                continue
            values.append(float(row["best_available_test_rmse_mean"]))
            labels.append(strategy_id)
        axis.bar(
            np.arange(len(values)), values, color=["#6c7a89", "#2a9d8f", "#e9c46a"][: len(values)]
        )
        axis.set_xticks(np.arange(len(values)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_title(dataset_id)
    axes_list[0].set_ylabel("held-out rmse")
    fig.suptitle("default vs winner vs challenger")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_head_comparison(head_summary: pd.DataFrame, *, out_path: Path) -> None:
    plot_rows = head_summary.loc[head_summary["model_type"] == "joint"].copy()
    if plot_rows.empty:
        plot_rows = head_summary.copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = ["gam", "elastic_net", "xgboost"]
    case_ids = plot_rows["case_id"].dropna().astype("string").unique().tolist()
    x = np.arange(len(methods), dtype=float)
    width = 0.24
    for index, case_id in enumerate(case_ids):
        case_rows = plot_rows.loc[plot_rows["case_id"] == case_id].set_index("method")
        values = [
            np.nan if method not in case_rows.index else float(case_rows.loc[method, "rmse_mean"])
            for method in methods
        ]
        ax.bar(x + (index - (len(case_ids) - 1) / 2.0) * width, values, width=width, label=case_id)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("held-out rmse")
    ax.set_title("fixed-profile head comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_deployment_quality(comparison: pd.DataFrame, *, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        comparison["actual_score"].astype(float),
        comparison["deployment_predicted_score"].astype(float),
        alpha=0.35,
    )
    min_value = min(
        comparison["actual_score"].astype(float).min(),
        comparison["deployment_predicted_score"].astype(float).min(),
    )
    max_value = max(
        comparison["actual_score"].astype(float).max(),
        comparison["deployment_predicted_score"].astype(float).max(),
    )
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    ax.set_xlabel("actual score")
    ax.set_ylabel("deployment predicted score")
    ax.set_title("deployment quality on held-out reduced responses")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_deployment_stability(comparison: pd.DataFrame, *, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        comparison["reference_predicted_score"].astype(float),
        comparison["deployment_predicted_score"].astype(float),
        alpha=0.35,
    )
    min_value = min(
        comparison["reference_predicted_score"].astype(float).min(),
        comparison["deployment_predicted_score"].astype(float).min(),
    )
    max_value = max(
        comparison["reference_predicted_score"].astype(float).max(),
        comparison["deployment_predicted_score"].astype(float).max(),
    )
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    ax.set_xlabel("calibration-stage prediction")
    ax.set_ylabel("deployment prediction")
    ax.set_title("deployment stability vs calibration predictions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _strategy_label(row: pd.Series) -> str:
    profile_id = str(row["profile_id"])
    method = str(row["preselection_method"])
    if profile_id == "baseline_current" and method == "random_cv":
        return "default"
    if profile_id == "reconstruction_relaxed" and method == "deterministic_info":
        return "winner"
    if profile_id == "minimal_cleaning" and method == "deterministic_info":
        return "challenger"
    return f"{profile_id} + {method}"


def _row_for_strategy(frame: pd.DataFrame, strategy_id: str) -> pd.Series | None:
    rows = frame.loc[frame["strategy_id"] == strategy_id].copy()
    if rows.empty:
        return None
    return rows.iloc[0]


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_no rows_"
    display = frame.copy()
    columns = display.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, divider]
    for _, row in display.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(_fmt_float(value))
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def _safe_correlation(actual: np.ndarray, predicted: np.ndarray, *, method: str) -> float | None:
    if len(actual) < 2:
        return None
    if np.allclose(actual, actual[0]) or np.allclose(predicted, predicted[0]):
        return None
    if method == "pearson":
        return float(pearsonr(actual, predicted).statistic)
    if method == "spearman":
        return float(spearmanr(actual, predicted).statistic)
    raise ValueError(f"unsupported correlation method: {method}")


def _fmt_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return None
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    main()
