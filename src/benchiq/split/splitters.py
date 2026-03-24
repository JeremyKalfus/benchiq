"""Model-level split logic and stage-03 artifact writing for BenchIQ."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from benchiq.io.load import Bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.logging import update_manifest
from benchiq.preprocess.scores import GRAND_MEAN_SCORE, SCORE_FULL, ScoreResult
from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID, SPLIT

GLOBAL_SPLIT = "global_split"
SPLIT_STRATUM = "split_stratum"
SPLIT_METHOD = "split_method"


@dataclass(slots=True)
class SplitResult:
    """Stage-03 split outputs."""

    splits_models: pd.DataFrame
    per_benchmark_splits: dict[str, pd.DataFrame]
    split_report: dict[str, Any]
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None

    @property
    def per_benchmark(self) -> dict[str, pd.DataFrame]:
        return self.per_benchmark_splits


def split_models(
    bundle: Bundle,
    score_result: ScoreResult,
    *,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> SplitResult:
    """Split models into global test and benchmark-local train/val partitions."""

    all_model_ids = (
        bundle.models[MODEL_ID]
        .dropna()
        .astype("string")
        .sort_values()
        .reset_index(drop=True)
        .tolist()
    )
    global_splits, global_report, global_test_ids = _build_global_splits(
        bundle,
        score_result=score_result,
        all_model_ids=all_model_ids,
    )
    benchmark_splits: dict[str, pd.DataFrame] = {}
    benchmark_reports: dict[str, dict[str, Any]] = {}
    warnings = list(global_report["warnings"])

    for benchmark_id in sorted(
        score_result.scores_full[BENCHMARK_ID].dropna().astype("string").unique()
    ):
        split_frame, benchmark_report = _build_benchmark_splits(
            bundle,
            score_result=score_result,
            benchmark_id=benchmark_id,
            global_test_ids=global_test_ids,
            global_test_enabled=global_report["enabled"],
        )
        benchmark_splits[benchmark_id] = split_frame
        benchmark_reports[benchmark_id] = benchmark_report
        warnings.extend(benchmark_report["warnings"])

    if not global_report["enabled"]:
        warnings.append("benchmark-local train/val only: global overlap is too small")

    split_report = {
        "warnings": warnings,
        "global_test": global_report,
        "benchmarks": benchmark_reports,
        "counts": {
            "benchmark_count": len(benchmark_splits),
            "global_test_enabled": global_report["enabled"],
            "global_test_model_count": len(global_test_ids),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    split_result = SplitResult(
        splits_models=global_splits,
        per_benchmark_splits=benchmark_splits,
        split_report=split_report,
    )
    run_root, manifest_path = _resolve_run_root(bundle, out_dir=out_dir, run_id=run_id)
    if run_root is not None:
        artifact_paths = _write_split_artifacts(split_result, run_root=run_root)
        split_result.artifact_paths = artifact_paths
        split_result.manifest_path = manifest_path
        if manifest_path is not None:
            update_manifest(
                manifest_path,
                {
                    "artifacts": {
                        "03_splits": {
                            "splits_models": str(artifact_paths["splits_models"]),
                            "split_report": str(artifact_paths["split_report"]),
                            "per_benchmark": {
                                benchmark_id: str(path)
                                for benchmark_id, path in sorted(
                                    artifact_paths["per_benchmark"].items(),
                                )
                            },
                        },
                    },
                },
            )

    return split_result


def _build_global_splits(
    bundle: Bundle,
    *,
    score_result: ScoreResult,
    all_model_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], set[str]]:
    score_lookup = score_result.scores_grand.set_index(MODEL_ID)[GRAND_MEAN_SCORE]
    eligible_model_ids = score_lookup.index.astype("string").tolist()
    fallback_reason = score_result.score_report["grand_scores"]["skip_reason"]
    warnings: list[str] = []
    test_model_ids: set[str] = set()
    split_method = "unavailable"
    split_strata = pd.Series(pd.NA, index=pd.Index(all_model_ids, dtype="string"), dtype="Int64")
    enabled = False
    stratification_used = False
    effective_bins = 0

    if eligible_model_ids:
        holdout_ids, split_info = _select_holdout_models(
            pd.Series(score_lookup.loc[eligible_model_ids].to_numpy(), index=eligible_model_ids),
            holdout_fraction=bundle.config.p_test,
            n_strata_bins=bundle.config.n_strata_bins,
            random_seed=bundle.config.random_seed,
        )
        test_model_ids = set(holdout_ids)
        split_method = split_info["method"]
        stratification_used = split_info["stratification_used"]
        effective_bins = split_info["effective_bins"]
        if split_info["warnings"]:
            warnings.extend(split_info["warnings"])
        if test_model_ids:
            enabled = True
            split_strata.loc[split_info["strata"].dropna().index.astype("string")] = split_info[
                "strata"
            ].dropna()
    else:
        warnings.append("global test split disabled: no grand scores available")

    split_rows: list[dict[str, Any]] = []
    for model_id in all_model_ids:
        if model_id in test_model_ids:
            global_split = "test"
        elif model_id in eligible_model_ids:
            global_split = "train_pool"
        else:
            global_split = "benchmark_local_only"
        split_rows.append(
            {
                MODEL_ID: model_id,
                GLOBAL_SPLIT: global_split,
                GRAND_MEAN_SCORE: score_lookup.get(model_id, pd.NA),
                SPLIT_STRATUM: split_strata.get(model_id, pd.NA),
                SPLIT_METHOD: split_method if enabled else "benchmark_local_only",
            },
        )

    split_frame = pd.DataFrame(split_rows)
    split_frame[MODEL_ID] = split_frame[MODEL_ID].astype("string")
    split_frame[GLOBAL_SPLIT] = split_frame[GLOBAL_SPLIT].astype("string")
    split_frame[GRAND_MEAN_SCORE] = pd.Series(split_frame[GRAND_MEAN_SCORE], dtype="Float64")
    split_frame[SPLIT_STRATUM] = pd.Series(split_frame[SPLIT_STRATUM], dtype="Int64")
    split_frame[SPLIT_METHOD] = split_frame[SPLIT_METHOD].astype("string")

    return (
        split_frame,
        {
            "enabled": enabled,
            "eligible_model_count": len(eligible_model_ids),
            "test_model_count": len(test_model_ids),
            "assigned_test_count": len(test_model_ids),
            "train_val_pool_count": len(eligible_model_ids) - len(test_model_ids),
            "assigned_train_pool_count": len(eligible_model_ids) - len(test_model_ids),
            "fallback_reason": None if enabled else fallback_reason or "no_grand_scores_available",
            "skip_reason": None if enabled else fallback_reason or "no_grand_scores_available",
            "split_method": split_method,
            "stratification_used": stratification_used,
            "effective_bins": effective_bins,
            "diagnostics": _score_diagnostics(
                split_frame,
                score_column=GRAND_MEAN_SCORE,
                split_column=GLOBAL_SPLIT,
            ),
            "warnings": warnings,
        },
        test_model_ids,
    )


def _build_benchmark_splits(
    bundle: Bundle,
    *,
    score_result: ScoreResult,
    benchmark_id: str,
    global_test_ids: set[str],
    global_test_enabled: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    benchmark_scores = score_result.scores_full.loc[
        score_result.scores_full[BENCHMARK_ID] == benchmark_id,
        [MODEL_ID, SCORE_FULL],
    ].copy()
    valid_scores = benchmark_scores.loc[benchmark_scores[SCORE_FULL].notna()].copy()
    valid_scores[MODEL_ID] = valid_scores[MODEL_ID].astype("string")
    valid_scores = valid_scores.sort_values(MODEL_ID).reset_index(drop=True)
    warnings: list[str] = []

    if valid_scores.empty:
        empty_frame = _empty_benchmark_split_frame(benchmark_id)
        return empty_frame, {
            "strategy": "benchmark_local_only",
            "valid_model_count": 0,
            "train_model_count": 0,
            "val_model_count": 0,
            "test_model_count": 0,
            "fallback_reason": "no_valid_scores",
            "val_split_method": "unavailable",
            "stratification_used": False,
            "effective_bins": 0,
            "warnings": [f"{benchmark_id}: no valid scores available for splitting"],
        }

    valid_model_ids = valid_scores[MODEL_ID].tolist()
    test_ids = sorted(set(valid_model_ids) & global_test_ids) if global_test_enabled else []
    pool_ids = [model_id for model_id in valid_model_ids if model_id not in test_ids]
    val_ids, split_info = _select_holdout_models(
        valid_scores.set_index(MODEL_ID).loc[pool_ids, SCORE_FULL],
        holdout_fraction=bundle.config.p_val,
        n_strata_bins=bundle.config.n_strata_bins,
        random_seed=bundle.config.random_seed + _benchmark_seed_offset(benchmark_id),
    )
    warnings.extend(split_info["warnings"])
    if not global_test_enabled:
        warnings.append(f"{benchmark_id}: global overlap too small; benchmark-local train/val only")

    split_lookup: dict[str, str] = {model_id: "test" for model_id in test_ids}
    split_lookup.update({model_id: "val" for model_id in val_ids})
    split_lookup.update(
        {model_id: "train" for model_id in pool_ids if model_id not in set(val_ids)},
    )
    strata_lookup = split_info["strata"].to_dict()
    split_frame = pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series([benchmark_id] * len(valid_model_ids), dtype="string"),
            MODEL_ID: pd.Series(valid_model_ids, dtype="string"),
            SCORE_FULL: valid_scores[SCORE_FULL].astype("Float64").to_numpy(),
            SPLIT: pd.Series(
                [split_lookup[model_id] for model_id in valid_model_ids], dtype="string"
            ),
            SPLIT_STRATUM: pd.Series(
                [strata_lookup.get(model_id, pd.NA) for model_id in valid_model_ids],
                dtype="Int64",
            ),
            SPLIT_METHOD: pd.Series(
                [split_info["method"]] * len(valid_model_ids),
                dtype="string",
            ),
        },
    )
    split_frame = split_frame.sort_values(MODEL_ID).reset_index(drop=True)

    return split_frame, {
        "strategy": "global_plus_local_val" if global_test_enabled else "benchmark_local_only",
        "valid_model_count": len(valid_model_ids),
        "train_model_count": int((split_frame[SPLIT] == "train").sum()),
        "val_model_count": int((split_frame[SPLIT] == "val").sum()),
        "test_model_count": int((split_frame[SPLIT] == "test").sum()),
        "fallback_reason": None if global_test_enabled else "benchmark_local_only",
        "val_split_method": split_info["method"],
        "stratification_used": split_info["stratification_used"],
        "effective_bins": split_info["effective_bins"],
        "diagnostics": _score_diagnostics(
            split_frame,
            score_column=SCORE_FULL,
            split_column=SPLIT,
        ),
        "warnings": warnings,
    }


def _select_holdout_models(
    scores: pd.Series,
    *,
    holdout_fraction: float,
    n_strata_bins: int,
    random_seed: int,
) -> tuple[list[str], dict[str, Any]]:
    score_series = scores.dropna().astype("Float64").sort_index()
    if len(score_series.index) < 2:
        return [], {
            "method": "all_train",
            "stratification_used": False,
            "effective_bins": 0,
            "strata": pd.Series(dtype="Int64"),
            "warnings": [
                "too few models for holdout split; assigned all remaining models to train"
            ],
        }

    strata, strat_info = _score_strata(
        score_series,
        holdout_fraction=holdout_fraction,
        n_strata_bins=n_strata_bins,
    )
    stratify_values = None if strata.empty else strata.to_numpy()
    method = "stratified" if stratify_values is not None else "random_fallback"
    warnings = list(strat_info["warnings"])
    if method == "random_fallback" and warnings:
        warnings = warnings

    _, holdout_ids = train_test_split(
        score_series.index.tolist(),
        test_size=holdout_fraction,
        random_state=random_seed,
        stratify=stratify_values,
    )
    return sorted(holdout_ids), {
        "method": method,
        "stratification_used": stratify_values is not None,
        "effective_bins": strat_info["effective_bins"],
        "strata": strata,
        "warnings": warnings,
    }


def _score_strata(
    scores: pd.Series,
    *,
    holdout_fraction: float,
    n_strata_bins: int,
) -> tuple[pd.Series, dict[str, Any]]:
    if scores.nunique(dropna=True) < 2:
        return pd.Series(dtype="Int64"), {
            "effective_bins": 0,
            "warnings": ["too few unique scores for stratification; used random fallback"],
        }

    holdout_size = max(1, ceil(len(scores.index) * holdout_fraction))
    remaining_size = len(scores.index) - holdout_size
    max_bins = min(
        n_strata_bins,
        int(scores.nunique(dropna=True)),
        len(scores.index) // 2,
        holdout_size,
        remaining_size,
    )
    if max_bins < 2:
        return pd.Series(dtype="Int64"), {
            "effective_bins": 0,
            "warnings": ["insufficient score bins for stratification; used random fallback"],
        }

    strata = pd.qcut(
        scores.rank(method="first"),
        q=max_bins,
        labels=False,
    ).astype("Int64")
    return strata, {"effective_bins": int(strata.nunique()), "warnings": []}


def _empty_benchmark_split_frame(benchmark_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            MODEL_ID: pd.Series(dtype="string"),
            SCORE_FULL: pd.Series(dtype="Float64"),
            SPLIT: pd.Series(dtype="string"),
            SPLIT_STRATUM: pd.Series(dtype="Int64"),
            SPLIT_METHOD: pd.Series(dtype="string"),
        },
    )


def _score_diagnostics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    split_column: str,
) -> dict[str, dict[str, float | int | None]]:
    diagnostics: dict[str, dict[str, float | int | None]] = {}
    for split_name, group in frame.groupby(split_column, dropna=True):
        scores = group[score_column].dropna()
        diagnostics[str(split_name)] = {
            "count": int(len(group.index)),
            "score_mean": float(scores.mean()) if not scores.empty else None,
            "score_min": float(scores.min()) if not scores.empty else None,
            "score_max": float(scores.max()) if not scores.empty else None,
        }
    return diagnostics


def _benchmark_seed_offset(benchmark_id: str) -> int:
    return sum(ord(character) for character in benchmark_id)


def _resolve_run_root(
    bundle: Bundle,
    *,
    out_dir: str | Path | None,
    run_id: str | None,
) -> tuple[Path | None, Path | None]:
    if out_dir is not None:
        resolved_run_id = run_id or bundle.run_id or _default_run_id()
        run_root = Path(out_dir) / resolved_run_id
        return run_root, run_root / "manifest.json"
    if bundle.manifest_path is not None:
        return bundle.manifest_path.parent, bundle.manifest_path
    return None, None


def _write_split_artifacts(
    result: SplitResult,
    *,
    run_root: Path,
) -> dict[str, Any]:
    stage_dir = run_root / "artifacts" / "03_splits"
    per_benchmark_paths: dict[str, Path] = {}
    for benchmark_id, split_frame in sorted(result.per_benchmark_splits.items()):
        per_benchmark_paths[benchmark_id] = write_parquet(
            split_frame,
            stage_dir / "per_benchmark" / benchmark_id / "splits_models.parquet",
        )

    return {
        "splits_models": write_parquet(result.splits_models, stage_dir / "splits_models.parquet"),
        "split_report": write_json(result.split_report, stage_dir / "split_report.json"),
        "per_benchmark": per_benchmark_paths,
    }


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
