#!/usr/bin/env python3
"""Run the strongest honest real-data metabench parity comparison from the frozen snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

import benchiq
from benchiq.config import BenchIQConfig
from benchiq.io.write import write_json, write_parquet
from benchiq.irt import estimate_theta_bundle, fit_irt_bundle
from benchiq.logging import update_manifest
from benchiq.preprocess.scores import GRAND_MEAN_SCORE, SCORE_FULL, ScoreResult
from benchiq.reconstruct.features import build_feature_tables
from benchiq.reconstruct.gam import cross_validate_gam, write_gam_artifacts
from benchiq.reconstruct.linear_predictor import fit_linear_predictor_bundle
from benchiq.reconstruct.reconstruction import reconstruct_scores
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID, SPLIT
from benchiq.select.information_filter import select_bundle
from benchiq.split.splitters import SplitResult
from benchiq.subsample.random_cv import BenchmarkSubsampleResult, SubsampleResult

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = REPO_ROOT / "docs" / "design" / "metabench_validation_full_profile.json"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_CACHE_DIR = REPO_ROOT / "out" / "metabench_real_source"
DEFAULT_OUT_DIR = REPO_ROOT / "out" / "metabench_real_validation"
DEFAULT_RUN_ID = "metabench-real-zenodo-12819251-parity"
BASELINE_RUN_ID = "metabench-real-zenodo-12819251"

MODEL_TYPE = "model_type"
ACTUAL_SCORE = "actual_score"
PREDICTED_SCORE = "predicted_score"
BASELINE_PREDICTION = "baseline_prediction"
RESIDUAL = "residual"
BASELINE_RESIDUAL = "baseline_residual"
RMSE = "rmse"
MAE = "mae"
PEARSON_R = "pearson_r"
SPEARMAN_R = "spearman_r"
BASELINE_RMSE = "baseline_rmse"
BASELINE_MAE = "baseline_mae"
ROW_COUNT = "row_count"
GRAND_MEAN_MODEL = "grand_mean"

DEFAULT_RELEASE_RDS = {
    "arc": "benchmark-data/arc-sub.rds",
    "gsm8k": "benchmark-data/gsm8k-sub.rds",
    "hellaswag": "benchmark-data/hellaswag-sub.rds",
    "mmlu": "benchmark-data/mmlu-sub.rds",
    "truthfulqa": "benchmark-data/truthfulqa-sub.rds",
    "winogrande": "benchmark-data/winogrande-sub.rds",
}


@dataclass(slots=True, frozen=True)
class ReleaseSubsetExport:
    responses_path: Path
    metadata_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ComparisonArtifacts:
    archive_path: Path
    responses_path: Path
    metadata_path: Path
    run_root: Path
    comparison_markdown: Path
    comparison_csv: Path
    notes_markdown: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the public metabench paper snapshot, export the release-default subsets, "
            "apply a parity-focused downstream BenchIQ run, and write a reviewer bundle."
        )
    )
    parser.add_argument("--profile-config", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", dest="out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--existing-run-root",
        type=Path,
        help="Optional existing completed parity run root. If set, skip recomputing stages.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args()

    profile = json.loads(args.profile_config.read_text(encoding="utf-8"))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    previous_comparison = load_previous_comparison(reports_dir=args.reports_dir)

    archive_path = ensure_frozen_archive(
        cache_dir=args.cache_dir,
        source_config=profile["source"],
        force_download=args.force_download,
    )
    exported = export_public_release_subset(
        archive_path=archive_path,
        cache_dir=args.cache_dir,
        profile=profile,
        force_build=args.force_build,
    )
    if args.existing_run_root is not None:
        run_root = args.existing_run_root
    else:
        run_root = run_release_subset_comparison(
            responses_path=exported.responses_path,
            metadata_path=exported.metadata_path,
            profile=profile,
            out_dir=args.out_dir,
            run_id=args.run_id,
        )

    comparison = build_comparison_payload(
        run_root=run_root,
        profile=profile,
        export_manifest=exported.manifest,
        previous_comparison=previous_comparison,
    )
    artifacts = write_comparison_bundle(
        profile=profile,
        comparison=comparison,
        archive_path=archive_path,
        responses_path=exported.responses_path,
        metadata_path=exported.metadata_path,
        run_root=run_root,
        reports_dir=args.reports_dir,
        previous_comparison=previous_comparison,
    )
    print(render_terminal_summary(comparison, artifacts))


def ensure_frozen_archive(
    *, cache_dir: Path, source_config: dict[str, Any], force_download: bool
) -> Path:
    archive_path = cache_dir / source_config["archive_name"]
    if archive_path.exists() and not force_download:
        verify_archive_hashes(archive_path, source_config)
        return archive_path

    download_path = archive_path.with_suffix(f"{archive_path.suffix}.part")
    for path in (archive_path, download_path):
        if path.exists():
            path.unlink()

    with (
        urllib.request.urlopen(source_config["archive_url"]) as response,
        download_path.open("wb") as handle,
    ):
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
    download_path.replace(archive_path)
    verify_archive_hashes(archive_path, source_config)
    return archive_path


def verify_archive_hashes(path: Path, source_config: dict[str, Any]) -> None:
    md5_digest = file_digest(path, "md5")
    sha256_digest = file_digest(path, "sha256")
    if md5_digest != source_config["archive_md5"]:
        raise RuntimeError(
            "archive md5 mismatch for "
            f"{path}: expected {source_config['archive_md5']}, got {md5_digest}"
        )
    if sha256_digest != source_config["archive_sha256"]:
        raise RuntimeError(
            "archive sha256 mismatch for "
            f"{path}: expected {source_config['archive_sha256']}, got {sha256_digest}"
        )


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def export_public_release_subset(
    *,
    archive_path: Path,
    cache_dir: Path,
    profile: dict[str, Any],
    force_build: bool,
) -> ReleaseSubsetExport:
    responses_path = cache_dir / "release_default_subset_responses_long.parquet"
    metadata_path = cache_dir / "release_default_subset_metadata.parquet"
    manifest_path = cache_dir / "release_default_subset_manifest.json"
    archive_sha256 = file_digest(archive_path, "sha256")

    if (
        responses_path.exists()
        and metadata_path.exists()
        and manifest_path.exists()
        and not force_build
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("archive_sha256") == archive_sha256:
            return ReleaseSubsetExport(
                responses_path=responses_path,
                metadata_path=metadata_path,
                manifest_path=manifest_path,
                manifest=manifest,
            )

    release_rds_dir = cache_dir / "release_default_rds"
    if release_rds_dir.exists() and force_build:
        shutil.rmtree(release_rds_dir)
    release_rds_dir.mkdir(parents=True, exist_ok=True)

    extract_release_subset_rds(
        archive_path=archive_path,
        release_rds_dir=release_rds_dir,
        benchmarks=profile["benchmarks"],
    )
    responses_csv = cache_dir / "release_default_subset_responses_long.csv"
    metadata_csv = cache_dir / "release_default_subset_metadata.csv"
    release_metrics_json = cache_dir / "release_default_subset_metrics.json"
    run_r_export(
        release_rds_dir=release_rds_dir,
        responses_csv=responses_csv,
        metadata_csv=metadata_csv,
        release_metrics_json=release_metrics_json,
        benchmarks=profile["benchmarks"],
    )
    responses_rows = csv_to_parquet(
        csv_path=responses_csv,
        parquet_path=responses_path,
        schema=pa.schema(
            [
                (BENCHMARK_ID, pa.string()),
                (MODEL_ID, pa.string()),
                (ITEM_ID, pa.string()),
                ("score", pa.int8()),
            ]
        ),
        dtypes={
            BENCHMARK_ID: "string",
            MODEL_ID: "string",
            ITEM_ID: "string",
            "score": "int8",
        },
    )
    metadata_rows = csv_to_parquet(
        csv_path=metadata_csv,
        parquet_path=metadata_path,
        schema=pa.schema(
            [
                (BENCHMARK_ID, pa.string()),
                (MODEL_ID, pa.string()),
                (SPLIT, pa.string()),
                ("score_full", pa.float64()),
                ("raw_score", pa.float64()),
                ("max_points", pa.float64()),
                ("selected_item_count", pa.int64()),
                ("release_rmse", pa.float64()),
            ]
        ),
        dtypes={
            BENCHMARK_ID: "string",
            MODEL_ID: "string",
            SPLIT: "string",
            "score_full": "float64",
            "raw_score": "float64",
            "max_points": "float64",
            "selected_item_count": "int64",
            "release_rmse": "float64",
        },
    )
    release_metrics = json.loads(release_metrics_json.read_text(encoding="utf-8"))
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "responses_path": str(responses_path),
        "metadata_path": str(metadata_path),
        "responses_row_count": responses_rows,
        "metadata_row_count": metadata_rows,
        "benchmarks": profile["benchmarks"],
        "input_mode": {
            "name": "public_release_default_subset",
            "description": (
                "the strongest honest in-session comparison path: use the primary Zenodo "
                "paper snapshot, export the release-default selected subsets and fixed "
                "train/test scores, then run downstream BenchIQ reconstruction"
            ),
        },
        "release_default_metrics": release_metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ReleaseSubsetExport(
        responses_path=responses_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def extract_release_subset_rds(
    *, archive_path: Path, release_rds_dir: Path, benchmarks: list[str]
) -> None:
    wanted = {benchmark_id: DEFAULT_RELEASE_RDS[benchmark_id] for benchmark_id in benchmarks}
    with tarfile.open(archive_path, "r:gz") as tar:
        available = {member.name for member in tar.getmembers()}
        for benchmark_id, member_name in wanted.items():
            if member_name not in available:
                raise FileNotFoundError(
                    f"release-default subset member missing for {benchmark_id}: {member_name}"
                )
            target_path = release_rds_dir / Path(member_name).name
            if target_path.exists():
                continue
            member = tar.extractfile(member_name)
            if member is None:
                raise FileNotFoundError(f"could not extract archive member: {member_name}")
            target_path.write_bytes(member.read())


def run_r_export(
    *,
    release_rds_dir: Path,
    responses_csv: Path,
    metadata_csv: Path,
    release_metrics_json: Path,
    benchmarks: list[str],
) -> None:
    if shutil.which("Rscript") is None:
        raise RuntimeError(
            "Rscript is required to unpack the public metabench .rds release artifacts. "
            "Install R to rerun this comparison."
        )

    benchmark_entries = ",\n".join(
        f'  "{benchmark_id}" = "{(release_rds_dir / f"{benchmark_id}-sub.rds").as_posix()}"'
        for benchmark_id in benchmarks
    )
    r_script = f"""
options(stringsAsFactors = FALSE)
benchmark_files <- c(
{benchmark_entries}
)
responses_csv <- {json.dumps(str(responses_csv))}
metadata_csv <- {json.dumps(str(metadata_csv))}
release_metrics_json <- {json.dumps(str(release_metrics_json))}

responses_parts <- list()
metadata_parts <- list()
release_metrics <- list()

for (benchmark_id in names(benchmark_files)) {{
  obj <- readRDS(benchmark_files[[benchmark_id]])
  data_train <- as.matrix(obj[["data.train"]])
  data_test <- as.matrix(obj[["data.test"]])
  train_long <- as.data.frame(as.table(data_train), stringsAsFactors = FALSE)
  names(train_long) <- c("model_id", "item_id", "score")
  train_long$benchmark_id <- benchmark_id
  train_long$score <- as.integer(train_long$score)

  test_long <- as.data.frame(as.table(data_test), stringsAsFactors = FALSE)
  names(test_long) <- c("model_id", "item_id", "score")
  test_long$benchmark_id <- benchmark_id
  test_long$score <- as.integer(test_long$score)

  responses_parts[[benchmark_id]] <- rbind(train_long, test_long)

  score_train <- as.numeric(obj[["scores.train"]])
  score_test <- as.numeric(obj[["scores.test"]])
  train_models <- rownames(data_train)
  test_models <- rownames(data_test)
  max_points <- as.numeric(obj[["max.points.orig"]])[1]
  release_rmse <- as.numeric(obj[["rmse.test"]])[1]
  selected_item_count <- ncol(data_train)

  metadata_parts[[benchmark_id]] <- rbind(
    data.frame(
      benchmark_id = benchmark_id,
      model_id = train_models,
      split = "train",
      score_full = (score_train / max_points) * 100,
      raw_score = score_train,
      max_points = max_points,
      selected_item_count = selected_item_count,
      release_rmse = release_rmse
    ),
    data.frame(
      benchmark_id = benchmark_id,
      model_id = test_models,
      split = "test",
      score_full = (score_test / max_points) * 100,
      raw_score = score_test,
      max_points = max_points,
      selected_item_count = selected_item_count,
      release_rmse = release_rmse
    )
  )

  release_metrics[[benchmark_id]] <- list(
    release_rmse = release_rmse,
    selected_item_count = selected_item_count,
    train_model_count = length(train_models),
    test_model_count = length(test_models)
  )
}}

responses <- do.call(rbind, responses_parts)
responses <- responses[, c("benchmark_id", "model_id", "item_id", "score")]
write.csv(responses, responses_csv, row.names = FALSE, quote = TRUE)

metadata <- do.call(rbind, metadata_parts)
metadata <- metadata[, c(
  "benchmark_id",
  "model_id",
  "split",
  "score_full",
  "raw_score",
  "max_points",
  "selected_item_count",
  "release_rmse"
)]
write.csv(metadata, metadata_csv, row.names = FALSE, quote = TRUE)

json_lines <- c("{{")
metric_names <- names(release_metrics)
for (i in seq_along(metric_names)) {{
  name <- metric_names[[i]]
  metric <- release_metrics[[name]]
  metric_json <- paste0(
    "{{",
    "\\"release_rmse\\":", format(metric$release_rmse, scientific = FALSE, digits = 16), ",",
    "\\"selected_item_count\\":", metric$selected_item_count, ",",
    "\\"train_model_count\\":", metric$train_model_count, ",",
    "\\"test_model_count\\":", metric$test_model_count,
    "}}"
  )
  suffix <- if (i < length(metric_names)) "," else ""
  json_lines <- c(json_lines, paste0("  \\"", name, "\\": ", metric_json, suffix))
}}
json_lines <- c(json_lines, "}}")
writeLines(json_lines, release_metrics_json)
"""
    subprocess.run(["Rscript", "-"], input=r_script, text=True, check=True)


def csv_to_parquet(
    *, csv_path: Path, parquet_path: Path, schema: pa.Schema, dtypes: dict[str, str]
) -> int:
    if parquet_path.exists():
        parquet_path.unlink()
    writer = pq.ParquetWriter(parquet_path, schema=schema, compression="zstd")
    row_count = 0
    try:
        for chunk in pd.read_csv(csv_path, chunksize=250_000):
            typed = chunk.astype(dtypes)
            writer.write_table(pa.Table.from_pandas(typed, schema=schema, preserve_index=False))
            row_count += len(typed.index)
    finally:
        writer.close()
    return row_count


def run_release_subset_comparison(
    *,
    responses_path: Path,
    metadata_path: Path,
    profile: dict[str, Any],
    out_dir: Path,
    run_id: str,
) -> Path:
    run_root = out_dir / run_id
    if run_root.exists():
        shutil.rmtree(run_root)

    config = BenchIQConfig.model_validate(profile["config"])
    bundle = benchiq.load_bundle(responses_path, config=config, out_dir=out_dir, run_id=run_id)
    metadata = pd.read_parquet(metadata_path).astype(
        {BENCHMARK_ID: "string", MODEL_ID: "string", SPLIT: "string"}
    )
    score_result = build_score_result(metadata)
    split_result = build_split_result(metadata)
    subsample_result = build_fixed_subset_result(bundle)

    irt_result = fit_irt_bundle(
        bundle, split_result, subsample_result, out_dir=out_dir, run_id=run_id
    )
    select_result = select_bundle(
        bundle,
        irt_result,
        k_final=resolve_published_final_counts(profile),
        n_bins=int(profile["stage_options"]["06_select"]["n_bins"]),
        theta_grid_size=int(profile["stage_options"]["06_select"]["theta_grid_size"]),
        out_dir=out_dir,
        run_id=run_id,
    )
    theta_result = estimate_theta_bundle(
        bundle,
        split_result,
        select_result,
        irt_result,
        theta_method="MAP",
        theta_grid_size=int(profile["stage_options"]["07_theta"]["theta_grid_size"]),
        out_dir=out_dir,
        run_id=run_id,
    )
    linear_result = fit_linear_predictor_bundle(
        bundle, score_result, split_result, select_result, out_dir=out_dir, run_id=run_id
    )
    feature_result = build_feature_tables(
        bundle,
        score_result,
        split_result,
        theta_result,
        linear_result,
        out_dir=out_dir,
        run_id=run_id,
    )
    reconstruct_scores(
        bundle,
        feature_result,
        lam_grid=tuple(profile["stage_options"]["09_reconstruct"]["lam_grid"]),
        cv_folds=int(profile["stage_options"]["09_reconstruct"]["cv_folds"]),
        n_splines=int(profile["stage_options"]["09_reconstruct"]["n_splines"]),
        out_dir=out_dir,
        run_id=run_id,
    )
    fit_grand_mean_reconstruction(
        run_root=run_root,
        score_result=score_result,
        feature_result=feature_result,
        lam_grid=tuple(profile["stage_options"]["09_reconstruct"]["lam_grid"]),
        cv_folds=int(profile["stage_options"]["09_reconstruct"]["cv_folds"]),
        n_splines=int(profile["stage_options"]["09_reconstruct"]["n_splines"]),
        random_seed=int(config.random_seed),
    )
    return run_root


def build_score_result(metadata: pd.DataFrame) -> ScoreResult:
    scores_full = metadata.rename(columns={"score_full": SCORE_FULL})[
        [BENCHMARK_ID, MODEL_ID, SCORE_FULL]
    ].copy()
    scores_full[SCORE_FULL] = pd.Series(scores_full[SCORE_FULL], dtype="Float64")
    benchmarks = sorted(scores_full[BENCHMARK_ID].dropna().astype("string").unique().tolist())
    overlap_sets = [
        set(scores_full.loc[scores_full[BENCHMARK_ID] == benchmark_id, MODEL_ID].astype(str))
        for benchmark_id in benchmarks
    ]
    overlap_models = sorted(set.intersection(*overlap_sets))
    scores_grand = (
        scores_full.loc[scores_full[MODEL_ID].astype(str).isin(overlap_models)]
        .groupby(MODEL_ID, sort=True)[SCORE_FULL]
        .mean()
        .reset_index()
    )
    scores_grand[MODEL_ID] = scores_grand[MODEL_ID].astype("string")
    scores_grand[GRAND_MEAN_SCORE] = pd.Series(scores_grand[SCORE_FULL], dtype="Float64")
    scores_grand["benchmark_count"] = len(benchmarks)
    scores_grand = scores_grand[[MODEL_ID, GRAND_MEAN_SCORE, "benchmark_count"]]
    score_report = {
        "grand_scores": {
            "skipped": False,
            "skip_reason": None,
            "overlap_model_ids": overlap_models,
            "overlap_model_count": len(overlap_models),
            "min_overlap_models_for_joint": 75,
            "source": "public_release_fixed_full_scores",
        }
    }
    return ScoreResult(
        scores_full=scores_full, scores_grand=scores_grand, score_report=score_report
    )


def build_split_result(metadata: pd.DataFrame) -> SplitResult:
    per_benchmark_splits: dict[str, pd.DataFrame] = {}
    split_frames: list[pd.DataFrame] = []
    benchmark_counts: dict[str, dict[str, int]] = {}
    for benchmark_id in sorted(metadata[BENCHMARK_ID].dropna().astype("string").unique().tolist()):
        frame = (
            metadata.loc[metadata[BENCHMARK_ID] == benchmark_id, [BENCHMARK_ID, MODEL_ID, SPLIT]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        per_benchmark_splits[benchmark_id] = frame
        split_frames.append(frame)
        benchmark_counts[benchmark_id] = {
            split_name: int(
                len(frame.loc[frame[SPLIT] == split_name, MODEL_ID].drop_duplicates().index)
            )
            for split_name in sorted(frame[SPLIT].dropna().astype("string").unique().tolist())
        }
    split_report = {
        "warnings": [],
        "global_test": {
            "enabled": False,
            "skip_reason": "public_release_fixed_train_test_split",
        },
        "benchmarks": {
            benchmark_id: {
                "method": "public_release_fixed_train_test_split",
                "split_counts": benchmark_counts[benchmark_id],
            }
            for benchmark_id in per_benchmark_splits
        },
        "counts": {
            "benchmark_count": len(per_benchmark_splits),
            "global_test_enabled": False,
            "global_test_model_count": 0,
        },
    }
    return SplitResult(
        splits_models=pd.concat(split_frames, ignore_index=True),
        per_benchmark_splits=per_benchmark_splits,
        split_report=split_report,
    )


def build_fixed_subset_result(bundle: benchiq.Bundle) -> SubsampleResult:
    benchmark_results: dict[str, BenchmarkSubsampleResult] = {}
    for benchmark_id in sorted(
        bundle.responses_long[BENCHMARK_ID].dropna().astype("string").unique().tolist()
    ):
        items = sorted(
            bundle.responses_long.loc[bundle.responses_long[BENCHMARK_ID] == benchmark_id, ITEM_ID]
            .dropna()
            .astype("string")
            .unique()
            .tolist()
        )
        preselect_items = pd.DataFrame(
            {
                BENCHMARK_ID: pd.Series([benchmark_id] * len(items), dtype="string"),
                ITEM_ID: pd.Series(items, dtype="string"),
            }
        )
        benchmark_results[benchmark_id] = BenchmarkSubsampleResult(
            benchmark_id=benchmark_id,
            preselect_items=preselect_items,
            cv_results=pd.DataFrame(),
            subsample_report={
                "benchmark_id": benchmark_id,
                "skipped": False,
                "skipped_reason": None,
                "selection_rule": "public_release_fixed_subset",
                "notes": [
                    "this comparison path preserves the release-default selected subset "
                    "from the frozen public metabench snapshot"
                ],
            },
        )
    return SubsampleResult(benchmarks=benchmark_results)


def resolve_published_final_counts(profile: dict[str, Any]) -> dict[str, int]:
    counts = profile["published_primary_targets"]["kept_item_counts"]
    return {str(benchmark_id): int(count) for benchmark_id, count in counts.items()}


def fit_grand_mean_reconstruction(
    *,
    run_root: Path,
    score_result: ScoreResult,
    feature_result: Any,
    lam_grid: tuple[float, ...],
    cv_folds: int,
    n_splines: int,
    random_seed: int,
) -> dict[str, Any]:
    stage_dir = run_root / "artifacts" / "09_reconstruct" / "grand_mean"
    manifest_path = run_root / "manifest.json"
    features_joint = feature_result.features_joint.copy()
    if features_joint.empty:
        report = {
            "model_type": GRAND_MEAN_MODEL,
            "skipped": True,
            "skip_reason": "joint_features_unavailable",
            "warnings": [
                {
                    "code": "grand_mean_skipped",
                    "message": (
                        "grand-mean reconstruction skipped because joint features were unavailable."
                    ),
                    "severity": "warning",
                }
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        report_path = write_json(report, stage_dir / "grand_mean_report.json")
        update_manifest(
            manifest_path,
            {
                "artifacts": {
                    "09_reconstruct": {
                        "grand_mean": {
                            "grand_mean_report": str(report_path),
                        }
                    }
                }
            },
        )
        return report

    theta_columns = sorted(
        column_name for column_name in features_joint.columns if column_name.startswith("theta_")
    )
    feature_columns = [*theta_columns, "grand_sub", "grand_lin"]
    model_features = (
        features_joint.loc[:, [MODEL_ID, SPLIT, *feature_columns]]
        .sort_values([MODEL_ID, SPLIT])
        .drop_duplicates(subset=[MODEL_ID], keep="first")
        .reset_index(drop=True)
    )
    grand_scores = score_result.scores_grand.loc[:, [MODEL_ID, GRAND_MEAN_SCORE]].copy()
    frame = model_features.merge(grand_scores, on=MODEL_ID, how="inner")
    frame = frame.dropna(subset=[MODEL_ID, SPLIT, GRAND_MEAN_SCORE, *feature_columns]).copy()
    frame[MODEL_ID] = frame[MODEL_ID].astype("string")
    frame[SPLIT] = frame[SPLIT].astype("string")
    frame[GRAND_MEAN_SCORE] = pd.Series(frame[GRAND_MEAN_SCORE], dtype="Float64")
    for column_name in feature_columns:
        frame[column_name] = pd.Series(frame[column_name], dtype="Float64")

    train_rows = frame.loc[frame[SPLIT] == "train"].copy()
    test_rows = frame.loc[frame[SPLIT] == "test"].copy()
    split_counts = (
        frame[SPLIT].value_counts(dropna=False).sort_index().astype(int).to_dict()
        if not frame.empty
        else {}
    )
    if len(train_rows.index) < 4 or len(test_rows.index) == 0:
        report = {
            "model_type": GRAND_MEAN_MODEL,
            "skipped": True,
            "skip_reason": "insufficient_rows_for_grand_mean_gam",
            "feature_columns": feature_columns,
            "split_counts": split_counts,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        report_path = write_json(report, stage_dir / "grand_mean_report.json")
        update_manifest(
            manifest_path,
            {
                "artifacts": {
                    "09_reconstruct": {
                        "grand_mean": {
                            "grand_mean_report": str(report_path),
                        }
                    }
                }
            },
        )
        return report

    effective_cv_folds = min(int(cv_folds), int(len(train_rows.index)))
    effective_n_splines = min(int(n_splines), int(len(train_rows.index)))
    cv_result = cross_validate_gam(
        train_rows.loc[:, feature_columns],
        train_rows[GRAND_MEAN_SCORE],
        lam_grid=lam_grid,
        cv_folds=effective_cv_folds,
        random_seed=random_seed,
        feature_names=feature_columns,
        target_name=GRAND_MEAN_SCORE,
        n_splines=effective_n_splines,
        X_test=test_rows.loc[:, feature_columns],
        y_test=test_rows[GRAND_MEAN_SCORE],
    )
    prediction_rows = frame.loc[:, [MODEL_ID, SPLIT, GRAND_MEAN_SCORE]].copy()
    prediction_rows[MODEL_TYPE] = GRAND_MEAN_MODEL
    prediction_rows[PREDICTED_SCORE] = cv_result.best_model.predict(frame.loc[:, feature_columns])
    train_mean = float(train_rows[GRAND_MEAN_SCORE].astype(float).mean())
    prediction_rows[BASELINE_PREDICTION] = train_mean
    prediction_rows[ACTUAL_SCORE] = pd.Series(prediction_rows[GRAND_MEAN_SCORE], dtype="Float64")
    prediction_rows[RESIDUAL] = prediction_rows[PREDICTED_SCORE].astype(float) - prediction_rows[
        ACTUAL_SCORE
    ].astype(float)
    prediction_rows[BASELINE_RESIDUAL] = prediction_rows[BASELINE_PREDICTION].astype(
        float
    ) - prediction_rows[ACTUAL_SCORE].astype(float)
    predictions = prediction_rows.loc[
        :,
        [
            MODEL_ID,
            SPLIT,
            MODEL_TYPE,
            ACTUAL_SCORE,
            PREDICTED_SCORE,
            BASELINE_PREDICTION,
            RESIDUAL,
            BASELINE_RESIDUAL,
        ],
    ].copy()
    predictions[MODEL_ID] = predictions[MODEL_ID].astype("string")
    predictions[SPLIT] = predictions[SPLIT].astype("string")
    predictions[MODEL_TYPE] = predictions[MODEL_TYPE].astype("string")
    for column_name in [
        ACTUAL_SCORE,
        PREDICTED_SCORE,
        BASELINE_PREDICTION,
        RESIDUAL,
        BASELINE_RESIDUAL,
    ]:
        predictions[column_name] = pd.Series(predictions[column_name], dtype="Float64")

    metrics = {
        split_name: split_metrics(predictions.loc[predictions[SPLIT] == split_name].copy())
        for split_name in ["train", "test"]
    }
    prediction_path = write_parquet(predictions, stage_dir / "predictions.parquet")
    gam_paths = write_gam_artifacts(
        cv_result,
        out_dir=stage_dir / "model",
        manifest_path=manifest_path,
        stage_key="09_reconstruct_grand",
    )
    plot_paths = write_prediction_plots(predictions, out_dir=stage_dir / "plots")
    report = {
        "model_type": GRAND_MEAN_MODEL,
        "skipped": False,
        "skip_reason": None,
        "feature_columns": feature_columns,
        "train_row_count": int(len(train_rows.index)),
        "test_row_count": int(len(test_rows.index)),
        "available_row_count": int(len(frame.index)),
        "split_counts": split_counts,
        "cv_report": cv_result.cv_report,
        "metrics": metrics,
        "artifacts": {
            "predictions": str(prediction_path),
            "plots": {name: str(path) for name, path in sorted(plot_paths.items())},
            "model": {name: str(path) for name, path in sorted(gam_paths.items())},
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = write_json(report, stage_dir / "grand_mean_report.json")
    update_manifest(
        manifest_path,
        {
            "artifacts": {
                "09_reconstruct": {
                    "grand_mean": {
                        "predictions": str(prediction_path),
                        "grand_mean_report": str(report_path),
                        **{name: str(path) for name, path in sorted(gam_paths.items())},
                        **{name: str(path) for name, path in sorted(plot_paths.items())},
                    }
                }
            }
        },
    )
    return report


def build_comparison_payload(
    *,
    run_root: Path,
    profile: dict[str, Any],
    export_manifest: dict[str, Any],
    previous_comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    benchmarks = profile["benchmarks"]
    reconstruction_summary = pd.read_parquet(
        run_root / "artifacts" / "09_reconstruct" / "reconstruction_summary.parquet"
    )
    marginal_test = reconstruction_summary.loc[
        (reconstruction_summary[MODEL_TYPE] == "marginal")
        & (reconstruction_summary[SPLIT] == "test")
    ].copy()
    joint_test = reconstruction_summary.loc[
        (reconstruction_summary[MODEL_TYPE] == "joint") & (reconstruction_summary[SPLIT] == "test")
    ].copy()
    published_targets = profile["published_primary_targets"]["joint_test_rmse_by_benchmark"]
    release_metrics = export_manifest["release_default_metrics"]

    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    kept_item_counts: dict[str, int] = {}
    marginal_rmse_by_benchmark: dict[str, float | None] = {}
    joint_rmse_by_benchmark: dict[str, float | None] = {}
    strong_band = float(profile["acceptance"]["strong_per_benchmark_abs_delta"])

    for benchmark_id in benchmarks:
        subset_path = (
            run_root
            / "artifacts"
            / "06_select"
            / "per_benchmark"
            / benchmark_id
            / "subset_final.parquet"
        )
        kept_item_count = int(len(pd.read_parquet(subset_path).index))
        kept_item_counts[benchmark_id] = kept_item_count
        published_target = float(published_targets[benchmark_id])
        release_rmse = float(release_metrics[benchmark_id]["release_rmse"])
        marginal_rmse = extract_metric(marginal_test, benchmark_id, RMSE)
        joint_rmse = extract_metric(joint_test, benchmark_id, RMSE)
        marginal_rmse_by_benchmark[benchmark_id] = marginal_rmse
        joint_rmse_by_benchmark[benchmark_id] = joint_rmse
        abs_delta = None if joint_rmse is None else abs(joint_rmse - published_target)
        if abs_delta is not None:
            deltas.append(abs_delta)
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "published_primary_target_rmse": published_target,
                "public_release_default_subset_rmse": release_rmse,
                "benchiq_marginal_rmse": marginal_rmse,
                "benchiq_joint_rmse": joint_rmse,
                "absolute_delta": abs_delta,
                "kept_item_count": kept_item_count,
                "within_strong_band": abs_delta is not None and abs_delta <= strong_band,
            }
        )

    mean_score_result = compute_mean_score_result(
        run_root=run_root,
        published_target=float(profile["published_primary_targets"]["open_llm_lb_mean_rmse"]),
        acceptable_band=float(profile["acceptance"]["acceptable_mean_score_abs_delta"]),
        strong_band=float(profile["acceptance"]["strong_mean_score_abs_delta"]),
    )
    mean_abs_delta = None if not deltas else sum(deltas) / len(deltas)
    benchmark_strong_pass = len(deltas) == len(benchmarks) and all(
        row["within_strong_band"] for row in rows
    )
    benchmark_acceptable_pass = mean_abs_delta is not None and mean_abs_delta <= float(
        profile["acceptance"]["acceptable_benchmark_mean_abs_delta"]
    )
    strong_pass = (
        benchmark_strong_pass
        and mean_score_result["available"] is True
        and mean_score_result["within_strong_band"] is True
    )
    acceptable_pass = (
        benchmark_acceptable_pass
        and mean_score_result["available"] is True
        and mean_score_result["within_acceptable_band"] is True
    )
    overall_pass = strong_pass or acceptable_pass
    verdict_reasons: list[str] = []
    if mean_abs_delta is None:
        verdict_reasons.append(
            "joint reconstruction rmse was unavailable for one or more benchmarks"
        )
        overall_pass = False
        strong_pass = False
        acceptable_pass = False
    elif mean_abs_delta > float(profile["acceptance"]["acceptable_benchmark_mean_abs_delta"]):
        acceptable_limit = float(profile["acceptance"]["acceptable_benchmark_mean_abs_delta"])
        verdict_reasons.append(
            f"mean absolute benchmark delta {mean_abs_delta:.3f} exceeded the acceptable "
            f"limit {acceptable_limit:.3f}"
        )
    if mean_score_result["available"] is not True:
        verdict_reasons.append(mean_score_result["reason"])
        overall_pass = False
        strong_pass = False
        acceptable_pass = False
    elif not overall_pass:
        verdict_reasons.append(
            "benchmark rmse deltas and/or the published mean-score parity band were exceeded"
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmarks": benchmarks,
        "rows": rows,
        "marginal_rmse_by_benchmark": marginal_rmse_by_benchmark,
        "joint_rmse_by_benchmark": joint_rmse_by_benchmark,
        "kept_item_counts": kept_item_counts,
        "mean_abs_delta": mean_abs_delta,
        "published_mean_score_rmse": float(
            profile["published_primary_targets"]["open_llm_lb_mean_rmse"]
        ),
        "mean_score_result": mean_score_result,
        "previous_comparison_analysis": analyze_previous_comparison(
            previous_comparison,
            comparison_rows=rows,
            current_mean_score_result=mean_score_result,
        ),
        "strong_pass": strong_pass,
        "acceptable_pass": acceptable_pass,
        "overall_pass": overall_pass,
        "verdict_reason": "; ".join(verdict_reasons) if verdict_reasons else None,
    }


def extract_metric(summary: pd.DataFrame, benchmark_id: str, metric: str) -> float | None:
    rows = summary.loc[summary[BENCHMARK_ID] == benchmark_id]
    if rows.empty:
        return None
    return float(rows.iloc[0][metric])


def compute_mean_score_result(
    *,
    run_root: Path,
    published_target: float,
    acceptable_band: float,
    strong_band: float,
) -> dict[str, Any]:
    report_path = (
        run_root / "artifacts" / "09_reconstruct" / "grand_mean" / "grand_mean_report.json"
    )
    prediction_path = (
        run_root / "artifacts" / "09_reconstruct" / "grand_mean" / "predictions.parquet"
    )
    if not report_path.exists() or not prediction_path.exists():
        return {
            "available": False,
            "reason": "dedicated grand-mean reconstruction artifacts were not written",
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("skipped"):
        return {
            "available": False,
            "reason": report.get("skip_reason") or "grand-mean reconstruction was skipped",
        }
    predictions = pd.read_parquet(prediction_path)
    test_rows = predictions.loc[predictions[SPLIT] == "test"].copy()
    if test_rows.empty:
        return {
            "available": False,
            "reason": "grand-mean reconstruction produced no held-out test rows",
        }
    rmse_value = float(report["metrics"]["test"][RMSE])
    abs_delta = abs(rmse_value - published_target)
    return {
        "available": True,
        "reason": "dedicated grand-mean GAM fit on fixed complete-overlap train/test models",
        "row_count": int(len(test_rows.index)),
        "rmse": rmse_value,
        "absolute_delta": abs_delta,
        "within_strong_band": abs_delta <= strong_band,
        "within_acceptable_band": abs_delta <= acceptable_band,
        "metrics": report["metrics"],
    }


def load_previous_comparison(*, reports_dir: Path) -> dict[str, Any] | None:
    baseline_run_root = DEFAULT_OUT_DIR / BASELINE_RUN_ID
    if baseline_run_root.exists():
        summary_path = (
            baseline_run_root / "artifacts" / "09_reconstruct" / "reconstruction_summary.parquet"
        )
        if summary_path.exists():
            summary = pd.read_parquet(summary_path)
            joint_test = summary.loc[
                (summary[MODEL_TYPE] == "joint") & (summary[SPLIT] == "test")
            ].copy()
            previous = {
                "joint_rmse_by_benchmark": {
                    str(row[BENCHMARK_ID]): float(row[RMSE])
                    for _, row in joint_test.iterrows()
                    if pd.notna(row[RMSE])
                }
            }
            legacy_mean_score = compute_legacy_mean_score_proxy_from_run_root(
                run_root=baseline_run_root,
                benchmarks=["arc", "gsm8k", "hellaswag", "mmlu", "truthfulqa", "winogrande"],
            )
            if legacy_mean_score is not None:
                previous["mean_score_rmse"] = legacy_mean_score
            return previous

    comparison_csv = reports_dir / "metabench_real_data_comparison.csv"
    comparison_md = reports_dir / "metabench_real_data_comparison.md"
    if not comparison_csv.exists():
        return None
    frame = pd.read_csv(comparison_csv)
    previous = {
        "joint_rmse_by_benchmark": {
            str(row["benchmark_id"]): float(row["benchiq_joint_rmse"])
            for _, row in frame.iterrows()
            if pd.notna(row.get("benchiq_joint_rmse"))
        }
    }
    if comparison_md.exists():
        text = comparison_md.read_text(encoding="utf-8")
        match = re.search(
            r"BenchIQ (?:derived mean-score proxy|dedicated grand-mean) RMSE: `([0-9.]+)`",
            text,
        )
        if match is not None:
            previous["mean_score_rmse"] = float(match.group(1))
    return previous


def compute_legacy_mean_score_proxy_from_run_root(
    *, run_root: Path, benchmarks: list[str]
) -> float | None:
    joined: pd.DataFrame | None = None
    for benchmark_id in benchmarks:
        prediction_path = (
            run_root
            / "artifacts"
            / "09_reconstruct"
            / "per_benchmark"
            / benchmark_id
            / "predictions.parquet"
        )
        if not prediction_path.exists():
            return None
        predictions = pd.read_parquet(prediction_path)
        joint_test = predictions.loc[
            (predictions[MODEL_TYPE] == "joint") & (predictions[SPLIT] == "test"),
            [MODEL_ID, ACTUAL_SCORE, PREDICTED_SCORE],
        ].copy()
        if joint_test.empty:
            return None
        joint_test = joint_test.rename(
            columns={
                ACTUAL_SCORE: f"actual_{benchmark_id}",
                PREDICTED_SCORE: f"predicted_{benchmark_id}",
            }
        )
        joined = (
            joint_test if joined is None else joined.merge(joint_test, on=MODEL_ID, how="inner")
        )

    if joined is None or joined.empty:
        return None
    actual_mean = joined[[f"actual_{benchmark_id}" for benchmark_id in benchmarks]].mean(axis=1)
    predicted_mean = joined[[f"predicted_{benchmark_id}" for benchmark_id in benchmarks]].mean(
        axis=1
    )
    return rmse(actual_mean.to_numpy(dtype=float), predicted_mean.to_numpy(dtype=float))


def analyze_previous_comparison(
    previous_comparison: dict[str, Any] | None,
    *,
    comparison_rows: list[dict[str, Any]],
    current_mean_score_result: dict[str, Any],
) -> dict[str, Any] | None:
    if previous_comparison is None:
        return None
    previous_joint = previous_comparison.get("joint_rmse_by_benchmark", {})
    if not previous_joint:
        return None
    previous_deltas: list[float] = []
    current_deltas: list[float] = []
    for row in comparison_rows:
        benchmark_id = str(row["benchmark_id"])
        if benchmark_id not in previous_joint or row["benchiq_joint_rmse"] is None:
            continue
        previous_deltas.append(
            abs(float(previous_joint[benchmark_id]) - float(row["published_primary_target_rmse"]))
        )
        current_deltas.append(
            abs(float(row["benchiq_joint_rmse"]) - float(row["published_primary_target_rmse"]))
        )
    if not previous_deltas or not current_deltas:
        return None
    analysis = {
        "previous_mean_abs_delta": float(sum(previous_deltas) / len(previous_deltas)),
        "current_mean_abs_delta": float(sum(current_deltas) / len(current_deltas)),
    }
    if "mean_score_rmse" in previous_comparison:
        analysis["previous_mean_score_rmse"] = float(previous_comparison["mean_score_rmse"])
    if current_mean_score_result["available"]:
        analysis["current_mean_score_rmse"] = float(current_mean_score_result["rmse"])
    return analysis


def write_comparison_bundle(
    *,
    profile: dict[str, Any],
    comparison: dict[str, Any],
    archive_path: Path,
    responses_path: Path,
    metadata_path: Path,
    run_root: Path,
    reports_dir: Path,
    previous_comparison: dict[str, Any] | None,
) -> ComparisonArtifacts:
    comparison_csv = reports_dir / "metabench_real_data_comparison.csv"
    comparison_markdown = reports_dir / "metabench_real_data_comparison.md"
    notes_markdown = reports_dir / "metabench_real_data_notes.md"
    pd.DataFrame.from_records(comparison["rows"]).to_csv(comparison_csv, index=False)
    comparison_markdown.write_text(
        build_comparison_markdown(
            comparison=comparison,
            profile=profile,
            archive_path=archive_path,
            responses_path=responses_path,
            metadata_path=metadata_path,
            run_root=run_root,
            previous_comparison=previous_comparison,
        ),
        encoding="utf-8",
    )
    notes_markdown.write_text(
        build_notes_markdown(
            comparison=comparison,
            profile=profile,
            archive_path=archive_path,
            responses_path=responses_path,
            metadata_path=metadata_path,
            run_root=run_root,
            previous_comparison=previous_comparison,
        ),
        encoding="utf-8",
    )
    return ComparisonArtifacts(
        archive_path=archive_path,
        responses_path=responses_path,
        metadata_path=metadata_path,
        run_root=run_root,
        comparison_markdown=comparison_markdown,
        comparison_csv=comparison_csv,
        notes_markdown=notes_markdown,
    )


def build_comparison_markdown(
    *,
    comparison: dict[str, Any],
    profile: dict[str, Any],
    archive_path: Path,
    responses_path: Path,
    metadata_path: Path,
    run_root: Path,
    previous_comparison: dict[str, Any] | None,
) -> str:
    del previous_comparison
    lines = [
        "# metabench real-data comparison",
        "",
        f"- generated_at: `{comparison['generated_at']}`",
        f"- data_source: `{profile['source']['label']}` ({profile['source']['doi']})",
        f"- release_tier: `{profile['source']['release_tier']}`",
        f"- archive_path: `{archive_path}`",
        f"- responses_path: `{responses_path}`",
        f"- metadata_path: `{metadata_path}`",
        f"- run_root: `{run_root}`",
        f"- overall_pass: `{comparison['overall_pass']}`",
        f"- strong_pass: `{comparison['strong_pass']}`",
        f"- acceptable_pass: `{comparison['acceptable_pass']}`",
        "",
        (
            "| benchmark | published target rmse | public release rmse | "
            "BenchIQ marginal rmse | BenchIQ joint rmse | absolute delta | "
            "kept items | within ±0.10 |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in comparison["rows"]:
        marginal = (
            "n/a" if row["benchiq_marginal_rmse"] is None else f"{row['benchiq_marginal_rmse']:.3f}"
        )
        joint = "n/a" if row["benchiq_joint_rmse"] is None else f"{row['benchiq_joint_rmse']:.3f}"
        delta = "n/a" if row["absolute_delta"] is None else f"{row['absolute_delta']:.3f}"
        lines.append(
            f"| {row['benchmark_id']} | {row['published_primary_target_rmse']:.3f} | "
            f"{row['public_release_default_subset_rmse']:.3f} | {marginal} | {joint} | "
            f"{delta} | {row['kept_item_count']} | {row['within_strong_band']} |"
        )

    mean_score_result = comparison["mean_score_result"]
    lines.extend(
        [
            "",
            "## mean-score comparison",
            "",
            (
                "- published Open LLM LB mean RMSE target: "
                f"`{comparison['published_mean_score_rmse']:.3f}`"
            ),
        ]
    )
    if mean_score_result["available"]:
        lines.extend(
            [
                f"- BenchIQ dedicated grand-mean RMSE: `{mean_score_result['rmse']:.3f}`",
                f"- absolute delta: `{mean_score_result['absolute_delta']:.3f}`",
                f"- overlap test models: `{mean_score_result['row_count']}`",
                f"- within ±0.05: `{mean_score_result['within_acceptable_band']}`",
            ]
        )
    else:
        lines.append("- BenchIQ dedicated grand-mean RMSE: `unavailable`")
    lines.append(f"- note: {mean_score_result['reason']}")

    previous_analysis = comparison["previous_comparison_analysis"]
    if previous_analysis is not None:
        lines.extend(
            [
                "",
                "## parity-repair delta vs previous reviewer pass",
                "",
                (
                    "- previous mean absolute benchmark delta: "
                    f"`{previous_analysis['previous_mean_abs_delta']:.3f}`"
                ),
                (
                    "- current mean absolute benchmark delta: "
                    f"`{previous_analysis['current_mean_abs_delta']:.3f}`"
                ),
            ]
        )
        if "previous_mean_score_rmse" in previous_analysis and mean_score_result["available"]:
            lines.extend(
                [
                    (
                        "- previous mean-score rmse: "
                        f"`{previous_analysis['previous_mean_score_rmse']:.3f}`"
                    ),
                    f"- current mean-score rmse: `{mean_score_result['rmse']:.3f}`",
                ]
            )

    lines.extend(
        [
            "",
            "## verdict",
            "",
            f"- mean absolute delta across the six benchmarks: `{comparison['mean_abs_delta']}`",
            f"- verdict_reason: {comparison['verdict_reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def build_notes_markdown(
    *,
    comparison: dict[str, Any],
    profile: dict[str, Any],
    archive_path: Path,
    responses_path: Path,
    metadata_path: Path,
    run_root: Path,
    previous_comparison: dict[str, Any] | None,
) -> str:
    del previous_comparison
    release_mismatch_lines = []
    for row in comparison["rows"]:
        release_mismatch_lines.append(
            f"- {row['benchmark_id']}: "
            f"published target `{row['published_primary_target_rmse']:.3f}`, "
            f"release-default rmse.test `{row['public_release_default_subset_rmse']:.3f}`"
        )

    mean_score_result = comparison["mean_score_result"]
    previous_analysis = comparison["previous_comparison_analysis"]
    previous_mean_abs_delta = (
        None if previous_analysis is None else previous_analysis.get("previous_mean_abs_delta")
    )
    current_mean_abs_delta = comparison["mean_abs_delta"]
    counts_materially_reduced = (
        previous_mean_abs_delta is not None
        and current_mean_abs_delta is not None
        and current_mean_abs_delta < previous_mean_abs_delta
    )
    previous_mean_score_rmse = (
        None if previous_analysis is None else previous_analysis.get("previous_mean_score_rmse")
    )
    grand_mean_material_change = (
        previous_mean_score_rmse is not None
        and mean_score_result["available"]
        and abs(mean_score_result["rmse"] - previous_mean_score_rmse) >= 0.05
    )

    remaining_cause = (
        "the frozen public snapshot exposes the release-default 350-item subsets but not the "
        "released final item identities, so BenchIQ still has to reconstruct the final subset "
        "with girth/pyGAM instead of replaying the original mirt/mgcv item path"
    )

    lines = [
        "# metabench real-data notes",
        "",
        "## source used",
        "",
        f"- source used: `{profile['source']['label']}`",
        f"- primary or secondary release: `{profile['source']['release_tier']}`",
        f"- zenodo doi: `{profile['source']['doi']}`",
        f"- zenodo record id: `{profile['source']['record_id']}`",
        f"- publication date: `{profile['source']['publication_date']}`",
        f"- frozen archive: `{archive_path}`",
        f"- archive md5: `{profile['source']['archive_md5']}`",
        f"- archive sha256: `{profile['source']['archive_sha256']}`",
        f"- exported responses_long: `{responses_path}`",
        f"- exported release metadata: `{metadata_path}`",
        f"- BenchIQ run root: `{run_root}`",
        "",
        "## what this run actually validates",
        "",
        (
            "- This pass uses the primary Zenodo paper snapshot, but it does not run "
            "the full raw 153M-row archive through BenchIQ end to end."
        ),
        (
            "- Instead, it exports the paper release's default `*-sub.rds` selected "
            "subsets and fixed train/test scores, then runs BenchIQ's downstream "
            "Python reconstruction stack on that real public split/subset."
        ),
        (
            "- This parity repair keeps the same frozen snapshot, preserves the public "
            "fixed split/subset behavior, then applies the paper's published final kept-item "
            "counts and a dedicated grand-mean GAM inside the validation harness."
        ),
        "",
        "## deviations from the original r metabench stack",
        "",
        "- BenchIQ is not claiming bit-for-bit parity with the original r pipeline.",
        (
            "- The frozen public data unpacking step in this harness uses `Rscript` "
            "to read the published `.rds` release artifacts. The downstream modeling "
            "stages remain BenchIQ's Python-first path."
        ),
        (
            "- Current BenchIQ therefore is **not** using only the Python-first path "
            "for this validation harness; the `.rds` import step is parity-specific, "
            "but the downstream modeling remains BenchIQ's Python-first path."
        ),
        (
            "- BenchIQ uses girth instead of mirt for 2PL fitting and pyGAM instead "
            "of mgcv for reconstruction."
        ),
        (
            "- The dedicated grand-mean GAM added here is parity-specific validation "
            "logic in this script. It does not change BenchIQ's generic product identity."
        ),
        "",
        "## snapshot-to-target mismatch to know about",
        "",
        (
            "- The user-requested primary targets are preserved below exactly as "
            "the acceptance baseline."
        ),
        (
            "- The frozen paper snapshot's default `rmse.test` values inside the "
            "public `*-sub.rds` artifacts do not exactly match those targets for "
            "every benchmark, especially HellaSwag."
        ),
        *release_mismatch_lines,
        "",
        "## likely explanations for any remaining gap",
        "",
        "- pyGAM smoothing selection will not exactly match mgcv's spline path.",
        "- girth's 2PL estimation and pathology handling differ from mirt.",
        (
            "- The public snapshot does not expose the original post-IRT/final-selection "
            "artifacts directly, so this harness still reconstructs the final selection "
            "inside BenchIQ from the public 350-item release subset."
        ),
        (
            "- The real-data comparison still does not replay every upstream r decision "
            "from the raw archive; it is the closest tractable like-for-like public path."
        ),
        "",
        "## direct answers for this parity repair",
        "",
        (
            "- Did matching the published kept-item counts materially reduce the RMSE deltas? "
            f"`{'yes' if counts_materially_reduced else 'no'}`"
        ),
    ]
    if previous_mean_abs_delta is not None and current_mean_abs_delta is not None:
        lines.extend(
            [
                f"- previous mean absolute benchmark delta: `{previous_mean_abs_delta:.3f}`",
                f"- current mean absolute benchmark delta: `{current_mean_abs_delta:.3f}`",
            ]
        )
    lines.append(
        (
            "- Did adding the dedicated grand-mean path materially change the mean-score "
            f"comparison? `{'yes' if grand_mean_material_change else 'no'}`"
        )
    )
    if previous_mean_score_rmse is not None and mean_score_result["available"]:
        lines.extend(
            [
                f"- previous mean-score rmse: `{previous_mean_score_rmse:.3f}`",
                f"- current dedicated grand-mean rmse: `{mean_score_result['rmse']:.3f}`",
            ]
        )
    lines.extend(
        [
            (
                "- Is the remaining gap now small enough to claim acceptance-grade parity "
                f"under the BenchIQ tolerance rule? `{comparison['overall_pass']}`"
            ),
            (
                "- If not, what is the smallest remaining cause of mismatch? "
                f"{remaining_cause if not comparison['overall_pass'] else 'none'}"
            ),
            "",
            "## acceptance outcome",
            "",
            f"- overall_pass: `{comparison['overall_pass']}`",
            f"- strong_pass: `{comparison['strong_pass']}`",
            f"- acceptable_pass: `{comparison['acceptable_pass']}`",
            f"- verdict_reason: {comparison['verdict_reason']}",
            "",
            "## smallest optional parity path if this still misses",
            "",
            (
                "- Add the smallest optional parity backend needed to replay the final "
                "mirt/mgcv behavior on the frozen snapshot, likely through an `rpy2` or "
                "explicit Rscript-backed validation-only mode, without changing BenchIQ's "
                "default Python-first product path."
            ),
            "",
            "## rerun command",
            "",
            (
                "- `.venv/bin/python "
                f"{REPO_ROOT / 'scripts' / 'run_metabench_real_data_comparison.py'} "
                f"--out {run_root.parent} --run-id {run_root.name}`"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_terminal_summary(comparison: dict[str, Any], artifacts: ComparisonArtifacts) -> str:
    mean_score_result = comparison["mean_score_result"]
    lines = [
        "metabench real-data comparison completed",
        f"run location: {artifacts.run_root}",
        f"comparison markdown: {artifacts.comparison_markdown}",
        f"comparison csv: {artifacts.comparison_csv}",
        f"notes: {artifacts.notes_markdown}",
        f"overall pass: {comparison['overall_pass']}",
        f"strong pass: {comparison['strong_pass']}",
        f"acceptable pass: {comparison['acceptable_pass']}",
        f"mean absolute delta: {comparison['mean_abs_delta']}",
    ]
    if mean_score_result["available"]:
        lines.append(f"dedicated grand-mean rmse: {mean_score_result['rmse']:.3f}")
    else:
        lines.append("dedicated grand-mean rmse: unavailable")
    return "\n".join(lines)


def split_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {
            RMSE: None,
            MAE: None,
            PEARSON_R: None,
            SPEARMAN_R: None,
            BASELINE_RMSE: None,
            BASELINE_MAE: None,
            ROW_COUNT: 0,
        }
    actual = predictions[ACTUAL_SCORE].astype(float).to_numpy()
    predicted = predictions[PREDICTED_SCORE].astype(float).to_numpy()
    baseline = predictions[BASELINE_PREDICTION].astype(float).to_numpy()
    return {
        RMSE: rmse(actual, predicted),
        MAE: mae(actual, predicted),
        PEARSON_R: correlation(actual, predicted, method="pearson"),
        SPEARMAN_R: correlation(actual, predicted, method="spearman"),
        BASELINE_RMSE: rmse(actual, baseline),
        BASELINE_MAE: mae(actual, baseline),
        ROW_COUNT: int(len(predictions.index)),
    }


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def correlation(actual: np.ndarray, predicted: np.ndarray, *, method: str) -> float | None:
    if actual.shape[0] < 2:
        return None
    if np.allclose(actual, actual[0]) or np.allclose(predicted, predicted[0]):
        return None
    if method == "pearson":
        return float(pearsonr(actual, predicted).statistic)
    return float(spearmanr(actual, predicted).statistic)


def write_prediction_plots(predictions: pd.DataFrame, *, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if predictions.empty:
        return {}

    actual = predictions[ACTUAL_SCORE].astype(float)
    min_value = float(actual.min())
    max_value = float(actual.max())

    calibration_path = out_dir / "calibration.png"
    figure, axis = plt.subplots(figsize=(6, 4))
    for split_name, split_frame in predictions.groupby(SPLIT, sort=True):
        axis.scatter(
            split_frame[ACTUAL_SCORE].astype(float),
            split_frame[PREDICTED_SCORE].astype(float),
            label=str(split_name),
            alpha=0.8,
        )
    axis.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--")
    axis.set_xlabel("actual mean score")
    axis.set_ylabel("predicted mean score")
    axis.set_title("grand-mean calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(calibration_path, dpi=150)
    plt.close(figure)

    residual_path = out_dir / "residual_histogram.png"
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(predictions[RESIDUAL].astype(float), bins=16, color="#4C72B0", alpha=0.85)
    axis.set_xlabel("prediction residual")
    axis.set_ylabel("count")
    axis.set_title("grand-mean residual histogram")
    figure.tight_layout()
    figure.savefig(residual_path, dpi=150)
    plt.close(figure)

    return {
        "calibration_plot": calibration_path,
        "residual_histogram": residual_path,
    }


if __name__ == "__main__":
    main()
