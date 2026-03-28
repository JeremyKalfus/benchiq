#!/usr/bin/env python3
"""Run the strongest honest real-data metabench comparison from the frozen paper snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import benchiq
from benchiq.config import BenchIQConfig
from benchiq.irt import estimate_theta_bundle, fit_irt_bundle
from benchiq.preprocess.scores import GRAND_MEAN_SCORE, SCORE_FULL, ScoreResult
from benchiq.reconstruct.features import build_feature_tables
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
DEFAULT_RUN_ID = "metabench-real-zenodo-12819251"

MODEL_TYPE = "model_type"
ACTUAL_SCORE = "actual_score"
PREDICTED_SCORE = "predicted_score"
RMSE = "rmse"
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
            "Freeze the primary public metabench snapshot, export the paper release's "
            "default selected subsets, and run the strongest honest BenchIQ comparison."
        ),
    )
    parser.add_argument("--profile-config", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", dest="out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--existing-run-root",
        type=Path,
        help="Optional existing completed comparison run root. If set, skip recomputing stages.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args()

    profile = json.loads(args.profile_config.read_text(encoding="utf-8"))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

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
    )
    artifacts = write_comparison_bundle(
        profile=profile,
        comparison=comparison,
        archive_path=archive_path,
        responses_path=exported.responses_path,
        metadata_path=exported.metadata_path,
        run_root=run_root,
        reports_dir=args.reports_dir,
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
            f"{path}: expected {source_config['archive_md5']}, got {md5_digest}",
        )
    if sha256_digest != source_config["archive_sha256"]:
        raise RuntimeError(
            "archive sha256 mismatch for "
            f"{path}: expected {source_config['archive_sha256']}, got {sha256_digest}",
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
                    f"release-default subset member missing for {benchmark_id}: {member_name}",
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
            "Install R to rerun this comparison.",
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
    if (run_root / "artifacts" / "09_reconstruct" / "reconstruction_summary.parquet").exists():
        return run_root

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
    k_final = {
        benchmark_id: max(1, len(result.irt_item_params.index))
        for benchmark_id, result in irt_result.benchmarks.items()
    }
    select_result = select_bundle(
        bundle, irt_result, k_final=k_final, out_dir=out_dir, run_id=run_id
    )
    theta_result = estimate_theta_bundle(
        bundle,
        split_result,
        select_result,
        irt_result,
        theta_method="MAP",
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
        lam_grid=(0.1, 1.0),
        cv_folds=5,
        n_splines=10,
        out_dir=out_dir,
        run_id=run_id,
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


def build_comparison_payload(
    *, run_root: Path, profile: dict[str, Any], export_manifest: dict[str, Any]
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

    mean_proxy = compute_mean_score_proxy(
        run_root=run_root,
        benchmarks=benchmarks,
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
        and mean_proxy["comparable_to_published"] is True
        and mean_proxy["within_strong_band"] is True
    )
    acceptable_pass = (
        benchmark_acceptable_pass
        and mean_proxy["comparable_to_published"] is True
        and mean_proxy["within_acceptable_band"] is True
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
    if mean_proxy["comparable_to_published"] is not True:
        verdict_reasons.append(mean_proxy["reason"])
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
        "mean_score_proxy": mean_proxy,
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


def compute_mean_score_proxy(
    *,
    run_root: Path,
    benchmarks: list[str],
    published_target: float,
    acceptable_band: float,
    strong_band: float,
) -> dict[str, Any]:
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
        predictions = pd.read_parquet(prediction_path)
        joint_test = predictions.loc[
            (predictions[MODEL_TYPE] == "joint") & (predictions[SPLIT] == "test"),
            [MODEL_ID, ACTUAL_SCORE, PREDICTED_SCORE],
        ].copy()
        if joint_test.empty:
            return {
                "available": False,
                "comparable_to_published": False,
                "reason": f"joint predictions were unavailable for benchmark {benchmark_id}",
            }
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
        return {
            "available": False,
            "comparable_to_published": False,
            "reason": "no complete-overlap joint test models were available for a mean-score proxy",
        }

    actual_columns = [f"actual_{benchmark_id}" for benchmark_id in benchmarks]
    predicted_columns = [f"predicted_{benchmark_id}" for benchmark_id in benchmarks]
    actual_mean = joined[actual_columns].mean(axis=1)
    predicted_mean = joined[predicted_columns].mean(axis=1)
    rmse = math.sqrt(float(((actual_mean - predicted_mean) ** 2).mean()))
    abs_delta = abs(rmse - published_target)
    return {
        "available": True,
        "comparable_to_published": False,
        "reason": (
            "BenchIQ v0.1 does not yet implement the published dedicated grand-mean GAM; "
            "the value below is a derived proxy from joint benchmark predictions and is "
            "reported for context only"
        ),
        "row_count": int(len(joined.index)),
        "rmse": rmse,
        "absolute_delta": abs_delta,
        "within_strong_band": abs_delta <= strong_band,
        "within_acceptable_band": abs_delta <= acceptable_band,
    }


def write_comparison_bundle(
    *,
    profile: dict[str, Any],
    comparison: dict[str, Any],
    archive_path: Path,
    responses_path: Path,
    metadata_path: Path,
    run_root: Path,
    reports_dir: Path,
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
) -> str:
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
    mean_proxy = comparison["mean_score_proxy"]
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
    if mean_proxy["available"]:
        lines.extend(
            [
                f"- BenchIQ derived mean-score proxy RMSE: `{mean_proxy['rmse']:.3f}`",
                f"- proxy absolute delta: `{mean_proxy['absolute_delta']:.3f}`",
                f"- proxy overlap models: `{mean_proxy['row_count']}`",
                f"- proxy within ±0.05: `{mean_proxy['within_acceptable_band']}`",
            ]
        )
    else:
        lines.append("- BenchIQ derived mean-score proxy RMSE: `unavailable`")
    lines.append(f"- note: {mean_proxy['reason']}")
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
) -> str:
    release_mismatch_lines = []
    for row in comparison["rows"]:
        release_mismatch_lines.append(
            f"- {row['benchmark_id']}: "
            f"published target `{row['published_primary_target_rmse']:.3f}`, "
            f"release-default rmse.test `{row['public_release_default_subset_rmse']:.3f}`"
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
            "- This is the strongest honest in-session comparison path because it "
            "preserves the public selected items and held-out evaluation split while "
            "staying computationally tractable."
        ),
        "",
        "## deviations from the original r metabench stack",
        "",
        "- BenchIQ is not claiming bit-for-bit parity with the original r pipeline.",
        (
            "- The frozen public data unpacking step in this harness uses `Rscript` "
            "to read the published `.rds` release artifacts. The downstream "
            "modeling stages remain BenchIQ's Python-first path."
        ),
        (
            "- Current BenchIQ therefore is **not** using only the Python-first "
            "path for this validation harness; the import step depends on the "
            "public r artifact format."
        ),
        (
            "- BenchIQ uses girth instead of mirt for 2PL fitting and pyGAM "
            "instead of mgcv for reconstruction."
        ),
        (
            "- BenchIQ v0.1 does not yet implement the published dedicated "
            "grand-mean GAM for the Open LLM Leaderboard mean score."
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
        "## likely explanations for any gap",
        "",
        "- pyGAM smoothing selection will not exactly match mgcv's spline path.",
        "- girth's 2PL estimation and pathology handling differ from mirt.",
        (
            "- BenchIQ's mean-score comparison is only a proxy because the "
            "dedicated grand-mean GAM is still missing."
        ),
        (
            "- The real-data comparison here starts from the public "
            "release-default subset artifacts, not from a full raw "
            "reimplementation of every upstream r decision."
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
            "- Add an optional parity-only validation mode that can call the "
            "original r/mirt/mgcv stack, likely via `rpy2` or explicit Rscript "
            "orchestration, without changing BenchIQ's product identity or "
            "default Python-first path."
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
    return "\n".join(lines)


def render_terminal_summary(comparison: dict[str, Any], artifacts: ComparisonArtifacts) -> str:
    mean_proxy = comparison["mean_score_proxy"]
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
    if mean_proxy["available"]:
        lines.append(f"derived mean-score proxy rmse: {mean_proxy['rmse']:.3f}")
    else:
        lines.append("derived mean-score proxy rmse: unavailable")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
