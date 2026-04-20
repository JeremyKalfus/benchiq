"""Materialization helpers for the narrowed public portfolio standing pass."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from benchiq.io.load import load_bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.portfolio.specs import (
    BenchmarkSourceSpec,
    BinaryMetricPolicy,
    MaterializationResult,
    PortfolioDatasetDef,
    SnapshotSpec,
)
from benchiq.portfolio.utils import (
    coerce_binary_metric,
    ensure_relative_path,
    gcs_api_path,
    http_get_json,
    infer_model_family,
    normalize_slug,
    prefixed_benchmark_id,
    prefixed_item_id,
    stable_hash,
    stable_sample,
)
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID, SCORE

PORTFOLIO_VERSION = "0.1"
PORTFOLIO_INDEX_FILENAME = "index_manifest.json"

_OPENEVAL_BENCHMARK_ALLOWLIST = ("boolq", "gpqa", "mmlu_pro", "omni_math", "ifeval")
_OPENEVAL_METRIC_POLICY = BinaryMetricPolicy(
    preferred_fragments=(
        "strict_accuracy",
        "exact_match",
        "accuracy",
        "acc",
        "correctness",
        "pass",
    ),
    rejected_fragments=(
        "rouge",
        "bleu",
        "f1",
        "judge",
        "preference",
        "win_rate",
    ),
)


class PortfolioSourceAdapter(Protocol):
    """One source-specific materializer."""

    adapter_id: str

    def materialize(
        self,
        *,
        source: BenchmarkSourceSpec,
        snapshot: SnapshotSpec,
        out_dir: Path,
    ) -> MaterializationResult:
        """Materialize or explicitly skip one source snapshot."""


@dataclass(slots=True)
class BundleTables:
    """Canonical bundle tables ready to be written."""

    responses_long: pd.DataFrame
    items: pd.DataFrame
    models: pd.DataFrame
    dataset: PortfolioDatasetDef
    details: dict[str, Any]


def materialize_catalog(
    *,
    source_specs: tuple[BenchmarkSourceSpec, ...],
    out_dir: str | Path,
) -> list[MaterializationResult]:
    """Materialize the narrowed public portfolio source catalog."""

    resolved_out_dir = Path(out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    results: list[MaterializationResult] = []
    adapters = _adapter_registry()
    for source in source_specs:
        adapter = adapters[source.adapter_id]
        for snapshot in source.snapshots:
            try:
                result = adapter.materialize(
                    source=source,
                    snapshot=snapshot,
                    out_dir=resolved_out_dir,
                )
            except Exception as exc:  # pragma: no cover - exercised via integration flow
                manifest_path = _write_status_manifest(
                    bundle_dir=resolved_out_dir / source.source_id / snapshot.snapshot_id,
                    payload={
                        "source_id": source.source_id,
                        "snapshot_id": snapshot.snapshot_id,
                        "role": snapshot.role,
                        "status": "failed",
                        "failure": {
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    },
                )
                result = MaterializationResult(
                    source_id=source.source_id,
                    snapshot_id=snapshot.snapshot_id,
                    role=snapshot.role,
                    status="failed",
                    manifest_path=str(manifest_path),
                    skip_reason=str(exc),
                    details={"traceback": traceback.format_exc()},
                )
            results.append(result)

    index_payload = {
        "portfolio_version": PORTFOLIO_VERSION,
        "results": [result.to_dict() for result in results],
    }
    write_json(index_payload, resolved_out_dir / PORTFOLIO_INDEX_FILENAME)
    return results


def load_portfolio_index(path: str | Path) -> list[MaterializationResult]:
    """Load a saved materialization index manifest."""

    payload = http_get_json(Path(path).resolve().as_uri()) if str(path).startswith("http") else None
    if payload is None:
        payload = _read_json(Path(path))
    results: list[MaterializationResult] = []
    for item in payload["results"]:
        dataset_payload = item.get("dataset")
        dataset = None
        if dataset_payload is not None:
            dataset = PortfolioDatasetDef(
                dataset_id=dataset_payload["dataset_id"],
                label=dataset_payload["label"],
                source_id=dataset_payload["source_id"],
                snapshot_id=dataset_payload["snapshot_id"],
                role=dataset_payload["role"],
                responses_path=dataset_payload["responses_path"],
                items_path=dataset_payload["items_path"],
                models_path=dataset_payload["models_path"],
                base_config=dataset_payload["base_config"],
                base_stage_options=dataset_payload["base_stage_options"],
                notes=tuple(dataset_payload.get("notes", [])),
            )
        results.append(
            MaterializationResult(
                source_id=item["source_id"],
                snapshot_id=item["snapshot_id"],
                role=item["role"],
                status=item["status"],
                manifest_path=item["manifest_path"],
                dataset=dataset,
                bundle_dir=item.get("bundle_dir"),
                skip_reason=item.get("skip_reason"),
                details=item.get("details", {}),
            )
        )
    return results


class OLLBV1LocalAdapter:
    """Materialize a reduced pinned extract from the existing local OLLB v1 export."""

    adapter_id = "ollb_v1_local"

    def materialize(
        self,
        *,
        source: BenchmarkSourceSpec,
        snapshot: SnapshotSpec,
        out_dir: Path,
    ) -> MaterializationResult:
        source_path = Path(snapshot.source_locator)
        if not source_path.exists():
            return _skipped_result(
                source=source,
                snapshot=snapshot,
                out_dir=out_dir,
                reason=f"local source export is missing: {source_path}",
            )

        frame = pd.read_parquet(
            source_path,
            columns=[BENCHMARK_ID, MODEL_ID, ITEM_ID, SCORE],
        )
        native_benchmarks = sorted(frame[BENCHMARK_ID].astype("string").unique().tolist())
        benchmark_counts = (
            frame[[MODEL_ID, BENCHMARK_ID]]
            .drop_duplicates()
            .groupby(MODEL_ID)[BENCHMARK_ID]
            .nunique()
        )
        complete_models = benchmark_counts.loc[
            benchmark_counts == len(native_benchmarks)
        ].index.astype("string").tolist()
        selected_models = stable_sample(
            complete_models,
            max_count=180,
            salt=f"{source.source_id}::{snapshot.snapshot_id}::models",
        )
        if len(selected_models) < 100:
            return _skipped_result(
                source=source,
                snapshot=snapshot,
                out_dir=out_dir,
                reason=(
                    "fewer than 100 complete-overlap models were available after applying the "
                    "pinned local export filter"
                ),
                details={"complete_overlap_model_count": len(complete_models)},
            )

        filtered_frames: list[pd.DataFrame] = []
        selected_item_counts: dict[str, int] = {}
        for native_benchmark_id in native_benchmarks:
            benchmark_frame = frame.loc[
                frame[BENCHMARK_ID].astype("string") == native_benchmark_id
            ].copy()
            benchmark_frame = benchmark_frame.loc[
                benchmark_frame[MODEL_ID].astype("string").isin(selected_models)
            ].copy()
            selected_items = _top_items_by_coverage(
                benchmark_frame,
                max_count=60,
                salt=f"{source.source_id}::{snapshot.snapshot_id}::{native_benchmark_id}::items",
            )
            selected_item_counts[native_benchmark_id] = len(selected_items)
            filtered_frames.append(
                benchmark_frame.loc[
                    benchmark_frame[ITEM_ID].astype("string").isin(selected_items)
                ].copy()
            )

        reduced = pd.concat(filtered_frames, ignore_index=True)
        reduced = _attach_prefixed_ids(
            reduced,
            source_id=source.source_id,
            snapshot_id=snapshot.snapshot_id,
            native_benchmark_column=BENCHMARK_ID,
            native_item_column=ITEM_ID,
        )
        tables = _build_bundle_tables(
            responses_long=reduced,
            dataset=_build_dataset_def(
                source=source,
                snapshot=snapshot,
                bundle_out_dir=out_dir / source.source_id / snapshot.snapshot_id,
                base_config={
                    "allow_low_n": False,
                    "min_models_per_benchmark": 100,
                    "warn_models_per_benchmark": 140,
                    "min_items_after_filtering": 40,
                    "min_models_per_item": 40,
                    "min_overlap_models_for_joint": 100,
                    "min_overlap_models_for_redundancy": 100,
                    "random_seed": 7,
                },
                base_stage_options=_portfolio_stage_options(k_preselect=50, k_final=24),
                notes=(
                    "reduced pinned extract from the local release-default source bundle",
                ),
            ),
            details={
                "native_benchmark_count": len(native_benchmarks),
                "selected_model_count": len(selected_models),
                "selected_item_counts": selected_item_counts,
                "source_path": str(source_path.resolve()),
            },
        )
        return _write_materialized_result(
            source=source,
            snapshot=snapshot,
            out_dir=out_dir,
            tables=tables,
        )


class OpenEvalObjectiveAdapter:
    """Materialize a reduced objective-only OpenEval extract."""

    adapter_id = "openeval_objective"
    _min_models_per_benchmark = 20

    def materialize(
        self,
        *,
        source: BenchmarkSourceSpec,
        snapshot: SnapshotSpec,
        out_dir: Path,
    ) -> MaterializationResult:
        dataset_meta = http_get_json("https://huggingface.co/api/datasets/human-centered-eval/OpenEval")
        base_url = "https://huggingface.co/datasets/human-centered-eval/OpenEval/resolve/main"
        bench_frame = pd.read_parquet(f"{base_url}/bench/train-00000-of-00001.parquet")
        item_frame = pd.read_parquet(f"{base_url}/item/train-00000-of-00001.parquet")
        response_files = [
            sibling["rfilename"]
            for sibling in dataset_meta.get("siblings", [])
            if sibling["rfilename"].startswith("response/")
            and sibling["rfilename"].endswith(".parquet")
        ]
        if not response_files:
            return _skipped_result(
                source=source,
                snapshot=snapshot,
                out_dir=out_dir,
                reason="no public response parquet shards were found for OpenEval",
            )

        bench_tags = {
            normalize_slug(row.benchmark_name): list(row.benchmark_tags)
            for row in bench_frame.itertuples(index=False)
        }
        item_lookup = {
            str(row.item_id): {
                "native_benchmark_id": _openeval_benchmark_from_item_id(str(row.item_id)),
                "item_content": row.item_content,
            }
            for row in item_frame.itertuples(index=False)
        }

        rows: list[dict[str, Any]] = []
        exclusion_counts: dict[str, int] = {}
        for response_file in sorted(response_files):
            shard = pd.read_parquet(f"{base_url}/{response_file}")
            for row in shard.itertuples(index=False):
                model_name = str(row.model["name"])
                item_id = _openeval_item_id_from_response_id(str(row.response_id), model_name)
                item_payload = item_lookup.get(item_id)
                if item_payload is None:
                    exclusion_counts["missing_item_lookup"] = (
                        exclusion_counts.get("missing_item_lookup", 0) + 1
                    )
                    continue
                native_benchmark_id = item_payload["native_benchmark_id"]
                if native_benchmark_id not in _OPENEVAL_BENCHMARK_ALLOWLIST:
                    exclusion_counts["unsupported_benchmark"] = (
                        exclusion_counts.get("unsupported_benchmark", 0) + 1
                    )
                    continue
                metric = _select_openeval_metric(row.scores)
                if metric is None:
                    exclusion_counts["no_binary_metric"] = (
                        exclusion_counts.get("no_binary_metric", 0) + 1
                    )
                    continue
                metric_name, score = metric
                rows.append(
                    {
                        "source_id": source.source_id,
                        "snapshot_id": snapshot.snapshot_id,
                        "native_benchmark_id": native_benchmark_id,
                        "native_item_id": item_id,
                        "native_metric_name": metric_name,
                        "native_model_id": model_name,
                        MODEL_ID: model_name,
                        BENCHMARK_ID: prefixed_benchmark_id(
                            source.source_id,
                            snapshot.snapshot_id,
                            native_benchmark_id,
                        ),
                        ITEM_ID: prefixed_item_id(
                            source.source_id,
                            snapshot.snapshot_id,
                            native_benchmark_id,
                            item_id,
                        ),
                        SCORE: score,
                    }
                )

        if not rows:
            return _skipped_result(
                source=source,
                snapshot=snapshot,
                out_dir=out_dir,
                reason="no objective-compatible binary response rows were recovered from OpenEval",
                details={"exclusions": exclusion_counts},
            )

        responses_long = pd.DataFrame.from_records(rows)
        responses_long, duplicate_details = _deduplicate_binary_responses(responses_long)
        benchmark_coverage = (
            responses_long[[MODEL_ID, BENCHMARK_ID]]
            .drop_duplicates()
            .groupby(MODEL_ID)[BENCHMARK_ID]
            .nunique()
        )
        complete_models = benchmark_coverage.loc[
            benchmark_coverage == responses_long[BENCHMARK_ID].nunique()
        ].index.astype("string").tolist()
        if len(complete_models) < self._min_models_per_benchmark:
            selected_models = (
                benchmark_coverage.sort_values(ascending=False).head(40).index.tolist()
            )
            selection_note = (
                "relaxed to top-coverage models because fewer than 20 "
                "complete-overlap models were available"
            )
        else:
            selected_models = stable_sample(
                complete_models,
                max_count=40,
                salt=f"{source.source_id}::{snapshot.snapshot_id}::models",
            )
            selection_note = "used complete-overlap models only"
        responses_long = responses_long.loc[
            responses_long[MODEL_ID].astype("string").isin(selected_models)
        ].copy()
        benchmark_model_support = (
            responses_long[[MODEL_ID, BENCHMARK_ID]]
            .drop_duplicates()
            .groupby(BENCHMARK_ID)[MODEL_ID]
            .nunique()
        )
        low_support_benchmarks = (
            benchmark_model_support.loc[
                benchmark_model_support < self._min_models_per_benchmark
            ]
            .index.astype("string")
            .tolist()
        )
        if low_support_benchmarks:
            exclusion_counts["low_model_support_benchmarks"] = len(low_support_benchmarks)
            responses_long = responses_long.loc[
                ~responses_long[BENCHMARK_ID].astype("string").isin(low_support_benchmarks)
            ].copy()
            selection_note = (
                selection_note + "; dropped low-support benchmarks after model selection"
            )
        if responses_long.empty:
            return _skipped_result(
                source=source,
                snapshot=snapshot,
                out_dir=out_dir,
                reason=(
                    "no OpenEval benchmarks retained enough models after enforcing the "
                    "portfolio support floor"
                ),
                details={"exclusions": exclusion_counts},
            )

        benchmark_coverage = (
            responses_long[[MODEL_ID, BENCHMARK_ID]]
            .drop_duplicates()
            .groupby(MODEL_ID)[BENCHMARK_ID]
            .nunique()
        )
        complete_models = benchmark_coverage.loc[
            benchmark_coverage == responses_long[BENCHMARK_ID].nunique()
        ].index.astype("string").tolist()
        if len(complete_models) >= self._min_models_per_benchmark:
            selected_models = stable_sample(
                complete_models,
                max_count=40,
                salt=f"{source.source_id}::{snapshot.snapshot_id}::models_refined",
            )
            responses_long = responses_long.loc[
                responses_long[MODEL_ID].astype("string").isin(selected_models)
            ].copy()

        filtered_frames: list[pd.DataFrame] = []
        selected_item_counts: dict[str, int] = {}
        for benchmark_id, benchmark_frame in responses_long.groupby(BENCHMARK_ID, sort=True):
            selected_items = _top_items_by_coverage(
                benchmark_frame,
                max_count=50,
                salt=f"{source.source_id}::{snapshot.snapshot_id}::{benchmark_id}::items",
            )
            selected_item_counts[str(benchmark_id)] = len(selected_items)
            filtered_frames.append(
                benchmark_frame.loc[
                    benchmark_frame[ITEM_ID].astype("string").isin(selected_items)
                ].copy()
            )
        responses_long = pd.concat(filtered_frames, ignore_index=True)
        tables = _build_bundle_tables(
            responses_long=responses_long,
            dataset=_build_dataset_def(
                source=source,
                snapshot=snapshot,
                bundle_out_dir=out_dir / source.source_id / snapshot.snapshot_id,
                base_config={
                    "allow_low_n": True,
                    "min_models_per_benchmark": 20,
                    "warn_models_per_benchmark": 30,
                    "min_items_after_filtering": 25,
                    "min_models_per_item": 15,
                    "min_overlap_models_for_joint": 15,
                    "min_overlap_models_for_redundancy": 15,
                    "random_seed": 7,
                },
                base_stage_options=_portfolio_stage_options(k_preselect=40, k_final=18),
                notes=(
                    "objective-only OpenEval extract",
                    selection_note,
                ),
            ),
            details={
                "exclusions": exclusion_counts,
                "duplicate_resolution": duplicate_details,
                "benchmark_model_support": {
                    str(key): int(value)
                    for key, value in (
                        responses_long[[MODEL_ID, BENCHMARK_ID]]
                        .drop_duplicates()
                        .groupby(BENCHMARK_ID)[MODEL_ID]
                        .nunique()
                        .items()
                    )
                },
                "selected_model_count": len(selected_models),
                "selected_item_counts": selected_item_counts,
                "admitted_benchmarks": sorted(
                    responses_long["native_benchmark_id"].drop_duplicates().astype("string").tolist()
                ),
                "bench_tags": bench_tags,
            },
        )
        return _write_materialized_result(
            source=source,
            snapshot=snapshot,
            out_dir=out_dir,
            tables=tables,
        )


class HELMObjectiveAdapter:
    """Materialize a reduced objective-only HELM extract from the public bucket."""

    adapter_id = "helm_capabilities_objective"

    _scenario_specs = (
        {
            "native_benchmark_id": "mmlu_pro",
            "run_prefix": "mmlu_pro:subset=all,use_chain_of_thought=true,use_few_shot=false",
            "metric_name": "chain_of_thought_correctness",
        },
        {
            "native_benchmark_id": "gpqa",
            "run_prefix": "gpqa:subset=gpqa_main,use_chain_of_thought=true,use_few_shot=false",
            "metric_name": "chain_of_thought_correctness",
        },
        {
            "native_benchmark_id": "ifeval",
            "run_prefix": "ifeval",
            "metric_name": "ifeval_strict_accuracy",
        },
    )

    def materialize(
        self,
        *,
        source: BenchmarkSourceSpec,
        snapshot: SnapshotSpec,
        out_dir: Path,
    ) -> MaterializationResult:
        scenario_models: dict[str, list[str]] = {}
        for spec in self._scenario_specs:
            scenario_models[spec["native_benchmark_id"]] = _helm_list_models(
                run_prefix=spec["run_prefix"],
            )

        admitted_specs = [
            spec
            for spec in self._scenario_specs
            if scenario_models[spec["native_benchmark_id"]]
        ]
        if not admitted_specs:
            return _skipped_result(
                source=source,
                snapshot=snapshot,
                out_dir=out_dir,
                reason="no public HELM runs were discovered for the configured objective subset",
            )

        model_intersection = set(scenario_models[admitted_specs[0]["native_benchmark_id"]])
        for spec in admitted_specs[1:]:
            model_intersection &= set(scenario_models[spec["native_benchmark_id"]])
        selected_models = stable_sample(
            model_intersection,
            max_count=24,
            salt=f"{source.source_id}::{snapshot.snapshot_id}::models",
        )
        if len(selected_models) < 12:
            return _skipped_result(
                source=source,
                snapshot=snapshot,
                out_dir=out_dir,
                reason=(
                    "fewer than 12 shared models were available across the configured HELM "
                    "objective scenarios"
                ),
                details={"shared_model_count": len(model_intersection)},
            )

        rows: list[dict[str, Any]] = []
        exclusion_counts: dict[str, int] = {}
        selected_item_counts: dict[str, int] = {}
        admitted_benchmarks: list[str] = []
        for spec in admitted_specs:
            native_benchmark_id = spec["native_benchmark_id"]
            first_model = selected_models[0]
            instances = _helm_fetch_instances(
                run_prefix=spec["run_prefix"],
                model_name=first_model,
            )
            instance_ids = [str(instance["id"]) for instance in instances]
            sampled_instance_ids = stable_sample(
                instance_ids,
                max_count=50,
                salt=f"{source.source_id}::{snapshot.snapshot_id}::{native_benchmark_id}::items",
            )
            benchmark_rows: list[dict[str, Any]] = []
            non_binary_count = 0
            for model_name in selected_models:
                predictions = _helm_fetch_display_predictions(
                    run_prefix=spec["run_prefix"],
                    model_name=model_name,
                )
                for prediction in predictions:
                    instance_id = str(prediction["instance_id"])
                    if instance_id not in sampled_instance_ids:
                        continue
                    raw_value = prediction.get("stats", {}).get(spec["metric_name"])
                    score = coerce_binary_metric(raw_value)
                    if score is None:
                        non_binary_count += 1
                        continue
                    benchmark_rows.append(
                        {
                            "source_id": source.source_id,
                            "snapshot_id": snapshot.snapshot_id,
                            "native_benchmark_id": native_benchmark_id,
                            "native_item_id": instance_id,
                            "native_metric_name": spec["metric_name"],
                            "native_model_id": model_name,
                            MODEL_ID: model_name,
                            BENCHMARK_ID: prefixed_benchmark_id(
                                source.source_id,
                                snapshot.snapshot_id,
                                native_benchmark_id,
                            ),
                            ITEM_ID: prefixed_item_id(
                                source.source_id,
                                snapshot.snapshot_id,
                                native_benchmark_id,
                                instance_id,
                            ),
                            SCORE: score,
                        }
                    )
            if benchmark_rows and non_binary_count == 0:
                rows.extend(benchmark_rows)
                selected_item_counts[native_benchmark_id] = len(sampled_instance_ids)
                admitted_benchmarks.append(native_benchmark_id)
            else:
                exclusion_counts[f"{native_benchmark_id}_non_binary_metric"] = non_binary_count

        if not rows:
            return _skipped_result(
                source=source,
                snapshot=snapshot,
                out_dir=out_dir,
                reason="no HELM objective scenarios produced strict binary instance scores",
                details={"exclusions": exclusion_counts},
            )

        responses_long = pd.DataFrame.from_records(rows)
        tables = _build_bundle_tables(
            responses_long=responses_long,
            dataset=_build_dataset_def(
                source=source,
                snapshot=snapshot,
                bundle_out_dir=out_dir / source.source_id / snapshot.snapshot_id,
                base_config={
                    "allow_low_n": True,
                    "min_models_per_benchmark": 12,
                    "warn_models_per_benchmark": 18,
                    "min_items_after_filtering": 20,
                    "min_models_per_item": 10,
                    "min_overlap_models_for_joint": 10,
                    "min_overlap_models_for_redundancy": 10,
                    "random_seed": 7,
                },
                base_stage_options=_portfolio_stage_options(k_preselect=28, k_final=12),
                notes=(
                    "objective-only HELM extract",
                    "scenarios with non-binary per-instance metrics were excluded",
                ),
            ),
            details={
                "selected_model_count": len(selected_models),
                "selected_item_counts": selected_item_counts,
                "admitted_benchmarks": admitted_benchmarks,
                "exclusions": exclusion_counts,
            },
        )
        return _write_materialized_result(
            source=source,
            snapshot=snapshot,
            out_dir=out_dir,
            tables=tables,
        )


class OLLBV2InspectionAdapter:
    """Inspect public OLLB v2 surfaces and skip when item-level details are unavailable."""

    adapter_id = "ollb_v2_details"

    def materialize(
        self,
        *,
        source: BenchmarkSourceSpec,
        snapshot: SnapshotSpec,
        out_dir: Path,
    ) -> MaterializationResult:
        search_url = "https://huggingface.co/api/datasets?author=open-llm-leaderboard&search=details&limit=1"
        search_results = http_get_json(search_url)
        if not search_results:
            return _skipped_result(
                source=source,
                snapshot=snapshot,
                out_dir=out_dir,
                reason=(
                    "no public OLLB details datasets were discoverable via the HF "
                    "dataset search api"
                ),
            )

        dataset_id = search_results[0]["id"]
        splits_url = f"https://datasets-server.huggingface.co/splits?dataset={dataset_id}"
        try:
            http_get_json(splits_url)
        except Exception as exc:
            return _skipped_result(
                source=source,
                snapshot=snapshot,
                out_dir=out_dir,
                reason=(
                    "OLLB v2 aggregate results are public, but the required item-level details "
                    f"dataset was not accessible without authentication: {dataset_id}"
                ),
                details={"details_dataset_id": dataset_id, "error": str(exc)},
            )

        return _skipped_result(
            source=source,
            snapshot=snapshot,
            out_dir=out_dir,
            reason=(
                "an accessible item-level details dataset was detected, but a parser for that "
                "wire shape has not yet been implemented in this checkout"
            ),
            details={"details_dataset_id": dataset_id},
        )


class LiveCodeBenchInspectionAdapter:
    """Inspect official public LiveCodeBench datasets and skip when no response matrix exists."""

    adapter_id = "livecodebench_public_inspection"

    def materialize(
        self,
        *,
        source: BenchmarkSourceSpec,
        snapshot: SnapshotSpec,
        out_dir: Path,
    ) -> MaterializationResult:
        dataset_ids = (
            "livecodebench/execution",
            "livecodebench/execution-v2",
        )
        inspected_features: dict[str, list[str]] = {}
        for dataset_id in dataset_ids:
            splits = http_get_json(
                f"https://datasets-server.huggingface.co/splits?dataset={dataset_id}"
            )["splits"]
            if not splits:
                continue
            split = splits[0]
            first_rows = http_get_json(
                "https://datasets-server.huggingface.co/first-rows"
                f"?dataset={dataset_id}&config={split['config']}&split={split['split']}"
            )
            inspected_features[dataset_id] = [
                feature["name"] for feature in first_rows.get("features", [])
            ]

        return _skipped_result(
            source=source,
            snapshot=snapshot,
            out_dir=out_dir,
            reason=(
                "official public LiveCodeBench datasets expose benchmark items and execution "
                "artifacts, but not a public many-model item-response matrix"
            ),
            details={"inspected_features": inspected_features},
        )


class BelebeleInspectionAdapter:
    """Inspect official public Belebele data and skip when no response matrix exists."""

    adapter_id = "belebele_public_inspection"

    def materialize(
        self,
        *,
        source: BenchmarkSourceSpec,
        snapshot: SnapshotSpec,
        out_dir: Path,
    ) -> MaterializationResult:
        split_payload = http_get_json(
            "https://datasets-server.huggingface.co/first-rows"
            "?dataset=facebook/belebele&config=eng_Latn&split=test"
        )
        inspected_features = [feature["name"] for feature in split_payload.get("features", [])]
        return _skipped_result(
            source=source,
            snapshot=snapshot,
            out_dir=out_dir,
            reason=(
                "the official public Belebele dataset exposes benchmark items only; no public "
                "many-model item-response bank was configured for BenchIQ ingestion"
            ),
            details={"inspected_features": inspected_features},
        )


def _adapter_registry() -> dict[str, PortfolioSourceAdapter]:
    return {
        OLLBV1LocalAdapter.adapter_id: OLLBV1LocalAdapter(),
        OLLBV2InspectionAdapter.adapter_id: OLLBV2InspectionAdapter(),
        OpenEvalObjectiveAdapter.adapter_id: OpenEvalObjectiveAdapter(),
        HELMObjectiveAdapter.adapter_id: HELMObjectiveAdapter(),
        LiveCodeBenchInspectionAdapter.adapter_id: LiveCodeBenchInspectionAdapter(),
        BelebeleInspectionAdapter.adapter_id: BelebeleInspectionAdapter(),
    }


def _build_bundle_tables(
    *,
    responses_long: pd.DataFrame,
    dataset: PortfolioDatasetDef,
    details: dict[str, Any],
) -> BundleTables:
    responses = responses_long.copy()
    responses[MODEL_ID] = responses[MODEL_ID].astype("string")
    responses[BENCHMARK_ID] = responses[BENCHMARK_ID].astype("string")
    responses[ITEM_ID] = responses[ITEM_ID].astype("string")
    responses[SCORE] = responses[SCORE].astype("Int8")
    items = (
        responses[[BENCHMARK_ID, ITEM_ID]]
        .drop_duplicates()
        .sort_values([BENCHMARK_ID, ITEM_ID])
        .reset_index(drop=True)
    )
    items["content_hash"] = items[ITEM_ID].astype("string").map(stable_hash).astype("string")
    models = (
        responses[[MODEL_ID]]
        .drop_duplicates()
        .sort_values([MODEL_ID])
        .reset_index(drop=True)
    )
    models["model_family"] = models[MODEL_ID].astype("string").map(infer_model_family)
    return BundleTables(
        responses_long=responses,
        items=items,
        models=models,
        dataset=dataset,
        details=details,
    )


def _write_materialized_result(
    *,
    source: BenchmarkSourceSpec,
    snapshot: SnapshotSpec,
    out_dir: Path,
    tables: BundleTables,
) -> MaterializationResult:
    bundle_dir = out_dir / source.source_id / snapshot.snapshot_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    responses_path = write_parquet(tables.responses_long, bundle_dir / "responses_long.parquet")
    items_path = write_parquet(tables.items, bundle_dir / "items.parquet")
    models_path = write_parquet(tables.models, bundle_dir / "models.parquet")

    load_bundle(
        responses_path,
        items_path,
        models_path,
        config=tables.dataset.base_config,
    )

    counts = {
        "rows": int(len(tables.responses_long.index)),
        "benchmark_count": int(tables.responses_long[BENCHMARK_ID].nunique()),
        "model_count": int(tables.responses_long[MODEL_ID].nunique()),
        "item_count": int(tables.responses_long[ITEM_ID].nunique()),
    }
    manifest_payload = {
        "portfolio_version": PORTFOLIO_VERSION,
        "source_id": source.source_id,
        "snapshot_id": snapshot.snapshot_id,
        "role": snapshot.role,
        "status": "materialized",
        "paths": {
            "responses_long": ensure_relative_path(responses_path, base=bundle_dir),
            "items": ensure_relative_path(items_path, base=bundle_dir),
            "models": ensure_relative_path(models_path, base=bundle_dir),
        },
        "counts": counts,
        "dataset": tables.dataset.to_dict(),
        "details": tables.details,
    }
    manifest_path = _write_status_manifest(bundle_dir=bundle_dir, payload=manifest_payload)
    return MaterializationResult(
        source_id=source.source_id,
        snapshot_id=snapshot.snapshot_id,
        role=snapshot.role,
        status="materialized",
        manifest_path=str(manifest_path),
        dataset=tables.dataset,
        bundle_dir=str(bundle_dir),
        details={"counts": counts, **tables.details},
    )


def _skipped_result(
    *,
    source: BenchmarkSourceSpec,
    snapshot: SnapshotSpec,
    out_dir: Path,
    reason: str,
    details: dict[str, Any] | None = None,
) -> MaterializationResult:
    bundle_dir = out_dir / source.source_id / snapshot.snapshot_id
    manifest_path = _write_status_manifest(
        bundle_dir=bundle_dir,
        payload={
            "portfolio_version": PORTFOLIO_VERSION,
            "source_id": source.source_id,
            "snapshot_id": snapshot.snapshot_id,
            "role": snapshot.role,
            "status": "skipped",
            "skip_reason": reason,
            "details": details or {},
        },
    )
    return MaterializationResult(
        source_id=source.source_id,
        snapshot_id=snapshot.snapshot_id,
        role=snapshot.role,
        status="skipped",
        manifest_path=str(manifest_path),
        bundle_dir=str(bundle_dir),
        skip_reason=reason,
        details=details or {},
    )


def _write_status_manifest(*, bundle_dir: Path, payload: dict[str, Any]) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return write_json(payload, bundle_dir / "source_manifest.json")


def _build_dataset_def(
    *,
    source: BenchmarkSourceSpec,
    snapshot: SnapshotSpec,
    bundle_out_dir: Path,
    base_config: dict[str, Any],
    base_stage_options: dict[str, dict[str, Any]],
    notes: tuple[str, ...],
) -> PortfolioDatasetDef:
    return PortfolioDatasetDef(
        dataset_id=f"{source.source_id}__{snapshot.snapshot_id}",
        label=f"{source.label} ({snapshot.label})",
        source_id=source.source_id,
        snapshot_id=snapshot.snapshot_id,
        role=snapshot.role,
        responses_path=str((bundle_out_dir / "responses_long.parquet").resolve()),
        items_path=str((bundle_out_dir / "items.parquet").resolve()),
        models_path=str((bundle_out_dir / "models.parquet").resolve()),
        base_config=base_config,
        base_stage_options=base_stage_options,
        notes=notes,
    )


def _portfolio_stage_options(*, k_preselect: int, k_final: int) -> dict[str, dict[str, Any]]:
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
        "07_theta": {"theta_grid_size": 151},
        "09_reconstruct": {
            "lam_grid": [0.1, 1.0],
            "cv_folds": 5,
            "n_splines": 8,
        },
        "10_redundancy": {
            "lam_grid": [0.1, 1.0],
            "cv_folds": 5,
            "n_splines": 8,
            "n_factors_to_try": [1, 2],
        },
    }


def _attach_prefixed_ids(
    frame: pd.DataFrame,
    *,
    source_id: str,
    snapshot_id: str,
    native_benchmark_column: str,
    native_item_column: str,
) -> pd.DataFrame:
    attached = frame.copy()
    attached["native_benchmark_id"] = attached[native_benchmark_column].astype("string")
    attached["native_item_id"] = attached[native_item_column].astype("string")
    attached["native_model_id"] = attached[MODEL_ID].astype("string")
    attached[BENCHMARK_ID] = attached["native_benchmark_id"].map(
        lambda native_id: prefixed_benchmark_id(source_id, snapshot_id, native_id)
    )
    attached[ITEM_ID] = attached.apply(
        lambda row: prefixed_item_id(
            source_id,
            snapshot_id,
            str(row["native_benchmark_id"]),
            str(row["native_item_id"]),
        ),
        axis=1,
    )
    return attached


def _top_items_by_coverage(
    frame: pd.DataFrame,
    *,
    max_count: int,
    salt: str,
) -> list[str]:
    coverage = (
        frame[[ITEM_ID, MODEL_ID]]
        .drop_duplicates()
        .groupby(ITEM_ID)[MODEL_ID]
        .nunique()
        .sort_values(ascending=False)
    )
    decorated = [
        (-int(count), stable_hash(f"{salt}::{item_id}"), str(item_id))
        for item_id, count in coverage.items()
    ]
    decorated.sort()
    return [item_id for _, _, item_id in decorated[:max_count]]


def _deduplicate_binary_responses(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    group_columns = [MODEL_ID, BENCHMARK_ID, ITEM_ID]
    duplicate_mask = frame.duplicated(group_columns, keep=False)
    if not duplicate_mask.any():
        return frame, {"duplicate_rows": 0, "conflicting_keys": 0}

    kept_rows: list[pd.DataFrame] = []
    conflicting_keys = 0
    duplicate_rows = int(duplicate_mask.sum())
    grouped = frame.groupby(group_columns, sort=False, dropna=False)
    for _, group in grouped:
        if len(group.index) == 1:
            kept_rows.append(group)
            continue
        if group[SCORE].nunique(dropna=True) == 1:
            kept_rows.append(group.head(1))
        else:
            conflicting_keys += 1

    deduped = pd.concat(kept_rows, ignore_index=True)
    return deduped, {
        "duplicate_rows": duplicate_rows,
        "conflicting_keys": conflicting_keys,
    }


def _openeval_benchmark_from_item_id(item_id: str) -> str:
    prefix = item_id.split("_", 1)[0]
    return normalize_slug(prefix)


def _openeval_item_id_from_response_id(response_id: str, model_name: str) -> str:
    marker = f"_{model_name}_"
    if marker in response_id:
        return response_id.rsplit(marker, 1)[0]
    return response_id.rsplit("_", 1)[0]


def _select_openeval_metric(scores_payload: Any) -> tuple[str, int] | None:
    if not isinstance(scores_payload, dict):
        return None
    metric_entries = list(scores_payload.get("metric", []))
    values = list(scores_payload.get("value", []))
    candidates: list[tuple[str, Any]] = []
    for metric_entry, value in zip(metric_entries, values, strict=False):
        if not isinstance(metric_entry, dict):
            continue
        candidates.append((str(metric_entry.get("name", "")), value))
    return _choose_binary_metric(candidates, policy=_OPENEVAL_METRIC_POLICY)


def _choose_binary_metric(
    candidates: list[tuple[str, Any]],
    *,
    policy: BinaryMetricPolicy,
) -> tuple[str, int] | None:
    ranked: list[tuple[int, str, int]] = []
    for metric_name, raw_value in candidates:
        lowered = metric_name.lower()
        if any(fragment in lowered for fragment in policy.rejected_fragments):
            continue
        score = coerce_binary_metric(raw_value)
        if score is None and policy.allow_numeric_binary:
            score = coerce_binary_metric(float(raw_value)) if raw_value is not None else None
        if score is None:
            continue
        rank = 99
        for index, fragment in enumerate(policy.preferred_fragments):
            if fragment in lowered:
                rank = index
                break
        ranked.append((rank, metric_name, score))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    _, metric_name, score = ranked[0]
    return metric_name, score


def _helm_list_models(*, run_prefix: str) -> list[str]:
    prefix = f"capabilities/benchmark_output/runs/v1.0.0/{run_prefix}"
    page_token: str | None = None
    models: set[str] = set()
    while True:
        url = f"https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o?prefix={prefix}"
        if page_token is not None:
            url = f"{url}&pageToken={page_token}"
        payload = http_get_json(url)
        for item in payload.get("items", []):
            name = str(item.get("name", ""))
            if not name.endswith("/display_predictions.json"):
                continue
            run_id = name.rsplit("/", 1)[0].rsplit("/", 1)[-1]
            if "model=" in run_id:
                models.add(run_id.rsplit("model=", 1)[1])
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return sorted(models)


def _helm_fetch_instances(*, run_prefix: str, model_name: str) -> list[dict[str, Any]]:
    run_id = f"{run_prefix},model={model_name}" if "model=" not in run_prefix else run_prefix
    if run_prefix == "ifeval":
        run_id = f"ifeval:model={model_name}"
    path = (
        "capabilities/benchmark_output/runs/v1.0.0/"
        f"{run_id}/instances.json"
    )
    url = (
        "https://storage.googleapis.com/download/storage/v1/b/crfm-helm-public/o/"
        f"{gcs_api_path(path)}?alt=media"
    )
    return http_get_json(url)


def _helm_fetch_display_predictions(*, run_prefix: str, model_name: str) -> list[dict[str, Any]]:
    run_id = f"{run_prefix},model={model_name}" if "model=" not in run_prefix else run_prefix
    if run_prefix == "ifeval":
        run_id = f"ifeval:model={model_name}"
    path = (
        "capabilities/benchmark_output/runs/v1.0.0/"
        f"{run_id}/display_predictions.json"
    )
    url = (
        "https://storage.googleapis.com/download/storage/v1/b/crfm-helm-public/o/"
        f"{gcs_api_path(path)}?alt=media"
    )
    return http_get_json(url)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
