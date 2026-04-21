from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchiq.portfolio.materialize import (
    HELMObjectiveAdapter,
    OLLBV1LocalAdapter,
    OpenEvalObjectiveAdapter,
    _choose_binary_metric,
    _deduplicate_binary_responses,
    _openeval_item_id_from_response_id,
)
from benchiq.portfolio.specs import BenchmarkSourceSpec, BinaryMetricPolicy, SnapshotSpec
from benchiq.portfolio.utils import infer_model_family, prefixed_benchmark_id, prefixed_item_id


def test_prefixed_ids_and_model_family_helpers_are_stable() -> None:
    assert (
        prefixed_benchmark_id("ollb_v1_metabench_source", "release_1", "MMLU-Pro")
        == "ollb_v1_metabench_source__release_1__mmlu_pro"
    )
    assert (
        prefixed_item_id("helm_objective", "v1", "gpqa", "id0")
        == "helm_objective__v1__gpqa__id0"
    )
    assert infer_model_family("meta-llama/Meta-Llama-3-8B") == "meta-llama"
    assert infer_model_family("amazon_nova-lite-v1:0") == "amazon_nova-lite-v1"


def test_choose_binary_metric_prefers_supported_objective_metrics() -> None:
    policy = BinaryMetricPolicy(
        preferred_fragments=("strict_accuracy", "exact_match", "accuracy", "acc"),
        rejected_fragments=("rouge", "judge"),
    )
    metric = _choose_binary_metric(
        [
            ("rougeL", 1.0),
            ("exact_match", 1.0),
            ("acc", 0.0),
        ],
        policy=policy,
    )
    assert metric == ("exact_match", 1)


def test_openeval_response_id_parser_and_duplicate_resolution() -> None:
    assert (
        _openeval_item_id_from_response_id(
            "boolq_20260305T211125Z_3_gemma-2b-it_0",
            "gemma-2b-it",
        )
        == "boolq_20260305T211125Z_3"
    )

    frame = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m2", "m2"],
            "benchmark_id": ["b1", "b1", "b1", "b1"],
            "item_id": ["i1", "i1", "i2", "i2"],
            "score": [1, 1, 0, 1],
        }
    )
    deduped, details = _deduplicate_binary_responses(frame)
    assert len(deduped.index) == 1
    assert details["duplicate_rows"] == 4
    assert details["conflicting_keys"] == 1


def test_ollb_v1_adapter_materializes_local_extract(tmp_path: Path) -> None:
    source_path = tmp_path / "ollb_v1.parquet"
    rows = []
    benchmark_ids = [f"bench_{index}" for index in range(6)]
    model_ids = [f"model_{index:03d}" for index in range(100)]
    for benchmark_id in benchmark_ids:
        for item_index in range(60):
            item_id = f"{benchmark_id}_item_{item_index:03d}"
            for model_index, model_id in enumerate(model_ids):
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "item_id": item_id,
                        "model_id": model_id,
                        "score": int((item_index + model_index) % 2 == 0),
                    }
                )
    pd.DataFrame.from_records(rows).to_parquet(source_path, index=False)

    source = BenchmarkSourceSpec(
        source_id="ollb_v1_metabench_source",
        label="OLLB v1",
        adapter_id="ollb_v1_local",
        role="optimize",
        snapshots=(),
    )
    snapshot = SnapshotSpec(
        snapshot_id="test_snapshot",
        label="test",
        release="test",
        source_locator=str(source_path),
        role="optimize",
    )
    result = OLLBV1LocalAdapter().materialize(source=source, snapshot=snapshot, out_dir=tmp_path)
    assert result.status == "materialized"
    assert result.dataset is not None
    assert result.dataset.base_stage_options["04_subsample"]["k_preselect"] == 44
    assert result.dataset.base_stage_options["06_select"]["k_final"] == 44
    assert (tmp_path / source.source_id / snapshot.snapshot_id / "responses_long.parquet").exists()


def test_openeval_and_helm_adapters_materialize_from_mocked_public_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_open = BenchmarkSourceSpec(
        source_id="openeval",
        label="OpenEval",
        adapter_id="openeval_objective",
        role="optimize",
        snapshots=(),
    )
    snapshot_open = SnapshotSpec(
        snapshot_id="test_snapshot",
        label="test",
        release="test",
        source_locator="https://huggingface.co/datasets/human-centered-eval/OpenEval",
        role="optimize",
    )

    bench_frame = pd.DataFrame(
        {
            "benchmark_name": ["boolq", "cnndm", "mmlu_pro"],
            "benchmark_tags": [
                ["question_answering"],
                ["summarization", "news"],
                ["multiple_choice"],
            ],
        }
    )
    item_frame = pd.DataFrame(
        {
            "item_id": ["boolq_20260305_0", "mmlu-pro_20260305_1"],
            "item_metadata": [
                {"source": "boolq"},
                {"source": "mmlu_pro"},
            ],
            "item_content": [
                {"input": ["q1"]},
                {"input": ["q2"]},
            ],
            "schema_version": ["v0.1.0", "v0.1.0"],
        }
    )
    response_rows = []
    model_names = [f"model-{index:02d}" for index in range(20)]
    for index, model_name in enumerate(model_names):
        response_rows.append(
            {
                "response_id": f"boolq_20260305_0_{model_name}_0",
                "model": {"name": model_name},
                "item_adaptation": {},
                "response_content": [],
                "scores": {"metric": [{"name": "accuracy"}], "value": [float(index % 2 == 0)]},
            }
        )
        response_rows.append(
            {
                "response_id": f"mmlu-pro_20260305_1_{model_name}_0",
                "model": {"name": model_name},
                "item_adaptation": {},
                "response_content": [],
                "scores": {
                    "metric": [{"name": "exact_match"}],
                    "value": [float(index % 2 == 1)],
                },
            }
        )
    response_frame = pd.DataFrame.from_records(response_rows)

    def fake_http_get_json(url: str):
        if "api/datasets/human-centered-eval/OpenEval" in url:
            return {
                "siblings": [
                    {"rfilename": "response/train-00000-of-00001.parquet"},
                ]
            }
        raise AssertionError(f"unexpected url: {url}")

    original_read_parquet = pd.read_parquet

    def fake_read_parquet(path, *args, **kwargs):
        path_str = str(path)
        if path_str.endswith("/bench/train-00000-of-00001.parquet"):
            return bench_frame.copy()
        if path_str.endswith("/item/train-00000-of-00001.parquet"):
            return item_frame.copy()
        if path_str.endswith("response/train-00000-of-00001.parquet"):
            return response_frame.copy()
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr("benchiq.portfolio.materialize.http_get_json", fake_http_get_json)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    open_result = OpenEvalObjectiveAdapter().materialize(
        source=source_open,
        snapshot=snapshot_open,
        out_dir=tmp_path,
    )
    assert open_result.status == "materialized"
    assert open_result.dataset is not None
    assert open_result.dataset.base_stage_options["04_subsample"]["k_preselect"] == 40
    assert open_result.dataset.base_stage_options["06_select"]["k_final"] == 26

    source_helm = BenchmarkSourceSpec(
        source_id="helm_objective",
        label="HELM",
        adapter_id="helm_capabilities_objective",
        role="optimize",
        snapshots=(),
    )
    snapshot_helm = SnapshotSpec(
        snapshot_id="test_snapshot",
        label="test",
        release="test",
        source_locator="gs://crfm-helm-public/capabilities/benchmark_output/releases/v1.0.0",
        role="optimize",
    )

    monkeypatch.setattr(
        "benchiq.portfolio.materialize._helm_list_models",
        lambda run_prefix: [
            "model_a",
            "model_b",
            "model_c",
            "model_d",
            "model_e",
            "model_f",
            "model_g",
            "model_h",
            "model_i",
            "model_j",
            "model_k",
            "model_l",
        ],
    )
    monkeypatch.setattr(
        "benchiq.portfolio.materialize._helm_fetch_instances",
        lambda run_prefix, model_name: [{"id": f"id{index}"} for index in range(60)],
    )
    monkeypatch.setattr(
        "benchiq.portfolio.materialize._helm_fetch_display_predictions",
        lambda run_prefix, model_name: [
            {
                "instance_id": f"id{index}",
                "stats": {
                    "chain_of_thought_correctness": int(index % 2 == 0),
                    "ifeval_strict_accuracy": 0.5,
                },
            }
            for index in range(60)
        ],
    )
    helm_result = HELMObjectiveAdapter().materialize(
        source=source_helm,
        snapshot=snapshot_helm,
        out_dir=tmp_path,
    )
    assert helm_result.status == "materialized"
    assert helm_result.dataset is not None
    assert helm_result.dataset.base_stage_options["04_subsample"]["k_preselect"] == 24
    assert helm_result.dataset.base_stage_options["06_select"]["k_final"] == 22
