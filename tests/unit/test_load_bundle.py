import json

import pandas as pd
import pytest

import benchiq
from benchiq.schema.checks import SchemaValidationError
from benchiq.schema.tables import RESPONSES_PRIMARY_KEY


def test_load_bundle_from_csv_writes_stage00_artifacts_and_manifest(tmp_path) -> None:
    responses_path = tmp_path / "responses.csv"
    pd.DataFrame(
        {
            "model_id": [" m1 ", "m2"],
            "benchmark_id": [" b1", "b1 "],
            "item_id": [" i1 ", "i1"],
            "score": [1, 0],
            "weight": [0.5, None],
        },
    ).to_csv(responses_path, index=False)

    out_dir = tmp_path / "out"
    bundle = benchiq.load_bundle(responses_path, out_dir=out_dir, run_id="toy-run")

    assert isinstance(bundle, benchiq.Bundle)
    assert bundle.responses_long.shape == (2, 5)
    assert bundle.items.shape == (1, 2)
    assert bundle.models.shape == (2, 1)
    assert bundle.responses_long["model_id"].tolist() == ["m1", "m2"]
    assert bundle.responses_long["benchmark_id"].tolist() == ["b1", "b1"]
    assert bundle.responses_long["item_id"].tolist() == ["i1", "i1"]

    stage_dir = out_dir / "toy-run" / "artifacts" / "00_canonical"
    assert (stage_dir / "responses_long.parquet").exists()
    assert (stage_dir / "items.parquet").exists()
    assert (stage_dir / "models.parquet").exists()
    assert (stage_dir / "canonicalization_report.json").exists()

    manifest = json.loads((out_dir / "toy-run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "toy-run"
    assert manifest["dependency_versions"]["pandas"]
    assert manifest["input_hashes"]["responses_long"]
    assert manifest["input_sources"]["items"]["derived"] is True
    assert manifest["resolved_config"]["duplicate_policy"] == "error"

    roundtrip = benchiq.load_bundle(
        stage_dir / "responses_long.parquet",
        items_path=stage_dir / "items.parquet",
        models_path=stage_dir / "models.parquet",
    )
    assert roundtrip.responses_long.shape == bundle.responses_long.shape
    assert roundtrip.items.shape == bundle.items.shape
    assert roundtrip.models.shape == bundle.models.shape
    assert not roundtrip.responses_long.duplicated(list(RESPONSES_PRIMARY_KEY)).any()


def test_load_bundle_supports_parquet_inputs(tmp_path) -> None:
    responses_path = tmp_path / "responses.parquet"
    items_path = tmp_path / "items.parquet"
    models_path = tmp_path / "models.parquet"

    pd.DataFrame(
        {
            "model_id": ["m1"],
            "benchmark_id": ["b1"],
            "item_id": ["i1"],
            "score": [1],
        },
    ).to_parquet(responses_path, index=False)
    pd.DataFrame({"benchmark_id": ["b1"], "item_id": ["i1"]}).to_parquet(items_path, index=False)
    pd.DataFrame({"model_id": ["m1"], "model_family": ["family-a"]}).to_parquet(
        models_path,
        index=False,
    )

    bundle = benchiq.load_bundle(responses_path, items_path=items_path, models_path=models_path)

    assert bundle.responses_long.shape == (1, 4)
    assert bundle.items.shape == (1, 2)
    assert bundle.models.shape == (1, 2)
    assert bundle.sources["responses_long"].file_format == "parquet"


def test_load_bundle_duplicate_failure_is_explicit_and_structured(tmp_path) -> None:
    responses_path = tmp_path / "responses.csv"
    pd.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "benchmark_id": ["b1", "b1"],
            "item_id": ["i1", "i1"],
            "score": [1, 0],
        },
    ).to_csv(responses_path, index=False)

    with pytest.raises(
        SchemaValidationError, match="responses_long failed schema validation"
    ) as exc:
        benchiq.load_bundle(responses_path)

    assert exc.value.report is not None
    assert exc.value.report.errors[0].code == "duplicate_primary_keys"
    assert exc.value.report.errors[0].context["duplicate_keys"] == 1


def test_load_bundle_non_binary_scores_fail_with_structured_error(tmp_path) -> None:
    responses_path = tmp_path / "responses.csv"
    pd.DataFrame(
        {
            "model_id": ["m1"],
            "benchmark_id": ["b1"],
            "item_id": ["i1"],
            "score": [2],
        },
    ).to_csv(responses_path, index=False)

    with pytest.raises(
        SchemaValidationError, match="responses_long failed schema validation"
    ) as exc:
        benchiq.load_bundle(responses_path)

    assert exc.value.report is not None
    assert exc.value.report.errors[0].code == "invalid_score_values"
    assert "pre-score your rubric into 0/1" in exc.value.report.errors[0].message
