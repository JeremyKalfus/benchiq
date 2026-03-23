import pandas as pd
import pytest

from benchiq.schema.checks import (
    ValidationCounts,
    coerce_items_table,
    coerce_models_table,
    coerce_responses_long,
)


@pytest.mark.parametrize(
    ("table_name", "coercer", "frame", "missing_column"),
    [
        (
            "responses_long",
            coerce_responses_long,
            pd.DataFrame({"model_id": ["m1"], "benchmark_id": ["b1"], "score": [1]}),
            "item_id",
        ),
        (
            "items",
            coerce_items_table,
            pd.DataFrame({"benchmark_id": ["b1"]}),
            "item_id",
        ),
        (
            "models",
            coerce_models_table,
            pd.DataFrame({"model_family": ["family"]}),
            "model_id",
        ),
    ],
)
def test_required_columns_are_enforced(table_name, coercer, frame, missing_column) -> None:
    coerced, report = coercer(frame)

    assert coerced is None
    assert not report.ok
    assert len(report.errors) == 1
    assert report.errors[0].code == "missing_required_columns"
    assert report.errors[0].table_name == table_name
    assert report.errors[0].context["missing_columns"] == [missing_column]
    assert report.errors[0].message == f"{table_name} is missing required columns: {missing_column}"


def test_responses_long_coerces_ids_and_score_dtypes() -> None:
    frame = pd.DataFrame(
        {
            "model_id": [101, 102],
            "benchmark_id": [" bench-a ", "bench-a"],
            "item_id": [" item-1 ", "item-2"],
            "score": [True, None],
            "weight": [0.5, None],
        },
    )

    coerced, report = coerce_responses_long(frame)

    assert coerced is not None
    assert str(coerced["model_id"].dtype) == "string"
    assert str(coerced["benchmark_id"].dtype) == "string"
    assert str(coerced["item_id"].dtype) == "string"
    assert str(coerced["score"].dtype) == "Int8"
    assert str(coerced["weight"].dtype) == "Float64"
    assert coerced.loc[0, "benchmark_id"] == "bench-a"
    assert coerced.loc[0, "item_id"] == "item-1"
    assert coerced.loc[0, "score"] == 1
    assert pd.isna(coerced.loc[1, "score"])
    assert report.ok


def test_duplicate_policy_error_returns_structured_error() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "benchmark_id": ["b1", "b1"],
            "item_id": ["i1", "i1"],
            "score": [0, 1],
        },
    )

    coerced, report = coerce_responses_long(frame, duplicate_policy="error")

    assert coerced is None
    assert not report.ok
    assert len(report.errors) == 1
    assert report.errors[0].code == "duplicate_primary_keys"
    assert report.errors[0].row_count == 2
    assert report.errors[0].context["duplicate_keys"] == 1


def test_duplicate_policy_first_write_wins_keeps_first_row() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "benchmark_id": ["b1", "b1"],
            "item_id": ["i1", "i1"],
            "score": [0, 1],
        },
    )

    coerced, report = coerce_responses_long(frame, duplicate_policy="first_write_wins")

    assert coerced is not None
    assert coerced.shape == (1, 4)
    assert coerced.loc[0, "score"] == 0
    assert len(report.warnings) == 1
    assert report.warnings[0].context["duplicate_keys"] == 1


def test_duplicate_policy_last_write_wins_keeps_last_row() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "benchmark_id": ["b1", "b1"],
            "item_id": ["i1", "i1"],
            "score": [0, 1],
        },
    )

    coerced, report = coerce_responses_long(frame, duplicate_policy="last_write_wins")

    assert coerced is not None
    assert coerced.shape == (1, 4)
    assert coerced.loc[0, "score"] == 1
    assert len(report.warnings) == 1


def test_items_and_models_require_unique_primary_keys() -> None:
    items_frame = pd.DataFrame({"benchmark_id": ["b1", "b1"], "item_id": ["i1", "i1"]})
    models_frame = pd.DataFrame({"model_id": ["m1", "m1"]})

    items_coerced, items_report = coerce_items_table(items_frame)
    models_coerced, models_report = coerce_models_table(models_frame)

    assert items_coerced is None
    assert not items_report.ok
    assert items_report.errors[0].code == "duplicate_primary_keys"
    assert items_report.errors[0].table_name == "items"

    assert models_coerced is None
    assert not models_report.ok
    assert models_report.errors[0].code == "duplicate_primary_keys"
    assert models_report.errors[0].table_name == "models"


def test_invalid_scores_return_structured_errors() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["m1", "m2"],
            "benchmark_id": ["b1", "b1"],
            "item_id": ["i1", "i2"],
            "score": [0, 2],
        },
    )

    coerced, report = coerce_responses_long(frame)

    assert coerced is None
    assert not report.ok
    assert len(report.errors) == 1
    assert report.errors[0].code == "invalid_score_values"
    assert report.errors[0].row_count == 1


@pytest.mark.parametrize(
    ("table_name", "coercer", "frame", "invalid_columns"),
    [
        (
            "responses_long",
            coerce_responses_long,
            pd.DataFrame(
                {
                    "model_id": [pd.NA],
                    "benchmark_id": ["b1"],
                    "item_id": ["i1"],
                    "score": [1],
                },
            ),
            {"model_id": 1},
        ),
        (
            "items",
            coerce_items_table,
            pd.DataFrame({"benchmark_id": ["   "], "item_id": ["i1"]}),
            {"benchmark_id": 1},
        ),
        (
            "models",
            coerce_models_table,
            pd.DataFrame({"model_id": [None]}),
            {"model_id": 1},
        ),
    ],
)
def test_required_key_columns_reject_null_or_blank_values(
    table_name,
    coercer,
    frame,
    invalid_columns,
) -> None:
    coerced, report = coercer(frame)

    assert coerced is None
    assert not report.ok
    assert report.errors[0].code == "null_required_key_values"
    assert report.errors[0].table_name == table_name
    assert report.errors[0].row_count == 1
    assert report.errors[0].context["invalid_columns"] == invalid_columns


def test_validation_report_exposes_counts_and_summary() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "benchmark_id": ["b1", "b1"],
            "item_id": ["i1", "i1"],
            "score": [0, 1],
        },
    )

    coerced, report = coerce_responses_long(frame, duplicate_policy="first_write_wins")

    assert coerced is not None
    assert report.counts == ValidationCounts(warning_count=1, error_count=0, table_count=1)
    assert report.summary == {
        "ok": True,
        "warning_count": 1,
        "error_count": 0,
        "table_count": 1,
        "tables": {
            "responses_long": {"rows": 1, "columns": 4},
        },
    }
