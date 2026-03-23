import pandas as pd
import pytest

from benchiq.schema.checks import (
    SchemaValidationError,
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
    with pytest.raises(
        SchemaValidationError,
        match=rf"{table_name} is missing required columns: {missing_column}",
    ):
        coercer(frame)


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


def test_duplicate_policy_error_raises() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "benchmark_id": ["b1", "b1"],
            "item_id": ["i1", "i1"],
            "score": [0, 1],
        },
    )

    with pytest.raises(
        SchemaValidationError,
        match="responses_long contains duplicate primary keys under duplicate_policy='error'",
    ):
        coerce_responses_long(frame, duplicate_policy="error")


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

    assert coerced.shape == (1, 4)
    assert coerced.loc[0, "score"] == 1
    assert len(report.warnings) == 1


def test_items_and_models_require_unique_primary_keys() -> None:
    items_frame = pd.DataFrame({"benchmark_id": ["b1", "b1"], "item_id": ["i1", "i1"]})
    models_frame = pd.DataFrame({"model_id": ["m1", "m1"]})

    with pytest.raises(SchemaValidationError, match="items contains duplicate primary keys"):
        coerce_items_table(items_frame)

    with pytest.raises(SchemaValidationError, match="models contains duplicate primary keys"):
        coerce_models_table(models_frame)
