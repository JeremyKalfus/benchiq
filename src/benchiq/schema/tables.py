"""Canonical table constants for BenchIQ."""

from __future__ import annotations

from typing import Final, Literal

MODEL_ID: Final = "model_id"
BENCHMARK_ID: Final = "benchmark_id"
ITEM_ID: Final = "item_id"
SCORE: Final = "score"
SPLIT: Final = "split"
WEIGHT: Final = "weight"
CONTENT_HASH: Final = "content_hash"
MODEL_FAMILY: Final = "model_family"

TableName = Literal["responses_long", "items", "models"]
DuplicatePolicy = Literal["error", "first_write_wins", "last_write_wins"]

TABLE_NAMES: Final[tuple[TableName, ...]] = ("responses_long", "items", "models")

RESPONSES_PRIMARY_KEY: Final[tuple[str, ...]] = (MODEL_ID, BENCHMARK_ID, ITEM_ID)
ITEMS_PRIMARY_KEY: Final[tuple[str, ...]] = (BENCHMARK_ID, ITEM_ID)
MODELS_PRIMARY_KEY: Final[tuple[str, ...]] = (MODEL_ID,)

REQUIRED_COLUMNS: Final[dict[TableName, tuple[str, ...]]] = {
    "responses_long": (*RESPONSES_PRIMARY_KEY, SCORE),
    "items": ITEMS_PRIMARY_KEY,
    "models": MODELS_PRIMARY_KEY,
}

STRING_COLUMNS: Final[dict[TableName, tuple[str, ...]]] = {
    "responses_long": (*RESPONSES_PRIMARY_KEY, SPLIT),
    "items": (BENCHMARK_ID, ITEM_ID, CONTENT_HASH),
    "models": (MODEL_ID, MODEL_FAMILY),
}

DEFAULT_DTYPES: Final[dict[TableName, dict[str, str]]] = {
    "responses_long": {
        MODEL_ID: "string",
        BENCHMARK_ID: "string",
        ITEM_ID: "string",
        SCORE: "Int8",
        SPLIT: "string",
        WEIGHT: "Float64",
    },
    "items": {
        BENCHMARK_ID: "string",
        ITEM_ID: "string",
        CONTENT_HASH: "string",
    },
    "models": {
        MODEL_ID: "string",
        MODEL_FAMILY: "string",
    },
}
