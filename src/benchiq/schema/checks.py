"""Schema coercion and validation checks for BenchIQ."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from benchiq.schema.tables import (
    DEFAULT_DTYPES,
    ITEMS_PRIMARY_KEY,
    MODELS_PRIMARY_KEY,
    REQUIRED_COLUMNS,
    RESPONSES_PRIMARY_KEY,
    SCORE,
    STRING_COLUMNS,
    WEIGHT,
    DuplicatePolicy,
    TableName,
)


class SchemaValidationError(ValueError):
    """Raised when a table fails schema validation."""


@dataclass(slots=True)
class ValidationIssue:
    """Structured validation issue."""

    level: Literal["warning", "error"]
    code: str
    message: str
    table_name: str | None = None
    row_count: int | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationReport:
    """Accumulated validation issues for a table or config check."""

    warnings: list[ValidationIssue] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)
    table_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_warning(
        self,
        *,
        code: str,
        message: str,
        table_name: str | None = None,
        row_count: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.warnings.append(
            ValidationIssue(
                level="warning",
                code=code,
                message=message,
                table_name=table_name,
                row_count=row_count,
                context=context or {},
            ),
        )

    def add_error(
        self,
        *,
        code: str,
        message: str,
        table_name: str | None = None,
        row_count: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.errors.append(
            ValidationIssue(
                level="error",
                code=code,
                message=message,
                table_name=table_name,
                row_count=row_count,
                context=context or {},
            ),
        )


def coerce_responses_long(
    frame: pd.DataFrame,
    *,
    duplicate_policy: DuplicatePolicy = "error",
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and coerce the canonical long response table."""

    coerced, report = _coerce_table(frame, table_name="responses_long")
    coerced = _coerce_score_column(coerced)

    if WEIGHT in coerced.columns:
        coerced[WEIGHT] = pd.to_numeric(coerced[WEIGHT], errors="raise").astype("Float64")

    duplicate_mask = coerced.duplicated(list(RESPONSES_PRIMARY_KEY), keep=False)
    if duplicate_mask.any():
        duplicate_rows = int(duplicate_mask.sum())
        duplicate_keys = int(
            coerced.loc[duplicate_mask, list(RESPONSES_PRIMARY_KEY)].drop_duplicates().shape[0],
        )
        if duplicate_policy == "error":
            raise SchemaValidationError(
                "responses_long contains duplicate primary keys under duplicate_policy='error'",
            )
        keep = "first" if duplicate_policy == "first_write_wins" else "last"
        coerced = coerced.drop_duplicates(list(RESPONSES_PRIMARY_KEY), keep=keep).reset_index(
            drop=True,
        )
        report.add_warning(
            code="duplicate_primary_key_resolved",
            message=f"resolved duplicate primary keys with duplicate_policy='{duplicate_policy}'",
            table_name="responses_long",
            row_count=duplicate_rows,
            context={"duplicate_keys": duplicate_keys},
        )

    report.table_shapes["responses_long"] = coerced.shape
    return coerced, report


def coerce_items_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and coerce the items table."""

    coerced, report = _coerce_table(frame, table_name="items")
    _raise_for_duplicate_primary_keys(coerced, ITEMS_PRIMARY_KEY, table_name="items")
    report.table_shapes["items"] = coerced.shape
    return coerced, report


def coerce_models_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and coerce the models table."""

    coerced, report = _coerce_table(frame, table_name="models")
    _raise_for_duplicate_primary_keys(coerced, MODELS_PRIMARY_KEY, table_name="models")
    report.table_shapes["models"] = coerced.shape
    return coerced, report


def _coerce_table(
    frame: pd.DataFrame,
    *,
    table_name: TableName,
) -> tuple[pd.DataFrame, ValidationReport]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{table_name} must be a pandas DataFrame")

    _raise_for_missing_columns(frame, table_name=table_name)

    coerced = frame.copy()
    report = ValidationReport()

    for column in STRING_COLUMNS[table_name]:
        if column in coerced.columns:
            coerced[column] = coerced[column].astype("string").str.strip()

    for column, dtype in DEFAULT_DTYPES[table_name].items():
        if column not in coerced.columns or column == SCORE or column == WEIGHT:
            continue
        coerced[column] = coerced[column].astype(dtype)

    return coerced.reset_index(drop=True), report


def _raise_for_missing_columns(frame: pd.DataFrame, *, table_name: TableName) -> None:
    missing_columns = sorted(set(REQUIRED_COLUMNS[table_name]) - set(frame.columns))
    if missing_columns:
        missing_string = ", ".join(missing_columns)
        raise SchemaValidationError(f"{table_name} is missing required columns: {missing_string}")


def _coerce_score_column(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_score = pd.to_numeric(frame[SCORE], errors="raise")
    valid_mask = numeric_score.isna() | numeric_score.isin([0, 1])
    if not valid_mask.all():
        raise SchemaValidationError("responses_long.score must contain only 0, 1, or null values")
    frame[SCORE] = pd.array(numeric_score, dtype="Int8")
    return frame


def _raise_for_duplicate_primary_keys(
    frame: pd.DataFrame,
    primary_key: tuple[str, ...],
    *,
    table_name: str,
) -> None:
    duplicate_mask = frame.duplicated(list(primary_key), keep=False)
    if duplicate_mask.any():
        raise SchemaValidationError(f"{table_name} contains duplicate primary keys")
