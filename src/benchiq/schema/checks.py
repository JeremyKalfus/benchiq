"""Schema coercion and validation checks for BenchIQ."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

    def __init__(self, message: str, *, report: "ValidationReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(slots=True)
class ValidationIssue:
    """Structured validation issue."""

    level: Literal["warning", "error"]
    code: str
    message: str
    table_name: str | None = None
    row_count: int | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class ValidationCounts:
    """Structured count totals for a validation report."""

    warning_count: int
    error_count: int
    table_count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class ValidationReport:
    """Accumulated validation issues for a table or config check."""

    warnings: list[ValidationIssue] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)
    table_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def counts(self) -> ValidationCounts:
        return ValidationCounts(
            warning_count=len(self.warnings),
            error_count=len(self.errors),
            table_count=len(self.table_shapes),
        )

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "warning_count": self.counts.warning_count,
            "error_count": self.counts.error_count,
            "table_count": self.counts.table_count,
            "tables": {
                table_name: {"rows": rows, "columns": columns}
                for table_name, (rows, columns) in sorted(self.table_shapes.items())
            },
        }

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

    def extend(self, other: "ValidationReport") -> None:
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        self.table_shapes.update(other.table_shapes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "counts": self.counts.to_dict(),
            "summary": self.summary,
            "warnings": [issue.to_dict() for issue in self.warnings],
            "errors": [issue.to_dict() for issue in self.errors],
        }


def coerce_responses_long(
    frame: Any,
    *,
    duplicate_policy: DuplicatePolicy = "error",
) -> tuple[pd.DataFrame | None, ValidationReport]:
    """Validate and coerce the canonical long response table."""

    coerced, report = _coerce_table(frame, table_name="responses_long")
    if coerced is None:
        return None, report

    coerced = _coerce_score_column(coerced, report)
    if coerced is None:
        return None, report

    if WEIGHT in coerced.columns:
        try:
            coerced[WEIGHT] = pd.to_numeric(coerced[WEIGHT], errors="raise").astype("Float64")
        except (TypeError, ValueError) as exc:
            report.add_error(
                code="invalid_weight_values",
                message="responses_long.weight must be numeric when provided",
                table_name="responses_long",
                context={"exception": str(exc)},
            )
            return None, report

    duplicate_mask = coerced.duplicated(list(RESPONSES_PRIMARY_KEY), keep=False)
    if duplicate_mask.any():
        duplicate_rows = int(duplicate_mask.sum())
        duplicate_key_frame = coerced.loc[
            duplicate_mask, list(RESPONSES_PRIMARY_KEY)
        ].drop_duplicates()
        duplicate_keys = int(duplicate_key_frame.shape[0])
        affected_keys = _serialize_duplicate_keys(duplicate_key_frame)
        if duplicate_policy == "error":
            report.add_error(
                code="duplicate_primary_keys",
                message=(
                    "responses_long contains duplicate primary keys under duplicate_policy='error'"
                ),
                table_name="responses_long",
                row_count=duplicate_rows,
                context={"duplicate_keys": duplicate_keys, "affected_keys": affected_keys},
            )
            return None, report

        keep = "first" if duplicate_policy == "first_write_wins" else "last"
        coerced = coerced.drop_duplicates(list(RESPONSES_PRIMARY_KEY), keep=keep).reset_index(
            drop=True,
        )
        report.add_warning(
            code="duplicate_primary_key_resolved",
            message=f"resolved duplicate primary keys with duplicate_policy='{duplicate_policy}'",
            table_name="responses_long",
            row_count=duplicate_rows,
            context={"duplicate_keys": duplicate_keys, "affected_keys": affected_keys},
        )
        if duplicate_rows / len(frame.index) > 0.001:
            report.add_warning(
                code="excessive_duplicates",
                message="responses_long has duplicates in more than 0.1% of rows",
                table_name="responses_long",
                row_count=duplicate_rows,
                context={
                    "duplicate_keys": duplicate_keys,
                    "affected_keys": affected_keys,
                    "duplicate_ratio": duplicate_rows / len(frame.index),
                },
            )

    report.table_shapes["responses_long"] = coerced.shape
    return coerced, report


def coerce_items_table(frame: Any) -> tuple[pd.DataFrame | None, ValidationReport]:
    """Validate and coerce the items table."""

    coerced, report = _coerce_table(frame, table_name="items")
    if coerced is None:
        return None, report

    if not _check_for_duplicate_primary_keys(
        coerced,
        ITEMS_PRIMARY_KEY,
        table_name="items",
        report=report,
    ):
        return None, report

    report.table_shapes["items"] = coerced.shape
    return coerced, report


def coerce_models_table(frame: Any) -> tuple[pd.DataFrame | None, ValidationReport]:
    """Validate and coerce the models table."""

    coerced, report = _coerce_table(frame, table_name="models")
    if coerced is None:
        return None, report

    if not _check_for_duplicate_primary_keys(
        coerced,
        MODELS_PRIMARY_KEY,
        table_name="models",
        report=report,
    ):
        return None, report

    report.table_shapes["models"] = coerced.shape
    return coerced, report


def _coerce_table(
    frame: Any,
    *,
    table_name: TableName,
) -> tuple[pd.DataFrame | None, ValidationReport]:
    report = ValidationReport()

    if not isinstance(frame, pd.DataFrame):
        report.add_error(
            code="not_a_dataframe",
            message=f"{table_name} must be a pandas DataFrame",
            table_name=table_name,
        )
        return None, report

    report.table_shapes[table_name] = frame.shape

    missing_columns = sorted(set(REQUIRED_COLUMNS[table_name]) - set(frame.columns))
    if missing_columns:
        report.add_error(
            code="missing_required_columns",
            message=f"{table_name} is missing required columns: {', '.join(missing_columns)}",
            table_name=table_name,
            context={"missing_columns": missing_columns},
        )
        return None, report

    coerced = frame.copy()
    for column in STRING_COLUMNS[table_name]:
        if column in coerced.columns:
            coerced[column] = coerced[column].astype("string").str.strip()

    for column, dtype in DEFAULT_DTYPES[table_name].items():
        if column not in coerced.columns or column == SCORE or column == WEIGHT:
            continue
        coerced[column] = coerced[column].astype(dtype)

    required_key_columns = {
        "responses_long": RESPONSES_PRIMARY_KEY,
        "items": ITEMS_PRIMARY_KEY,
        "models": MODELS_PRIMARY_KEY,
    }[table_name]
    if not _check_for_missing_key_values(
        coerced,
        required_key_columns,
        table_name=table_name,
        report=report,
    ):
        return None, report

    return coerced.reset_index(drop=True), report


def _coerce_score_column(
    frame: pd.DataFrame,
    report: ValidationReport,
) -> pd.DataFrame | None:
    try:
        numeric_score = pd.to_numeric(frame[SCORE], errors="raise")
    except (TypeError, ValueError) as exc:
        report.add_error(
            code="invalid_score_values",
            message=(
                "responses_long.score must contain only numeric 0, 1, or null values; "
                "v0.1 requires dichotomous item scores, so pre-score your rubric into 0/1"
            ),
            table_name="responses_long",
            context={"exception": str(exc)},
        )
        return None

    valid_mask = numeric_score.isna() | numeric_score.isin([0, 1])
    if not valid_mask.all():
        report.add_error(
            code="invalid_score_values",
            message=(
                "responses_long.score must contain only 0, 1, or null values; "
                "v0.1 requires dichotomous item scores, so pre-score your rubric into 0/1"
            ),
            table_name="responses_long",
            row_count=int((~valid_mask).sum()),
        )
        return None

    frame[SCORE] = pd.array(numeric_score, dtype="Int8")
    return frame


def _check_for_missing_key_values(
    frame: pd.DataFrame,
    key_columns: tuple[str, ...],
    *,
    table_name: str,
    report: ValidationReport,
) -> bool:
    invalid_columns: dict[str, int] = {}
    invalid_mask = pd.Series(False, index=frame.index, dtype="boolean")
    for column in key_columns:
        column_mask = frame[column].isna()
        if pd.api.types.is_string_dtype(frame[column].dtype):
            column_mask = column_mask | frame[column].eq("").fillna(False)
        if bool(column_mask.any()):
            invalid_columns[column] = int(column_mask.sum())
            invalid_mask = invalid_mask | column_mask

    if invalid_columns:
        report.add_error(
            code="null_required_key_values",
            message=(
                f"{table_name} contains null or blank values in required key columns: "
                f"{', '.join(sorted(invalid_columns))}"
            ),
            table_name=table_name,
            row_count=int(invalid_mask.sum()),
            context={"invalid_columns": invalid_columns},
        )
        return False

    return True


def _check_for_duplicate_primary_keys(
    frame: pd.DataFrame,
    primary_key: tuple[str, ...],
    *,
    table_name: str,
    report: ValidationReport,
) -> bool:
    duplicate_mask = frame.duplicated(list(primary_key), keep=False)
    if duplicate_mask.any():
        duplicate_key_frame = frame.loc[duplicate_mask, list(primary_key)].drop_duplicates()
        report.add_error(
            code="duplicate_primary_keys",
            message=f"{table_name} contains duplicate primary keys",
            table_name=table_name,
            row_count=int(duplicate_mask.sum()),
            context={
                "duplicate_keys": int(duplicate_key_frame.shape[0]),
                "affected_keys": _serialize_duplicate_keys(duplicate_key_frame),
            },
        )
        return False

    return True


def _serialize_duplicate_keys(frame: pd.DataFrame, *, limit: int = 25) -> list[dict[str, Any]]:
    return frame.head(limit).to_dict(orient="records")
