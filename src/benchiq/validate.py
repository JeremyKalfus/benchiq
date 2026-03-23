"""Public validation entrypoint for canonical in-memory bundles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchiq.config import BenchIQConfig
from benchiq.schema.checks import (
    ValidationReport,
    coerce_items_table,
    coerce_models_table,
    coerce_responses_long,
)
from benchiq.schema.tables import TABLE_NAMES


def validate(
    bundle: Any,
    config: BenchIQConfig | Mapping[str, Any] | None = None,
) -> ValidationReport:
    """Validate the canonical bundle tables and return a structured report."""

    resolved_config = BenchIQConfig.model_validate({} if config is None else config)
    report = ValidationReport()
    for table_name in TABLE_NAMES:
        table = _get_bundle_table(bundle, table_name)
        if table is None:
            report.add_error(
                code="missing_required_table",
                message=f"bundle is missing required table: {table_name}",
                table_name=table_name,
            )
            continue

        if table_name == "responses_long":
            _, table_report = coerce_responses_long(
                table,
                duplicate_policy=resolved_config.duplicate_policy,
            )
        elif table_name == "items":
            _, table_report = coerce_items_table(table)
        else:
            _, table_report = coerce_models_table(table)

        report.extend(table_report)

    return report


def _get_bundle_table(bundle: Any, table_name: str) -> Any | None:
    if isinstance(bundle, Mapping):
        return bundle.get(table_name)

    table_mapping = getattr(bundle, "tables", None)
    if isinstance(table_mapping, Mapping):
        return table_mapping.get(table_name)

    return getattr(bundle, table_name, None)
