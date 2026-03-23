"""BenchIQ package bootstrap."""

from benchiq.config import BenchIQConfig
from benchiq.schema.checks import (
    SchemaValidationError,
    ValidationCounts,
    ValidationIssue,
    ValidationReport,
)
from benchiq.validate import validate

__all__ = [
    "__version__",
    "BenchIQConfig",
    "SchemaValidationError",
    "ValidationCounts",
    "ValidationIssue",
    "ValidationReport",
    "validate",
]

__version__ = "0.1.0a0"
