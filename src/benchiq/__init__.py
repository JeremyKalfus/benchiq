"""BenchIQ package bootstrap."""

from benchiq.config import BenchIQConfig
from benchiq.schema.checks import SchemaValidationError, ValidationIssue, ValidationReport

__all__ = [
    "__version__",
    "BenchIQConfig",
    "SchemaValidationError",
    "ValidationIssue",
    "ValidationReport",
]

__version__ = "0.1.0a0"
