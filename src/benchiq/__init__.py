"""BenchIQ package bootstrap."""

from benchiq.config import BenchIQConfig
from benchiq.io import Bundle, BundleSource, load_bundle
from benchiq.runner import BenchIQRunner, RunResult, run
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
    "Bundle",
    "BundleSource",
    "BenchIQRunner",
    "SchemaValidationError",
    "RunResult",
    "ValidationCounts",
    "ValidationIssue",
    "ValidationReport",
    "load_bundle",
    "run",
    "validate",
]

__version__ = "0.1.0a0"
