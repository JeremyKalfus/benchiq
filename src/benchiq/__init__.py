"""BenchIQ package bootstrap."""

from benchiq.calibration import CalibrationResult, calibrate
from benchiq.config import BenchIQConfig
from benchiq.deployment import (
    CalibrationBenchmarkSpec,
    LoadedCalibrationBundle,
    PredictionResult,
    load_calibration_bundle,
    predict,
)
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
    "CalibrationBenchmarkSpec",
    "CalibrationResult",
    "LoadedCalibrationBundle",
    "PredictionResult",
    "SchemaValidationError",
    "RunResult",
    "ValidationCounts",
    "ValidationIssue",
    "ValidationReport",
    "calibrate",
    "load_bundle",
    "load_calibration_bundle",
    "predict",
    "run",
    "validate",
]

__version__ = "0.1.0a0"
