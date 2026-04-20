"""Public BenchIQ package surface.

BenchIQ exposes four high-level workflows at the package root:

- ``validate(...)`` for schema checks and canonicalization-only runs
- ``calibrate(...)`` to fit and export a reusable ``calibration_bundle/``
- ``predict(...)`` for deployment-time scoring from a saved bundle
- ``run(...)`` for the historical full end-to-end local pipeline

Package-level helpers for the calibration / deployment path are also public:

- ``load_calibration_bundle(...)`` to validate and inspect a saved bundle
- ``deploy(...)`` as a readability alias for ``predict(...)``
- ``public_workflows()`` as a quick discovery helper from ``import benchiq``
"""

from __future__ import annotations

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
from benchiq.profiles import (
    BenchIQProfile,
    build_psychometric_default_profile,
    build_reconstruction_first_profile,
    load_profile,
    product_profiles,
)
from benchiq.runner import BenchIQRunner, RunResult, run
from benchiq.schema.checks import (
    SchemaValidationError,
    ValidationCounts,
    ValidationIssue,
    ValidationReport,
)
from benchiq.validate import validate

deploy = predict

_PUBLIC_WORKFLOWS = {
    "validate": "canonicalize inputs, run schema checks, and write validation artifacts",
    "calibrate": "fit the reusable calibration stack and export calibration_bundle/",
    "predict": "score new reduced responses from a saved calibration bundle",
    "deploy": "alias for predict() when you want deployment wording in python code",
    "run": "execute the full end-to-end pipeline, including secondary redundancy analysis",
    "reconstruction_first": "load the default reconstruction-first product profile",
}


def public_workflows() -> dict[str, str]:
    """Return the supported package-level workflows and when to use them."""

    return dict(_PUBLIC_WORKFLOWS)


__all__ = [
    "__version__",
    "BenchIQConfig",
    "Bundle",
    "BundleSource",
    "BenchIQRunner",
    "BenchIQProfile",
    "CalibrationBenchmarkSpec",
    "CalibrationResult",
    "LoadedCalibrationBundle",
    "PredictionResult",
    "SchemaValidationError",
    "RunResult",
    "ValidationCounts",
    "ValidationIssue",
    "ValidationReport",
    "build_psychometric_default_profile",
    "build_reconstruction_first_profile",
    "calibrate",
    "deploy",
    "load_bundle",
    "load_calibration_bundle",
    "load_profile",
    "predict",
    "product_profiles",
    "public_workflows",
    "run",
    "validate",
]

__version__ = "0.1.0a0"
