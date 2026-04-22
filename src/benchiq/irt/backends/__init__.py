"""IRT backend adapters and backend selection for BenchIQ."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from benchiq.irt.backends.bayes_backend import (
    BayesMCMC2PLResult,
    build_bayes_observed_responses,
    fit_bayes_mcmc_2pl,
)
from benchiq.irt.backends.common import (
    SUPPORTED_IRT_BACKENDS,
    IRT2PLResult,
    IRTBackendDependencyError,
    UnknownIRTBackendError,
    normalize_backend_name,
)
from benchiq.irt.backends.girth_backend import (
    Girth2PLResult,
    build_girth_response_matrix,
    fit_girth_2pl,
)


def fit_irt_backend(
    responses_long: pd.DataFrame,
    *,
    benchmark_id: str,
    item_ids: Sequence[str],
    model_ids: Sequence[str],
    backend: str = "girth",
    options: dict[str, Any] | None = None,
) -> IRT2PLResult:
    """Fit one normalized stage-05 backend by name."""

    resolved_backend = normalize_backend_name(backend)
    if resolved_backend == "girth":
        return fit_girth_2pl(
            responses_long,
            benchmark_id=benchmark_id,
            item_ids=item_ids,
            model_ids=model_ids,
            options=options,
        )
    if resolved_backend == "bayes_mcmc":
        return fit_bayes_mcmc_2pl(
            responses_long,
            benchmark_id=benchmark_id,
            item_ids=item_ids,
            model_ids=model_ids,
            options=options,
        )
    raise UnknownIRTBackendError(f"unsupported stage-05 IRT backend: {backend!r}")


__all__ = [
    "BayesMCMC2PLResult",
    "Girth2PLResult",
    "IRT2PLResult",
    "IRTBackendDependencyError",
    "SUPPORTED_IRT_BACKENDS",
    "UnknownIRTBackendError",
    "build_bayes_observed_responses",
    "build_girth_response_matrix",
    "fit_bayes_mcmc_2pl",
    "fit_girth_2pl",
    "fit_irt_backend",
    "normalize_backend_name",
]
