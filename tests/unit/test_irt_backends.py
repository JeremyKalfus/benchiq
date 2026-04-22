import builtins
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import benchiq.irt.backends.bayes_backend as bayes_backend
import benchiq.irt.fit as fit_module
from benchiq.irt.backends import (
    IRTBackendDependencyError,
    UnknownIRTBackendError,
    fit_irt_backend,
)
from benchiq.irt.backends.bayes_backend import BayesPosteriorSummary, fit_bayes_mcmc_2pl
from benchiq.irt.backends.common import IRT2PLResult


def _responses_long() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"benchmark_id": "b1", "item_id": "i1", "model_id": "m1", "score": 1},
            {"benchmark_id": "b1", "item_id": "i1", "model_id": "m2", "score": 0},
            {"benchmark_id": "b1", "item_id": "i2", "model_id": "m1", "score": 1},
            {"benchmark_id": "b1", "item_id": "i2", "model_id": "m2", "score": 1},
            {"benchmark_id": "b1", "item_id": "i3", "model_id": "m1", "score": 0},
        ]
    ).astype(
        {
            "benchmark_id": "string",
            "item_id": "string",
            "model_id": "string",
            "score": "Int64",
        }
    )


def _normalized_item_params(backend: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "benchmark_id": pd.Series(["b1", "b1"], dtype="string"),
            "item_id": pd.Series(["i1", "i3"], dtype="string"),
            "irt_backend": pd.Series([backend, backend], dtype="string"),
            "discrimination": pd.Series([1.0, 1.4], dtype="Float64"),
            "difficulty": pd.Series([-0.3, 0.7], dtype="Float64"),
            "pathology_warning": pd.Series([False, False], dtype=bool),
            "pathology_warning_reasons": pd.Series([[], []], dtype=object),
            "pathology_excluded": pd.Series([False, False], dtype=bool),
            "pathology_excluded_reasons": pd.Series([[], []], dtype=object),
        }
    )


def _empty_dropped_frame() -> pd.DataFrame:
    return _normalized_item_params("bayes_mcmc").iloc[0:0].copy()


def _ability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "benchmark_id": pd.Series(["b1", "b1"], dtype="string"),
            "model_id": pd.Series(["m1", "m2"], dtype="string"),
            "ability_eap": pd.Series([0.2, -0.1], dtype="Float64"),
        }
    )


def test_fit_irt_backend_rejects_unknown_backend() -> None:
    with pytest.raises(UnknownIRTBackendError, match="unsupported stage-05 IRT backend"):
        fit_irt_backend(
            _responses_long(),
            benchmark_id="b1",
            item_ids=["i1"],
            model_ids=["m1"],
            backend="not_a_backend",
        )


def test_fit_bayes_mcmc_2pl_raises_explicit_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(  # type: ignore[no-untyped-def]
        name,
        globals=None,
        locals=None,
        fromlist=(),
        level=0,
    ):
        if name in {"arviz", "pymc"}:
            raise ImportError(f"missing optional dependency: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(IRTBackendDependencyError, match=r"\.\[bayes\]"):
        fit_bayes_mcmc_2pl(
            _responses_long(),
            benchmark_id="b1",
            item_ids=["i1", "i2", "i3"],
            model_ids=["m1", "m2"],
        )


def test_fit_bayes_mcmc_2pl_normalizes_posterior_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_summary(**_: object) -> BayesPosteriorSummary:
        return BayesPosteriorSummary(
            discrimination_mean=np.array([0.9, 0.01, 1.5], dtype=float),
            difficulty_mean=np.array([-0.6, np.inf, 0.8], dtype=float),
            ability_mean=np.array([0.25, -0.15], dtype=float),
            diagnostics={
                "status_available": True,
                "status": "ok",
                "rhat_max": 1.003,
                "ess_bulk_min": 321.0,
                "divergence_count": 0,
                "warning_code": None,
                "warning_message": None,
            },
        )

    monkeypatch.setattr(bayes_backend, "_sample_posterior_summary", fake_summary)

    result = fit_bayes_mcmc_2pl(
        _responses_long(),
        benchmark_id="b1",
        item_ids=["i1", "i2", "i3"],
        model_ids=["m1", "m2"],
        options={"draws": 50, "tune": 25, "chains": 2, "random_seed": 7},
    )

    assert result.item_params["item_id"].tolist() == ["i1", "i3"]
    assert result.item_params["irt_backend"].astype(str).tolist() == ["bayes_mcmc", "bayes_mcmc"]
    assert result.dropped_pathological_items["item_id"].tolist() == ["i2"]
    assert list(result.ability_estimates.columns) == ["benchmark_id", "model_id", "ability_eap"]
    assert result.fit_report["irt_backend"] == "bayes_mcmc"
    assert result.fit_report["parameter_summary"] == "posterior_mean"
    assert result.fit_report["backend_options"]["draws"] == 50
    assert result.fit_report["convergence"]["status_available"] is True
    assert result.fit_report["convergence"]["status"] == "ok"
    assert result.fit_report["counts"]["pathology_excluded_count"] == 1
    assert result.fit_report["pathology"]["excluded_item_ids"] == ["i2"]
    assert result.fit_report["fit_metrics"]["aic"] is None
    assert result.fit_report["fit_metrics"]["bic"] is None


def test_fit_irt_bundle_threads_requested_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fit_irt_backend(  # type: ignore[no-untyped-def]
        responses_long,
        *,
        benchmark_id,
        item_ids,
        model_ids,
        backend,
        options=None,
    ):
        captured["benchmark_id"] = benchmark_id
        captured["item_ids"] = list(item_ids)
        captured["model_ids"] = list(model_ids)
        captured["backend"] = backend
        captured["options"] = options
        return IRT2PLResult(
            item_params=_normalized_item_params(backend),
            dropped_pathological_items=_empty_dropped_frame(),
            fit_report={
                "benchmark_id": benchmark_id,
                "irt_backend": backend,
                "model": "2pl",
                "parameter_summary": "posterior_mean",
                "skipped": False,
                "skipped_reason": None,
                "warnings": [],
                "backend_options": options or {},
                "convergence": {
                    "status": "ok",
                    "backend_exposes_flag": False,
                    "status_available": True,
                    "warning_code": None,
                },
                "counts": {
                    "train_model_count": len(model_ids),
                    "preselect_item_count": len(item_ids),
                    "valid_response_count": 5,
                    "missing_response_count": 1,
                    "pathology_warning_count": 0,
                    "pathology_excluded_count": 0,
                    "retained_item_count": len(item_ids),
                },
                "pathology": {
                    "warning_item_ids": [],
                    "excluded_item_ids": [],
                    "retained_item_ids": list(item_ids),
                    "exclusion_is_data_dependent": True,
                    "exclusion_note": "fixture",
                    "excluded_items": [],
                },
                "artifacts": {
                    "plots_written": False,
                    "plots_reason": "written_by_stage05_artifact_writer",
                    "dropped_pathological_items_written": False,
                    "item_parameter_scatter": None,
                },
                "fit_metrics": {
                    "runtime_seconds": 0.01,
                },
                "generated_at": "2026-04-21T00:00:00+00:00",
            },
            ability_estimates=_ability_frame(),
        )

    monkeypatch.setattr(fit_module, "fit_irt_backend", fake_fit_irt_backend)

    bundle = SimpleNamespace(
        responses_long=_responses_long(),
        manifest_path=None,
        run_id=None,
    )
    split_result = SimpleNamespace(
        per_benchmark_splits={
            "b1": pd.DataFrame(
                {
                    "model_id": pd.Series(["m1", "m2"], dtype="string"),
                    "split": pd.Series(["train", "train"], dtype="string"),
                }
            )
        }
    )
    subsample_result = SimpleNamespace(
        benchmarks={
            "b1": SimpleNamespace(
                preselect_items=pd.DataFrame(
                    {"item_id": pd.Series(["i1", "i3"], dtype="string")}
                ),
                subsample_report={"skipped": False, "skipped_reason": None},
            )
        }
    )

    result = fit_module.fit_irt_bundle(
        bundle,
        split_result,
        subsample_result,
        backend="bayes_mcmc",
        backend_options={"draws": 12},
    )

    assert captured["benchmark_id"] == "b1"
    assert captured["item_ids"] == ["i1", "i3"]
    assert captured["model_ids"] == ["m1", "m2"]
    assert captured["backend"] == "bayes_mcmc"
    assert captured["options"] == {"draws": 12}
    benchmark_result = result.benchmarks["b1"]
    assert benchmark_result.irt_fit_report["irt_backend"] == "bayes_mcmc"
    assert benchmark_result.irt_item_params["irt_backend"].astype(str).tolist() == [
        "bayes_mcmc",
        "bayes_mcmc",
    ]
