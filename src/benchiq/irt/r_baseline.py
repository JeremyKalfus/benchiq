"""Optional R-baseline comparison helpers for IRT parity work."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from girth.synthetic import create_synthetic_irt_dichotomous
from scipy.stats import pearsonr, spearmanr

from benchiq.io.write import write_json
from benchiq.irt.backends import fit_irt_backend
from benchiq.irt.theta import estimate_theta_responses
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID

DEFAULT_DISCRIMINATION = (0.7, 0.9, 1.1, 1.3, 1.6, 1.9)
DEFAULT_DIFFICULTY = (-1.4, -0.8, -0.2, 0.4, 0.9, 1.5)
DEFAULT_THETA_MIN = -4.0
DEFAULT_THETA_MAX = 4.0
DEFAULT_THETA_GRID_SIZE = 161
DEFAULT_OUTPUT_DIR = Path("reports") / "irt_r_baseline"
DEFAULT_PARITY_GATE_THRESHOLDS = {
    "theta_pearson_min": 0.95,
    "theta_spearman_min": 0.95,
    "icc_mean_rmse_max": 0.08,
}


@dataclass(slots=True)
class IRTBaselineComparisonResult:
    """Saved outputs from the optional R-baseline harness."""

    out_dir: Path
    summary_path: Path
    table_path: Path
    report: dict[str, Any]


def run_r_baseline_comparison(
    *,
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
    random_seed: int = 11,
    model_count: int = 300,
    theta_method: str = "EAP",
    difficulty: tuple[float, ...] = DEFAULT_DIFFICULTY,
    discrimination: tuple[float, ...] = DEFAULT_DISCRIMINATION,
    backend: str = "girth",
    backend_options: Mapping[str, Any] | None = None,
    gate_thresholds: Mapping[str, float] | None = None,
) -> IRTBaselineComparisonResult:
    """Compare BenchIQ's 2PL path to an optional R mirt baseline."""

    resolved_out_dir = Path(out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    resolved_gate_thresholds = _resolve_gate_thresholds(gate_thresholds)
    environment = _collect_environment_metadata()
    simulated = _simulate_fixture(
        difficulty=np.asarray(difficulty, dtype=float),
        discrimination=np.asarray(discrimination, dtype=float),
        model_count=model_count,
        random_seed=random_seed,
    )
    benchiq_result = _fit_benchiq_fixture(
        simulated=simulated,
        theta_method=theta_method,
        backend=backend,
        backend_options=backend_options,
    )

    skip_reason = _baseline_skip_reason(environment["r"])
    if skip_reason is not None:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "skipped",
            "skip_reason": skip_reason,
            "simulation": {
                "model_count": model_count,
                "item_count": int(len(difficulty)),
                "random_seed": random_seed,
                "theta_method": theta_method,
                "backend": backend,
                "backend_options": dict(backend_options or {}),
            },
            "environment": environment,
        }
        report["gate"] = _build_gate_report(report=report, gate_thresholds=resolved_gate_thresholds)
        table_path = _write_comparison_table(
            resolved_out_dir,
            pd.DataFrame(
                {
                    "item_id": pd.Series(dtype="string"),
                    "benchiq_discrimination": pd.Series(dtype="Float64"),
                    "r_discrimination_aligned": pd.Series(dtype="Float64"),
                    "benchiq_difficulty": pd.Series(dtype="Float64"),
                    "r_difficulty_aligned": pd.Series(dtype="Float64"),
                    "icc_rmse": pd.Series(dtype="Float64"),
                }
            ),
        )
        summary_path = _write_summary_artifacts(
            out_dir=resolved_out_dir,
            report=report,
            table_path=table_path,
        )
        return IRTBaselineComparisonResult(
            out_dir=resolved_out_dir,
            summary_path=summary_path,
            table_path=table_path,
            report=report,
        )

    r_result = _fit_r_mirt_fixture(simulated=simulated)
    aligned = _align_r_baseline_to_benchiq(
        benchiq_item_params=benchiq_result["item_params"],
        benchiq_theta=benchiq_result["theta"],
        r_item_params=r_result["item_params"],
        r_theta=r_result["theta"],
    )
    comparison_table = _build_comparison_table(
        benchiq_item_params=benchiq_result["item_params"],
        aligned_item_params=aligned["item_params_aligned"],
    )
    metrics = _compute_parity_metrics(
        benchiq_item_params=benchiq_result["item_params"],
        aligned_item_params=aligned["item_params_aligned"],
        benchiq_theta=benchiq_result["theta"],
        aligned_theta=aligned["theta_aligned"],
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "skip_reason": None,
        "simulation": {
            "model_count": model_count,
            "item_count": int(len(difficulty)),
            "random_seed": random_seed,
            "theta_method": theta_method,
            "backend": backend,
            "backend_options": dict(backend_options or {}),
        },
        "environment": environment,
        "alignment": aligned["alignment"],
        "metrics": metrics,
    }
    report["gate"] = _build_gate_report(report=report, gate_thresholds=resolved_gate_thresholds)
    table_path = _write_comparison_table(resolved_out_dir, comparison_table)
    summary_path = _write_summary_artifacts(
        out_dir=resolved_out_dir,
        report=report,
        table_path=table_path,
    )
    return IRTBaselineComparisonResult(
        out_dir=resolved_out_dir,
        summary_path=summary_path,
        table_path=table_path,
        report=report,
    )


def align_r_baseline_to_benchiq(
    *,
    benchiq_item_params: pd.DataFrame,
    benchiq_theta: pd.DataFrame,
    r_item_params: pd.DataFrame,
    r_theta: pd.DataFrame,
) -> dict[str, Any]:
    """Public wrapper for the alignment logic used by the parity harness."""

    return _align_r_baseline_to_benchiq(
        benchiq_item_params=benchiq_item_params,
        benchiq_theta=benchiq_theta,
        r_item_params=r_item_params,
        r_theta=r_theta,
    )


def _simulate_fixture(
    *,
    difficulty: np.ndarray,
    discrimination: np.ndarray,
    model_count: int,
    random_seed: int,
) -> dict[str, Any]:
    thetas = np.linspace(-2.5, 2.5, model_count, dtype=float)
    responses = create_synthetic_irt_dichotomous(
        difficulty=difficulty,
        discrimination=discrimination,
        thetas=thetas,
        seed=random_seed,
    )
    item_ids = [f"i{index + 1}" for index in range(responses.shape[0])]
    model_ids = [f"m{index + 1:03d}" for index in range(responses.shape[1])]

    rows: list[dict[str, object]] = []
    for item_index, item_id in enumerate(item_ids):
        for model_index, model_id in enumerate(model_ids):
            rows.append(
                {
                    BENCHMARK_ID: "sim",
                    ITEM_ID: item_id,
                    MODEL_ID: model_id,
                    "score": int(responses[item_index, model_index]),
                }
            )
    responses_long = pd.DataFrame(rows).astype(
        {
            BENCHMARK_ID: "string",
            ITEM_ID: "string",
            MODEL_ID: "string",
            "score": "Int64",
        }
    )
    response_matrix = (
        pd.DataFrame(
            responses.T,
            index=pd.Index(model_ids, dtype="string"),
            columns=pd.Index(item_ids, dtype="string"),
        )
        .astype(int)
        .sort_index()
    )
    return {
        "item_ids": item_ids,
        "model_ids": model_ids,
        "responses_long": responses_long,
        "response_matrix": response_matrix,
    }


def _fit_benchiq_fixture(
    *,
    simulated: dict[str, Any],
    theta_method: str,
    backend: str,
    backend_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    resolved_backend_options = dict(backend_options or {})
    if backend == "girth":
        resolved_backend_options.setdefault("max_iteration", 60)
    fit_result = fit_irt_backend(
        simulated["responses_long"],
        benchmark_id="sim",
        item_ids=simulated["item_ids"],
        model_ids=simulated["model_ids"],
        backend=backend,
        options=resolved_backend_options,
    )
    theta_grid = np.linspace(
        DEFAULT_THETA_MIN,
        DEFAULT_THETA_MAX,
        DEFAULT_THETA_GRID_SIZE,
        dtype=float,
    )
    theta_rows: list[dict[str, Any]] = []
    response_matrix = simulated["response_matrix"]
    for model_id in simulated["model_ids"]:
        estimate = estimate_theta_responses(
            responses=response_matrix.loc[model_id].astype("Float64"),
            item_params=fit_result.item_params,
            theta_method=theta_method,
            theta_grid=theta_grid,
            missing_heavy_threshold=0.5,
        )
        theta_rows.append(
            {
                MODEL_ID: model_id,
                "theta_hat": estimate["theta_hat"],
                "theta_se": estimate["theta_se"],
            }
        )
    theta = pd.DataFrame(theta_rows).astype(
        {
            MODEL_ID: "string",
            "theta_hat": "Float64",
            "theta_se": "Float64",
        }
    )
    return {
        "item_params": fit_result.item_params.copy(),
        "theta": theta,
    }


def _baseline_skip_reason(r_environment: Mapping[str, Any]) -> str | None:
    if not bool(r_environment["available"]):
        return "Rscript is not available in this environment"
    if r_environment.get("package_check_error") is not None:
        return "Rscript is available, but checking for the mirt package failed"
    if not bool(r_environment["mirt_installed"]):
        return "R package `mirt` is not installed in this environment"
    return None


def _collect_environment_metadata() -> dict[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            "arviz": _package_version_or_none("arviz"),
            "pymc": _package_version_or_none("pymc"),
            "numpy": _package_version_or_none("numpy"),
            "pandas": _package_version_or_none("pandas"),
            "scipy": _package_version_or_none("scipy"),
            "girth": _package_version_or_none("girth"),
        },
        "r": _probe_r_environment(),
    }


def _probe_r_environment() -> dict[str, Any]:
    rscript_path = shutil.which("Rscript")
    if rscript_path is None:
        return {
            "available": False,
            "rscript_path": None,
            "version": None,
            "version_error": None,
            "mirt_installed": False,
            "mirt_version": None,
            "package_check_error": None,
        }

    version_result = subprocess.run(
        [rscript_path, "-e", "cat(R.version.string)"],
        capture_output=True,
        check=False,
        text=True,
    )
    version_text = version_result.stdout.strip() if version_result.returncode == 0 else None
    version_error = (
        None if version_result.returncode == 0 else _subprocess_error_text(version_result)
    )

    package_check = subprocess.run(
        [rscript_path, "-e", "cat(requireNamespace('mirt', quietly=TRUE))"],
        capture_output=True,
        check=False,
        text=True,
    )
    package_check_error = (
        None if package_check.returncode == 0 else _subprocess_error_text(package_check)
    )
    mirt_installed = (
        package_check.stdout.strip().lower() == "true"
        if package_check_error is None
        else False
    )
    mirt_version = None
    if mirt_installed:
        package_version_result = subprocess.run(
            [rscript_path, "-e", "cat(as.character(utils::packageVersion('mirt')))"],
            capture_output=True,
            check=False,
            text=True,
        )
        if package_version_result.returncode == 0:
            mirt_version = package_version_result.stdout.strip() or None

    return {
        "available": True,
        "rscript_path": rscript_path,
        "version": version_text,
        "version_error": version_error,
        "mirt_installed": mirt_installed,
        "mirt_version": mirt_version,
        "package_check_error": package_check_error,
    }


def _package_version_or_none(distribution_name: str) -> str | None:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None


def _subprocess_error_text(result: subprocess.CompletedProcess[str]) -> str:
    return result.stderr.strip() or result.stdout.strip() or "no error output captured"


def _resolve_gate_thresholds(
    gate_thresholds: Mapping[str, float] | None,
) -> dict[str, float]:
    resolved = dict(DEFAULT_PARITY_GATE_THRESHOLDS)
    if gate_thresholds is None:
        return resolved

    unknown = sorted(set(gate_thresholds) - set(DEFAULT_PARITY_GATE_THRESHOLDS))
    if unknown:
        raise ValueError(f"unsupported gate threshold keys: {', '.join(unknown)}")

    for key, value in gate_thresholds.items():
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"gate threshold `{key}` must be finite")
        resolved[key] = numeric_value
    return resolved


def _build_gate_report(
    *,
    report: Mapping[str, Any],
    gate_thresholds: Mapping[str, float],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    status_passed = report["status"] == "ok"
    checks.append(
        {
            "metric": "status",
            "observed": report["status"],
            "comparator": "==",
            "threshold": "ok",
            "passed": status_passed,
        }
    )
    if not status_passed:
        reason = report.get("skip_reason") or "comparison did not complete"
        failures.append(f"comparison status was `{report['status']}`: {reason}")
        return {
            "thresholds": dict(gate_thresholds),
            "checks": checks,
            "passed": False,
            "failure_count": len(failures),
            "failures": failures,
        }

    metric_checks = [
        (
            "theta.pearson",
            float(report["metrics"]["theta"]["pearson"]),
            ">=",
            float(gate_thresholds["theta_pearson_min"]),
        ),
        (
            "theta.spearman",
            float(report["metrics"]["theta"]["spearman"]),
            ">=",
            float(gate_thresholds["theta_spearman_min"]),
        ),
        (
            "icc.mean_rmse",
            float(report["metrics"]["icc"]["mean_rmse"]),
            "<=",
            float(gate_thresholds["icc_mean_rmse_max"]),
        ),
    ]
    for metric_name, observed, comparator, threshold in metric_checks:
        passed = observed >= threshold if comparator == ">=" else observed <= threshold
        checks.append(
            {
                "metric": metric_name,
                "observed": observed,
                "comparator": comparator,
                "threshold": threshold,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(
                f"{metric_name} was {observed:.6f}, required {comparator} {threshold:.6f}"
            )

    return {
        "thresholds": dict(gate_thresholds),
        "checks": checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _fit_r_mirt_fixture(*, simulated: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="benchiq-r-baseline-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        matrix_path = temp_dir / "responses_matrix.csv"
        item_params_path = temp_dir / "r_item_params.csv"
        theta_path = temp_dir / "r_theta.csv"
        script_path = temp_dir / "fit_mirt.R"

        simulated["response_matrix"].to_csv(matrix_path, index=True)
        script_path.write_text(_r_script_source(), encoding="utf-8")
        result = subprocess.run(
            [
                shutil.which("Rscript") or "Rscript",
                str(script_path),
                str(matrix_path),
                str(item_params_path),
                str(theta_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"R baseline execution failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        item_params = pd.read_csv(item_params_path).astype(
            {
                ITEM_ID: "string",
                "discrimination": "Float64",
                "difficulty": "Float64",
            }
        )
        theta = pd.read_csv(theta_path).astype(
            {
                MODEL_ID: "string",
                "theta_hat": "Float64",
                "theta_se": "Float64",
            }
        )
        return {
            "item_params": item_params,
            "theta": theta,
        }


def _align_r_baseline_to_benchiq(
    *,
    benchiq_item_params: pd.DataFrame,
    benchiq_theta: pd.DataFrame,
    r_item_params: pd.DataFrame,
    r_theta: pd.DataFrame,
) -> dict[str, Any]:
    theta_merged = benchiq_theta.merge(r_theta, on=MODEL_ID, suffixes=("_benchiq", "_r"))
    if theta_merged.empty:
        raise ValueError(
            "could not align R theta estimates because no shared model_id values were found"
        )
    raw_theta_corr = float(
        spearmanr(
            theta_merged["theta_hat_benchiq"].astype(float).to_numpy(),
            theta_merged["theta_hat_r"].astype(float).to_numpy(),
        ).statistic
    )
    sign = -1.0 if raw_theta_corr < 0.0 else 1.0

    r_item_signed = r_item_params.copy()
    r_item_signed["discrimination"] = r_item_signed["discrimination"].astype(float) * sign
    r_item_signed["difficulty"] = r_item_signed["difficulty"].astype(float) * sign

    r_theta_signed = r_theta.copy()
    r_theta_signed["theta_hat"] = r_theta_signed["theta_hat"].astype(float) * sign
    r_theta_signed["theta_se"] = r_theta_signed["theta_se"].astype(float)

    signed_theta = benchiq_theta.merge(r_theta_signed, on=MODEL_ID, suffixes=("_benchiq", "_r"))
    if signed_theta.empty:
        raise ValueError(
            "could not align signed R theta estimates because no shared model_id values were found"
        )
    slope, intercept = np.polyfit(
        signed_theta["theta_hat_benchiq"].astype(float).to_numpy(),
        signed_theta["theta_hat_r"].astype(float).to_numpy(),
        deg=1,
    )
    if abs(slope) < 1e-8:
        raise ValueError("theta scale alignment slope was too close to zero")

    theta_aligned = r_theta_signed.copy()
    theta_aligned["theta_hat"] = (
        theta_aligned["theta_hat"].astype(float) - float(intercept)
    ) / float(slope)
    theta_aligned["theta_se"] = theta_aligned["theta_se"].astype(float) / abs(float(slope))

    item_params_aligned = r_item_signed.copy()
    item_params_aligned["difficulty"] = (
        item_params_aligned["difficulty"].astype(float) - float(intercept)
    ) / float(slope)
    item_params_aligned["discrimination"] = item_params_aligned["discrimination"].astype(
        float
    ) * float(slope)

    return {
        "item_params_aligned": item_params_aligned.astype(
            {
                ITEM_ID: "string",
                "discrimination": "Float64",
                "difficulty": "Float64",
            }
        ),
        "theta_aligned": theta_aligned.astype(
            {
                MODEL_ID: "string",
                "theta_hat": "Float64",
                "theta_se": "Float64",
            }
        ),
        "alignment": {
            "sign_applied": sign,
            "theta_slope": float(slope),
            "theta_intercept": float(intercept),
            "raw_theta_spearman": raw_theta_corr,
        },
    }


def _build_comparison_table(
    *,
    benchiq_item_params: pd.DataFrame,
    aligned_item_params: pd.DataFrame,
) -> pd.DataFrame:
    merged = benchiq_item_params.loc[:, [ITEM_ID, "discrimination", "difficulty"]].merge(
        aligned_item_params.loc[:, [ITEM_ID, "discrimination", "difficulty"]],
        on=ITEM_ID,
        suffixes=("_benchiq", "_r"),
    )
    theta_grid = np.linspace(DEFAULT_THETA_MIN, DEFAULT_THETA_MAX, 81, dtype=float)
    icc_rmses: list[float] = []
    for row in merged.itertuples(index=False):
        benchiq_probability = _probability_2pl(
            theta_grid,
            discrimination=float(getattr(row, "discrimination_benchiq")),
            difficulty=float(getattr(row, "difficulty_benchiq")),
        )
        r_probability = _probability_2pl(
            theta_grid,
            discrimination=float(getattr(row, "discrimination_r")),
            difficulty=float(getattr(row, "difficulty_r")),
        )
        icc_rmses.append(float(np.sqrt(np.mean((benchiq_probability - r_probability) ** 2))))
    merged["icc_rmse"] = pd.Series(icc_rmses, dtype="Float64")
    return merged.rename(
        columns={
            "discrimination_benchiq": "benchiq_discrimination",
            "difficulty_benchiq": "benchiq_difficulty",
            "discrimination_r": "r_discrimination_aligned",
            "difficulty_r": "r_difficulty_aligned",
        }
    ).astype(
        {
            ITEM_ID: "string",
            "benchiq_discrimination": "Float64",
            "benchiq_difficulty": "Float64",
            "r_discrimination_aligned": "Float64",
            "r_difficulty_aligned": "Float64",
            "icc_rmse": "Float64",
        }
    )


def _compute_parity_metrics(
    *,
    benchiq_item_params: pd.DataFrame,
    aligned_item_params: pd.DataFrame,
    benchiq_theta: pd.DataFrame,
    aligned_theta: pd.DataFrame,
) -> dict[str, Any]:
    items = benchiq_item_params.loc[:, [ITEM_ID, "discrimination", "difficulty"]].merge(
        aligned_item_params.loc[:, [ITEM_ID, "discrimination", "difficulty"]],
        on=ITEM_ID,
        suffixes=("_benchiq", "_r"),
    )
    theta = benchiq_theta.merge(aligned_theta, on=MODEL_ID, suffixes=("_benchiq", "_r"))
    comparison_table = _build_comparison_table(
        benchiq_item_params=benchiq_item_params,
        aligned_item_params=aligned_item_params,
    )
    return {
        "item_parameter_mae": {
            "discrimination": float(
                np.mean(
                    np.abs(
                        items["discrimination_benchiq"].astype(float).to_numpy()
                        - items["discrimination_r"].astype(float).to_numpy()
                    )
                )
            ),
            "difficulty": float(
                np.mean(
                    np.abs(
                        items["difficulty_benchiq"].astype(float).to_numpy()
                        - items["difficulty_r"].astype(float).to_numpy()
                    )
                )
            ),
        },
        "item_parameter_pearson": {
            "discrimination": float(
                pearsonr(
                    items["discrimination_benchiq"].astype(float).to_numpy(),
                    items["discrimination_r"].astype(float).to_numpy(),
                ).statistic
            ),
            "difficulty": float(
                pearsonr(
                    items["difficulty_benchiq"].astype(float).to_numpy(),
                    items["difficulty_r"].astype(float).to_numpy(),
                ).statistic
            ),
        },
        "theta": {
            "pearson": float(
                pearsonr(
                    theta["theta_hat_benchiq"].astype(float).to_numpy(),
                    theta["theta_hat_r"].astype(float).to_numpy(),
                ).statistic
            ),
            "spearman": float(
                spearmanr(
                    theta["theta_hat_benchiq"].astype(float).to_numpy(),
                    theta["theta_hat_r"].astype(float).to_numpy(),
                ).statistic
            ),
        },
        "icc": {
            "mean_rmse": float(comparison_table["icc_rmse"].astype(float).mean()),
            "max_rmse": float(comparison_table["icc_rmse"].astype(float).max()),
        },
    }


def _write_comparison_table(out_dir: Path, table: pd.DataFrame) -> Path:
    path = out_dir / "irt_r_baseline_item_comparison.csv"
    table.to_csv(path, index=False)
    return path


def _write_summary_artifacts(
    *,
    out_dir: Path,
    report: dict[str, Any],
    table_path: Path,
) -> Path:
    write_json(report, out_dir / "irt_r_baseline_summary.json")
    summary_path = out_dir / "irt_r_baseline_summary.md"
    summary_path.write_text(
        _summary_markdown(report=report, table_path=table_path),
        encoding="utf-8",
    )
    return summary_path


def _summary_markdown(*, report: dict[str, Any], table_path: Path) -> str:
    lines = [
        "# irt r-baseline comparison",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- status: `{report['status']}`",
        f"- table: `{table_path}`",
        f"- model_count: `{report['simulation']['model_count']}`",
        f"- item_count: `{report['simulation']['item_count']}`",
        f"- random_seed: `{report['simulation']['random_seed']}`",
        f"- theta_method: `{report['simulation']['theta_method']}`",
        f"- backend: `{report['simulation']['backend']}`",
        f"- backend_options: `{report['simulation']['backend_options']}`",
    ]
    if report["status"] == "skipped":
        lines.append(f"- skip_reason: `{report['skip_reason']}`")
    lines.extend(
        [
            "",
            "## environment",
            "",
            f"- python version: `{report['environment']['python']['version']}`",
            f"- python implementation: `{report['environment']['python']['implementation']}`",
            f"- python executable: `{report['environment']['python']['executable']}`",
            f"- python platform: `{report['environment']['python']['platform']}`",
            f"- arviz version: `{report['environment']['packages']['arviz']}`",
            f"- pymc version: `{report['environment']['packages']['pymc']}`",
            f"- numpy version: `{report['environment']['packages']['numpy']}`",
            f"- pandas version: `{report['environment']['packages']['pandas']}`",
            f"- scipy version: `{report['environment']['packages']['scipy']}`",
            f"- girth version: `{report['environment']['packages']['girth']}`",
            f"- rscript available: `{report['environment']['r']['available']}`",
            f"- rscript path: `{report['environment']['r']['rscript_path']}`",
            f"- r version: `{report['environment']['r']['version']}`",
            f"- r version_error: `{report['environment']['r']['version_error']}`",
            f"- mirt installed: `{report['environment']['r']['mirt_installed']}`",
            f"- mirt version: `{report['environment']['r']['mirt_version']}`",
            f"- mirt check_error: `{report['environment']['r']['package_check_error']}`",
        ]
    )
    if report["status"] == "ok":
        lines.extend(
            [
                "",
                "## alignment",
                "",
                f"- sign_applied: `{report['alignment']['sign_applied']}`",
                f"- theta_slope: `{report['alignment']['theta_slope']}`",
                f"- theta_intercept: `{report['alignment']['theta_intercept']}`",
                "",
                "## metrics",
                "",
                f"- theta pearson: `{report['metrics']['theta']['pearson']}`",
                f"- theta spearman: `{report['metrics']['theta']['spearman']}`",
                "- discrimination mae: "
                + f"`{report['metrics']['item_parameter_mae']['discrimination']}`",
                f"- difficulty mae: `{report['metrics']['item_parameter_mae']['difficulty']}`",
                f"- mean icc rmse: `{report['metrics']['icc']['mean_rmse']}`",
                f"- max icc rmse: `{report['metrics']['icc']['max_rmse']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## gate",
            "",
            f"- passed: `{report['gate']['passed']}`",
            f"- failure_count: `{report['gate']['failure_count']}`",
            f"- theta_pearson_min: `{report['gate']['thresholds']['theta_pearson_min']}`",
            f"- theta_spearman_min: `{report['gate']['thresholds']['theta_spearman_min']}`",
            f"- icc_mean_rmse_max: `{report['gate']['thresholds']['icc_mean_rmse_max']}`",
            "",
            "### checks",
            "",
        ]
    )
    for check in report["gate"]["checks"]:
        lines.append(
            "- "
            + f"{check['metric']}: `{check['observed']}` {check['comparator']} "
            + f"`{check['threshold']}` -> `{check['passed']}`"
        )
    if report["gate"]["failures"]:
        lines.extend(["", "### failures", ""])
        for failure in report["gate"]["failures"]:
            lines.append(f"- {failure}")
    lines.append("")
    return "\n".join(lines)


def _probability_2pl(theta: np.ndarray, *, discrimination: float, difficulty: float) -> np.ndarray:
    logits = discrimination * (theta - difficulty)
    return 1.0 / (1.0 + np.exp(-logits))


def _r_script_source() -> str:
    return """
args <- commandArgs(trailingOnly = TRUE)
matrix_path <- args[[1]]
item_path <- args[[2]]
theta_path <- args[[3]]

suppressPackageStartupMessages(library(mirt))
dat <- read.csv(matrix_path, row.names = 1, check.names = FALSE)
mod <- mirt(dat, 1, itemtype = '2PL', verbose = FALSE)

item_pars <- as.data.frame(coef(mod, IRTpars = TRUE, simplify = TRUE)$items)
item_pars$item_id <- rownames(item_pars)
disc_col <- if ('a' %in% names(item_pars)) {
  'a'
} else if ('a1' %in% names(item_pars)) {
  'a1'
} else {
  stop('could not find discrimination column')
}
if ('b' %in% names(item_pars)) {
  item_pars$difficulty <- item_pars[['b']]
} else if ('d' %in% names(item_pars)) {
  item_pars$difficulty <- -item_pars[['d']] / item_pars[[disc_col]]
} else {
  stop('could not find difficulty column')
}
item_out <- data.frame(
  item_id = item_pars$item_id,
  discrimination = item_pars[[disc_col]],
  difficulty = item_pars$difficulty
)
write.csv(item_out, item_path, row.names = FALSE)

score_df <- as.data.frame(
  fscores(mod, method = 'EAP', full.scores = TRUE, full.scores.SE = TRUE)
)
theta_col <- if ('F1' %in% names(score_df)) 'F1' else names(score_df)[[1]]
se_col <- if ('SE_F1' %in% names(score_df)) {
  'SE_F1'
} else if (ncol(score_df) >= 2) {
  names(score_df)[[2]]
} else {
  NA
}
theta_out <- data.frame(
  model_id = rownames(dat),
  theta_hat = score_df[[theta_col]],
  theta_se = if (is.na(se_col)) rep(NA_real_, nrow(score_df)) else score_df[[se_col]]
)
write.csv(theta_out, theta_path, row.names = FALSE)
"""
