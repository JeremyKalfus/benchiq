"""Optional R-baseline comparison helpers for IRT parity work."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from girth.synthetic import create_synthetic_irt_dichotomous
from scipy.stats import pearsonr, spearmanr

from benchiq.io.write import write_json
from benchiq.irt.backends.girth_backend import fit_girth_2pl
from benchiq.irt.theta import estimate_theta_responses
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID, MODEL_ID

DEFAULT_DISCRIMINATION = (0.7, 0.9, 1.1, 1.3, 1.6, 1.9)
DEFAULT_DIFFICULTY = (-1.4, -0.8, -0.2, 0.4, 0.9, 1.5)
DEFAULT_THETA_MIN = -4.0
DEFAULT_THETA_MAX = 4.0
DEFAULT_THETA_GRID_SIZE = 161
DEFAULT_OUTPUT_DIR = Path("reports") / "irt_r_baseline"


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
) -> IRTBaselineComparisonResult:
    """Compare BenchIQ's 2PL path to an optional R mirt baseline."""

    resolved_out_dir = Path(out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    simulated = _simulate_fixture(
        difficulty=np.asarray(difficulty, dtype=float),
        discrimination=np.asarray(discrimination, dtype=float),
        model_count=model_count,
        random_seed=random_seed,
    )
    benchiq_result = _fit_benchiq_fixture(simulated=simulated, theta_method=theta_method)

    skip_reason = _baseline_skip_reason()
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
            },
        }
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
        },
        "alignment": aligned["alignment"],
        "metrics": metrics,
    }
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


def _fit_benchiq_fixture(*, simulated: dict[str, Any], theta_method: str) -> dict[str, Any]:
    fit_result = fit_girth_2pl(
        simulated["responses_long"],
        benchmark_id="sim",
        item_ids=simulated["item_ids"],
        model_ids=simulated["model_ids"],
        options={"max_iteration": 60},
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


def _baseline_skip_reason() -> str | None:
    rscript_path = shutil.which("Rscript")
    if rscript_path is None:
        return "Rscript is not available in this environment"
    package_check = subprocess.run(
        [rscript_path, "-e", "cat(requireNamespace('mirt', quietly=TRUE))"],
        capture_output=True,
        check=False,
        text=True,
    )
    if package_check.returncode != 0:
        return "Rscript is available, but checking for the mirt package failed"
    if package_check.stdout.strip().lower() != "true":
        return "R package `mirt` is not installed in this environment"
    return None


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
    ]
    if report["status"] == "skipped":
        lines.append(f"- skip_reason: `{report['skip_reason']}`")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            f"- sign_applied: `{report['alignment']['sign_applied']}`",
            f"- theta_slope: `{report['alignment']['theta_slope']}`",
            f"- theta_intercept: `{report['alignment']['theta_intercept']}`",
            "",
            "## metrics",
            "",
            f"- theta pearson: `{report['metrics']['theta']['pearson']}`",
            f"- theta spearman: `{report['metrics']['theta']['spearman']}`",
            f"- discrimination mae: `{report['metrics']['item_parameter_mae']['discrimination']}`",
            f"- difficulty mae: `{report['metrics']['item_parameter_mae']['difficulty']}`",
            f"- mean icc rmse: `{report['metrics']['icc']['mean_rmse']}`",
            f"- max icc rmse: `{report['metrics']['icc']['max_rmse']}`",
            "",
        ]
    )
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
  model_id = rownames(score_df),
  theta_hat = score_df[[theta_col]],
  theta_se = if (is.na(se_col)) rep(NA_real_, nrow(score_df)) else score_df[[se_col]]
)
write.csv(theta_out, theta_path, row.names = FALSE)
"""
