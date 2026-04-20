#!/usr/bin/env python3
"""Compare random and deterministic preselection on the compact regression fixture."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import benchiq
from benchiq.reconstruct.reconstruction import JOINT_MODEL, MARGINAL_MODEL
from benchiq.schema.tables import BENCHMARK_ID, SPLIT

METHODS = ("random_cv", "deterministic_info")
SEEDS = (7, 11, 19)


def main() -> None:
    config_payload = json.loads(Path("tests/data/tiny_example/config.json").read_text())
    reports_dir = Path("reports/selection_comparison")
    workdir = reports_dir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for method in METHODS:
        for seed in SEEDS:
            config = dict(config_payload["config"])
            config["random_seed"] = seed
            stage_options = json.loads(json.dumps(config_payload["stage_options"]))
            stage_options.setdefault("04_subsample", {})
            stage_options["04_subsample"]["method"] = method
            if method == "random_cv":
                stage_options["04_subsample"]["n_iter"] = 12
                stage_options["04_subsample"]["checkpoint_interval"] = 4

            run_id = f"{method}-seed-{seed}"
            run_result = benchiq.run(
                "tests/data/compact_validation/responses_long.csv",
                config=config,
                out_dir=workdir,
                run_id=run_id,
                stage_options=stage_options,
                stop_after="09_reconstruct",
            )
            selection_rows.extend(
                _selection_rows(
                    method=method,
                    seed=seed,
                    subsample_result=run_result.stage_results["04_subsample"],
                )
            )
            run_rows.extend(
                _run_metric_rows(
                    method=method,
                    seed=seed,
                    run_result=run_result,
                )
            )

    selection_frame = pd.DataFrame(selection_rows)
    run_frame = pd.DataFrame(run_rows)
    stability_frame = _stability_frame(selection_frame)
    summary_frame = _summary_frame(run_frame, stability_frame)

    reports_dir.mkdir(parents=True, exist_ok=True)
    selection_frame.to_parquet(reports_dir / "selection_sets.parquet", index=False)
    selection_frame.to_csv(reports_dir / "selection_sets.csv", index=False)
    run_frame.to_parquet(reports_dir / "selection_metrics.parquet", index=False)
    run_frame.to_csv(reports_dir / "selection_metrics.csv", index=False)
    stability_frame.to_parquet(reports_dir / "stability_summary.parquet", index=False)
    stability_frame.to_csv(reports_dir / "stability_summary.csv", index=False)
    summary_frame.to_parquet(reports_dir / "summary.parquet", index=False)
    summary_frame.to_csv(reports_dir / "summary.csv", index=False)

    report = {
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "winner": _winner(summary_frame),
    }
    (reports_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot_summary(
        summary_frame,
        value_column="best_available_test_rmse_mean",
        ylabel="best-available test rmse",
        title="selection method quality",
        out_path=reports_dir / "best_available_rmse_by_method.png",
    )
    _plot_summary(
        summary_frame,
        value_column="stage04_runtime_mean_seconds",
        ylabel="stage-04 runtime (s)",
        title="selection method runtime",
        out_path=reports_dir / "stage04_runtime_by_method.png",
    )
    _plot_summary(
        summary_frame,
        value_column="selection_stability_mean",
        ylabel="pairwise jaccard",
        title="selection stability by method",
        out_path=reports_dir / "selection_stability_by_method.png",
    )
    (reports_dir / "summary.md").write_text(
        _summary_markdown(summary_frame, report=report),
        encoding="utf-8",
    )
    metadata = {
        "source_responses": "tests/data/compact_validation/responses_long.csv",
        "source_config": "tests/data/tiny_example/config.json",
        "workdir": str(workdir),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
    }
    (reports_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(reports_dir / "summary.md")


def _selection_rows(*, method: str, seed: int, subsample_result) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for benchmark_id, benchmark_result in sorted(subsample_result.benchmarks.items()):
        item_ids = (
            benchmark_result.preselect_items["item_id"]
            .dropna()
            .astype("string")
            .sort_values()
            .tolist()
        )
        rows.append(
            {
                "method": method,
                "seed": seed,
                BENCHMARK_ID: benchmark_id,
                "selected_items": json.dumps(item_ids),
                "selected_item_count": len(item_ids),
            }
        )
    return rows


def _run_metric_rows(*, method: str, seed: int, run_result) -> list[dict[str, object]]:
    stage_records = run_result.summary()["stage_records"]
    stage04_runtime = float(stage_records["04_subsample"]["duration_seconds"])
    reconstruction_summary = run_result.stage_results["09_reconstruct"].reconstruction_summary
    rows: list[dict[str, object]] = []
    for benchmark_id in (
        reconstruction_summary[BENCHMARK_ID].dropna().astype("string").unique().tolist()
    ):
        marginal_rmse = _split_metric(
            reconstruction_summary,
            benchmark_id=benchmark_id,
            model_type=MARGINAL_MODEL,
            split_name="test",
        )
        joint_rmse = _split_metric(
            reconstruction_summary,
            benchmark_id=benchmark_id,
            model_type=JOINT_MODEL,
            split_name="test",
        )
        rows.append(
            {
                "method": method,
                "seed": seed,
                BENCHMARK_ID: benchmark_id,
                "marginal_test_rmse": marginal_rmse,
                "joint_test_rmse": joint_rmse,
                "best_available_test_rmse": joint_rmse if joint_rmse is not None else marginal_rmse,
                "stage04_runtime_seconds": stage04_runtime,
            }
        )
    return rows


def _split_metric(
    reconstruction_summary: pd.DataFrame,
    *,
    benchmark_id: str,
    model_type: str,
    split_name: str,
) -> float | None:
    rows = reconstruction_summary.loc[
        (reconstruction_summary[BENCHMARK_ID] == benchmark_id)
        & (reconstruction_summary["model_type"] == model_type)
        & (reconstruction_summary[SPLIT] == split_name)
    ].copy()
    if rows.empty or rows["rmse"].isna().all():
        return None
    return float(rows["rmse"].dropna().iloc[0])


def _stability_frame(selection_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in sorted(selection_frame["method"].dropna().unique().tolist()):
        method_rows = selection_frame.loc[selection_frame["method"] == method].copy()
        for benchmark_id in sorted(method_rows[BENCHMARK_ID].dropna().unique().tolist()):
            benchmark_rows = method_rows.loc[method_rows[BENCHMARK_ID] == benchmark_id].copy()
            item_sets = [
                set(json.loads(payload))
                for payload in benchmark_rows["selected_items"].astype(str).tolist()
            ]
            pairwise = [
                _jaccard(left, right) for left, right in itertools.combinations(item_sets, 2)
            ]
            rows.append(
                {
                    "method": method,
                    BENCHMARK_ID: benchmark_id,
                    "pairwise_jaccard_mean": (
                        1.0 if not pairwise else float(sum(pairwise) / len(pairwise))
                    ),
                    "pairwise_jaccard_min": 1.0 if not pairwise else float(min(pairwise)),
                    "seed_count": int(len(item_sets)),
                }
            )
    return pd.DataFrame(rows).astype(
        {
            "method": "string",
            BENCHMARK_ID: "string",
            "pairwise_jaccard_mean": "Float64",
            "pairwise_jaccard_min": "Float64",
            "seed_count": "Int64",
        }
    )


def _summary_frame(run_frame: pd.DataFrame, stability_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        run_frame.groupby("method", dropna=False)
        .agg(
            best_available_test_rmse_mean=("best_available_test_rmse", "mean"),
            best_available_test_rmse_std=("best_available_test_rmse", "std"),
            stage04_runtime_mean_seconds=("stage04_runtime_seconds", "mean"),
            stage04_runtime_std_seconds=("stage04_runtime_seconds", "std"),
            marginal_test_rmse_mean=("marginal_test_rmse", "mean"),
            joint_test_rmse_mean=("joint_test_rmse", "mean"),
            runs=("best_available_test_rmse", "count"),
        )
        .reset_index()
    )
    seed_spread = (
        run_frame.groupby(["method", "seed"], dropna=False)["best_available_test_rmse"]
        .mean()
        .groupby(level=0)
        .std()
        .rename("seed_rmse_std")
        .reset_index()
    )
    stability_summary = (
        stability_frame.groupby("method", dropna=False)
        .agg(
            selection_stability_mean=("pairwise_jaccard_mean", "mean"),
            selection_stability_min=("pairwise_jaccard_min", "min"),
        )
        .reset_index()
    )
    return (
        grouped.merge(seed_spread, on="method", how="left")
        .merge(
            stability_summary,
            on="method",
            how="left",
        )
        .astype(
            {
                "method": "string",
                "best_available_test_rmse_mean": "Float64",
                "best_available_test_rmse_std": "Float64",
                "stage04_runtime_mean_seconds": "Float64",
                "stage04_runtime_std_seconds": "Float64",
                "marginal_test_rmse_mean": "Float64",
                "joint_test_rmse_mean": "Float64",
                "runs": "Int64",
                "seed_rmse_std": "Float64",
                "selection_stability_mean": "Float64",
                "selection_stability_min": "Float64",
            }
        )
        .sort_values(["best_available_test_rmse_mean", "stage04_runtime_mean_seconds"])
        .reset_index(drop=True)
    )


def _winner(summary_frame: pd.DataFrame) -> dict[str, object] | None:
    if summary_frame.empty:
        return None
    best_row = summary_frame.sort_values(
        [
            "best_available_test_rmse_mean",
            "selection_stability_mean",
            "stage04_runtime_mean_seconds",
        ],
        ascending=[True, False, True],
    ).iloc[0]
    return {
        "method": best_row["method"],
        "best_available_test_rmse_mean": best_row["best_available_test_rmse_mean"],
        "selection_stability_mean": best_row["selection_stability_mean"],
        "stage04_runtime_mean_seconds": best_row["stage04_runtime_mean_seconds"],
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _plot_summary(
    summary_frame: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        summary_frame["method"].astype(str).tolist(),
        summary_frame[value_column].astype(float).tolist(),
        color=["#2d6a4f", "#bc6c25"],
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summary_markdown(summary_frame: pd.DataFrame, *, report: dict[str, object]) -> str:
    lines = ["# selection comparison", "", "## winner", ""]
    winner = report["winner"]
    if winner is None:
        lines.append("- no valid comparison rows were produced")
    else:
        lines.append(
            "- "
            f"`{winner['method']}` "
            f"(best_available_test_rmse_mean={float(winner['best_available_test_rmse_mean']):.4f}, "
            f"selection_stability_mean={float(winner['selection_stability_mean']):.4f}, "
            f"stage04_runtime_mean_seconds={float(winner['stage04_runtime_mean_seconds']):.4f})"
        )
    lines.extend(["", "## summary", ""])
    for row in summary_frame.to_dict(orient="records"):
        lines.append(
            "- "
            f"`{row['method']}`: "
            f"best_available_test_rmse_mean={float(row['best_available_test_rmse_mean']):.4f}, "
            f"best_available_test_rmse_std={_fmt_float(row['best_available_test_rmse_std'])}, "
            f"seed_rmse_std={_fmt_float(row['seed_rmse_std'])}, "
            f"selection_stability_mean={float(row['selection_stability_mean']):.4f}, "
            f"stage04_runtime_mean_seconds={float(row['stage04_runtime_mean_seconds']):.4f}"
        )
    return "\n".join(lines) + "\n"


def _fmt_float(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
