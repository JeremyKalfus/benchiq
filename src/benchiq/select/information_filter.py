"""Stage-06 Fisher-information item selection and artifact writing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchiq.io.load import Bundle
from benchiq.io.write import write_json, write_parquet
from benchiq.irt import IRTResult
from benchiq.irt.info import (
    FISHER_INFORMATION,
    PROBABILITY_CORRECT,
    THETA,
    build_information_grid,
    build_theta_grid,
)
from benchiq.logging import update_manifest
from benchiq.schema.tables import BENCHMARK_ID, ITEM_ID

SELECTION_ORDER = "selection_order"
SELECTION_ROUND = "selection_round"
SELECTION_BIN = "selection_bin"
SELECTION_THETA = "selection_theta"
SELECTION_INFORMATION = "selection_information"


@dataclass(slots=True)
class BenchmarkSelectResult:
    """Per-benchmark stage-06 selection outputs."""

    benchmark_id: str
    subset_final: pd.DataFrame
    info_grid: pd.DataFrame
    selection_report: dict[str, Any]
    artifact_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(slots=True)
class SelectResult:
    """Stage-06 selection outputs."""

    benchmarks: dict[str, BenchmarkSelectResult]
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None


def select_bundle(
    bundle: Bundle,
    irt_result: IRTResult,
    *,
    k_final: int | Mapping[str, int],
    n_bins: int | None = None,
    info_min: float = 0.0,
    theta_grid_type: str = "difficulty_range",
    theta_grid_size: int = 401,
    theta_min: float | None = None,
    theta_max: float | None = None,
    out_dir: str | Path | None = None,
    run_id: str | None = None,
) -> SelectResult:
    """Select final item subsets from retained T09 item parameters."""

    benchmark_results: dict[str, BenchmarkSelectResult] = {}
    for benchmark_id in sorted(irt_result.benchmarks):
        benchmark_results[benchmark_id] = select_benchmark(
            irt_result,
            benchmark_id=benchmark_id,
            k_final=_resolve_k_final(k_final, benchmark_id=benchmark_id),
            n_bins=n_bins,
            info_min=info_min,
            theta_grid_type=theta_grid_type,
            theta_grid_size=theta_grid_size,
            theta_min=theta_min,
            theta_max=theta_max,
        )

    result = SelectResult(benchmarks=benchmark_results)
    run_root, manifest_path = _resolve_run_root(bundle, out_dir=out_dir, run_id=run_id)
    if run_root is not None:
        artifact_paths = _write_select_artifacts(result, run_root=run_root)
        result.artifact_paths = artifact_paths
        result.manifest_path = manifest_path
        if manifest_path is not None:
            update_manifest(
                manifest_path,
                {
                    "artifacts": {
                        "06_select": {
                            "per_benchmark": {
                                benchmark_id: {
                                    name: str(path) for name, path in sorted(paths.items())
                                }
                                for benchmark_id, paths in sorted(
                                    artifact_paths["per_benchmark"].items()
                                )
                            },
                        },
                    },
                },
            )
    return result


def select_benchmark(
    irt_result: IRTResult,
    *,
    benchmark_id: str,
    k_final: int,
    n_bins: int | None = None,
    info_min: float = 0.0,
    theta_grid_type: str = "difficulty_range",
    theta_grid_size: int = 401,
    theta_min: float | None = None,
    theta_max: float | None = None,
) -> BenchmarkSelectResult:
    """Select final items for one benchmark from retained IRT item parameters."""

    if k_final < 1:
        raise ValueError("k_final must be at least 1")
    if info_min < 0.0:
        raise ValueError("info_min must be non-negative")

    benchmark_irt = irt_result.benchmarks[benchmark_id]
    if benchmark_irt.irt_fit_report["skipped"]:
        return _skipped_benchmark_result(
            benchmark_id,
            skipped_reason="benchmark_skipped_in_irt",
            requested_k_final=k_final,
        )

    retained_items = benchmark_irt.irt_item_params.copy()
    candidate_item_count = int(len(retained_items.index))
    if candidate_item_count == 0:
        return _skipped_benchmark_result(
            benchmark_id,
            skipped_reason="no_retained_irt_items_available",
            requested_k_final=k_final,
        )

    effective_k_final = min(k_final, candidate_item_count)
    effective_n_bins = max(1, min(n_bins or min(effective_k_final, 250), effective_k_final))
    theta_grid = build_theta_grid(
        retained_items,
        grid_type=theta_grid_type,
        grid_size=theta_grid_size,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    info_grid = build_information_grid(retained_items, theta_grid=theta_grid)
    subset_final = _select_across_theta_bins(
        retained_items,
        info_grid,
        k_final=effective_k_final,
        n_bins=effective_n_bins,
        info_min=info_min,
    )
    selection_report = _build_selection_report(
        benchmark_id=benchmark_id,
        retained_items=retained_items,
        subset_final=subset_final,
        info_grid=info_grid,
        requested_k_final=k_final,
        effective_k_final=effective_k_final,
        n_bins=effective_n_bins,
        info_min=info_min,
        theta_grid_type=theta_grid_type,
        theta_grid=theta_grid,
    )
    return BenchmarkSelectResult(
        benchmark_id=benchmark_id,
        subset_final=subset_final,
        info_grid=info_grid,
        selection_report=selection_report,
    )


def _select_across_theta_bins(
    retained_items: pd.DataFrame,
    info_grid: pd.DataFrame,
    *,
    k_final: int,
    n_bins: int,
    info_min: float,
) -> pd.DataFrame:
    theta_values = np.sort(info_grid[THETA].astype(float).unique())
    bin_indices = np.array_split(np.arange(len(theta_values)), n_bins)
    difficulty_lookup = retained_items.set_index(ITEM_ID)["difficulty"].astype(float)
    discrimination_lookup = retained_items.set_index(ITEM_ID)["discrimination"].astype(float)

    selected_item_ids: set[str] = set()
    selected_records: list[dict[str, Any]] = []
    round_index = 0
    while len(selected_item_ids) < k_final:
        added_in_round = False
        for bin_number, theta_index_group in enumerate(bin_indices, start=1):
            if len(selected_item_ids) >= k_final or len(theta_index_group) == 0:
                break
            theta_slice = theta_values[theta_index_group]
            bin_frame = info_grid.loc[
                info_grid[THETA].astype(float).isin(theta_slice)
                & ~info_grid[ITEM_ID].isin(selected_item_ids)
            ].copy()
            if bin_frame.empty:
                continue
            item_max_info = (
                bin_frame.groupby(ITEM_ID, sort=True)[FISHER_INFORMATION]
                .max()
                .sort_values(ascending=False)
            )
            if item_max_info.empty:
                continue
            best_item_id = str(item_max_info.index[0])
            best_info = float(item_max_info.iloc[0])
            if best_info < info_min:
                continue
            best_row = (
                bin_frame.loc[bin_frame[ITEM_ID] == best_item_id]
                .sort_values(FISHER_INFORMATION, ascending=False)
                .iloc[0]
            )
            selected_item_ids.add(best_item_id)
            selected_records.append(
                {
                    BENCHMARK_ID: str(best_row[BENCHMARK_ID]),
                    ITEM_ID: best_item_id,
                    SELECTION_ORDER: len(selected_records) + 1,
                    SELECTION_ROUND: round_index + 1,
                    SELECTION_BIN: bin_number,
                    SELECTION_THETA: float(best_row[THETA]),
                    SELECTION_INFORMATION: best_info,
                    "discrimination": float(discrimination_lookup.loc[best_item_id]),
                    "difficulty": float(difficulty_lookup.loc[best_item_id]),
                    "theta_bin_min": float(theta_slice.min()),
                    "theta_bin_max": float(theta_slice.max()),
                }
            )
            added_in_round = True
        if not added_in_round:
            break
        round_index += 1

    subset_final = pd.DataFrame.from_records(selected_records)
    if subset_final.empty:
        return _empty_subset_frame()
    return subset_final.astype(
        {
            BENCHMARK_ID: "string",
            ITEM_ID: "string",
            SELECTION_ORDER: "Int64",
            SELECTION_ROUND: "Int64",
            SELECTION_BIN: "Int64",
            SELECTION_THETA: "Float64",
            SELECTION_INFORMATION: "Float64",
            "discrimination": "Float64",
            "difficulty": "Float64",
            "theta_bin_min": "Float64",
            "theta_bin_max": "Float64",
        }
    )


def _build_selection_report(
    *,
    benchmark_id: str,
    retained_items: pd.DataFrame,
    subset_final: pd.DataFrame,
    info_grid: pd.DataFrame,
    requested_k_final: int,
    effective_k_final: int,
    n_bins: int,
    info_min: float,
    theta_grid_type: str,
    theta_grid: np.ndarray,
) -> dict[str, Any]:
    selected_item_ids = subset_final[ITEM_ID].astype("string").tolist()
    warnings: list[dict[str, Any]] = []
    if effective_k_final < requested_k_final:
        warnings.append(
            {
                "code": "k_final_capped_by_candidates",
                "message": (
                    f"requested k_final={requested_k_final} exceeds retained item count; "
                    f"using {effective_k_final} candidates."
                ),
                "severity": "warning",
            }
        )
    if len(selected_item_ids) < effective_k_final:
        warnings.append(
            {
                "code": "selection_shortfall",
                "message": (
                    f"selected {len(selected_item_ids)} items below target {effective_k_final} "
                    "because no remaining item met the information threshold."
                ),
                "severity": "warning",
            }
        )
    if len(selected_item_ids) < 20:
        warnings.append(
            {
                "code": "small_final_subset",
                "message": (
                    f"selected only {len(selected_item_ids)} final items; reduced benchmark may "
                    "be too small for stable downstream reconstruction."
                ),
                "severity": "warning",
            }
        )

    theta_curve_full = (
        info_grid.groupby(THETA, sort=True)[FISHER_INFORMATION].sum().astype(float).to_dict()
    )
    theta_curve_selected = (
        info_grid.loc[info_grid[ITEM_ID].isin(selected_item_ids)]
        .groupby(THETA, sort=True)[FISHER_INFORMATION]
        .sum()
        .astype(float)
        .to_dict()
    )
    return {
        "benchmark_id": benchmark_id,
        "skipped": False,
        "skipped_reason": None,
        "warnings": warnings,
        "parameters": {
            "requested_k_final": requested_k_final,
            "effective_k_final": effective_k_final,
            "n_bins": n_bins,
            "info_min": info_min,
            "theta_grid_type": theta_grid_type,
            "theta_grid_size": int(len(theta_grid)),
        },
        "counts": {
            "candidate_item_count": int(len(retained_items.index)),
            "selected_item_count": int(len(selected_item_ids)),
            "info_grid_row_count": int(len(info_grid.index)),
        },
        "selected_item_ids": selected_item_ids,
        "theta_grid": {
            "min_theta": float(theta_grid.min()),
            "max_theta": float(theta_grid.max()),
            "grid_size": int(len(theta_grid)),
        },
        "expected_test_information": {
            "full_sum_max": float(max(theta_curve_full.values(), default=0.0)),
            "reduced_sum_max": float(max(theta_curve_selected.values(), default=0.0)),
        },
        "artifacts": {
            "plots_written": False,
            "plots_reason": "written_by_stage06_artifact_writer",
            "expected_test_information_plot": None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _skipped_benchmark_result(
    benchmark_id: str,
    *,
    skipped_reason: str,
    requested_k_final: int,
) -> BenchmarkSelectResult:
    return BenchmarkSelectResult(
        benchmark_id=benchmark_id,
        subset_final=_empty_subset_frame(),
        info_grid=_empty_info_grid_frame(),
        selection_report={
            "benchmark_id": benchmark_id,
            "skipped": True,
            "skipped_reason": skipped_reason,
            "warnings": [],
            "parameters": {
                "requested_k_final": requested_k_final,
                "effective_k_final": 0,
                "n_bins": 0,
                "info_min": 0.0,
                "theta_grid_type": None,
                "theta_grid_size": 0,
            },
            "counts": {
                "candidate_item_count": 0,
                "selected_item_count": 0,
                "info_grid_row_count": 0,
            },
            "selected_item_ids": [],
            "theta_grid": {
                "min_theta": None,
                "max_theta": None,
                "grid_size": 0,
            },
            "expected_test_information": {
                "full_sum_max": 0.0,
                "reduced_sum_max": 0.0,
            },
            "artifacts": {
                "plots_written": False,
                "plots_reason": "no_plot_written_for_skipped_selection",
                "expected_test_information_plot": None,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _write_select_artifacts(
    result: SelectResult,
    *,
    run_root: Path,
) -> dict[str, Any]:
    stage_dir = run_root / "artifacts" / "06_select"
    per_benchmark_paths: dict[str, dict[str, Path]] = {}
    for benchmark_id, benchmark_result in sorted(result.benchmarks.items()):
        benchmark_dir = stage_dir / "per_benchmark" / benchmark_id
        if benchmark_result.selection_report["skipped"]:
            benchmark_result.selection_report["artifacts"]["plots_written"] = False
            benchmark_result.selection_report["artifacts"]["plots_reason"] = (
                "no_plot_written_for_skipped_selection"
            )
            benchmark_result.selection_report["artifacts"]["expected_test_information_plot"] = None
            benchmark_paths = {
                "subset_final": write_parquet(
                    benchmark_result.subset_final,
                    benchmark_dir / "subset_final.parquet",
                ),
                "info_grid": write_parquet(
                    benchmark_result.info_grid,
                    benchmark_dir / "info_grid.parquet",
                ),
                "selection_report": write_json(
                    benchmark_result.selection_report,
                    benchmark_dir / "selection_report.json",
                ),
            }
        else:
            plot_path = _write_expected_test_information_plot(
                benchmark_result.info_grid,
                subset_final=benchmark_result.subset_final,
                path=benchmark_dir / "plots" / "expected_test_information.png",
            )
            benchmark_result.selection_report["artifacts"]["plots_written"] = True
            benchmark_result.selection_report["artifacts"]["plots_reason"] = None
            benchmark_result.selection_report["artifacts"]["expected_test_information_plot"] = str(
                plot_path
            )
            benchmark_paths = {
                "subset_final": write_parquet(
                    benchmark_result.subset_final,
                    benchmark_dir / "subset_final.parquet",
                ),
                "info_grid": write_parquet(
                    benchmark_result.info_grid,
                    benchmark_dir / "info_grid.parquet",
                ),
                "selection_report": write_json(
                    benchmark_result.selection_report,
                    benchmark_dir / "selection_report.json",
                ),
                "expected_test_information_plot": plot_path,
            }
        per_benchmark_paths[benchmark_id] = benchmark_paths
    return {"per_benchmark": per_benchmark_paths}


def _write_expected_test_information_plot(
    info_grid: pd.DataFrame,
    *,
    subset_final: pd.DataFrame,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_item_ids = subset_final[ITEM_ID].astype("string").tolist()
    full_curve = (
        info_grid.groupby(THETA, sort=True)[FISHER_INFORMATION].sum().astype(float).reset_index()
    )
    reduced_curve = (
        info_grid.loc[info_grid[ITEM_ID].isin(selected_item_ids)]
        .groupby(THETA, sort=True)[FISHER_INFORMATION]
        .sum()
        .astype(float)
        .reset_index()
    )

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        full_curve[THETA].astype(float),
        full_curve[FISHER_INFORMATION].astype(float),
        label="full retained set",
        color="#4c78a8",
    )
    axis.plot(
        reduced_curve[THETA].astype(float),
        reduced_curve[FISHER_INFORMATION].astype(float),
        label="final subset",
        color="#f58518",
    )
    axis.set_xlabel("theta")
    axis.set_ylabel("expected test information")
    axis.set_title("Expected Test Information")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _resolve_k_final(k_final: int | Mapping[str, int], *, benchmark_id: str) -> int:
    if isinstance(k_final, Mapping):
        if benchmark_id not in k_final:
            raise KeyError(f"k_final is missing benchmark_id={benchmark_id!r}")
        resolved = int(k_final[benchmark_id])
    else:
        resolved = int(k_final)
    if resolved < 1:
        raise ValueError("k_final must be at least 1")
    return resolved


def _resolve_run_root(
    bundle: Bundle,
    *,
    out_dir: str | Path | None,
    run_id: str | None,
) -> tuple[Path | None, Path | None]:
    if out_dir is not None:
        resolved_run_id = run_id or bundle.run_id or _default_run_id()
        run_root = Path(out_dir) / resolved_run_id
        return run_root, run_root / "manifest.json"
    if bundle.manifest_path is not None:
        return bundle.manifest_path.parent, bundle.manifest_path
    return None, None


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _empty_subset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            ITEM_ID: pd.Series(dtype="string"),
            SELECTION_ORDER: pd.Series(dtype="Int64"),
            SELECTION_ROUND: pd.Series(dtype="Int64"),
            SELECTION_BIN: pd.Series(dtype="Int64"),
            SELECTION_THETA: pd.Series(dtype="Float64"),
            SELECTION_INFORMATION: pd.Series(dtype="Float64"),
            "discrimination": pd.Series(dtype="Float64"),
            "difficulty": pd.Series(dtype="Float64"),
            "theta_bin_min": pd.Series(dtype="Float64"),
            "theta_bin_max": pd.Series(dtype="Float64"),
        }
    )


def _empty_info_grid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BENCHMARK_ID: pd.Series(dtype="string"),
            ITEM_ID: pd.Series(dtype="string"),
            THETA: pd.Series(dtype="Float64"),
            PROBABILITY_CORRECT: pd.Series(dtype="Float64"),
            FISHER_INFORMATION: pd.Series(dtype="Float64"),
        }
    )
