"""Pairwise benchmark-correlation helpers for stage-10 redundancy analysis."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from scipy.stats import spearmanr

from benchiq.schema.tables import BENCHMARK_ID, MODEL_ID

OTHER_BENCHMARK_ID = "other_benchmark_id"
CORRELATION = "correlation"
OVERLAP_COUNT = "overlap_count"
VALUE_KIND = "value_kind"
METHOD = "method"


def build_pairwise_correlation_table(
    frame: pd.DataFrame,
    *,
    value_column: str,
    benchmark_ids: list[str] | None = None,
    method: Literal["spearman"] = "spearman",
) -> pd.DataFrame:
    """Compute pairwise benchmark correlations with overlap counts in long form."""

    if method != "spearman":
        raise ValueError(f"unsupported correlation method: {method}")

    if benchmark_ids is None:
        benchmark_ids = sorted(frame[BENCHMARK_ID].dropna().astype("string").unique())

    value_wide = (
        frame.loc[:, [BENCHMARK_ID, MODEL_ID, value_column]]
        .dropna(subset=[value_column])
        .pivot(index=MODEL_ID, columns=BENCHMARK_ID, values=value_column)
        .astype("Float64")
    )
    records: list[dict[str, object]] = []
    for benchmark_id in benchmark_ids:
        left = (
            value_wide[benchmark_id]
            if benchmark_id in value_wide.columns
            else pd.Series(dtype=float)
        )
        for other_benchmark_id in benchmark_ids:
            right = (
                value_wide[other_benchmark_id]
                if other_benchmark_id in value_wide.columns
                else pd.Series(dtype=float)
            )
            overlap = pd.concat([left, right], axis=1).dropna()
            overlap_count = int(len(overlap.index))
            if benchmark_id == other_benchmark_id and overlap_count > 0:
                correlation = 1.0
            elif overlap_count >= 2:
                correlation = float(spearmanr(overlap.iloc[:, 0], overlap.iloc[:, 1]).statistic)
            else:
                correlation = None
            records.append(
                {
                    BENCHMARK_ID: benchmark_id,
                    OTHER_BENCHMARK_ID: other_benchmark_id,
                    CORRELATION: correlation,
                    OVERLAP_COUNT: overlap_count,
                    METHOD: method,
                }
            )

    result = pd.DataFrame.from_records(records).astype(
        {
            BENCHMARK_ID: "string",
            OTHER_BENCHMARK_ID: "string",
            CORRELATION: "Float64",
            OVERLAP_COUNT: "Int64",
            METHOD: "string",
        }
    )
    return result


def summarize_correlation_table(correlation_table: pd.DataFrame) -> dict[str, float | int | None]:
    """Return compact summary stats for a pairwise correlation table."""

    off_diagonal = correlation_table.loc[
        correlation_table[BENCHMARK_ID] != correlation_table[OTHER_BENCHMARK_ID]
    ].copy()
    finite = off_diagonal[CORRELATION].dropna().astype(float)
    return {
        "pair_count": int(len(off_diagonal.index)),
        "finite_pair_count": int(len(finite.index)),
        "mean_abs_correlation": float(finite.abs().mean()) if not finite.empty else None,
        "max_abs_correlation": float(finite.abs().max()) if not finite.empty else None,
        "min_overlap_count": (
            int(off_diagonal[OVERLAP_COUNT].min()) if not off_diagonal.empty else None
        ),
        "max_overlap_count": (
            int(off_diagonal[OVERLAP_COUNT].max()) if not off_diagonal.empty else None
        ),
    }
