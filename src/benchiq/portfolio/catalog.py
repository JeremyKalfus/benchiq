"""Source catalog for the narrowed public portfolio standing pass."""

from __future__ import annotations

from benchiq.portfolio.specs import BenchmarkSourceSpec, SnapshotSpec


def narrowed_public_portfolio_catalog() -> tuple[BenchmarkSourceSpec, ...]:
    """Return the fixed narrowed source catalog for the standing pass."""

    return (
        BenchmarkSourceSpec(
            source_id="ollb_v1_metabench_source",
            label="Open LLM Leaderboard v1 / MetaBench-source",
            adapter_id="ollb_v1_local",
            role="optimize",
            snapshots=(
                SnapshotSpec(
                    snapshot_id="release_default_subset_20260405",
                    label="local release-default subset export",
                    release="2026-04-05",
                    source_locator="out/release_bundle_source/release_default_subset_responses_long.parquet",
                    role="optimize",
                    notes=(
                        "uses the existing local export as the baseline source of truth",
                    ),
                ),
            ),
        ),
        BenchmarkSourceSpec(
            source_id="ollb_v2",
            label="Open LLM Leaderboard v2",
            adapter_id="ollb_v2_details",
            role="optimize",
            snapshots=(
                SnapshotSpec(
                    snapshot_id="results_20250315",
                    label="public results dataset plus details datasets",
                    release="2025-03-15",
                    source_locator="https://huggingface.co/datasets/open-llm-leaderboard/results",
                    role="optimize",
                    notes=(
                        "requires item-level details datasets, not only aggregate results json",
                    ),
                ),
            ),
        ),
        BenchmarkSourceSpec(
            source_id="openeval",
            label="OpenEval item-centered repository",
            adapter_id="openeval_objective",
            role="optimize",
            snapshots=(
                SnapshotSpec(
                    snapshot_id="hf_94f8112_20260314",
                    label="Human-Centered Eval OpenEval public dataset snapshot",
                    release="2026-03-14",
                    source_locator="https://huggingface.co/datasets/human-centered-eval/OpenEval",
                    role="optimize",
                    notes=(
                        "objective-compatible subset only",
                    ),
                ),
            ),
        ),
        BenchmarkSourceSpec(
            source_id="helm_objective",
            label="HELM objective subset",
            adapter_id="helm_capabilities_objective",
            role="optimize",
            snapshots=(
                SnapshotSpec(
                    snapshot_id="capabilities_v1_0_0",
                    label="HELM Capabilities v1.0.0 objective subset",
                    release="v1.0.0",
                    source_locator="gs://crfm-helm-public/capabilities/benchmark_output/releases/v1.0.0",
                    role="optimize",
                    notes=(
                        "item-level objective scenarios only",
                    ),
                ),
            ),
        ),
        BenchmarkSourceSpec(
            source_id="livecodebench",
            label="LiveCodeBench",
            adapter_id="livecodebench_public_inspection",
            role="validate",
            snapshots=(
                SnapshotSpec(
                    snapshot_id="public_release_202604",
                    label="public official datasets and leaderboard surface",
                    release="2026-04",
                    source_locator="https://huggingface.co/livecodebench/datasets",
                    role="validate",
                    is_dynamic=True,
                    notes=(
                        "requires a public many-model item-response bank to be BenchIQ-usable",
                    ),
                ),
            ),
        ),
        BenchmarkSourceSpec(
            source_id="belebele",
            label="Belebele",
            adapter_id="belebele_public_inspection",
            role="validate",
            snapshots=(
                SnapshotSpec(
                    snapshot_id="facebook_20230503",
                    label="official public dataset snapshot",
                    release="2023-05-03",
                    source_locator="https://huggingface.co/datasets/facebook/belebele",
                    role="validate",
                    notes=(
                        "multilingual validation requires public many-model response "
                        "data, not only the item set",
                    ),
                ),
            ),
        ),
    )
