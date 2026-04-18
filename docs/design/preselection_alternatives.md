# preselection alternatives

The original BenchIQ v0.1 path uses random cross-validated subset search at stage 04.

That remains the baseline, but the pivot adds one less-stochastic alternative:

- `deterministic_info`

## deterministic_info

`deterministic_info` ranks retained items by a simple deterministic proxy for early-item
usefulness:

`abs(point_biserial)^2 * mean * (1 - mean) * item_coverage`

Rationale:

- `abs(point_biserial)` acts as an early discrimination proxy
- `mean * (1 - mean)` rewards non-saturated binary items
- `item_coverage` penalizes items that are weakly observed

This is intentionally narrow. It is not presented as a full replacement by assumption. It is an
artifact-first comparison target that can be benchmarked honestly against the existing random-CV
path on:

- reconstruction quality
- runtime
- seed variance
- selected-item stability

The saved comparison outputs live under `reports/selection_comparison/`.
