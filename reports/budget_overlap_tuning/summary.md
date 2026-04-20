# budget and overlap tuning

## decision

- explicit decision: keep the current dataset-specific budgets
- adoption status: do not adopt a smaller `k_preselect` / `k_final` profile
- reason: smaller budgets improved complete-feature coverage in the synthetic dense regime, but not enough to beat current held-out RMSE, and a milder real-bundle shrink already degraded real held-out RMSE materially

## dense overlap

Seed-7 sweep on `synthetic_dense_overlap` with `reconstruction_relaxed` + `deterministic_info`:

| k_preselect | k_final | best_available_test_rmse_mean | joint_complete_feature_model_count | joint_available_count | note |
| --- | --- | --- | --- | --- | --- |
| 70 | 45 | 4.5663 | 4 | 0 | current budget |
| 40 | 20 | 6.5596 | 40 | 0 | more complete coverage, worse RMSE |
| 25 | 10 | 8.6483 | 116 | 0 | coverage rises, RMSE worsens sharply |
| 15 | 6 | 10.1961 | 179 | 0 | nearly clears joint threshold, RMSE much worse |
| 12 | 5 | 6.6283 | 197 | 5 | joint turns on, but overall RMSE is still worse than baseline |

Takeaway:

- shrinking `k_final` does protect stage-08 complete-feature overlap
- that rescue is not enough to offset the information loss from selecting too few items
- the current dense-overlap budget stays better on held-out RMSE

## sparse overlap

Seed-7 sweep on `synthetic_sparse_overlap` with the same profile:

- current stage-02 complete-overlap model count is `80`, below the required joint threshold `90`
- the split output is benchmark-local train / val only, so the reconstruction stage has `0` test rows per benchmark
- smaller budgets improved mean linear-predictor coverage from about `12.8` predicted models per benchmark at `55/35` to about `122.5` at `15/6`
- despite that coverage gain, all sparse candidates still had `0` informative benchmarks for held-out RMSE and `0` joint-available benchmarks

Takeaway:

- this stress bundle is non-informative for budget tuning because overlap has already collapsed before stage-04 / stage-06 can rescue it
- smaller budgets cannot create a held-out test surface or restore joint reconstruction once the raw overlap threshold is already missed

## real bundle

Seed-7 confirmation on `large_release_default_subset`:

| k_preselect | k_final | best_available_test_rmse_mean | joint_available_count | note |
| --- | --- | --- | --- | --- |
| 350 | 250 | 0.9015 | 6 | current budget |
| 300 | 200 | 1.1260 | 6 | milder shrink, materially worse RMSE |

Takeaway:

- the real bundle does not want a smaller budget in this direction
- because a mild shrink already hurts, there is no evidence-based reason to promote a smaller real-data budget profile now

## implication

If overlap protection is worth revisiting later, the next honest surface is not "shrink the item budgets more." The synthetic diagnostics point instead toward:

- split policy when overlap is already thin
- joint-threshold policy for stress bundles
- linear-predictor coverage policy if we want more models to stay scorable without requiring full selected-item coverage

Those are different product decisions than `k_preselect` / `k_final`, and they should be evaluated separately.
