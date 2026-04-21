# PLANS.md

## purpose

- this is the live execution ledger for BenchIQ.
- keep it short.
- detailed historical rationale belongs in `docs/design/` and the saved report bundles under `reports/`.

## source of truth

- product and methodology: [docs/specs/benchiq_v0_1_spec.md](/Users/jeremykalfus/CodingProjects/BenchIQ/docs/specs/benchiq_v0_1_spec.md)
- repo working rules: [AGENTS.md](/Users/jeremykalfus/CodingProjects/BenchIQ/AGENTS.md)

## current repo state

- BenchIQ v0.1 is implemented end to end through `validate`, `calibrate`, `predict`, and `run`.
- the runtime default is `reconstruction_first`.
- the explicit alternate baseline is `psychometric_default`.
- the promoted stage-04 method is `deterministic_info`.
- the latest narrowed public-portfolio best-so-far is
  `psychometric_default__deterministic_info` under
  `reports/portfolio_optimization_cycles/best_so_far.{md,json}`; this is current optimization
  evidence, not yet a shipped default change.
- the main saved report bundles live under:
  - `reports/preprocessing_optimization/`
  - `reports/generalization_optimization/`
  - `reports/preprocessing_variation_followup/`
  - `reports/deployment_validation/`
  - `reports/selection_comparison/`
  - `reports/experiments/reconstruction_heads/`
  - `reports/portfolio_standing/`
  - `reports/portfolio_optimization_cycles/`
- there is no standing open multi-step roadmap right now. add a short active-work section below when a new approved ticket starts.

## active work

- T29 portfolio rmse optimization
- goal: minimize equal-source-weight held-out RMSE on the narrowed public optimization portfolio, targeting `< 2.0` without reducing informative-source coverage.
- dependencies: `reports/portfolio_standing/*`, `src/benchiq/portfolio/*`, reconstruction and preprocessing internals, saved public source materializations under `out/portfolio_sources/`.
- acceptance checks: updated tests for behavior changes; `ruff check .`; `pytest`; fresh saved evidence under `reports/portfolio_optimization_cycles/`; refreshed comparison against the current standing winner with per-source and leave-one-out deltas.
- status: in progress; cycle 010 lowered the best saved standing rmse to `2.9344` with `psychometric_default__deterministic_info` after raising OLLB to `44/44` while keeping HELM at `24/22`, OpenEval at `40/26`, and the low-support chooser in place; current frontier experiments are whether OLLB still improves with an even larger retained final subset and whether HELM/OpenEval need deeper reconstruction changes rather than more budget tuning.

## ticket summaries

- T01 scaffold and toolchain: created the package skeleton, project metadata, test scaffolding, and working lint/build/test commands.
- T02 config and schema foundation: added `BenchIQConfig`, canonical table/schema helpers, and structured validation errors.
- T03 bundle loading and canonicalization: implemented bundle loading, canonical long-table validation, and stage-00 artifact writing.
- T04 preprocessing filters: added benchmark-wise item stats, filtering, and stage-01 preprocessing artifacts.
- T05 score tables: added full-score tables, overlap-aware grand scores, and stage-02 score artifacts.
- T06 model-level splits: implemented train/val/test splitting with split diagnostics and fallback reporting.
- T07 GAM backend and cv harness: added the reusable GAM wrapper and reconstruction cross-validation utilities.
- T08 preselection search: implemented stage-04 candidate subsampling with progress, fold metrics, and checkpoint artifacts.
- T09 benchmark-local IRT: added 2PL fitting with `girth`, retained/dropped item diagnostics, and stage-05 plots.
- T10 final item selection: added Fisher-information ranking, final subset selection, and stage-06 selection artifacts.
- T11 theta estimation: added MAP/EAP theta estimation, standard errors, and stage-07 theta outputs.
- T12 reduced predictors: added benchmark-local reduced subscores, no-intercept linear predictors, and fallback diagnostics.
- T13 feature tables: added marginal and joint feature-table assembly plus explicit joint-skip reporting.
- T14 reconstruction: added marginal and overlap-gated joint GAM reconstruction, predictions, residuals, and summary metrics.
- T15 redundancy analysis: added correlation, factor, and compressibility outputs as secondary benchmark-level analysis.
- T16 runner orchestration: added the deterministic runner, stage manifests, and partial rerun support.
- T17 artifact-first CLI: added the stable `benchiq validate` and `benchiq run` CLI flows with integration coverage.
- T18 retired: no live ticket in the current checkout uses this number.
- T19 calibration / deployment split: added reusable `calibrate` / `predict` APIs, calibration bundles, and fit-once / score-later coverage.
- T20 optional R baseline harness: added the optional simulated-data IRT parity harness and clean skip behavior when R packages are absent.
- T21 reconstruction head experiments: added GAM vs Elastic Net vs XGBoost comparison harnesses and saved report bundles.
- T22 less-stochastic preselection alternative: added `deterministic_info` and the saved stability/runtime/quality comparison bundle.
- T23 factor-analysis demotion: updated docs and public framing so distillation, calibration, deployment, and reconstruction lead the product story.
- T24 stable report bundles: standardized saved markdown, tables, plots, and report JSON locations for the main experiment families.
- T25 preprocessing optimization: ran the broader preprocessing sweep, real-data confirmation, and method checks that promoted the reconstruction-first path.
- T26 generalization and promotion: ran the multi-bundle validation and deployment checks that promoted `reconstruction_first` as the product default path.
- T26 follow-up guardrails: kept `deterministic_info` for stage 04, kept head experiments comparison-only, and adopted the light `drop_low_tail_models_quantile=0.002` follow-up improvement after confirmation.
- T27 spec refresh: rewrote `docs/specs/benchiq_v0_1_spec.md` into a short current source-of-truth spec aligned with the shipped `validate` / `calibrate` / `predict` / `run` surface and runtime defaults.
- T28 narrowed public portfolio standing pass: added internal portfolio source specs and adapters, wired the narrowed public materialization and standing-pass scripts, and saved `reports/portfolio_standing/` as the initial baseline snapshot for the narrowed public portfolio.
- T29 narrowed public portfolio optimization cycles: iterated beyond the initial standing snapshot, saved the cycle bundle family under `reports/portfolio_optimization_cycles/`, and currently recommends `psychometric_default__deterministic_info` in `best_so_far.{md,json}`.
- T30 docs/status reconciliation: updated the major docs so the shipped runtime default, the historical standing baseline, and the current narrowed public-portfolio best-so-far record are distinguished explicitly.

## notes

- `docs/design/*plan*.md` are historical design notes, not the live execution ledger.
- when a new approved multi-step ticket starts, add only:
  - ticket id
  - goal
  - dependencies
  - acceptance checks
  - status
