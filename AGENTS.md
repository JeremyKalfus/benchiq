# AGENTS.md

## purpose

BenchIQ is a python tool for benchmark-bundle distillation and overlap analysis. The current goal of BenchIQ is to improve scores as much as possible and publish our results in NeurIPS. BenchIQ is a research project.

The source of truth for product scope and methodology is [docs/specs/benchiq_v0_1_spec.md](/Users/jeremykalfus/CodingProjects/BenchIQ/docs/specs/benchiq_v0_1_spec.md).

BenchIQ is a general benchmark-bundle distillation tool. It is not an external-benchmark parity harness.

## repo layout

BenchIQ targets this stable v0.1 layout:

- `AGENTS.md`: repo-level agent contract.
- `PLANS.md`: live execution ledger for long multi-step work. keep it short and current.
- `docs/specs/benchiq_v0_1_spec.md`: locked v0.1 source-of-truth spec copied from the user-provided document.
- `docs/design/`: design notes, schema notes, validation notes, cli docs.
- `src/benchiq/`: package code.
- `src/benchiq/io/`: loading, writing, manifests.
- `src/benchiq/schema/`: canonical tables and validation checks.
- `src/benchiq/preprocess/`: item stats, filters, score tables.
- `src/benchiq/split/`: model-level split logic.
- `src/benchiq/subsample/`: cross-validated preselection.
- `src/benchiq/irt/`: 2pl fit, fisher information, theta estimation.
- `src/benchiq/select/`: information-based final subset selection.
- `src/benchiq/reconstruct/`: linear predictors, feature building, gam fitting, metrics.
- `src/benchiq/redundancy/`: benchmark-level overlap and compressibility analysis.
- `src/benchiq/cli/`: artifact-first cli entrypoints.
- `tests/unit/`, `tests/integration/`, `tests/regression/`, `tests/data/`: fast unit coverage, end-to-end fixture coverage, regression coverage, and local test data only.
- `out/` or user-provided run directories: disk-backed run artifacts. Keep every major stage inspectable.

## current product notes

- public workflows: `validate`, `calibrate`, `predict`, and `run`
- runtime default profile: `reconstruction_first`
- explicit alternate baseline: `psychometric_default`
- promoted stage-04 method: `deterministic_info`

## canonical commands

These are the commands this repo should standardize on and keep working once scaffolding lands:

- setup: `python -m pip install -e '.[dev]'`
- build: `python -m build`
- test: `pytest`
- targeted test: `pytest path/to/test_file.py -k pattern`
- lint: `ruff check .`
- format check: `ruff format --check .`

If one of these commands does not work yet, the current ticket should add the missing project scaffolding before deeper implementation continues.

## coding conventions

- target python `>=3.10,<3.15`.
- keep `responses_long` as the canonical internal source of truth.
- keep all major stages inspectable, deterministic, and disk-backed.
- use parquet for tables and json for manifests/reports unless the spec says otherwise.
- use `pathlib` over ad hoc string paths.
- prefer explicit config objects, typed function signatures, and small composable functions.
- model-level splits only. never do item-level train/val/test splits.
- treat warnings, refusal reasons, and skipped stages as first-class artifacts.
- no silent fallback behavior in code or workflow. if something fails, surface it and stop.
- ALWAYS keep code comments lowercase and human-like.
- ALWAYS keep commit messages lowercase, short, and humanlike.

## BenchIQ v0.1 constraints

Keep v0.1 narrow and spec-aligned:

- benchmark bundle input is user-chosen.
- binary item scores only: `score in {0,1}` with nullable missing values allowed.
- canonical long-format item-response table first.
- benchmark-wise preprocessing with variance, ceiling, discrimination, and coverage filters.
- model-level train/val/test splits, including global test split when grand-overlap permits.
- stage-04 preselection must stay explicit and evidence-backed. both `random_cv` and `deterministic_info` exist in the repo; do not silently swap methods without saved comparison evidence.
- benchmark-specific unidimensional 2pl irt only.
- fisher-information item selection to `k_final`.
- latent ability estimation with uncertainty.
- score reconstruction to normalized full-benchmark percent scores with gams.
- benchmark-level factor, redundancy, and compressibility analysis only.
- artifact-first cli and python api.

Explicit non-goals for v0.1:

- no dashboards, guis, or hosted platform features.
- no cat workflows, item banking, or arbitrary irt family expansion.
- no multimodal features.
- no automatic data collection in v0.1 unless explicitly requested by the user. BenchIQ operates on user-provided item-level response data.
- no item-level multidimensional factor models.
- no broad psychometrics framework scope creep.

## execution rules

- use `PLANS.md` for any long multi-step work. update current status, active work, and ticket summaries when the repo state changes.
- keep changes small and ticket-sized.
- after each ticket, add or update tests and run them before moving on.
- always check and self-test before moving on. no fallbacks, ever. if something fails, tell the user instead of guessing.
- if stuck, use the `find-skills` skill before inventing a broad workaround.
- do not silently widen scope beyond the approved ticket.

## definition of done

A ticket is done only when all of the following are true:

- the implementation matches [docs/specs/benchiq_v0_1_spec.md](/Users/jeremykalfus/CodingProjects/BenchIQ/docs/specs/benchiq_v0_1_spec.md) and the current ticket in [PLANS.md](/Users/jeremykalfus/CodingProjects/BenchIQ/PLANS.md).
- the change stays within BenchIQ v0.1 scope.
- new or changed tests exist and pass locally.
- relevant canonical commands still pass, or any failure has been reported immediately.
- artifacts, manifests, warnings, and refusal reasons are inspectable on disk.
- if the user-facing story changed, the relevant top-level docs and reports were updated to match.
- skipped stages are explicit and justified. nothing is silently skipped.
- the user can be told exactly what changed, which files changed, which tests ran, and whether acceptance criteria passed.
