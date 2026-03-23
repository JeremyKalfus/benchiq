# AGENTS.md

## purpose

BenchIQ is a python tool for benchmark-bundle distillation and overlap analysis.

The source of truth for product scope and methodology is [docs/specs/benchiq_v0_1_spec.md](/Users/jeremykalfus/CodingProjects/MedARC_FMRI/Returning%20Soldier%20Effect/smellm/benchIQ/docs/specs/benchiq_v0_1_spec.md). Do not redefine the product as a metabench-only reproduction. metabench is the methodological reference case and validation harness.

## repo layout

The repo is currently being bootstrapped. Build it toward this stable v0.1 layout and keep it coherent:

- `AGENTS.md`: repo-level agent contract.
- `PLANS.md`: ordered execution plan. use this for long multi-step work.
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
- `tests/unit/`, `tests/integration/`, `tests/regression/`, `tests/data/`: fast unit coverage, end-to-end fixtures, metabench validation fixtures, and tiny synthetic data only.
- `out/` or user-provided run directories: disk-backed run artifacts. keep every major stage inspectable.

## canonical commands

These are the commands this repo should standardize on and keep working once scaffolding lands:

- setup: `python -m pip install -e .[dev]`
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

- benchmark bundle input is user-chosen, not metabench-only.
- binary item scores only: `score in {0,1}` with nullable missing values allowed.
- canonical long-format item-response table first.
- benchmark-wise preprocessing with variance, ceiling, discrimination, and coverage filters.
- model-level train/val/test splits, including global test split when grand-overlap permits.
- cross-validated random subsampling to fixed `k_preselect` per benchmark.
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

- use `PLANS.md` for any long multi-step work. update it when ticket order, dependencies, or blockers change.
- keep changes small and ticket-sized.
- after each ticket, add or update tests and run them before moving on.
- ALWAYS check and self-test before moving on. no fallbacks, ever. if something fails, tell the user instead of guessing.
- ALWAYS use subagents liberally and keep context clean when the active session allows delegation. keep each subagent narrow and merge carefully.
- if stuck, use the `find-skills` skill before inventing a broad workaround.
- do not silently widen scope beyond the approved ticket.

## definition of done

A ticket is done only when all of the following are true:

- the implementation matches [docs/specs/benchiq_v0_1_spec.md](/Users/jeremykalfus/CodingProjects/MedARC_FMRI/Returning%20Soldier%20Effect/smellm/benchIQ/docs/specs/benchiq_v0_1_spec.md) and the current ticket in [PLANS.md](/Users/jeremykalfus/CodingProjects/MedARC_FMRI/Returning%20Soldier%20Effect/smellm/benchIQ/PLANS.md).
- the change stays within BenchIQ v0.1 scope.
- new or changed tests exist and pass locally.
- relevant canonical commands still pass, or any failure has been reported immediately.
- artifacts, manifests, warnings, and refusal reasons are inspectable on disk.
- skipped stages are explicit and justified. nothing is silently skipped.
- the user can be told exactly what changed, which files changed, which tests ran, and whether acceptance criteria passed.
