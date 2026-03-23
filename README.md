# BenchIQ

BenchIQ is an artifact-first python workflow for llm benchmark distillation and overlap analysis on user-provided item-level response data.

The source of truth for v0.1 scope and methodology is [docs/specs/benchiq_v0_1_spec.md](/Users/jeremykalfus/CodingProjects/BenchIQ/docs/specs/benchiq_v0_1_spec.md).

## status

This repository is in early bootstrap. T01 establishes the package scaffold, tooling, and test loop so later tickets can implement the pipeline stage by stage.

## canonical commands

- install: `python -m pip install -e .[dev]`
- build: `python -m build`
- test: `pytest`
- lint: `ruff check .`
- format check: `ruff format --check .`

## contributor notes

- follow [AGENTS.md](/Users/jeremykalfus/CodingProjects/BenchIQ/AGENTS.md)
- follow [PLANS.md](/Users/jeremykalfus/CodingProjects/BenchIQ/PLANS.md)
- keep changes ticket-sized and test after every ticket
