# BenchIQ v0.1 Scope

BenchIQ v0.1 is a narrow, inspectable workflow for benchmark-bundle distillation and overlap analysis from user-provided item-level model responses.

## Product Identity

BenchIQ is:

- a generic benchmark-bundle tool
- artifact-first and deterministic
- designed for many-model, many-checkpoint evaluation settings
- focused on reduced subsets, score reconstruction, and cross-benchmark redundancy

BenchIQ is not:

- a metabench-only reproduction
- a hosted platform
- a dashboard product
- a broad psychometrics toolkit
- a generic data-collection system

metabench remains the methodological reference case and validation harness, not the product identity.

## In Scope for v0.1

- canonical `responses_long` schema
- csv/parquet input loading
- benchmark-wise preprocessing with explicit refusal reasons
- model-level train, validation, and test splits
- cross-validated random preselection to fixed `k_preselect`
- benchmark-specific unidimensional 2PL IRT
- Fisher-information selection to fixed `k_final`
- theta estimation with uncertainty
- reduced subscores and benchmark-local linear predictors
- marginal and joint feature tables
- GAM-based full-score reconstruction
- benchmark-level redundancy, factor, and compressibility analysis
- artifact-first python API and CLI
- metabench validation mode and reduced regression fixture

## Out of Scope for v0.1

- dashboards, GUIs, or hosted services
- multidimensional item models
- arbitrary IRT family expansion
- CAT workflows or item banking
- multimodal inputs
- automatic benchmark scraping or collection
- broad experiment tracking infrastructure
- exact R-stack parity claims for metabench

## Validation Position

BenchIQ v0.1 does implement the methodology end to end.

BenchIQ v0.1 does not yet claim acceptance-grade metabench parity in the default Python-first path. The real-data reviewer bundle exists so users can inspect the current gap honestly:

- frozen public snapshot
- reproducible validation script
- reported deltas against published targets
- explicit failure verdict when the tolerance rule is not met
