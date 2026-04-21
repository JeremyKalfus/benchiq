# BenchIQ v0.1 Scope

BenchIQ v0.1 is a narrow, inspectable workflow for benchmark-bundle distillation and overlap analysis from user-provided item-level model responses.

## Product Identity

BenchIQ is:

- a generic benchmark-bundle tool
- artifact-first and deterministic
- designed for many-model, many-checkpoint evaluation settings
- focused on reduced subsets, score reconstruction, and cross-benchmark redundancy

BenchIQ is not:

- a hosted platform
- a dashboard product
- a broad psychometrics toolkit
- a generic data-collection system

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
- saved optimization and deployment evaluation bundles under `reports/`

## Out of Scope for v0.1

- dashboards, GUIs, or hosted services
- multidimensional item models
- arbitrary IRT family expansion
- CAT workflows or item banking
- multimodal inputs
- automatic benchmark scraping or collection
- broad experiment tracking infrastructure
- parity-only benchmark-comparison work as a product goal

## Validation Position

BenchIQ v0.1 does implement the methodology end to end.

BenchIQ v0.1 evaluates itself with held-out reconstruction quality, reproducibility, and deployment usefulness. The repo keeps saved product-facing evidence bundles under `reports/` so users can inspect the current state honestly:

- saved optimization summaries
- saved deployment validation artifacts
- the narrowed public-portfolio standing baseline under `reports/portfolio_standing/`
- the iterative narrowed public-portfolio best-so-far record under
  `reports/portfolio_optimization_cycles/`
- explicit recommendation and default-profile updates only when the evidence improves
