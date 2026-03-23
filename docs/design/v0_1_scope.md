# BenchIQ v0.1 scope

BenchIQ v0.1 is a reusable python workflow for user-chosen benchmark bundles. It is not a metabench-only reproduction.

metabench remains the methodological reference case and the primary validation harness.

v0.1 stays narrow:

- canonical long-format item-response table
- benchmark-wise preprocessing
- model-level train, validation, and test splits
- cross-validated preselection to fixed benchmark budgets
- benchmark-specific unidimensional 2pl irt
- fisher-information item selection
- latent ability estimation with uncertainty
- gam-based score reconstruction
- benchmark-level redundancy and compressibility analysis
- artifact-first cli and python api

v0.1 does not include dashboards, guis, cat workflows, multimodal support, automatic data collection, or broad psychometrics-framework ambitions.
