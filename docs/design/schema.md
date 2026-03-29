# BenchIQ Schema

BenchIQ v0.1 is centered on a canonical long-format item-response table called `responses_long`.

## Required Tables

BenchIQ accepts up to three tables:

- `responses_long` required
- `items` optional
- `models` optional

If `items` or `models` are omitted, BenchIQ derives them from `responses_long` and records that derivation in `manifest.json`.

## responses_long

Required columns:

- `model_id` string, required, non-null
- `benchmark_id` string, required, non-null
- `item_id` string, required, non-null
- `score` nullable binary integer in `{0,1}`

Optional columns currently recognized by the canonical schema:

- `split` string
- `weight` numeric

Primary key:

- `(model_id, benchmark_id, item_id)`

Rules:

- duplicate primary keys are a hard failure when `duplicate_policy="error"`
- whitespace is canonicalized away from string key fields during stage-00 coercion
- non-binary `score` values are a hard validation failure
- missing key fields are a hard validation failure

## items

Required columns:

- `benchmark_id` string, required, non-null
- `item_id` string, required, non-null

Optional columns currently recognized:

- `content_hash` string

Primary key:

- `(benchmark_id, item_id)`

## models

Required columns:

- `model_id` string, required, non-null

Optional columns currently recognized:

- `model_family` string

Primary key:

- `(model_id,)`

## Accepted File Formats

BenchIQ v0.1 accepts only:

- `.csv`
- `.parquet`

The internal artifact format is:

- parquet for tables
- json for reports, manifests, and diagnostics

## Stage-00 Outputs

`benchiq validate` and `benchiq run` both start by writing:

- `artifacts/00_canonical/responses_long.parquet`
- `artifacts/00_canonical/items.parquet`
- `artifacts/00_canonical/models.parquet`
- `artifacts/00_canonical/canonicalization_report.json`

The canonicalization report includes:

- input source metadata
- file hashes where applicable
- validation report payload with structured warnings/errors
- resolved duplicate policy

## Validation Objects

BenchIQ surfaces validation state through structured reports rather than free-form logs.

Important objects:

- `ValidationIssue`
  - `level`
  - `code`
  - `message`
  - `table_name`
  - `row_count`
  - `context`
- `ValidationReport`
  - `ok`
  - `counts`
  - `summary`
  - `warnings`
  - `errors`

This is what powers:

- `reports/validation_report.json`
- `reports/validation_summary.md`
- stage-specific refusal and warning artifacts

## Inspectable Artifact Contract

BenchIQ treats these as first-class outputs:

- validation failures
- skipped stages
- refused benchmarks
- warning payloads
- stage manifests and resolved config

If a benchmark or stage is skipped, the reason should appear in a json report instead of being hidden in console output.
