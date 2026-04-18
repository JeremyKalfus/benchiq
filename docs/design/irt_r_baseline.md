# irt r-baseline parity note

This note defines what the repo means by an "R-baseline comparison" for the optional IRT parity
work.

## what identical means

The comparison uses the exact same simulated dichotomous response matrix for both implementations.

- the same item discrimination values
- the same item difficulty values
- the same latent ability vector used to generate responses
- the same realized binary response matrix

The R baseline and the BenchIQ path therefore start from the same observed data, not merely the
same random seed.

## why exact parameter equality is not the right target

Unidimensional 2PL models are not identifiable in a naive byte-for-byte sense across
implementations.

Two fitted solutions can represent the same item characteristic curves while differing by:

- latent sign
- latent scale
- latent location

Because of that, this comparison aligns the R baseline onto the BenchIQ scale before comparing item
parameters directly.

## alignment used in the harness

The harness:

1. compares BenchIQ and R theta estimates on the same response patterns
2. flips the R orientation if the raw theta rank correlation is negative
3. fits a linear transform from BenchIQ theta to R theta
4. maps R item difficulties and discriminations back onto the BenchIQ scale

After alignment, the repo compares:

- item discrimination agreement
- item difficulty agreement
- theta Pearson and Spearman agreement
- item characteristic curve RMSE on a fixed theta grid

## tolerance philosophy

The goal is honest implementation comparison, not a fake "exact match" claim.

The harness therefore treats these as the primary acceptance signals:

- strong theta ordering agreement
- small item-characteristic-curve differences
- reasonable aligned item-parameter error

If `Rscript` or the R package `mirt` is unavailable locally, the harness writes a clear skipped
report instead of silently falling back.
