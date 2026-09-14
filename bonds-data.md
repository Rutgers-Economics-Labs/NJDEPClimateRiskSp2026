# Bonds Data

## Goal

Build the dependent variable for the project: municipal bond spreads for New Jersey issuers.

## Core Output

A bond-issue-level dataset where each row is a municipal bond issue or maturity tranche and the main outcome is:

`bond_spread = offering_yield - benchmark_AAA_yield`

## Preferred Source

- `MSRB EMMA`

EMMA is the official source for municipal bond issuance and trade information. The first-pass dataset should focus on new issues rather than secondary trades because spread measurement is cleaner at issuance.

## Unit of Observation

- Preferred: `CUSIP maturity tranche`
- Acceptable fallback: `bond issue`

Using tranche-level data is better because maturity-specific yield differences matter for spread construction.

## Variables to Collect

### Bond pricing and structure

- `issuer_name`
- `issuer_type`
- `issue_date`
- `sale_date`
- `CUSIP`
- `par_amount`
- `maturity_date`
- `years_to_maturity`
- `coupon`
- `offering_yield`
- `callable`
- `tax_status`
- `insured`
- `sale_type` (competitive vs negotiated)
- `security_type` (GO, revenue, utility, school, etc.)

### Credit quality

- `rating_moodys`
- `rating_sp`
- `rating_fitch`
- `rating_numeric` (harmonized scale)

### Derived variables

- `benchmark_date`
- `benchmark_maturity`
- `benchmark_AAA_yield`
- `bond_spread`
- `log_par_amount`

## Benchmark for Spread Construction

Use a same-day municipal AAA benchmark curve matched as closely as possible to the bond's maturity.

Examples:

- EMMA benchmark products if accessible
- Tradeweb AAA municipal curve
- another documented AAA municipal benchmark consistently available across the sample

The benchmark source must be the same for all bonds.

## Sample Restrictions

Recommended first-pass sample:

- New Jersey municipal issuers only
- primary market issues
- tax-exempt bonds
- general obligation and essential-service revenue bonds
- exclude variable-rate and highly structured bonds if spread measurement is noisy

## Key Challenges

- Full EMMA bulk access may be limited
- Some issues have multiple maturities and complex structures
- Ratings may be missing or inconsistent
- Spread measurement will be noisy if issue yields are not aligned with the benchmark date

## Deliverable

A cleaned table named conceptually like:

`bond_issues_clean`

with one row per bond tranche and a defensible spread measure.
