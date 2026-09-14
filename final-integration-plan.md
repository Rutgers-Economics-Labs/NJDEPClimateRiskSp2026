# Final Integration Plan

## Research Question

How do short-term flood risk and resilience investment affect New Jersey municipal bond spreads?

More specifically:

- Does higher near-term flood risk increase borrowing costs?
- Does visible resilience investment through CHAMP reduce that penalty?

## Final Dataset Design

The merged analysis file should be at the:

- `bond tranche x issue date`

level.

Each bond observation will be matched to municipality-level flood risk and municipality-level CHAMP commitment measures.

## Merge Strategy

### Step 1: Prepare the bond table

Start with the cleaned bonds dataset and standardize:

- `issuer_name`
- `issue_date`
- `county`
- issuer type labels

Create a municipality crosswalk so issuer names can be linked to standardized municipal names.

### Step 2: Prepare the CHAMP table

Aggregate CHAMP projects to the municipality level and convert them into an `as-of-date` treatment measure.

For each bond issue date, compute:

- `CHAMP_dollars_asof_issue_date`
- `CHAMP_dollars_flood_related_asof_issue_date`
- `CHAMP_any_asof_issue_date`

This ensures the bond is only matched to information that would have been observable at the time of issuance.

### Step 3: Prepare the flood risk table

Assign each municipality a short-horizon flood risk measure, ideally for 2030 and possibly 2035.

Primary variable:

- `flood_risk_primary`

Examples:

- `share_of_tax_base_exposed_2030`
- `share_of_parcels_exposed_2030`

### Step 4: Merge all three

For each bond observation, merge:

- bond spread and bond controls
- municipality flood risk
- municipality CHAMP resilience measure

The merged table should contain:

- `bond_spread`
- `flood_risk_primary`
- `CHAMP_resilience_measure`
- `flood_risk_primary x CHAMP_resilience_measure`
- bond-level controls
- municipality-level controls

## Main Estimating Equation

The baseline specification is:

`bond_spread_i = beta1*flood_risk_m + beta2*CHAMP_m + beta3*(flood_risk_m * CHAMP_m) + controls_i + FE + error_i`

where:

- `i` indexes bond tranches
- `m` indexes municipalities

## Interpretation

- `beta1 > 0`: higher flood risk is associated with wider bond spreads
- `beta2 < 0`: resilience investment is associated with lower spreads
- `beta3 < 0`: CHAMP spending attenuates the spread penalty from flood risk

The interaction term is the core coefficient for the resilience question.

## Controls

### Bond-level controls

- rating
- maturity
- par amount
- coupon
- callable status
- insured status
- tax status
- sale type
- issuer type

### Municipality-level controls

- population
- median income
- debt burden
- fund balance if available
- coastal indicator
- county fixed effects or municipality fixed effects if sample size allows

### Time controls

- year fixed effects
- month or quarter fixed effects if issuance volume supports them

## Empirical Sequence

1. Estimate the relationship between flood risk and bond spreads without CHAMP.
2. Add CHAMP as a direct resilience measure.
3. Add the interaction between flood risk and CHAMP.
4. Run robustness checks with alternative CHAMP and flood-risk definitions.
5. Test whether results differ by issuer type, coastal status, or bond rating tier.

## Main Risks to Manage

- limited number of CHAMP-treated municipalities
- noisy issuer-to-municipality matching
- inconsistent timing between project commitment dates and bond issue dates
- too many bond types in the initial sample

## Expected Conclusion Structure

The final conclusion should answer four points:

1. whether short-term flood risk is priced into New Jersey municipal bonds
2. whether CHAMP-linked resilience commitments are associated with lower borrowing costs
3. whether resilience investment weakens the market penalty from flood exposure
4. how strong the evidence is given sample size and timing limitations

## Final Deliverable

The project should end with:

- one merged estimation dataset
- one baseline regression table
- a small set of robustness checks
- a concise written conclusion on flood risk, resilience, and municipal borrowing costs
