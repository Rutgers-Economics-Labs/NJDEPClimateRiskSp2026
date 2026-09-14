# Flood Risk Data

## Goal

Measure short-term flood risk over roughly a 5-10 year horizon so the project captures near-term climate exposure rather than distant end-of-century scenarios.

## Core Output

A municipality-level flood exposure dataset with a planning horizon anchored around:

- `2030`
- `2035`

These horizons match the project's short-term objective.

## Preferred Sources

- `First Street` projected flood risk data
- `NJFloodMapper` for New Jersey-specific spatial reference
- additional parcel, land value, or tax-base layers if needed for exposure weighting

## Unit of Observation

- Primary: `municipality`
- Raw inputs may be at parcel, building, grid-cell, or census-geography level

The raw spatial data should be aggregated to the municipality level because the bond and CHAMP data will merge most naturally there.

## Candidate Risk Measures

At least one of these should be chosen as the primary flood regressor:

- `share_of_parcels_exposed`
- `share_of_buildings_exposed`
- `expected_annual_loss`
- `average_flood_probability`
- `share_of_land_area_exposed`
- `share_of_tax_base_exposed`

## Recommended Primary Variable

Best first choice:

- `share_of_tax_base_exposed_2030`

If tax-base matching is too difficult, use:

- `share_of_parcels_exposed_2030`

This keeps the interpretation clear and the data pipeline manageable.

## Variables to Collect or Derive

- `municipality_name`
- `county`
- `flood_horizon_year`
- `parcels_total`
- `parcels_exposed`
- `buildings_exposed`
- `land_area_exposed`
- `assessed_value_exposed`
- `expected_annual_loss`
- `flood_risk_measure_primary`

## Aggregation Logic

If the raw data are spatial:

1. assign each parcel or exposure unit to a municipality
2. calculate exposure counts or dollars
3. normalize by municipality size, population, or tax base
4. produce one municipality-level record per horizon year

## Key Challenges

- Flood datasets may use different risk definitions and return periods
- Some sources emphasize coastal flooding while others include pluvial or riverine flooding
- Municipality boundary matching must be consistent across all sources
- A single exposure metric should be fixed before estimation to avoid specification drift

## Deliverable

A cleaned table conceptually named:

`flood_risk_muni`

with municipality-level short-horizon flood exposure measures for 2030 and, if available, 2035.
