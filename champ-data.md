# CHAMP Data

## Goal

Measure municipality-level resilience investment using exact dollar amounts committed through CHAMP as a proxy for flood and stormwater mitigation.

## Core Output

A municipality-level panel of CHAMP-related commitments where each row is a `municipality x date` or `municipality x fiscal year` observation.

## Program Context

CHAMP is recent, so the project should treat it as a forward-looking resilience signal rather than a long historical series. The relevant measure is not just whether a town participates, but how many dollars are tied to resilience-oriented projects.

## Preferred Sources

- `NJ CHAMP annual reports`
- `Project Priority Lists (PPLs)`
- `amended PPLs`
- supporting NJ Infrastructure Bank or NJOEM documents

## Unit of Observation

Two useful structures should be built:

1. `project-level table`
2. `municipality-level aggregated table`

The project-level table is the raw layer. The municipality-level table is what will merge into the bond dataset.

## Variables to Collect

### Project-level fields

- `project_id`
- `municipality_name`
- `county`
- `project_title`
- `project_description`
- `program_year`
- `announcement_date`
- `estimated_total_cost`
- `estimated_CHAMP_loan_amount`
- `project_status`
- `project_category`

### Classification fields

- `flood_related`
- `stormwater_related`
- `drainage_related`
- `resilience_related`

These should be coded from project descriptions so that the final measure captures the subset of CHAMP spending most relevant to flood mitigation.

### Municipality-level derived variables

- `CHAMP_dollars_total`
- `CHAMP_dollars_flood_related`
- `CHAMP_projects_count`
- `CHAMP_any`
- `CHAMP_dollars_per_capita`
- `CHAMP_dollars_per_square_mile`
- `CHAMP_dollars_asof_date`

## Recommended Construction

Primary treatment variable:

- `CHAMP_dollars_flood_related_asof_issue_date`

Robustness measures:

- `estimated_total_cost` instead of loan amount
- indicator for `any CHAMP project`
- count of CHAMP projects
- per-capita scaling

## Key Challenges

- CHAMP is new, so the number of treated municipalities may be small
- Some project descriptions may be vague and require manual classification
- Municipality names may not exactly match bond issuer names
- Commitment date must be defined consistently: PPL date, approval date, or financial close date

## Deliverable

Two cleaned tables conceptually named:

- `champ_projects_clean`
- `champ_muni_panel`

The municipality-level panel should be ready to merge by municipality and issue date.
