# NJDEP Climate Risk and Resilience Project

This repository assembles New Jersey municipality-level climate, resilience, demographic, and bond market data into a single workflow for exploring whether resilience and flood exposure are reflected in municipal borrowing costs.

The current dashboard focuses on a fast demo question:

"For any municipality, county, or the full state, how does observed bond pricing move relative to a synthetic AAA benchmark over time, and what climate or resilience context sits behind that geography?"

## What is in this repo

- Climate exposure processing from municipal boundaries and sea level rise layers
- FEMA CRS resilience cleaning and municipality-level resilience flags
- Available CHAMP applicant PDF parsing and municipality-level cohort flags
- Census and finance cleaning for municipality characteristics
- WRDS bond extraction and matching for New Jersey municipal trades
- A FastAPI + React dashboard for interactive exploration

## End-to-end workflow

### 1. Build municipality reference data

The project first creates a common municipality reference layer so all downstream sources can be joined at the municipal level.

- `data_processing/process_boundaries.py`
  Builds cleaned New Jersey municipal boundary outputs and municipality lists.
- `data_processing/process_climate.py`
  Intersects municipalities with sea level rise flood surfaces and produces municipality flood-area summaries.
- `data_processing/process_fema_crs.py`
  Cleans FEMA Community Rating System data and derives municipality-level CRS participation variables.
- `data_processing/process_census.py`
  Adds demographic and income controls.
- `data_processing/process_finance.py`
  Processes available municipal finance files.
- `data_processing/process_champ_scores.py`
  Parses available CHAMP applicant PDFs into raw applicant rows for municipal matching.
- `data_processing/build_master_panel.py`
  Merges the cleaned municipality-level components and bond trades into `data/data_cleaned/final_panel_master.csv`.

The key municipality master file currently includes:

- flood exposure at 2ft, 5ft, and 7ft SLR scenarios
- CRS class and CRS discount information
- CRS and CHAMP-derived resilience indicators
- selected census controls

### 2. Build bond analytics

- `data_processing/process_wrds_data.py`
  Reads yearly WRDS trade files, filters likely New Jersey municipal bond observations, attaches a deterministic synthetic AAA benchmark, computes spreads, and produces dashboard-ready time series.

Primary bond outputs:

- `data/data_cleaned/nj_bonds_analytics.csv`
- `data/data_cleaned/premium_timeseries_muni.csv`
- `data/data_cleaned/premium_timeseries_county.csv`
- `data/data_cleaned/premium_timeseries_state.csv`
- `data/data_cleaned/premium_lookup.csv`

### 3. Serve the dashboard

- `dashboard/backend/main.py`
  Serves FastAPI endpoints for options, summaries, and time series.
- `dashboard/frontend/src/App.jsx`
  Provides the municipality/county/state explorer, premium chart, synthetic AAA chart, and climate/resilience detail panels.

## Important current scope notes

- The dashboard currently uses a synthetic AAA proxy rather than a live market AAA benchmark.
- The bond matching layer is usable for interactive exploration, but it is still conservative and does not yet recover all municipalities or a full validated 2015-2025 panel.
- Available CHAMP applicant data is integrated as an exploratory cohort/timing flag, but it is not a complete validated I-Bank treatment history.
- The dashboard is optimized for exploratory demo use, not final econometric estimation.

## Key commands

Install dependencies:

```bash
make install
```

Run the full processing pipeline:

```bash
make process-all
```

Start the dashboard locally:

```bash
make dev
```

Frontend URL:

```text
http://127.0.0.1:5173
```

Backend URL:

```text
http://127.0.0.1:8000
```

## Data folders

- `data/data_raw`
  Raw source files including WRDS yearly trade extracts, CHAMP PDFs, Water Bank/PPL files, and climate inputs.
- `data/data_cleaned`
  Processed municipality characteristics, bond analytics, and dashboard-ready outputs.
- `data_processing/diagnostics`
  One-off diagnostic scripts that are not part of the main processing pipeline.
- `docs`
  Project scope documents and narrative references.
- `dashboard/frontend/public`
  Static dashboard assets including the REL tab icon and NJDEP logo.

## Repository layout

```text
.
├── data/
│   ├── data_raw/          # Source data: WRDS, ACS, CHAMP, Water Bank, GIS inputs
│   ├── data_cleaned/      # Generated panels, finance outputs, climate outputs
│   └── data_zipped/       # Archived source downloads
├── data_processing/       # Processing, regression, export, and diagnostics scripts
├── dashboard/             # FastAPI backend and React frontend
├── docs/                  # Scope documents and project references
└── results/               # Written model outputs and summaries
```

## Suggested next steps after the demo

- Replace exploratory CHAMP PDF-derived flags with a validated I-Bank / Intended Use Plan treatment history
- Improve post-2021 WRDS issuer matching
- Replace the synthetic AAA proxy with a true benchmark series
- Add formal econometric outputs once the market panel is fully validated
