# NJDEP Climate Risk and Municipal Bond Pricing

This repository builds a New Jersey municipality-level panel linking climate exposure, resilience program participation, fiscal controls, and municipal bond secondary-market spreads. The current analysis is an exploratory screening workflow for whether sea-level-rise exposure and CHAMP participation are reflected in municipal borrowing-cost proxies.

## Current Research Design

- **Outcome:** secondary-market municipal bond spread over a synthetic AAA benchmark, in basis points.
- **Climate exposure:** share of municipal market value exposed under the 4-foot sea-level-rise scenario.
- **Resilience / policy cohort:** municipalities appearing in available parsed NJ CHAMP applicant PDFs.
- **Policy timing:** `is_post_2023`, equal to 1 for trades on or after January 1, 2023, when CHAMP is treated as active in this analysis.
- **Controls and structure:** time to maturity, selected fiscal and demographic controls, municipality fixed effects, year fixed effects, and municipality-clustered standard errors.

The latest regression outputs are in `results/did_results.txt`, with slide-ready table images in `results/tables/`.

## Main Interpretation

The fixed-effects DiD models show post-2023 spread compression for CHAMP-cohort municipalities relative to non-CHAMP municipalities. The coastal resilience triple interaction, `ever_champ x post_2023 x slr_exposure_pct`, is small and not statistically significant, so the current evidence does not show that the CHAMP association is stronger specifically for higher-SLR-exposure municipalities.

This should be presented as exploratory evidence, not a final causal estimate. The pre-trend diagnostic shows CHAMP-cohort towns were already moving differently before 2023.

## Key Limitations

- The bond panel uses secondary-market trades, not primary issuance yields.
- The AAA benchmark is synthetic, not a Bloomberg/MMD AAA curve.
- The local bond panel is missing 2020; the analysis mitigates this by using multiple pre-treatment years, multiple post-2023 years, municipality fixed effects, and year fixed effects rather than relying on a single before/after year.
- The CHAMP cohort is based on locally parsed applicant PDFs and should be replaced with a fully validated NJ I-Bank / Intended Use Plan treatment history for policy-grade work.
- Important bond-level controls remain incomplete, including rating, insurance, tax status, callability, issue size, trade size, and liquidity.

## Repository Structure

```text
.
├── data/
│   ├── data_raw/          # Source data: WRDS, ACS, CHAMP, Water Bank, GIS inputs
│   ├── data_cleaned/      # Generated panels, finance outputs, climate outputs
│   └── data_zipped/       # Archived source downloads
├── data_processing/       # Processing, regression, export, and diagnostic scripts
├── dashboard/             # FastAPI backend and React frontend
├── docs/                  # Scope documents and project references
└── results/               # Model outputs, summaries, and presentation tables
```

## Reproduce the Current Analysis

Install dependencies:

```bash
make install
```

Run the full processing pipeline:

```bash
make process-all
```

Rebuild the regression-ready panel only:

```bash
python3 data_processing/build_master_panel.py
```

Rerun the DiD regressions:

```bash
python3 data_processing/run_did_regression.py
```

Regenerate dashboard coefficients and presentation tables:

```bash
python3 data_processing/export_did_coefficients.py
python3 data_processing/export_paper_tables.py
```

## Dashboard

Start the local dashboard:

```bash
make dev
```

Frontend: `http://127.0.0.1:5173`

Backend: `http://127.0.0.1:8000`

The dashboard supports exploratory municipality, county, and statewide views of secondary-market spread movements relative to the synthetic AAA benchmark.

## Important Outputs

- `data/data_cleaned/final_panel_master.csv`
- `results/did_results.txt`
- `results/project_summary.txt`
- `results/tables/regression_results_current.png`
- `results/tables/data_dictionary_current.png`
- `dashboard/frontend/public/data/did_coefficients.json`

## Recommended Next Steps

- Validate CHAMP treatment timing and cohort membership against NJ I-Bank source records.
- Replace the synthetic AAA proxy with a market-standard benchmark curve.
- Add bond-level controls for ratings, insurance, callability, tax status, issue size, trade size, and liquidity.
- Improve post-2021 WRDS issuer matching and document unmatched municipalities.
