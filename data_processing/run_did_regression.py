"""
run_did_regression.py
======================
Executes the Two-Way Fixed Effects (TWFE) Difference-in-Differences estimator.

Model:
    spread_bps[i,t] = α_i  +  λ_t
                    + β1 · is_resilient[i]
                    + β2 · is_post_2022[t]
                    + β3 · (is_resilient[i] × is_post_2022[t])   ← KEY DiD coefficient
                    + β4 · debt_to_gdp[i,t]
                    + β5 · median_income[i,t]
                    + β6 · slr_exposure_pct[i]
                    + β7 · crs_class[i]
                    + ε[i,t]

    where α_i = municipality fixed effects  (absorb all time-invariant credit quality)
          λ_t = year fixed effects          (absorb aggregate interest rate movements)

Interpretation of β3 ("Resilience Premium"):
    The causal estimate of how much the 2022 CHAMP/STORM Act policy pivot
    changed the bond spread (in bps) for resilient municipalities relative
    to non-resilient municipalities, over and above any aggregate time trend.
    Negative β3 = market rewarded resilient towns with tighter spreads post-2022.

⚠ WARNING: slr_exposure_pct is currently a PROXY (flooded_pct_5ft from chars).
  Re-run this script once test_ocean_county.py finishes and you replace that column
  in final_panel_master.csv with the true tax-base-weighted SLR exposure.

Usage:
    python3 data_processing/run_did_regression.py
"""
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEANED  = os.path.join(PROJECT_ROOT, "data", "data_cleaned")
PANEL_FILE    = os.path.join(DATA_CLEANED, "final_panel_master.csv")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_regression():
    print("=" * 60)
    print("DiD REGRESSION: CLIMATE RESILIENCE PREMIUM")
    print("=" * 60)

    # ── Load panel ─────────────────────────────────────────────────────────
    df = pd.read_csv(PANEL_FILE, low_memory=False)
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    df['year']       = df['issue_date'].dt.year

    # ── Normalise numeric controls ─────────────────────────────────────────
    # Median income → $10k units (prevents tiny coefficients)
    df['median_income_10k'] = df['median_income'] / 10_000

    print(f"\n  Panel: {len(df):,} obs | {df['muni_name'].nunique()} towns | {df['year'].min()}–{df['year'].max()}")
    print(f"  Resilient towns: {df[df['is_resilient']==1]['muni_name'].nunique()}")
    print(f"  Non-resilient:   {df[df['is_resilient']==0]['muni_name'].nunique()}")

    # ── PRE-REGRESSION SANITY CHECK: Parallel Trends table ────────────────
    print("\n" + "-" * 60)
    print("PRE-TREATMENT PARALLEL TRENDS CHECK")
    print("Avg spread by group × period (should be parallel pre-2022)")
    print("-" * 60)
    pivot = (df.groupby(['is_resilient', 'is_post_2022'])['spread_bps']
               .agg(['mean','count'])
               .rename(columns={'mean':'Avg Spread (bps)', 'count':'N Trades'}))
    pivot.index = pd.MultiIndex.from_tuples(
        [(('Resilient' if r else 'Control'), ('Post-2022' if p else 'Pre-2022'))
         for r,p in pivot.index], names=['Group', 'Period'])
    print(pivot.to_string())

    # ── Model 1: Pooled OLS (The Baseline Premium) ─────────────────────────
    # To answer: "Does the market care about CRS and Flood Risk?"
    # We drop Municipal Fixed Effects to see the cross-sectional signals.
    formula_pooled = (
        "spread_bps ~ is_resilient + crs_class + slr_exposure_pct"
        " + debt_to_gdp + median_income_10k"
        " + C(year)"
    )

    # ── Model 2: TWFE DiD (The 2022 Pivot) ──────────────────────────────────
    # To answer: "Did the 2022 CHAMP policy compress spreads?"
    # We keep Municipal Fixed Effects but DROP static variables (resilience, CRS, SLR)
    # which are time-invariant and would be absorbed/distorted by the FE.
    formula_did = (
        "spread_bps ~ is_resilient:is_post_2022"
        " + debt_to_gdp + median_income_10k"
        " + C(muni_name) + C(year)"
    )

    # ── Model 3: TWFE + SLR Interaction ─────────────────────────────────────
    # Tests if the policy effect varies by the magnitude of tax base at risk.
    formula_slr_int = (
        "spread_bps ~ is_resilient:is_post_2022"
        " + is_post_2022:slr_exposure_pct"
        " + debt_to_gdp + median_income_10k"
        " + C(muni_name) + C(year)"
    )

    print("\n" + "-" * 60)
    print("Running regressions…")
    print("-" * 60)

    models = {}

    # Model 1 – Pooled
    print("  [1/3] Model A: Pooled OLS (Climate Premiums)…")
    m1 = smf.ols(formula_pooled, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['muni_name']}
    )
    models['(A) Pooled OLS'] = m1

    # Model 2 – TWFE DiD
    print("  [2/3] Model B: TWFE DiD (Causal Pivot)…")
    m2 = smf.ols(formula_did, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['muni_name']}
    )
    models['(B) TWFE DiD'] = m2

    # Model 3 – TWFE SLR
    print("  [3/3] Model C: TWFE + SLR Interaction…")
    m3 = smf.ols(formula_slr_int, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['muni_name']}
    )
    models['(C) TWFE + SLR Int'] = m3

    # ── Key Coefficient Extraction ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    
    # Model A checks
    print("  Model A (Baseline):")
    for var in ['crs_class', 'slr_exposure_pct', 'median_income_10k']:
        if var in m1.params:
            print(f"    {var:20s} β = {m1.params[var]:+.3f}  p={m1.pvalues[var]:.4f}")

    # Model B/C checks
    print("\n  Model B/C (DiD):")
    for name, m in {'B': m2, 'C': m3}.items():
        coef = m.params.get('is_resilient:is_post_2022', 0)
        p = m.pvalues.get('is_resilient:is_post_2022', 1)
        print(f"    Model {name} β3 (DiD)      = {coef:+.3f}  p={p:.4f}")

    # ── Full Summary Table ─────────────────────────────────────────────────
    # Only report the key non-FE coefficients for readability
    info_dict = {'N': lambda m: f"{int(m.nobs):,}", 'R²': lambda m: f"{m.rsquared:.3f}"}
    regressor_order = [
        'Intercept',
        'is_resilient', 'is_post_2022', 'is_resilient:is_post_2022',
        'debt_to_gdp', 'median_income_10k', 'slr_exposure_pct', 'crs_class',
        'is_post_2022:slr_exposure_pct',
    ]

    # Filter to only variables that actually exist across models
    existing = [r for r in regressor_order
                if any(r in m.params.index for m in models.values())]

    print("\n" + "=" * 60)
    print("FULL RESULTS TABLE (FE coefficients suppressed)")
    print("=" * 60)
    table = summary_col(
        list(models.values()),
        model_names=list(models.keys()),
        stars=True,
        regressor_order=existing,
        drop_omitted=True,
        info_dict=info_dict,
    )
    print(table)

    # ── Save outputs ───────────────────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, "did_results.txt")
    with open(results_path, 'w') as f:
        f.write("DiD Regression: NJ Climate Resilience Bond Spread Premium\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(table))
        f.write("\n\n[FULL TWFE MODEL]\n")
        f.write(m2.summary().as_text())

    print(f"\nFull results saved → {results_path}")
    print("\n⚠ REMINDER: Re-run after replacing slr_exposure_pct with real SLR data.")

if __name__ == '__main__':
    run_regression()
