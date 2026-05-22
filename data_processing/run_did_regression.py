"""
run_did_regression.py
======================
Executes exploratory regressions for the NJ climate risk / CHAMP panel.

Important limits:
    The dependent variable is built from secondary-market WRDS trades and a
    synthetic AAA benchmark. These models are screening diagnostics, not final
    causal estimates of issuance costs.

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


def fmt_result(model, term):
    if term not in model.params:
        return "not included in this specification"
    coef = model.params[term]
    p = model.pvalues[term]
    return f"{coef:+.3f} bps (p={p:.4f})"


def write_model_interpretations(handle, models, pretrend_coef, pretrend_p):
    m1 = models['(A) Pooled OLS']
    m2 = models['(B) TWFE DiD']
    m3 = models['(C) TWFE Coastal DDD']
    m4 = models['(D) Event Study']

    handle.write("\n\n[MODEL INTERPRETATION]\n")
    handle.write("Model A: Pooled OLS\n")
    handle.write(
        "This model compares all observations after controlling for the post-2023 CHAMP period, "
        "time to maturity, median income, and debt-to-AV. It does not include municipality "
        "fixed effects, so cross-town differences may still reflect unobserved credit quality, "
        "wealth, geography, or bond-market composition. "
    )
    handle.write(f"The CHAMP cohort coefficient is {fmt_result(m1, 'ever_champ')}; ")
    handle.write(f"the SLR exposure coefficient is {fmt_result(m1, 'slr_exposure_pct')}; ")
    handle.write(f"the debt-to-AV coefficient is {fmt_result(m1, 'debt_to_av')}.\n\n")

    handle.write("Model B: Two-way fixed effects DiD\n")
    handle.write(
        "This model asks whether CHAMP-cohort municipalities changed differently in the post-2023 CHAMP period, "
        "using municipality and year fixed effects and controlling for time to maturity. "
        "Debt-to-AV is intentionally excluded from this specification. "
    )
    handle.write(
        f"The CHAMP cohort x post-2023 coefficient is {fmt_result(m2, 'ever_champ:is_post_2023')}. "
        "Interpret it as the average post-2023 spread change for CHAMP-cohort municipalities "
        "relative to non-cohort municipalities within the exploratory secondary-trade panel.\n\n"
    )

    handle.write("Model C: Coastal triple-difference\n")
    handle.write(
        "This model extends Model B by asking whether the post-2023 CHAMP-cohort change varies "
        "with 4-foot SLR tax-base exposure. It includes municipality and year fixed effects and "
        "time to maturity; debt-to-AV is intentionally excluded. "
    )
    handle.write(f"The post-2023 x SLR coefficient is {fmt_result(m3, 'is_post_2023:slr_exposure_pct')}; ")
    handle.write(
        f"the CHAMP cohort x post-2023 x SLR coefficient is "
        f"{fmt_result(m3, 'ever_champ:is_post_2023:slr_exposure_pct')}. "
        "The triple interaction is the main coastal-resilience diagnostic: it is measured in "
        "basis points per one percentage point of SLR-exposed municipal market value.\n\n"
    )

    handle.write("Model D: Event-study style CHAMP-by-year model (within-demeaned)\n")
    handle.write(
        "This model interacts CHAMP-cohort status with each year to inspect whether the cohort "
        "already had different spread patterns before the post-2023 CHAMP period. It is primarily a "
        "diagnostic for pre-trends rather than the main treatment estimate. "
    )
    handle.write(
        f"The separate linear pre-trend check (with muni FEs) gives {pretrend_coef:+.3f} bps/year "
        f"(p={pretrend_p:.4f}). "
    )
    if pretrend_p < 0.10:
        handle.write("A statistically meaningful pre-trend weakens a causal DiD "
                     "interpretation of Models B and C.\n")
    else:
        handle.write("The absence of a statistically significant pre-trend supports "
                     "the parallel trends assumption underlying Models B and C.\n")


def run_regression():
    print("=" * 60)
    print("REGRESSION: CLIMATE RISK AND CHAMP RESILIENCE PREMIUM")
    print("=" * 60)

    # ── Load panel ─────────────────────────────────────────────────────────
    df = pd.read_csv(PANEL_FILE, low_memory=False)
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    df['year']       = df['issue_date'].dt.year
    if "debt_to_gdp" in df.columns and "debt_to_av" not in df.columns:
        df = df.rename(columns={"debt_to_gdp": "debt_to_av"})

    required = [
        "spread_bps", "time_to_maturity", "debt_to_av", "median_income",
        "slr_exposure_pct", "ever_champ", "is_post_2023", "muni_name",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{PANEL_FILE} is missing required columns {missing}. "
            "Re-run data_processing/build_master_panel.py after process_wrds_data.py."
        )

    # ── Normalise numeric controls ─────────────────────────────────────────
    # Median income → $10k units (prevents tiny coefficients)
    df['median_income_10k'] = df['median_income'] / 10_000
    df["ever_champ"] = df["ever_champ"].fillna(0).astype(int)
    df["is_resilient"] = df.get("is_resilient", 0)

    support = df.groupby('muni_name')['is_post_2023'].agg(
        pre=lambda s: int((s == 0).sum()),
        post=lambda s: int((s == 1).sum()),
    )
    both_period_munis = support[(support['pre'] > 0) & (support['post'] > 0)].index
    df = df[df["muni_name"].isin(both_period_munis)].copy()

    low, high = df["spread_bps"].quantile([0.01, 0.99])
    df["spread_bps_winsor"] = df["spread_bps"].clip(lower=low, upper=high)

    print(f"\n  Panel: {len(df):,} obs | {df['muni_name'].nunique()} towns | {df['year'].min()}–{df['year'].max()}")
    print(f"  Ever-CHAMP cohort: mean={df['ever_champ'].mean():.2f}")
    print(f"  Time-varying CHAMP active: mean={df['is_resilient'].mean():.2f}")
    print(f"  SLR exposure: mean={df['slr_exposure_pct'].mean():.2f}%, max={df['slr_exposure_pct'].max():.2f}%")
    print(f"  Avg raw spread: all={df['spread_bps'].mean():.2f} bps | pre={df.loc[df['is_post_2023']==0, 'spread_bps'].mean():.2f} | post={df.loc[df['is_post_2023']==1, 'spread_bps'].mean():.2f}")
    print(f"  Winsorized spread bounds: p1={low:.2f} bps | p99={high:.2f} bps")

    both_periods = ((support['pre'] > 0) & (support['post'] > 0)).sum()
    pre_only = ((support['pre'] > 0) & (support['post'] == 0)).sum()
    post_only = ((support['pre'] == 0) & (support['post'] > 0)).sum()
    print(f"  Town support: {both_periods} both-period | {pre_only} pre-only | {post_only} post-only")
    print("\n  Observations by year:")
    print(df['year'].value_counts().sort_index().to_string())

    # ── PRE-REGRESSION SANITY CHECK: pre-treatment trends table ───────────
    print("\n" + "-" * 60)
    print("PRE-TREATMENT TREND CHECK")
    print("Avg spread by CHAMP resilience × pre-treatment year")
    print("-" * 60)
    df['resilience_group'] = df['ever_champ'].map({0: 'Not CHAMP Cohort', 1: 'Ever CHAMP Cohort'})
    pre = df[df['is_post_2023'] == 0].copy()
    pre_trends = (pre.groupby(['year', 'resilience_group'], observed=True)['spread_bps']
                    .agg(['mean', 'count'])
                    .rename(columns={'mean': 'Avg Spread (bps)', 'count': 'N Trades'}))
    print(pre_trends.to_string())

    # Linear pre-trend with municipality FEs for consistency with TWFE models
    pre['pre_year_index'] = pre['year'] - pre['year'].min()
    trend_formula = (
        "spread_bps_winsor ~ ever_champ:pre_year_index"
        " + time_to_maturity"
        " + C(muni_name) + C(year)"
    )
    pre_trend_model = smf.ols(trend_formula, data=pre).fit(
        cov_type='cluster', cov_kwds={'groups': pre['muni_name']}
    )
    trend_term = 'ever_champ:pre_year_index'
    pretrend_coef = np.nan
    pretrend_p = np.nan
    if trend_term in pre_trend_model.params:
        pretrend_coef = pre_trend_model.params[trend_term]
        pretrend_p = pre_trend_model.pvalues[trend_term]
        print(f"\n  Linear CHAMP differential pre-trend (with muni FE): β={pretrend_coef:+.3f} bps/year  p={pretrend_p:.4f}")

    # ── Model 1: Pooled OLS (The Baseline Premium) ─────────────────────────
    # To answer: "Does the market care about CHAMP participation and SLR exposure?"
    formula_pooled = (
        "spread_bps_winsor ~ ever_champ + slr_exposure_pct"
        " + debt_to_av + median_income_10k + time_to_maturity"
        " + is_post_2023"
    )

    # ── Model 2: TWFE DiD (The post-2023 CHAMP period) ─────────────────────
    # To answer: "Did the post-2023 period change the CHAMP/spread relationship?"
    formula_did = (
        "spread_bps_winsor ~ ever_champ:is_post_2023"
        " + time_to_maturity"
        " + C(muni_name) + C(year)"
    )

    # ── Model 3: TWFE Triple-DiD (Coastal Resilience) ──────────────────────
    # Tests if CHAMP resilience specifically reduced spreads for coastal towns post-2023
    formula_slr_int = (
        "spread_bps_winsor ~ ever_champ:is_post_2023"
        " + is_post_2023:slr_exposure_pct"
        " + ever_champ:is_post_2023:slr_exposure_pct"
        " + time_to_maturity"
        " + C(muni_name) + C(year)"
    )

    # ── Model 4: Event Study ───────────────────────────────────────────────
    # Uses within-municipality demeaning to avoid perfect collinearity between
    # time-invariant ever_champ and C(muni_name) in the unbalanced panel.
    # Reference year = 2021 (last pre-treatment year).
    #
    # For the event study we also require municipalities to have observations
    # in at least 3 distinct years, so that the within-demeaning is well-
    # identified and muni-year gaps don't create near-singular columns.
    es_yr_coverage = df.groupby('muni_name')['year'].nunique()
    es_munis = es_yr_coverage[es_yr_coverage >= 3].index
    df_es = df[df['muni_name'].isin(es_munis)].copy()
    print(f"  Event study panel: {len(df_es):,} obs | {df_es['muni_name'].nunique()} towns "
          f"(dropped {df['muni_name'].nunique() - df_es['muni_name'].nunique()} with <3 years)")

    # Within-municipality demeaning: subtract muni-level means from both
    # the dependent variable and the time-varying controls.
    muni_means = df_es.groupby('muni_name')[['spread_bps_winsor', 'time_to_maturity']].transform('mean')
    df_es['spread_dm'] = df_es['spread_bps_winsor'] - muni_means['spread_bps_winsor']
    df_es['ttm_dm'] = df_es['time_to_maturity'] - muni_means['time_to_maturity']

    # Build explicit CHAMP × year dummies, omitting the reference year (2021)
    ref_year = 2021
    all_years = sorted(df_es['year'].unique())
    event_years = [y for y in all_years if y != ref_year]
    for y in event_years:
        df_es[f'champ_x_{y}'] = (df_es['ever_champ'] * (df_es['year'] == y)).astype(int)

    champ_year_terms = ' + '.join(f'champ_x_{y}' for y in event_years)
    year_dummies = ' + '.join(f'I(year == {y})' for y in event_years)  # ref year omitted
    formula_event = (
        f"spread_dm ~ {champ_year_terms} + ttm_dm + {year_dummies}"
    )

    print("\n" + "-" * 60)
    print("Running regressions…")
    print("-" * 60)

    models = {}

    # Model 1 – Pooled
    print("  [1/4] Model A: Pooled OLS (winsorized secondary-trade spread)…")
    m1 = smf.ols(formula_pooled, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['muni_name']}
    )
    models['(A) Pooled OLS'] = m1

    # Model 2 – TWFE DiD
    print("  [2/4] Model B: TWFE DiD CHAMP × Post-2023…")
    m2 = smf.ols(formula_did, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['muni_name']}
    )
    models['(B) TWFE DiD'] = m2

    # Model 3 – TWFE Triple-DiD SLR
    print("  [3/4] Model C: TWFE Triple-DiD (CHAMP × Post × SLR)…")
    m3 = smf.ols(formula_slr_int, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['muni_name']}
    )
    models['(C) TWFE Coastal DDD'] = m3

    # Model 4 – Event Study (demeaned)
    print("  [4/4] Model D: Event Study (within-demeaned, CHAMP × Year, ref=2021)…")
    m4 = smf.ols(formula_event, data=df_es).fit(
        cov_type='cluster', cov_kwds={'groups': df_es['muni_name']}
    )
    models['(D) Event Study'] = m4

    # ── Key Coefficient Extraction ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    
    # Model A checks
    print("  Model A (Baseline):")
    for var in ['ever_champ', 'slr_exposure_pct', 'median_income_10k']:
        if var in m1.params:
            print(f"    {var:20s} β = {m1.params[var]:+.3f}  p={m1.pvalues[var]:.4f}")

    # Model B/C checks
    print("\n  Model B/C (DiD):")
    for name, m in {'B': m2, 'C': m3}.items():
        coef = m.params.get('ever_champ:is_post_2023', 0)
        p = m.pvalues.get('ever_champ:is_post_2023', 1)
        print(f"    Model {name} CHAMP cohort × post = {coef:+.3f}  p={p:.4f}")
        if 'is_post_2023:slr_exposure_pct' in m.params:
            slr_coef = m.params['is_post_2023:slr_exposure_pct']
            slr_p = m.pvalues['is_post_2023:slr_exposure_pct']
            print(f"    Model {name} SLR × post     = {slr_coef:+.3f}  p={slr_p:.4f}")
        if 'ever_champ:is_post_2023:slr_exposure_pct' in m.params:
            ddd_coef = m.params['ever_champ:is_post_2023:slr_exposure_pct']
            ddd_p = m.pvalues['ever_champ:is_post_2023:slr_exposure_pct']
            print(f"    Model {name} Triple-DiD     = {ddd_coef:+.3f}  p={ddd_p:.4f}")

    # ── Event Study Coefficients ────────────────────────────────────────────
    print("\n  Model D (Event Study, ref=2021):")
    for y in event_years:
        term = f'champ_x_{y}'
        if term in m4.params:
            print(f"    CHAMP × {y}: β = {m4.params[term]:+.3f}  p={m4.pvalues[term]:.4f}")

    # ── Full Summary Table ─────────────────────────────────────────────────
    # Only report the key non-FE coefficients for readability
    info_dict = {'N': lambda m: f"{int(m.nobs):,}", 'R²': lambda m: f"{m.rsquared:.3f}"}
    regressor_order = [
        'Intercept', 'is_post_2023', 'ever_champ', 'slr_exposure_pct',
        'ever_champ:is_post_2023', 'is_post_2023:slr_exposure_pct',
        'ever_champ:is_post_2023:slr_exposure_pct',
        'debt_to_av', 'time_to_maturity', 'median_income_10k',
    ] + [f"ever_champ:C(year)[T.{y}]" for y in sorted(df['year'].unique())]

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
        f.write("Regression: NJ Climate Risk and CHAMP Cohort Secondary-Trade Spread Diagnostics\n")
        f.write("=" * 60 + "\n\n")
        f.write("IMPORTANT: Uses secondary-market WRDS trades and a synthetic AAA benchmark. ")
        f.write("Primary specifications use 1st/99th percentile winsorized spreads and towns with both pre/post observations. ")
        f.write("Treat coefficients as exploratory associations, not final causal estimates.\n\n")
        f.write(str(table))
        write_model_interpretations(f, models, pretrend_coef, pretrend_p)
        f.write("\n\n[FULL TWFE DiD MODEL (B)]\n")
        f.write(m2.summary().as_text())
        f.write("\n\n[FULL EVENT STUDY MODEL (D)]\n")
        f.write(m4.summary().as_text())

    print(f"\nFull results saved → {results_path}")

if __name__ == '__main__':
    run_regression()
