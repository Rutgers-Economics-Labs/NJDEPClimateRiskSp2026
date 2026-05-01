"""
build_master_panel.py
======================
Constructs the final regression-ready panel by merging:

  Track A: WRDS bond trades       → spread_bps (dependent variable)
  Track B: SLR flood exposure     → slr_exposure_pct (flooded_pct_5ft from chars, 5-ft scenario)
  Track C: CHAMP resilience       → is_resilient  (binary)
           FEMA CRS               → crs_class     (1-10)
           UFB debt panel         → debt_to_gdp   (debt-to-assessed-value)
           ACS census             → median_income

Key crosswalk:
  WRDS 'MATCHED_MUNI' == chars['MUN']  (e.g. "ABERDEEN TWP")
  All other datasets use a stripped base name (e.g. "ABERDEEN").
  We derive that base from chars['MUN'] via get_base_name().

NOTE on slr_exposure_pct:
  Currently uses flooded_pct_5ft from nj_municipality_characteristics.csv
  as a PROXY (% of municipal land area flooded at 5-ft SLR).
  Replace with the tax-base-weighted output of test_ocean_county.py
  once that script finishes. The column name is unchanged.

Policy pivot date: 2022-07-01 (start of NJ SFY2023, first full CHAMP/STORM Act year)

Usage:
  python3 data_processing/build_master_panel.py
"""
import os, re
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEANED  = os.path.join(PROJECT_ROOT, "data", "data_cleaned")

WRDS_FILE   = os.path.join(DATA_CLEANED, "finance", "nj_muni_spreads_full_trades.csv")
CHARS_FILE  = os.path.join(DATA_CLEANED, "nj_municipality_characteristics.csv")
UFB_FILE    = os.path.join(DATA_CLEANED, "finance", "nj_ufb_debt_panel.csv")
CENSUS_FILE = os.path.join(DATA_CLEANED, "census", "nj_census_dp03_combined.csv")
OUTPUT_FILE = os.path.join(DATA_CLEANED, "final_panel_master.csv")

PIVOT_DATE = "2022-07-01"   # NJ CHAMP/STORM Act treatment cutoff

# ── Name normaliser ────────────────────────────────────────────────────────
def get_base_name(s):
    """Strip municipal suffix variants → plain name used by UFB & Census."""
    s = str(s).upper().strip()
    # Handle "ABSECON, CITY OF" style
    s = re.sub(r',?\s*(CITY OF|BOROUGH OF|TOWNSHIP OF|TOWN OF|VILLAGE OF)\s*$', '', s)
    # Handle plain suffixes
    s = re.sub(r'\s+(TOWNSHIP|TWP|BOROUGH|BORO|CITY|TOWN)\s*$', '', s).strip()
    return s

def build_panel():
    print("=" * 60)
    print("MASTER PANEL CONSTRUCTION")
    print("=" * 60)

    # ── 1. WRDS Base Panel ─────────────────────────────────────────────────
    print("\n[1/5] Loading WRDS base panel...")
    df = pd.read_csv(WRDS_FILE, low_memory=False)
    df = df.rename(columns={
        'CUSIP':         'bond_id',
        'MATCHED_MUNI':  'mun',          # full canonical name, e.g. "ABERDEEN TWP"
        'TRADE_DATE_DT': 'issue_date',
        'SPREAD_BPS':    'spread_bps',
    })
    df['issue_date']   = pd.to_datetime(df['issue_date'])
    df['year']         = df['issue_date'].dt.year
    df['is_post_2022'] = (df['issue_date'] > PIVOT_DATE).astype(int)

    # Derive the base name on WRDS side for downstream joins
    df['muni_name'] = df['mun'].apply(get_base_name)
    print(f"   {len(df):,} trades across {df['mun'].nunique()} towns.")

    # ── 2. Chars Crosswalk (CRS, SLR, Resilience) ─────────────────────────
    print("\n[2/5] Merging chars (CRS class, SLR exposure, resilience flag)...")
    chars = pd.read_csv(CHARS_FILE)
    chars['mun']       = chars['MUN'].str.upper().str.strip()
    chars['is_resilient'] = chars['is_resilient'].map(
        {True: 1, False: 0, 'True': 1, 'False': 0}
    ).fillna(0).astype(int)

    chars_slim = chars[['mun', 'COUNTY', 'crs_class', 'flooded_pct_5ft', 'is_resilient']].rename(columns={
        'COUNTY':           'county',
        'flooded_pct_5ft':  'slr_exposure_pct',   # ← PROXY; replace when SLR script finishes
    })

    df = pd.merge(df, chars_slim, on='mun', how='left')
    print(f"   Matched {df['crs_class'].notna().sum():,} rows with CRS/SLR data.")

    # ── 3. UFB Fiscal Controls ─────────────────────────────────────────────
    print("\n[3/5] Merging UFB fiscal controls...")
    ufb = pd.read_csv(UFB_FILE)
    ufb['muni_name'] = ufb['municipality'].apply(get_base_name)
    ufb = ufb.rename(columns={'ufb_year': 'year', 'debt_to_assessed_value': 'debt_to_gdp'})

    df = pd.merge(df, ufb[['muni_name', 'year', 'debt_to_gdp']], on=['muni_name', 'year'], how='left')
    matched_ufb = df['debt_to_gdp'].notna().sum()
    print(f"   UFB matched: {matched_ufb:,} rows ({matched_ufb/len(df)*100:.1f}%).")

    # ── 4. Census Demographics ─────────────────────────────────────────────
    print("\n[4/5] Merging ACS census controls...")
    census = pd.read_csv(CENSUS_FILE)
    census['muni_name']   = census['municipality'].apply(get_base_name)
    census = census.rename(columns={
        'acs_year':               'year',
        'median_household_income':'median_income',
    })
    census['year'] = pd.to_numeric(census['year'], errors='coerce').astype('Int64')

    # Forward-fill 2022 ACS values for 2023-2025
    latest = census[census['year'] == 2022][['muni_name', 'median_income']].copy()
    fills  = pd.concat([latest.assign(year=y) for y in [2023, 2024, 2025]], ignore_index=True)
    census_ext = pd.concat([census[['muni_name', 'year', 'median_income']], fills], ignore_index=True)
    census_ext = census_ext.drop_duplicates(subset=['muni_name', 'year'])

    df = pd.merge(df, census_ext, on=['muni_name', 'year'], how='left')
    matched_cen = df['median_income'].notna().sum()
    print(f"   Census matched: {matched_cen:,} rows ({matched_cen/len(df)*100:.1f}%).")

    # ── 5. Finalise ────────────────────────────────────────────────────────
    print("\n[5/5] Selecting columns and dropping incomplete rows...")
    final_cols = [
        'bond_id', 'muni_name', 'county', 'issue_date',
        'spread_bps',
        'slr_exposure_pct',  # PROXY: flooded_pct_5ft (replace with SLR script output)
        'crs_class',         # REAL:  FEMA CRS class
        'is_resilient',      # REAL:  CHAMP binary flag
        'is_post_2022',      # ENGINEERED: 1 if trade date > 2022-07-01
        'debt_to_gdp',       # REAL:  UFB debt-to-assessed-value ratio
        'median_income',     # REAL:  ACS median household income (2023-25 fwd-filled)
    ]
    df_final = df[final_cols].copy()

    initial_n = len(df_final)
    df_final  = df_final.dropna(subset=['debt_to_gdp', 'median_income', 'spread_bps'])
    dropped   = initial_n - len(df_final)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MASTER PANEL SUMMARY")
    print("=" * 60)
    print(f"  Total rows:          {len(df_final):,}")
    print(f"  Dropped (missing):   {dropped:,}")
    print(f"  Unique towns:        {df_final['muni_name'].nunique()}")
    print(f"  Year range:          {df_final['issue_date'].min().year} – {df_final['issue_date'].max().year}")
    print(f"  Resilient towns:     {df_final[df_final['is_resilient']==1]['muni_name'].nunique()}")
    print(f"  Post-2022 trades:    {df_final['is_post_2022'].sum():,}")
    print(f"  Avg spread (all):    {df_final['spread_bps'].mean():.2f} bps")
    print(f"  Avg spread (pre):    {df_final[df_final['is_post_2022']==0]['spread_bps'].mean():.2f} bps")
    print(f"  Avg spread (post):   {df_final[df_final['is_post_2022']==1]['spread_bps'].mean():.2f} bps")
    print(f"\n  ⚠  slr_exposure_pct = flooded_pct_5ft (PROXY)")
    print(f"     Replace with test_ocean_county.py output when ready.\n")

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved → {OUTPUT_FILE}")

if __name__ == '__main__':
    build_panel()
