"""
build_master_panel.py
======================
Constructs the final regression-ready panel by merging:

  Track A: WRDS bond trades       → spread_bps (dependent variable)
  Track B: SLR flood exposure     → slr_exposure_pct (flooded_pct_5ft from chars, 5-ft scenario)
  Track C: CHAMP resilience       → is_resilient        (binary)
           MS4 stormwater infra   → ms4_outfall_density  (outfalls per sq mile)
           UFB debt panel         → debt_to_gdp          (debt-to-assessed-value)
           ACS census             → median_income

Key crosswalk:
  WRDS 'MATCHED_MUNI' == chars['MUN']  (e.g. "ABERDEEN TWP")
  All other datasets use a stripped base name (e.g. "ABERDEEN").
  We derive that base from chars['MUN'] via get_base_name().

NOTE on slr_exposure_pct:
  REAL METRIC = FRM_RATIO_4FT * 100
  = % of the municipal tax base (assessed property value) at risk under a 4-ft SLR scenario.
  This comes from the final spatial-financial join (nj_statewide_frm_scores_FINAL.csv).
  This is the gold-standard metric for the DiD model as it captures 
  the economic severity of climate risk, not just geographic area.

Policy pivot date: 2022-07-01 (start of NJ SFY2023, first full CHAMP/STORM Act year)

Usage:
  python3 data_processing/build_master_panel.py
"""
import os, re
import pandas as pd
import numpy as np
from process_wrds_data import generate_synthetic_benchmark

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEANED  = os.path.join(PROJECT_ROOT, "data", "data_cleaned")

WRDS_FILE   = os.path.join(DATA_CLEANED, "finance", "nj_muni_spreads_full_trades.csv")
WRDS_TXT_FILE = os.path.join(DATA_CLEANED, "finance", "nj_muni_spreads_full_trades.txt")
CHARS_FILE  = os.path.join(DATA_CLEANED, "nj_municipality_characteristics.csv")
UFB_FILE    = os.path.join(DATA_CLEANED, "finance", "nj_ufb_debt_panel.csv")
CENSUS_FILE = os.path.join(DATA_CLEANED, "census", "nj_census_dp03_combined.csv")
SLR_FINAL_FILE = os.path.join(PROJECT_ROOT, "nj_statewide_frm_scores_FINAL.csv")
MS4_FILE    = os.path.join(DATA_CLEANED, "finance", "nj_ms4_resilience.csv")
OUTPUT_FILE = os.path.join(DATA_CLEANED, "final_panel_master.csv")

PIVOT_DATE = "2022-07-01"   # NJ CHAMP/STORM Act treatment cutoff

# ── Name normalisers / crosswalks ──────────────────────────────────────────
MUNI_TYPE_SUFFIXES = {
    "CITY": "CITY",
    "BOROUGH": "BORO",
    "BORO": "BORO",
    "TOWNSHIP": "TWP",
    "TWP": "TWP",
    "TOWN": "TOWN",
    "VILLAGE": "VILLAGE",
}


def clean_upper(s):
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).upper().strip())


def normalize_mun_key(s):
    """Normalize a full NJ municipal key while preserving legal suffixes."""
    s = clean_upper(s)
    s = re.sub(r",?\s*(CITY OF|BOROUGH OF|TOWNSHIP OF|TOWN OF|VILLAGE OF)\s+(.+)$",
               lambda m: f"{m.group(2)} {MUNI_TYPE_SUFFIXES[m.group(1).split()[0]]}", s)
    s = re.sub(r"\bTOWNSHIP$", "TWP", s)
    s = re.sub(r"\bBOROUGH$", "BORO", s)
    return clean_upper(s)


def get_base_name(s):
    """Strip legal suffix variants for fallback matching only."""
    s = str(s).upper().strip()
    # Handle "ABSECON, CITY OF" style
    s = re.sub(r',?\s*(CITY OF|BOROUGH OF|TOWNSHIP OF|TOWN OF|VILLAGE OF)\s*$', '', s)
    # Handle plain suffixes (double pass for cases like "ATLANTIC CITY CITY")
    for _ in range(2):
        s = re.sub(r'\s+(TOWNSHIP|TWP|BOROUGH|BORO|CITY|TOWN)\s*$', '', s).strip()
    return s


def extract_census_county(raw_name):
    if not isinstance(raw_name, str):
        return ""
    m = re.search(r",\s*([^,]+)\s+COUNTY\s*,\s*NEW JERSEY$", raw_name, flags=re.IGNORECASE)
    return clean_upper(m.group(1)) if m else ""


def extract_census_type(raw_name):
    if not isinstance(raw_name, str):
        return ""
    name = re.sub(r",.*$", "", raw_name).strip().upper()
    for suffix in ["CITY", "BOROUGH", "TOWNSHIP", "TOWN", "VILLAGE"]:
        if re.search(rf"\s+{suffix}$", name):
            return MUNI_TYPE_SUFFIXES[suffix]
    return ""


def build_municipality_crosswalk(chars):
    """Create a one-row-per-MUN crosswalk used to resolve weak source names."""
    xwalk = chars[["MUN", "MUN_LABEL", "COUNTY"]].copy()
    xwalk["mun"] = xwalk["MUN"].apply(normalize_mun_key)
    xwalk["county"] = xwalk["COUNTY"].apply(clean_upper)
    xwalk["base_name"] = xwalk["mun"].apply(get_base_name)
    xwalk["label_base"] = xwalk["MUN_LABEL"].apply(get_base_name)
    xwalk = xwalk.drop_duplicates(subset=["mun", "county"])
    return xwalk[["mun", "county", "base_name", "label_base"]]


def attach_mun_from_crosswalk(source, xwalk, source_name_col, source_county_col=None):
    """
    Attach canonical MUN to a source table.

    Direct full-key matches are preferred. Fallbacks use county + base name,
    which keeps names like OCEAN, UNION, FRANKLIN, WASHINGTON from matching
    every municipality with the same stripped base across the state.
    """
    out = source.copy()
    out["_source_full"] = out[source_name_col].apply(normalize_mun_key)
    out["_source_base"] = out[source_name_col].apply(get_base_name)
    out["_source_county"] = (
        out[source_county_col].apply(clean_upper) if source_county_col else ""
    )

    out["mun"] = pd.NA
    if source_county_col:
        direct_county = xwalk[["mun", "county"]].rename(
            columns={"mun": "_county_mun", "county": "_source_county"}
        )
        out = out.merge(
            direct_county,
            left_on=["_source_full", "_source_county"],
            right_on=["_county_mun", "_source_county"],
            how="left",
            validate="m:1",
        )
        out["mun"] = out["_county_mun"]

    direct = (
        xwalk["mun"].value_counts()
        .loc[lambda s: s == 1]
        .index
        .to_series()
        .reset_index(drop=True)
        .to_frame("mun")
        .rename(columns={"mun": "_direct_mun"})
    )
    out = out.merge(direct, left_on="_source_full", right_on="_direct_mun", how="left", validate="m:1")
    out["mun"] = out["mun"].fillna(out["_direct_mun"])

    needs_fallback = out["mun"].isna() & out["_source_county"].ne("")
    if needs_fallback.any():
        fallback = xwalk[["mun", "county", "base_name"]].copy()
        unique_fallback_keys = (
            fallback.groupby(["county", "base_name"]).size().loc[lambda s: s == 1].index
        )
        fallback = fallback.set_index(["county", "base_name"]).loc[unique_fallback_keys].reset_index()
        fallback = fallback.rename(
            columns={"mun": "_fallback_mun", "county": "_source_county", "base_name": "_source_base"}
        )
        tmp = out.loc[needs_fallback].drop(columns=["_fallback_mun"], errors="ignore").merge(
            fallback,
            on=["_source_county", "_source_base"],
            how="left",
            validate="m:1",
        )
        out.loc[needs_fallback, "mun"] = tmp["_fallback_mun"].values

    needs_label_fallback = out["mun"].isna() & out["_source_county"].ne("")
    if needs_label_fallback.any():
        fallback = xwalk[["mun", "county", "label_base"]].copy()
        unique_fallback_keys = (
            fallback.groupby(["county", "label_base"]).size().loc[lambda s: s == 1].index
        )
        fallback = fallback.set_index(["county", "label_base"]).loc[unique_fallback_keys].reset_index()
        fallback = fallback.rename(
            columns={"mun": "_label_mun", "county": "_source_county", "label_base": "_source_base"}
        )
        tmp = out.loc[needs_label_fallback].drop(columns=["_label_mun"], errors="ignore").merge(
            fallback,
            on=["_source_county", "_source_base"],
            how="left",
            validate="m:1",
        )
        out.loc[needs_label_fallback, "mun"] = tmp["_label_mun"].values

    helper_cols = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=helper_cols)


def assert_unique(df, keys, name):
    dup = df[df.duplicated(keys, keep=False)]
    if not dup.empty:
        sample = dup[keys].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"{name} is not unique on {keys}. Sample duplicate keys: {sample}")

def build_panel():
    print("=" * 60)
    print("MASTER PANEL CONSTRUCTION (FINAL 4FT SLR)")
    print("=" * 60)

    # ── 1. WRDS Base Panel ─────────────────────────────────────────────────
    print("\n[1/6] Loading WRDS base panel...")
    wrds_file = WRDS_FILE if os.path.exists(WRDS_FILE) else WRDS_TXT_FILE
    df = pd.read_csv(wrds_file, low_memory=False)
    df = df.rename(columns={
        'CUSIP':         'bond_id',
        'MATCHED_MUNI':  'mun',          # full canonical name, e.g. "ABERDEEN TWP"
        'TRADE_DATE_DT': 'issue_date',
        'SPREAD_BPS':    'spread_bps',
        'YIELD':         'yield',
    })
    df['issue_date']   = pd.to_datetime(df['issue_date'])
    df['year']         = df['issue_date'].dt.year
    df['is_post_2022'] = (df['issue_date'] > PIVOT_DATE).astype(int)
    df['yield']        = pd.to_numeric(df.get('yield'), errors='coerce')
    df['mun']          = df['mun'].apply(normalize_mun_key)

    if 'spread_bps' not in df.columns:
        print("   SPREAD_BPS not found; computing spread from synthetic AAA benchmark...")
        benchmark = generate_synthetic_benchmark().rename(columns={'trade_date': 'issue_date'})
        df = df.merge(benchmark, on='issue_date', how='left')
        df['spread_bps'] = (df['yield'] - df['synthetic_aaa']) * 100

    raw_trade_key = ['bond_id', 'issue_date', 'mun', 'spread_bps']
    raw_unique_trades = df.drop_duplicates(subset=raw_trade_key).shape[0]
    print(f"   {len(df):,} trade rows across {df['mun'].nunique()} towns.")
    print(f"   {raw_unique_trades:,} unique bond/date/mun/spread rows before controls.")

    # ── 2. Chars & Final SLR (Resilience) ─────────────────────────────────
    print("\n[2/6] Merging chars and final SLR 4ft scores...")
    chars = pd.read_csv(CHARS_FILE)
    chars['mun']       = chars['MUN'].apply(normalize_mun_key)
    chars['county']    = chars['COUNTY'].apply(clean_upper)
    chars['is_resilient'] = chars['is_resilient'].map(
        {True: 1, False: 0, 'True': 1, 'False': 0}
    ).fillna(0).astype(int)
    xwalk = build_municipality_crosswalk(chars)
    unique_muns = xwalk["mun"].value_counts().loc[lambda s: s == 1].index
    ambiguous_muns = sorted(set(xwalk["mun"]) - set(unique_muns))
    ambiguous_rows = df["mun"].isin(ambiguous_muns).sum()
    if ambiguous_rows:
        print(
            f"   Dropping {ambiguous_rows:,} WRDS rows from {df.loc[df['mun'].isin(ambiguous_muns), 'mun'].nunique()} "
            "statewide-ambiguous municipality keys without county."
        )
        df = df[~df["mun"].isin(ambiguous_muns)].copy()
    control_base_rows = len(df)

    chars_slim = chars[['mun', 'county', 'is_resilient']]
    chars_slim = chars_slim[chars_slim['mun'].isin(unique_muns)].copy()
    assert_unique(chars_slim, ['mun'], 'Characteristics merge input')

    # Load Final SLR (Tax-Base Weighted)
    slr = pd.read_csv(SLR_FINAL_FILE)
    slr['MUN_NAME'] = slr['MUN_NAME'].apply(normalize_mun_key)
    slr['COUNTY'] = slr['COUNTY'].apply(clean_upper)
    slr = attach_mun_from_crosswalk(slr, xwalk, 'MUN_NAME', 'COUNTY')
    slr['slr_exposure_pct'] = slr['FRM_RATIO_4FT'] * 100.0  # Convert to percentage points
    slr_slim = slr[['mun', 'slr_exposure_pct']].dropna(subset=['mun'])
    slr_slim = slr_slim[slr_slim['mun'].isin(unique_muns)].copy()
    assert_unique(slr_slim, ['mun'], 'SLR merge input')

    df = pd.merge(df, chars_slim, on='mun', how='left', validate='m:1')
    df = pd.merge(df, slr_slim, on='mun', how='left', validate='m:1')

    # Fill NaN SLR exposure for inland towns with 0
    df['slr_exposure_pct'] = df['slr_exposure_pct'].fillna(0)

    print(f"   Matched {df['is_resilient'].notna().sum():,} rows with CHAMP resilience.")
    print(f"   Matched {df['slr_exposure_pct'].notna().sum():,} rows with Final SLR (4ft).")

    # ── 3. MS4 Stormwater Infrastructure Resilience ────────────────────────
    print("\n[3/6] Merging MS4 outfall resilience metrics...")
    ms4 = pd.read_csv(MS4_FILE)
    ms4['mun'] = ms4['mun'].apply(normalize_mun_key)
    ms4 = attach_mun_from_crosswalk(ms4, xwalk, 'mun')
    ms4_slim = ms4[['mun', 'ms4_outfall_density', 'ms4_outfall_count']].dropna(subset=['mun'])
    ms4_slim = ms4_slim[ms4_slim['mun'].isin(unique_muns)].copy()
    assert_unique(ms4_slim, ['mun'], 'MS4 merge input')
    df = pd.merge(df, ms4_slim, on='mun', how='left', validate='m:1')
    df['ms4_outfall_density'] = df['ms4_outfall_density'].fillna(0.0)
    df['ms4_outfall_count']   = df['ms4_outfall_count'].fillna(0).astype(int)
    print(f"   Matched {(df['ms4_outfall_density'] > 0).sum():,} rows with MS4 outfall data.")

    # ── 4. UFB Fiscal Controls ─────────────────────────────────────────────
    print("\n[4/6] Merging UFB fiscal controls...")
    ufb = pd.read_csv(UFB_FILE)
    ufb = ufb.rename(columns={'ufb_year': 'year', 'debt_to_assessed_value': 'debt_to_gdp'})
    if 'county' not in ufb.columns:
        raise ValueError("UFB debt panel is missing county. Re-run data_processing/process_ufb.py.")
    ufb_name_col = 'municipality_raw' if 'municipality_raw' in ufb.columns else 'municipality'
    ufb = attach_mun_from_crosswalk(ufb, xwalk, ufb_name_col, 'county')
    ufb_slim = ufb[['mun', 'year', 'debt_to_gdp']].dropna(subset=['mun'])
    ufb_slim = ufb_slim[ufb_slim['mun'].isin(unique_muns)].copy()
    assert_unique(ufb_slim, ['mun', 'year'], 'UFB merge input')

    df = pd.merge(df, ufb_slim, on=['mun', 'year'], how='left', validate='m:1')
    matched_ufb = df['debt_to_gdp'].notna().sum()
    print(f"   UFB matched:       {matched_ufb:,} rows ({matched_ufb/len(df)*100:.1f}%).")

    # ── 5. Census Demographics ─────────────────────────────────────────────
    print("\n[5/6] Merging ACS census controls...")
    census = pd.read_csv(CENSUS_FILE)
    census['county'] = census['municipality_raw'].apply(extract_census_county)
    census = census.rename(columns={
        'acs_year':               'year',
        'median_household_income':'median_income',
    })
    census['year'] = pd.to_numeric(census['year'], errors='coerce').astype('Int64')
    census = attach_mun_from_crosswalk(census, xwalk, 'municipality', 'county')

    # Forward-fill 2022 ACS values for 2023-2025
    latest = census[census['year'] == 2022][['mun', 'median_income']].copy()
    fills  = pd.concat([latest.assign(year=y) for y in [2023, 2024, 2025]], ignore_index=True)
    census_ext = pd.concat([census[['mun', 'year', 'median_income']], fills], ignore_index=True)
    census_ext = census_ext.dropna(subset=['mun'])
    census_ext = census_ext[census_ext['mun'].isin(unique_muns)].copy()
    census_ext = census_ext.drop_duplicates(subset=['mun', 'year'])
    assert_unique(census_ext, ['mun', 'year'], 'Census merge input')

    df = pd.merge(df, census_ext, on=['mun', 'year'], how='left', validate='m:1')
    matched_cen = df['median_income'].notna().sum()
    print(f"   Census matched: {matched_cen:,} rows ({matched_cen/len(df)*100:.1f}%).")

    # ── 6. Finalise ────────────────────────────────────────────────────────
    print("\n[6/6] Selecting columns and dropping incomplete rows...")
    final_cols = [
        'bond_id', 'mun', 'county', 'issue_date',
        'spread_bps',
        'slr_exposure_pct',      # % of total market value at risk under 4-ft SLR
        'is_resilient',          # CHAMP binary flag
        'ms4_outfall_density',   # MS4 outfalls per sq mile (stormwater infrastructure intensity)
        'ms4_outfall_count',     # raw outfall count (secondary)
        'is_post_2022',          # 1 if trade date > 2022-07-01
        'debt_to_gdp',           # UFB debt-to-assessed-value ratio
        'median_income',         # ACS median household income (2023-25 fwd-filled)
    ]
    df_final = df[final_cols].copy()
    df_final = df_final.rename(columns={'mun': 'muni_name'})

    initial_n = len(df_final)
    df_final  = df_final.dropna(subset=['debt_to_gdp', 'median_income', 'spread_bps'])
    dropped   = initial_n - len(df_final)

    if initial_n > control_base_rows:
        raise ValueError(
            "Control merges increased the number of WRDS rows. "
            f"Before controls={control_base_rows:,}, after controls={initial_n:,}. Check merge keys."
        )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MASTER PANEL SUMMARY")
    print("=" * 60)
    print(f"  Total rows:          {len(df_final):,}")
    print(f"  Dropped (missing):   {dropped:,}")
    print(f"  Unique towns:        {df_final['muni_name'].nunique()}")
    print(f"  Year range:          {df_final['issue_date'].min().year} – {df_final['issue_date'].max().year}")
    print(f"  CHAMP resilient towns: {df_final[df_final['is_resilient']==1]['muni_name'].nunique()}")
    print(f"  MS4 density (mean):  {df_final['ms4_outfall_density'].mean():.2f} outfalls/sq mi")
    print(f"  Post-2022 trades:    {df_final['is_post_2022'].sum():,}")
    print(f"  Avg spread (all):    {df_final['spread_bps'].mean():.2f} bps")
    print(f"  Avg spread (pre):    {df_final[df_final['is_post_2022']==0]['spread_bps'].mean():.2f} bps")
    print(f"  Avg spread (post):   {df_final[df_final['is_post_2022']==1]['spread_bps'].mean():.2f} bps")
    print(f"\n  ✅  slr_exposure_pct = % of Total Market Value @ 4-ft SLR (FINAL)")
    print(f"     Mean={df_final['slr_exposure_pct'].mean():.2f}%  Max={df_final['slr_exposure_pct'].max():.2f}%\n")

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved → {OUTPUT_FILE}")

if __name__ == '__main__':
    build_panel()
