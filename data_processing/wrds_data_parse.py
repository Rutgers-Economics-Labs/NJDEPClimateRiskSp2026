import os
import glob
import pandas as pd
import numpy as np
import requests
import io
import re
import sys
import gc

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw", "wrds")
CHARS_FILE = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "nj_municipality_characteristics.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "finance", "nj_muni_spreads_full_trades.csv")

# Constants
SEARCH_PATTERN = r"NJ|NEW JERSEY|N J"
GO_PATTERN = r"\b(?:GO|OBLIG|GEN OBL|GENERAL OBLIGATION|GENL OBLIG|G O|GEN IMPT|VAR PURP|IMP|IMPT)\b"
STRICT_EXCLUSION = r"UNIVERSITY|AUTHORITY|AUTH|REV|REVENUE|FACILITIES|FACS|TRANSPORTATION|TURNPIKE|WATER|SEWER|UTIL|HOUSING|COUNTY|CNTY|STATE|PORT|BRIDGE|TUNNEL|HSA"

def fetch_treasury_curve():
    """Fetch Treasury Par Yield Curve from FRED."""
    print("\nFetching Treasury Curve from FRED...")
    series_map = {1: 'DGS1', 2: 'DGS2', 5: 'DGS5', 10: 'DGS10', 20: 'DGS20', 30: 'DGS30'}
    curve_data = []
    
    for maturity, sid in series_map.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        resp = requests.get(url)
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch {sid}")
            
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [str(c).strip().upper() for c in df.columns]
        date_col = [c for c in df.columns if 'DATE' in c][0]
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df.columns = [maturity]
        curve_data.append(df)
        
    df_curve = pd.concat(curve_data, axis=1).sort_index()
    df_curve = df_curve.apply(pd.to_numeric, errors='coerce').interpolate(method='time').dropna()
    df_curve = df_curve[~df_curve.index.duplicated(keep='first')]
    return df_curve

def calculate_spreads(trades_df, curve_df):
    """Vectorized spread calculation."""
    print("Interpolating yields based on TRADE_DATE and Maturity...")
    # Find nearest date in the curve
    curve_at_trades = curve_df.reindex(trades_df['TRADE_DATE_DT'], method='nearest')
    curve_maturities = np.array([1, 2, 5, 10, 20, 30])
    curve_values = curve_at_trades.values
    maturities = trades_df['YEARS_TO_MATURITY'].values
    
    bench_yields = [np.interp(maturities[i], curve_maturities, curve_values[i]) for i in range(len(trades_df))]
    
    # 0.85 multiplier is the tax-equivalent adjustment for munis vs treasuries
    # Usually handled if comparing taxable to tax-exempt, we'll keep the notebook's logic.
    return np.array(bench_yields) * 0.85

def parse_wrds_data():
    print("=" * 60)
    print("WRDS DATA PARSING PIPELINE")
    print("=" * 60)
    
    # 1. LOAD RAW DATA
    gz_files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, "*.csv*")))
    if not gz_files:
        print(f"No raw files found in {DATA_RAW_DIR}")
        return
        
    print(f"Found {len(gz_files)} raw data files.")
    
    dfs = []
    essential_cols = ['CUSIP', 'TRADE_DATE', 'MATURITY_DATE', 'YIELD', 'SECURITY_DESCRIPTION', 'DATED_DATE']
    
    for f in gz_files:
        fname = os.path.basename(f)
        if 'MSRB' in fname: continue
        
        print(f"  Loading {fname}...", end=" ")
        # Load in chunks to filter
        chunks = pd.read_csv(f, chunksize=250000, low_memory=False)
        matched_chunks = []
        for chunk in chunks:
            chunk.columns = [str(c).upper() for c in chunk.columns]
            
            # Keep essential cols if they exist
            avail_cols = [c for c in essential_cols if c in chunk.columns]
            chunk = chunk[avail_cols]
            
            # Filter for NJ
            if 'SECURITY_DESCRIPTION' in chunk.columns:
                mask = chunk['SECURITY_DESCRIPTION'].str.contains(SEARCH_PATTERN, case=False, na=False)
                matched_chunks.append(chunk[mask])
                
        if matched_chunks:
            df_file = pd.concat(matched_chunks, ignore_index=True)
            dfs.append(df_file)
            print(f"({len(df_file):,} NJ trades)")
        else:
            print("(0 NJ trades)")
            
        # Free memory
        del chunks
        del matched_chunks
        gc.collect()
            
    df_master = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    print(f"\nTotal NJ Raw Trades Extracted: {len(df_master):,}")
    
    # 2. FILTERING
    print("\nApplying filters (GO Bonds, Exclude Authorities)...")
    is_go = df_master['SECURITY_DESCRIPTION'].str.contains(GO_PATTERN, case=False, na=False, regex=True)
    is_not_noise = ~df_master['SECURITY_DESCRIPTION'].str.contains(STRICT_EXCLUSION, case=False, na=False, regex=True)
    
    df_clean = df_master[is_go & is_not_noise].copy()
    
    # Jersey City vs State of Jersey
    is_jersey_state = (df_clean['SECURITY_DESCRIPTION'].str.contains('JERSEY', case=False)) & \
                      (~df_clean['SECURITY_DESCRIPTION'].str.contains('CITY', case=False))
    df_clean = df_clean[~is_jersey_state].copy()
    
    # 3. DATE PARSING & WINDOW FILTERING
    print("\nParsing dates...")
    df_clean['TRADE_DATE_DT'] = pd.to_datetime(df_clean['TRADE_DATE'].astype(str).str.split('.').str[0], errors='coerce', format='ISO8601')
    df_clean.loc[df_clean['TRADE_DATE_DT'].isna(), 'TRADE_DATE_DT'] = pd.to_datetime(
        df_clean.loc[df_clean['TRADE_DATE_DT'].isna(), 'TRADE_DATE'].astype(str).str.split('.').str[0], errors='coerce', format='%Y%m%d'
    )
    
    df_clean['MATURITY_DATE_DT'] = pd.to_datetime(df_clean['MATURITY_DATE'].astype(str).str.split('.').str[0], errors='coerce', format='ISO8601')
    df_clean.loc[df_clean['MATURITY_DATE_DT'].isna(), 'MATURITY_DATE_DT'] = pd.to_datetime(
        df_clean.loc[df_clean['MATURITY_DATE_DT'].isna(), 'MATURITY_DATE'].astype(str).str.split('.').str[0], errors='coerce', format='%Y%m%d'
    )
    
    # *** CRITICAL FIX ***
    # Filter by TRADE_DATE_DT (when the bond was traded in the secondary market)
    # NOT by DATED_DATE (when it was issued)
    df_clean = df_clean[(df_clean['TRADE_DATE_DT'] >= '2015-01-01') & (df_clean['TRADE_DATE_DT'] <= '2025-12-31')].copy()
    
    df_clean['YEARS_TO_MATURITY'] = (df_clean['MATURITY_DATE_DT'] - df_clean['TRADE_DATE_DT']).dt.days / 365.25
    df_clean = df_clean.dropna(subset=['YEARS_TO_MATURITY', 'YIELD'])
    df_clean = df_clean[df_clean['YEARS_TO_MATURITY'] > 0].copy()
    
    print(f"Trades strictly between 2015-2025: {len(df_clean):,}")
    
    # 4. TOWN MAPPING
    print("\nMapping bonds to NJ Municipalities...")
    chars = pd.read_csv(CHARS_FILE)
    
    # Build clean map base names
    def get_base_name(n):
        return str(n).upper().replace(' TOWNSHIP', '').replace(' TWP', '').replace(' BOROUGH', '').replace(' BORO', '').replace(' CITY', '').replace(' TOWN', '').strip()

    ambiguous_groups = {}
    clean_map = {}
    
    for _, row in chars.iterrows():
        full_name = row['MUN'].upper()
        base = get_base_name(full_name)
        info = {'mun': full_name, 'label': row['MUN_LABEL'], 'county': row['COUNTY']}
        clean_map[full_name] = info
        
        if base not in ambiguous_groups:
            ambiguous_groups[base] = []
        ambiguous_groups[base].append(info)
        
    # Cache compiled regexes for speed
    sorted_bases = sorted(ambiguous_groups.keys(), key=len, reverse=True)
    base_regexes = {base: re.compile(rf"\b{re.escape(base)}\b") for base in sorted_bases}
    
    def smart_match(desc):
        desc = str(desc).upper()
        
        matched_base = None
        for base in sorted_bases:
            if base_regexes[base].search(desc):
                matched_base = base
                break
                
        if not matched_base:
            return None
            
        options = ambiguous_groups[matched_base]
        if len(options) == 1:
            return options[0]
            
        # Ambiguous resolution
        if 'TWP' in desc or 'TOWNSHIP' in desc:
            for opt in options:
                if 'TWP' in opt['mun'] or 'TOWNSHIP' in opt['mun']: return opt
        if 'BORO' in desc or 'BOROUGH' in desc:
            for opt in options:
                if 'BORO' in opt['mun'] or 'BOROUGH' in opt['mun']: return opt
        if 'CITY' in desc:
            for opt in options:
                if 'CITY' in opt['mun']: return opt
        
        return None

    mapped_info = df_clean['SECURITY_DESCRIPTION'].apply(smart_match)
    df_clean['MATCHED_MUNI'] = mapped_info.apply(lambda x: x['mun'] if x else None)
    
    df_final = df_clean.dropna(subset=['MATCHED_MUNI']).copy()
    print(f"Mapped {len(df_final):,} trades to {df_final['MATCHED_MUNI'].nunique()} unique towns.")
    
    # 5. SPREAD CALCULATION
    df_curve = fetch_treasury_curve()
    df_final['BENCHMARK_AAA_YIELD'] = calculate_spreads(df_final, df_curve)
    df_final['SPREAD_BPS'] = (df_final['YIELD'] - df_final['BENCHMARK_AAA_YIELD']) * 100
    
    df_final = df_final.dropna(subset=['SPREAD_BPS'])
    
    print("\n--- FINAL PANEL ---")
    print(f"Total Rows: {len(df_final):,}")
    print(f"Average Spread: {df_final['SPREAD_BPS'].mean():.2f} bps")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    parse_wrds_data()
