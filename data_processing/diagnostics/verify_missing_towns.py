import pandas as pd
import re

print("Loading data...")
# Load all towns
chars = pd.read_csv("data/data_cleaned/nj_municipality_characteristics.csv")
all_towns = chars['MUN'].unique()

# Load our final mapped trades
final_df = pd.read_csv("data/data_cleaned/finance/nj_muni_spreads_full_trades.csv")
found_towns = final_df['MATCHED_MUNI'].unique()

# Identify missing towns
missing_towns = set(all_towns) - set(found_towns)
print(f"Total Towns: {len(all_towns)}")
print(f"Mapped Towns: {len(found_towns)}")
print(f"Missing Towns: {len(missing_towns)}")

# Load the RAW NJ Trades (pre-filtering)
print("\nLoading raw NJ trades (10M+ rows)...")
raw_df = pd.read_csv("data/data_cleaned/extracted_nj_trades.csv", usecols=['SECURITY_DESCRIPTION', 'TRADE_DATE'], dtype=str)

print("Filtering raw trades to 2015-2025 window...")
raw_df['TRADE_DATE_DT'] = pd.to_datetime(raw_df['TRADE_DATE'].str.split('.').str[0], errors='coerce', format='ISO8601')
raw_df.loc[raw_df['TRADE_DATE_DT'].isna(), 'TRADE_DATE_DT'] = pd.to_datetime(
    raw_df.loc[raw_df['TRADE_DATE_DT'].isna(), 'TRADE_DATE'].str.split('.').str[0], errors='coerce', format='%Y%m%d'
)
raw_df = raw_df[(raw_df['TRADE_DATE_DT'] >= '2015-01-01') & (raw_df['TRADE_DATE_DT'] <= '2025-12-31')].copy()

# Critical optimization: unique descriptions
unique_desc_series = pd.Series(raw_df['SECURITY_DESCRIPTION'].dropna().unique()).str.upper()
print(f"Unique descriptions to scan: {len(unique_desc_series)}")

# Define the exact patterns used in our final script
GO_PATTERN = re.compile(r"\b(?:GO|OBLIG|GEN OBL|GENERAL OBLIGATION|GENL OBLIG|G O|GEN IMPT|VAR PURP|IMP|IMPT)\b")
STRICT_EXCLUSION = re.compile(r"UNIVERSITY|AUTHORITY|AUTH|REV|REVENUE|FACILITIES|FACS|TRANSPORTATION|TURNPIKE|WATER|SEWER|UTIL|HOUSING|COUNTY|CNTY|STATE|PORT|BRIDGE|TUNNEL|HSA")

def get_base_name(n):
    return str(n).upper().replace(' TOWNSHIP', '').replace(' TWP', '').replace(' BOROUGH', '').replace(' BORO', '').replace(' CITY', '').replace(' TOWN', '').strip()

print("\nScanning raw trades for missing towns...")
results = []
for town in missing_towns:
    base = get_base_name(town)
    pattern = re.compile(rf"\b{re.escape(base)}\b")
    
    # Check if this town exists ANYWHERE in the unique descriptions
    matches = unique_desc_series[unique_desc_series.apply(lambda x: bool(pattern.search(x)))]
    
    if len(matches) == 0:
        results.append({'Town': town, 'Status': 'No Trades in 2015-2025'})
    else:
        # It has trades! Let's see why it was dropped
        passes_go = matches.apply(lambda x: bool(GO_PATTERN.search(x)))
        passes_noise = matches.apply(lambda x: not bool(STRICT_EXCLUSION.search(x)))
        
        valid_trades = matches[passes_go & passes_noise]
        
        if len(valid_trades) > 0:
            results.append({'Town': town, 'Status': f'Missed by Mapper ({len(valid_trades)} valid trades)'})
        else:
            if not passes_go.any():
                results.append({'Town': town, 'Status': 'Dropped: Has trades, but ZERO are GO Bonds (Likely Revenue/Special)'})
            elif not passes_noise.any():
                results.append({'Town': town, 'Status': 'Dropped: Caught in Noise Filter (Authority/Utility)'})
            else:
                results.append({'Town': town, 'Status': 'Dropped: Failed GO/Noise filters'})

results_df = pd.DataFrame(results)
print("\n--- Diagnostic Summary ---")
print(results_df['Status'].value_counts())

missed = results_df[results_df['Status'].str.contains('Missed')]
if not missed.empty:
    print("\nTowns that SHOULD have been mapped but were missed:")
    print(missed)
else:
    print("\nSUCCESS: 0 valid GO towns were missed by the mapper!")

print("\nCoastal Town Check (Sample of Missing Towns):")
coastal = ['AVALON', 'STONE HARBOR', 'MARGATE CITY', 'LONG BEACH TOWNSHIP', 'CAPE MAY CITY']
for c in coastal:
    if c in results_df['Town'].values:
        print(f"{c}: {results_df[results_df['Town'] == c]['Status'].values[0]}")
    elif c in found_towns:
        print(f"{c}: MAPPED SUCCESSFULLY")
    else:
        print(f"{c}: Unknown")

results_df.to_csv("data/data_cleaned/finance/missing_town_diagnostics.csv", index=False)
