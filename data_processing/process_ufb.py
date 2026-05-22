import os
import glob
import pandas as pd
import sys

# Add current dir to path to import process_census
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from process_census import standardize_muni_name

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "finance")
OUTPUT_DIR = INPUT_DIR

def clean_currency(x):
    """Clean currency strings into floats."""
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.replace('$', '').replace(',', '').strip()
        if x == '' or x == '-':
            return 0.0
    try:
        return float(x)
    except:
        return None

def process_ufb_debt():
    files = glob.glob(os.path.join(INPUT_DIR, "ufb_*_Summary.csv"))
    
    all_years = []
    
    for f in files:
        # Extract year from filename (e.g. ufb_2022_Summary.csv)
        basename = os.path.basename(f)
        year_str = basename.split('_')[1]
        try:
            year = int(year_str)
        except ValueError:
            print(f"Skipping {basename}: cannot parse year")
            continue
            
        print(f"Processing UFB Debt for Year: {year}")
        
        # The UFB summaries have 4 rows of title/metadata headers
        df = pd.read_csv(f, low_memory=False, header=4)
        
        # We need Municipality, County, Net Debt, Assessed Value (or Property Valuation)
        # Columns might have slightly different exact names, so let's map them
        col_muni = None
        col_county = None
        col_net_debt = None
        col_assessed_val = None
        
        for c in df.columns:
            c_upper = str(c).upper().strip()
            if c_upper == 'MUNICIPALITY':
                col_muni = c
            elif c_upper == 'COUNTY':
                col_county = c
            elif c_upper == 'NET DEBT':
                col_net_debt = c
            elif 'PROPERTY VALUATION' in c_upper or 'ASSESSED VALUE' in c_upper or 'EQUALIZED VALUATION' in c_upper:
                if 'NET DEBT AS' not in c_upper:
                    col_assessed_val = c
                
        if not col_muni or not col_county or not col_net_debt or not col_assessed_val:
            print(f"  WARNING: Missing columns in {year}. (M:{col_muni}, C:{col_county}, ND:{col_net_debt}, AV:{col_assessed_val})")
            continue
            
        df_subset = df[[col_muni, col_county, col_net_debt, col_assessed_val]].copy()
        
        # Standardize muni
        df_subset['municipality_raw'] = df_subset[col_muni].astype(str).str.strip()
        df_subset['municipality'] = df_subset[col_muni].apply(standardize_muni_name)
        df_subset['county'] = df_subset[col_county].astype(str).str.upper().str.strip()
        df_subset = df_subset.dropna(subset=['municipality'])
        df_subset = df_subset[df_subset['municipality'] != '']
        
        # Clean currency
        df_subset['net_debt'] = df_subset[col_net_debt].apply(clean_currency)
        df_subset['assessed_value'] = df_subset[col_assessed_val].apply(clean_currency)
        
        # Drop rows where both are null/0
        df_subset = df_subset.dropna(subset=['net_debt', 'assessed_value'])
        
        # Calculate debt to tax base (assessed value)
        def calc_ratio(row):
            if row['assessed_value'] and row['assessed_value'] > 0:
                return row['net_debt'] / row['assessed_value']
            return None
            
        df_subset['debt_to_assessed_value'] = df_subset.apply(calc_ratio, axis=1)
        
        df_subset['ufb_year'] = year
        
        # Keep final columns
        df_final = df_subset[['municipality_raw', 'municipality', 'county', 'ufb_year', 'net_debt', 'assessed_value', 'debt_to_assessed_value']]
        all_years.append(df_final)

    if all_years:
        combined = pd.concat(all_years, ignore_index=True)
        # Drop duplicate municipality/year pairs just in case
        combined = combined.drop_duplicates(subset=['municipality_raw', 'municipality', 'county', 'ufb_year'])
        
        out_path = os.path.join(OUTPUT_DIR, "nj_ufb_debt_panel.csv")
        combined.to_csv(out_path, index=False)
        print(f"\nSuccessfully saved {len(combined)} panel records to {os.path.basename(out_path)}")
    else:
        print("No data extracted.")

if __name__ == '__main__':
    process_ufb_debt()
