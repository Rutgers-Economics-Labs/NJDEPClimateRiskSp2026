import os
import pandas as pd
from thefuzz import process, fuzz

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned")

FLOOD_FILE = os.path.join(DATA_CLEANED_DIR, "climate", "nj_municipality_flood_area_summary.csv")
FEMA_FILE = os.path.join(DATA_CLEANED_DIR, "finance", "nj_fema_crs_cleaned.csv")
CENSUS_FILE = os.path.join(DATA_CLEANED_DIR, "census", "nj_census_dp03_combined.csv")

OUTPUT_FILE = os.path.join(DATA_CLEANED_DIR, "nj_municipality_characteristics.csv")

def clean_fema_name(name):
    """
    Cleans FEMA names like 'ABERDEEN, TOWNSHIP OF' -> 'ABERDEEN TOWNSHIP'
    """
    if pd.isna(name):
        return ""
    name = str(name).upper().strip()
    if ", " in name:
        parts = name.split(", ", 1)
        # Check if the second part is something like 'TOWNSHIP OF', 'CITY OF', 'BOROUGH OF'
        suffix = parts[1].replace(" OF", "").strip()
        if suffix in ["TOWNSHIP", "CITY", "BOROUGH", "TOWN", "VILLAGE"]:
            # E.g. "ABERDEEN TOWNSHIP"
            # The flood file uses "BORO" instead of "BOROUGH" usually.
            if suffix == "BOROUGH":
                suffix = "BORO"
            name = f"{parts[0]} {suffix}"
        else:
            name = f"{parts[0]} {parts[1]}"
    return name

def clean_county_name(name):
    if pd.isna(name):
        return ""
    return str(name).upper().replace(" COUNTY", "").strip()

def match_mun_names(source_muns, target_muns):
    """
    Uses fuzzy matching to align names if exact matching fails.
    """
    matches = {}
    target_choices = list(target_muns)
    
    for muni in source_muns:
        # Exact match attempt
        if muni in target_choices:
            matches[muni] = muni
            continue
            
        # Fuzzy match
        best_match, score = process.extractOne(muni, target_choices, scorer=fuzz.token_sort_ratio)
        if score >= 85:  # Tolerance threshold
            matches[muni] = best_match
        else:
            matches[muni] = None
            
    return matches

def main():
    print("Loading data files...")
    df_flood = pd.read_csv(FLOOD_FILE)
    df_fema = pd.read_csv(FEMA_FILE)
    df_census = pd.read_csv(CENSUS_FILE)
    
    # 1. Base DataFrame is the Flood Data (since we have exact areas for everybody)
    # Ensure MUN and COUNTY are clean
    df_flood["MUN"] = df_flood["MUN"].str.upper().str.strip()
    df_flood["COUNTY"] = df_flood["COUNTY"].str.upper().str.strip()
    
    # 2. Merge FEMA Data
    print("Merging FEMA CRS data...")
    df_fema["fmt_county"] = df_fema["county"].apply(clean_county_name)
    df_fema["fmt_mun"] = df_fema["municipality"].apply(clean_fema_name)
    
    # We match within counties to prevent cross-county duplicate names (e.g., Washington Twp)
    fema_merged = pd.DataFrame()
    for county in df_flood["COUNTY"].unique():
        flood_county = df_flood[df_flood["COUNTY"] == county].copy()
        fema_county = df_fema[df_fema["fmt_county"] == county].copy()
        
        if fema_county.empty:
            fema_merged = pd.concat([fema_merged, flood_county])
            continue
            
        match_dict = match_mun_names(flood_county["MUN"], fema_county["fmt_mun"])
        
        # Apply the match mapping
        flood_county["fema_mun_match"] = flood_county["MUN"].map(match_dict)
        
        # Merge
        merged = pd.merge(flood_county, fema_county, left_on="fema_mun_match", right_on="fmt_mun", how="left")
        merged.drop(columns=["fema_mun_match", "fmt_county", "fmt_mun"], inplace=True)
        fema_merged = pd.concat([fema_merged, merged])

    # 3. Merge Census Data
    print("Merging Census ACS data (2023 Cross-section)...")
    # Take latest year for cross-sectional controls
    df_census_latest = df_census[df_census["acs_year"] == 2023].copy()
    
    # Census only has 'municipality' without suffix (e.g. 'ABERDEEN')
    # Because there are duplicate names (like "WASHINGTON") an exact match might miss the county context.
    # Let's map census data smartly. The flood 'MUN' is 'ABERDEEN TWP'. 
    # We extract the base name.
    
    base_panel = fema_merged.copy()
    base_panel["MUN_BASE"] = base_panel["MUN"].str.replace(r"\\s+(TWP|BORO|CITY|TOWN|VILLAGE)$", "", regex=True)
    
    census_merged = pd.DataFrame()
    match_dict = match_mun_names(base_panel["MUN_BASE"], df_census_latest["municipality"])
    base_panel["census_mun_match"] = base_panel["MUN_BASE"].map(match_dict)
    
    final_panel = pd.merge(base_panel, df_census_latest, left_on="census_mun_match", right_on="municipality", how="left", suffixes=("", "_census"))
    
    final_panel.drop(columns=["MUN_BASE", "census_mun_match"], inplace=True)
    
    # 4. Cleanup & Derived Metrics Classification
    print("Classifying Resilience vs Non-Resilient cohorts...")
    final_panel["crs_class"] = final_panel["crs_class"].fillna(10) # 10 is non-participating
    final_panel["is_resilient"] = final_panel["crs_class"] < 10 # lower class means better rating
    
    final_panel.to_csv(OUTPUT_FILE, index=False)
    print(f"Master characteristics panel saved to {OUTPUT_FILE}")
    print(f"Total Municipalities tracked: {len(final_panel)}")
    
if __name__ == "__main__":
    main()
