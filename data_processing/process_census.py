"""
process_census.py
-----------------
Locates US Census ACS DP03 CSV files in the project, cleans the
double-header format, filters for NJ municipalities, extracts key
economic variables, and standardizes municipality names for joining.
Outputs cleaned CSV to data_cleaned/census/.
"""

import os
import re
import glob
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "census")

# Key columns to extract (Census variable codes)
KEY_COLUMNS = {
    "GEO_ID": "geo_id",
    "NAME": "municipality_raw",
    "DP03_0062E": "median_household_income",
    "DP03_0009PE": "unemployment_rate_pct",
    "DP03_0063E": "mean_household_income",
    "DP03_0119E": "poverty_rate_families_pct",
}

# NJ municipality type suffixes to strip during standardization
MUNI_SUFFIXES = [
    "city", "borough", "township", "town", "village",
    "CDP", "cdp",
]


def find_census_files():
    """Search the data_raw directory for ACS DP03 data CSV files."""
    pattern = os.path.join(DATA_RAW_DIR, "**", "ACSDP5Y*.DP03-Data.csv")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def standardize_muni_name(raw_name):
    """
    Clean Census municipality names for cross-dataset joining.
    'Sea Isle City city, Cape May County, New Jersey' → 'SEA ISLE CITY'
    """
    if not isinstance(raw_name, str):
        return raw_name

    # Remove state name
    name = re.sub(r",\s*New Jersey\s*$", "", raw_name, flags=re.IGNORECASE)

    # Remove county name (everything after last comma)
    name = re.sub(r",\s*[^,]+County\s*$", "", name, flags=re.IGNORECASE)

    # Remove municipality type suffixes
    for suffix in MUNI_SUFFIXES:
        name = re.sub(rf"\s+{suffix}\s*$", "", name, flags=re.IGNORECASE)

    # Clean whitespace and uppercase
    name = name.strip().upper()

    return name


def extract_year_from_filename(filepath):
    """Extract the ACS year from filename like ACSDP5Y2023.DP03-Data.csv."""
    basename = os.path.basename(filepath)
    match = re.search(r"ACSDP5Y(\d{4})", basename)
    return match.group(1) if match else "unknown"


def process_census_file(filepath):
    """Process a single Census DP03 CSV file."""
    year = extract_year_from_filename(filepath)
    print(f"\n  Processing ACS {year} from: {os.path.relpath(filepath, PROJECT_ROOT)}")

    # Read with first row as header (Census code names)
    # Skip the second row which contains human-readable descriptions
    df = pd.read_csv(filepath, dtype=str, low_memory=False)

    # The first row after the code-header is the description row — skip it
    # Census CSVs: Row 0 = codes (used as header by pandas), Row 1 = descriptions
    # Check if the first data row looks like descriptions
    if df.iloc[0]["GEO_ID"] == "Geography":
        df = df.iloc[1:].reset_index(drop=True)
        print(f"  Skipped description header row")

    # Filter for NJ County Subdivisions (municipalities)
    # GEO_ID format: 0600000US34XXXYYYYY (06 = county subdivision, 34 = NJ FIPS)
    nj_mask = df["GEO_ID"].str.startswith("0600000US34")
    df_nj = df[nj_mask].copy()
    print(f"  Found {len(df_nj)} NJ municipalities")

    # Select only key columns (those that exist in data)
    available_cols = [c for c in KEY_COLUMNS.keys() if c in df_nj.columns]
    missing_cols = [c for c in KEY_COLUMNS.keys() if c not in df_nj.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns: {missing_cols}")

    df_clean = df_nj[available_cols].copy()

    # Rename columns to clean names
    df_clean = df_clean.rename(columns={k: v for k, v in KEY_COLUMNS.items() if k in available_cols})

    # Standardize municipality names
    df_clean["municipality"] = df_clean["municipality_raw"].apply(standardize_muni_name)

    # Convert numeric columns
    for col in ["median_household_income", "unemployment_rate_pct",
                 "mean_household_income", "poverty_rate_families_pct"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Add year column
    df_clean["acs_year"] = int(year)

    # Sort by municipality
    df_clean = df_clean.sort_values("municipality").reset_index(drop=True)

    return df_clean


def process_census():
    """Main processing pipeline for Census data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("CENSUS / ACS DATA PROCESSING")
    print("=" * 60)

    census_files = find_census_files()
    if not census_files:
        print("ERROR: No ACS DP03 CSV files found in project tree!")
        print(f"Searched under: {DATA_RAW_DIR}")
        return []

    print(f"\nFound {len(census_files)} Census data file(s):")
    for f in census_files:
        print(f"  • {os.path.relpath(f, PROJECT_ROOT)}")

    output_files = []
    all_frames = []

    for filepath in census_files:
        df_clean = process_census_file(filepath)
        all_frames.append(df_clean)

        # Export individual year file
        year = extract_year_from_filename(filepath)
        out_path = os.path.join(OUTPUT_DIR, f"nj_census_dp03_{year}.csv")
        df_clean.to_csv(out_path, index=False)
        print(f"  ✓ Exported: {os.path.relpath(out_path, PROJECT_ROOT)}")
        print(f"    Rows: {len(df_clean)}, Columns: {list(df_clean.columns)}")
        output_files.append(out_path)

    # Also export a combined file with all years
    if len(all_frames) > 1:
        combined = pd.concat(all_frames, ignore_index=True)
        out_combined = os.path.join(OUTPUT_DIR, "nj_census_dp03_combined.csv")
        combined.to_csv(out_combined, index=False)
        print(f"\n  ✓ Combined file: {os.path.relpath(out_combined, PROJECT_ROOT)}")
        print(f"    Total rows: {len(combined)}")
        output_files.append(out_combined)

    return output_files


if __name__ == "__main__":
    output_files = process_census()
    print(f"\n{'=' * 60}")
    print(f"Census processing complete. {len(output_files)} file(s) created.")
