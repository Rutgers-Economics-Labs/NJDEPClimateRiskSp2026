"""
process_finance.py
------------------
Locates the UFB (Utility Finance Board) Database Excel workbook in the
project, reads relevant sheets, and exports cleaned CSV data to
data_cleaned/finance/.
"""

import os
import glob
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "finance")


def find_finance_files():
    """Search project tree for financial data files (.xlsm, .xlsx, .csv)."""
    results = []
    for ext in ["*.xlsm", "*.xlsx"]:
        pattern = os.path.join(DATA_RAW_DIR, "**", ext)
        results.extend(glob.glob(pattern, recursive=True))

    # Also look for bond issuance CSVs
    csv_pattern = os.path.join(DATA_RAW_DIR, "**", "bond_issuances*.csv")
    results.extend(glob.glob(csv_pattern, recursive=True))

    # Look for CRS scores and I-Bank data
    crs_pattern = os.path.join(DATA_RAW_DIR, "**", "crs_scores*")
    results.extend(glob.glob(crs_pattern, recursive=True))

    ibank_pattern = os.path.join(DATA_RAW_DIR, "**", "ibank*")
    results.extend(glob.glob(ibank_pattern, recursive=True))

    return results


def process_xlsm(filepath):
    """Process an Excel workbook, exporting each sheet as CSV."""
    print(f"\n  Processing: {os.path.relpath(filepath, PROJECT_ROOT)}")

    output_files = []
    try:
        # Read sheet names first
        xl = pd.ExcelFile(filepath, engine="openpyxl")
        sheet_names = xl.sheet_names
        print(f"  Sheets found: {sheet_names}")

        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(filepath, sheet_name=sheet_name,
                                   engine="openpyxl")

                # Skip empty sheets
                if df.empty or (len(df) < 2 and df.iloc[0].isna().all()):
                    print(f"    Skipping empty sheet: {sheet_name}")
                    continue

                # Clean sheet name for filename
                safe_name = sheet_name.strip().replace(" ", "_").replace("/", "-")
                safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")

                # Export
                out_path = os.path.join(OUTPUT_DIR, f"ufb_{safe_name}.csv")
                df.to_csv(out_path, index=False)
                print(f"    ✓ Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} cols → {os.path.basename(out_path)}")
                output_files.append(out_path)

            except Exception as e:
                print(f"    WARNING: Could not read sheet '{sheet_name}': {e}")

    except Exception as e:
        print(f"  ERROR: Could not open workbook: {e}")

    return output_files


def process_csv(filepath):
    """Copy and clean a standard CSV data file."""
    print(f"\n  Processing: {os.path.relpath(filepath, PROJECT_ROOT)}")

    try:
        df = pd.read_csv(filepath, dtype=str, low_memory=False)
        basename = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{basename}_cleaned.csv")
        df.to_csv(out_path, index=False)
        print(f"    ✓ {len(df)} rows → {os.path.basename(out_path)}")
        return [out_path]
    except Exception as e:
        print(f"    ERROR: {e}")
        return []


def process_finance():
    """Main processing pipeline for financial data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("FINANCIAL DATA PROCESSING")
    print("=" * 60)

    files = find_finance_files()
    if not files:
        print("WARNING: No financial data files found in project tree.")
        print(f"Searched under: {PROJECT_ROOT}")
        print("Expected files: UFB Database .xlsm, bond_issuances.csv,")
        print("                crs_scores.csv, ibank_priority_list.xlsx")
        return []

    print(f"\nFound {len(files)} financial data file(s):")
    for f in files:
        print(f"  • {os.path.relpath(f, PROJECT_ROOT)}")

    output_files = []
    for filepath in files:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in [".xlsm", ".xlsx"]:
            output_files.extend(process_xlsm(filepath))
        elif ext == ".csv":
            output_files.extend(process_csv(filepath))
        else:
            print(f"\n  Skipping unsupported format: {os.path.basename(filepath)}")

    return output_files


if __name__ == "__main__":
    output_files = process_finance()
    print(f"\n{'=' * 60}")
    print(f"Finance processing complete. {len(output_files)} file(s) created.")
