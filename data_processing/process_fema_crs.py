"""
process_fema_crs.py
-------------------
Parses the FEMA Community Rating System (CRS) HTML report for NJ to
extract community flood insurance discount information.
Input: data/data_raw/fema_nj_crs.html
Output: data/data_cleaned/finance/nj_fema_crs_cleaned.csv
"""

import os
import re
import pandas as pd
from bs4 import BeautifulSoup

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "finance")
INPUT_FILE = "fema_nj_crs.html"


def find_crs_html():
    """Locate the FEMA CRS HTML file."""
    # Try exact path first
    path = os.path.join(DATA_RAW_DIR, INPUT_FILE)
    if os.path.exists(path):
        return path
    
    # Fallback search
    for root, _, files in os.walk(DATA_RAW_DIR):
        if INPUT_FILE in files:
            return os.path.join(root, INPUT_FILE)
            
    return None


def clean_text(text):
    """Clean extracted text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def parse_crs_html(filepath):
    """Parse the FEMA CRS HTML file."""
    print(f"  Parsing: {os.path.relpath(filepath, PROJECT_ROOT)}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        
    data = []
    rows = soup.find_all('tr')
    
    print(f"  Found {len(rows)} table rows. Extracting community data...")
    
    for row in rows:
        cells = row.find_all('td')
        if not cells:
            continue
            
        # Get text from all cells
        cell_texts = [clean_text(cell.get_text()) for cell in cells]
        
        # Filter out empty/spacer cells
        # The HTML has many empty spacer cells. We need to identify the content cells.
        # Based on inspection:
        # CID is usually the first non-empty cell if it matches pattern 34xxxx
        
        content_cells = [c for c in cell_texts if c]
        
        if not content_cells:
            continue
            
        first_cell = content_cells[0]
        
        # Check if row looks like a community record (CID starts with 34, length 6 or 7)
        # e.g. 340312C, 340001A
        if re.match(r'^34\d{4}[A-Z#]?$', first_cell):
            # Detailed row structure extraction based on inspection
            # 0: CID (340312C)
            # 1: Community Name (ABERDEEN, TOWNSHIP OF)
            # 2: County (MONMOUTH COUNTY)
            # ... variable dates ...
            # Second to last meaningful cell is usually Class (e.g. 8)
            # Last meaningful cell is usually Discount (e.g. 10%)
            
            # We can rely on the order of content_cells
            # Expected content items:
            # CID, Name, County, Date1, Date2, Date3, Status, Date4, Date5, Date6, Class, Discount%
            
            if len(content_cells) >= 5:
                # Extract key fields
                cid = first_cell
                name = content_cells[1]
                county = content_cells[2]
                
                # Discount is typically the last item with '%'
                discount = next((c for c in reversed(content_cells) if '%' in c), "0%")
                
                # Class is usually the item before discount, or just a single digit/number
                # Scan from end, skip discount, find first single/double digit number
                crs_class = "10" # Default/No discount class
                
                # Iterate backwards to find class
                for item in reversed(content_cells):
                    if item == discount: continue
                    if re.match(r'^\d{1,2}$', item):
                        crs_class = item
                        break
                        
                # Handling status "Rescinded" cases
                if "Rescinded" in content_cells:
                    status = "Rescinded"
                else:
                    status = "Active" if discount != "0%" else "Not Participating"

                data.append({
                    "cid": cid,
                    "municipality": name,
                    "county": county,
                    "crs_class": crs_class,
                    "crs_discount": discount,
                    "status": status
                })

    return pd.DataFrame(data)


def process_fema_crs():
    """Main processing pipeline for FEMA CRS data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("FEMA CRS DATA PROCESSING")
    print("=" * 60)

    html_path = find_crs_html()
    if not html_path:
        print("ERROR: fema_nj_crs.html not found in data/data_raw/!")
        return []

    df = parse_crs_html(html_path)
    
    if df.empty:
        print("  WARNING: No data extracted.")
        return []
        
    print(f"  Extracted {len(df)} community records.")
    
    # Clean up fields
    # Percentage to float
    df['crs_discount_pct'] = df['crs_discount'].str.replace('%', '').astype(float)
    
    # Class to int
    df['crs_class'] = pd.to_numeric(df['crs_class'], errors='coerce').fillna(10).astype(int)
    
    # Clean municipality names
    # Reuse standardization logic if needed, but keeping it simple first
    
    out_path = os.path.join(OUTPUT_DIR, "nj_fema_crs_cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"  ✓ Exported: {os.path.relpath(out_path, PROJECT_ROOT)}")
    print(f"    Rows: {len(df)}, Columns: {list(df.columns)}")
    
    return [out_path]


if __name__ == "__main__":
    output_files = process_fema_crs()
    print(f"\n{'=' * 60}")
    print(f"FEMA CRS processing complete. {len(output_files)} file(s) created.")
