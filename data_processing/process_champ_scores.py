import os
import glob
import pdfplumber
import pandas as pd
import re
import sys

# Add current dir to path to import process_census
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from process_census import standardize_muni_name

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw", "champ")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "finance")

MUNI_TYPE_SUFFIXES = {
    "CITY": "CITY",
    "BOROUGH": "BORO",
    "BORO": "BORO",
    "TOWNSHIP": "TWP",
    "TWP": "TWP",
    "TOWN": "TOWN",
    "VILLAGE": "VILLAGE",
}


def normalize_applicant_name(raw_name):
    """Normalize applicant names without stripping words that are part of names."""
    if not isinstance(raw_name, str):
        return ""
    name = re.sub(r"\s+", " ", raw_name.upper().strip())
    name = re.sub(r"\bTOWNSHIP$", "TWP", name)
    name = re.sub(r"\bBOROUGH$", "BORO", name)
    name = re.sub(r"\bCITY OF\s+(.+)$", r"\1 CITY", name)
    name = re.sub(r"\bBOROUGH OF\s+(.+)$", r"\1 BORO", name)
    name = re.sub(r"\bTOWNSHIP OF\s+(.+)$", r"\1 TWP", name)
    return name


def extract_sfy(source_file):
    match = re.search(r"SFY\s*(?:20)?(\d{2})", source_file, flags=re.IGNORECASE)
    if match:
        return int(f"20{match.group(1)}")
    return pd.NA

def extract_champ_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    
    all_projects = []
    
    for pdf_path in pdf_files:
        print(f"Processing: {os.path.basename(pdf_path)}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for t in tables:
                        # Find the header row
                        header_idx = -1
                        for r_idx, row in enumerate(t):
                            cols = [str(c).upper().strip() for c in row if c]
                            if any("APPLICANT" in c for c in cols) and any("RANK" in c or "PROJECT" in c for c in cols):
                                header_idx = r_idx
                                break
                        
                        if header_idx != -1:
                            headers = [str(c).upper().replace('\n', ' ').strip() for c in t[header_idx]]
                            
                            # Find index of APPLICANT and DESCRIPTION
                            app_idx = -1
                            desc_idx = -1
                            rank_idx = -1
                            
                            for c_idx, h in enumerate(headers):
                                if "APPLICANT" in h:
                                    app_idx = c_idx
                                elif "DESCRIPTION" in h:
                                    desc_idx = c_idx
                                elif "RANK" in h:
                                    rank_idx = c_idx
                            
                            if app_idx != -1:
                                for row in t[header_idx+1:]:
                                    if not row or not any(row):
                                        continue
                                    
                                    # Handle rows with fewer columns
                                    if len(row) <= app_idx:
                                        continue
                                        
                                    applicant = str(row[app_idx]).replace('\n', ' ').strip()
                                    if not applicant or applicant.upper() == 'NONE' or 'APPLICANT' in applicant.upper() or 'TOTAL' in applicant.upper() or len(applicant) > 40 or 'CRITERION' in applicant.upper() or 'TIEBREAKER' in applicant.upper() or 'PROJECT AREA' in applicant.upper():
                                        continue
                                        
                                    desc = str(row[desc_idx]).replace('\n', ' ').strip() if desc_idx != -1 and len(row) > desc_idx else ""
                                    rank = str(row[rank_idx]).replace('\n', ' ').strip() if rank_idx != -1 and len(row) > rank_idx else ""
                                    
                                    all_projects.append({
                                        "source_file": os.path.basename(pdf_path),
                                        "rank": rank,
                                        "applicant_raw": applicant,
                                        "description": desc
                                    })
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

    df = pd.DataFrame(all_projects)
    
    if len(df) == 0:
        print("No CHAMP projects extracted. Please review PDF table structures.")
        return
        
    # Keep both fields: the old stripped value is useful for inspection, but
    # applicant_key preserves names like ATLANTIC CITY and JERSEY CITY.
    df["municipality"] = df["applicant_raw"].apply(standardize_muni_name)
    df["applicant_key"] = df["applicant_raw"].apply(normalize_applicant_name)
    df["sfy"] = df["source_file"].apply(extract_sfy)
    
    # Save raw scores/projects
    raw_out = os.path.join(OUTPUT_DIR, "nj_champ_raw_scores.csv")
    df.to_csv(raw_out, index=False)
    print(f"Saved {len(df)} raw projects to {raw_out}")
    
    # Create the binary flag: is_resilient = 1 if they appear in this list
    # We drop NA municipalities (e.g. state authorities that don't match our list)
    # or keep them just in case.
    resilient_munis = df[["applicant_key"]].drop_duplicates().copy()
    resilient_munis = resilient_munis[resilient_munis["applicant_key"] != ""]
    resilient_munis["is_resilient"] = 1
    
    # Also attach their minimum rank just for context
    # Rank might be "N/A" or missing, so we safely convert
    df["rank_num"] = pd.to_numeric(df["rank"], errors='coerce')
    min_ranks = df.groupby("applicant_key")["rank_num"].min().reset_index()
    first_sfy = df.groupby("applicant_key")["sfy"].min().reset_index()
    resilient_munis = resilient_munis.merge(min_ranks, on="applicant_key", how="left")
    resilient_munis = resilient_munis.merge(first_sfy, on="applicant_key", how="left")
    resilient_munis.rename(columns={"rank_num": "best_champ_rank"}, inplace=True)
    
    flag_out = os.path.join(OUTPUT_DIR, "nj_champ_resilience_flags.csv")
    resilient_munis.to_csv(flag_out, index=False)
    print(f"Saved {len(resilient_munis)} resilient flags to {flag_out}")

if __name__ == '__main__':
    extract_champ_data()
