"""
process_ms4_tiers.py
--------------------
Parses the NJPDES Municipal Stormwater Regulation Program tier PDF and
exports a municipality-level Tier A indicator.

Input:
  data/data_raw/ms4_municipal_tier_list.pdf

Outputs:
  data/data_cleaned/finance/nj_ms4_tier_flags.csv
  updates data/data_cleaned/finance/nj_ms4_resilience.csv when present
"""

import os
import re
import subprocess
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "data_raw")
DATA_CLEANED = os.path.join(PROJECT_ROOT, "data", "data_cleaned")
FINANCE_DIR = os.path.join(DATA_CLEANED, "finance")

PDF_FILE = os.path.join(DATA_RAW, "ms4_municipal_tier_list.pdf")
MUN_LIST = os.path.join(DATA_CLEANED, "boundaries", "nj_municipality_list.csv")
TIER_OUTPUT = os.path.join(FINANCE_DIR, "nj_ms4_tier_flags.csv")
MS4_RESILIENCE = os.path.join(FINANCE_DIR, "nj_ms4_resilience.csv")


ALIASES = {
    ("ESSEX", "CITY OF ORANGE TWP"): "CITY OF ORANGE TWP",
    ("ESSEX", "ORANGE CITY"): "CITY OF ORANGE TWP",
    ("ESSEX", "SOUTH ORANGE VILLAGE TWP"): "SOUTH ORANGE VILLAGE",
    ("SOMERSET", "PEAPACK AND GLADSTONE BORO"): "PEAPACK-GLADSTONE BORO",
}


def attach_county_to_ms4(ms4, mun_list):
    """Attach county to the existing MS4 file without collapsing duplicate town names."""
    if "county" in ms4.columns:
        ms4["county"] = ms4["county"].str.upper().str.strip()
        return ms4

    keys = mun_list[["MUN", "COUNTY"]].rename(columns={"MUN": "mun", "COUNTY": "county"})
    keys["county"] = keys["county"].str.upper().str.strip()

    # nj_ms4_resilience.csv is built from the same municipality list, so row order
    # is the only unambiguous key for repeated names like Hamilton Twp or Ocean Twp.
    if len(ms4) == len(keys) and ms4["mun"].astype(str).reset_index(drop=True).equals(
        keys["mun"].astype(str).reset_index(drop=True)
    ):
        ms4 = ms4.copy()
        insert_at = 2 if "municipality" in ms4.columns else 1
        ms4.insert(insert_at, "county", keys["county"].values)
        return ms4

    duplicate_muns = keys.loc[keys["mun"].duplicated(keep=False), "mun"].unique()
    if ms4["mun"].isin(duplicate_muns).any():
        problem = sorted(ms4.loc[ms4["mun"].isin(duplicate_muns), "mun"].dropna().unique())[:10]
        raise ValueError(
            "Cannot attach county to MS4 rows by municipality name alone because names are duplicated: "
            f"{problem}"
        )

    return ms4.merge(keys, on="mun", how="left", validate="m:1")


def normalize_muni_name(raw):
    """Normalize PDF municipality names to the canonical MUN style."""
    if not isinstance(raw, str):
        return ""

    name = raw.upper().replace("*", "").strip()
    name = re.sub(r"\s+", " ", name)
    name = name.replace("AVON-BY-THE-SEA", "AVON BY THE SEA")

    replacements = [
        (r"\bTOWNSHIP$", "TWP"),
        (r"\bBOROUGH$", "BORO"),
    ]
    for pattern, repl in replacements:
        name = re.sub(pattern, repl, name)

    return name.strip()


def extract_pdf_text(pdf_path):
    """Use poppler's pdftotext to extract readable text from the tier PDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"MS4 tier PDF not found: {pdf_path}")
    try:
        return subprocess.check_output(["pdftotext", pdf_path, "-"], text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("pdftotext is required to parse ms4_municipal_tier_list.pdf") from exc


def parse_tier_a_rows(text, valid_counties):
    """Return raw county/name rows listed under each Tier A section."""
    rows = []
    current_county = None
    current_tier = None

    ignored = {
        "NJPDES MUNICIPAL STORMWATER REGULATION PROGRAM",
        "TIERLSTL",
        "4/20/17",
        "NONE",
        "*",
    }

    for raw_line in text.splitlines():
        line = raw_line.strip().strip("\x0c").strip()
        if not line or line.isdigit():
            continue

        upper = line.upper()
        if upper in ignored:
            continue
        if upper.startswith("IT IS THE DEPARTMENT") or upper.startswith("BOROUGH,"):
            continue
        if upper.startswith("WEST NEW YORK TOWN DO NOT OPERATE"):
            continue
        if upper.startswith("7:14A-25") or upper.startswith("STORMWATER REGULATION"):
            continue

        if line.endswith(" County"):
            county = line[:-7].upper().strip()
            if county in valid_counties:
                current_county = county
                current_tier = None
            continue

        if upper == "TIER A MUNICIPALITIES":
            current_tier = "A"
            continue
        if upper == "TIER B MUNICIPALITIES":
            current_tier = "B"
            continue

        if current_county and current_tier == "A":
            rows.append({"county": current_county, "pdf_municipality": line})

    return pd.DataFrame(rows)


def process_ms4_tiers():
    os.makedirs(FINANCE_DIR, exist_ok=True)

    print("=" * 60)
    print("MS4 MUNICIPAL TIER PARSING")
    print("=" * 60)

    mun_list = pd.read_csv(MUN_LIST)
    mun_list["county"] = mun_list["COUNTY"].str.upper().str.strip()
    mun_list["mun_norm"] = mun_list["MUN"].apply(normalize_muni_name)

    text = extract_pdf_text(PDF_FILE)
    tier_a_raw = parse_tier_a_rows(text, set(mun_list["county"]))
    tier_a_raw["mun_norm"] = tier_a_raw["pdf_municipality"].apply(normalize_muni_name)
    tier_a_raw["mun_norm"] = tier_a_raw.apply(
        lambda row: ALIASES.get((row["county"], row["mun_norm"]), row["mun_norm"]),
        axis=1,
    )

    matches = tier_a_raw.merge(
        mun_list[["MUN", "county", "mun_norm"]],
        on=["county", "mun_norm"],
        how="left",
        validate="m:1",
    )

    missing = matches[matches["MUN"].isna()]
    if not missing.empty:
        sample = missing[["county", "pdf_municipality", "mun_norm"]].to_dict("records")
        raise ValueError(f"Could not match {len(missing)} Tier A rows to canonical municipalities: {sample}")

    tier_a_keys = set(zip(matches["county"], matches["MUN"]))
    result = mun_list[["MUN", "COUNTY", "municipality"]].copy()
    result = result.rename(columns={"MUN": "mun", "COUNTY": "county"})
    result["county"] = result["county"].str.upper().str.strip()
    result["ms4_tier_a"] = result.apply(
        lambda row: int((row["county"], row["mun"]) in tier_a_keys),
        axis=1,
    )

    result.to_csv(TIER_OUTPUT, index=False)

    outputs = [TIER_OUTPUT]
    if os.path.exists(MS4_RESILIENCE):
        ms4 = pd.read_csv(MS4_RESILIENCE)
        ms4 = attach_county_to_ms4(ms4, mun_list)
        ms4 = ms4.drop(columns=["ms4_tier_a"], errors="ignore")
        ms4 = ms4.merge(result[["mun", "county", "ms4_tier_a"]], on=["mun", "county"], how="left", validate="m:1")
        ms4["ms4_tier_a"] = ms4["ms4_tier_a"].fillna(0).astype(int)
        ms4.to_csv(MS4_RESILIENCE, index=False)
        outputs.append(MS4_RESILIENCE)

    print(f"  Parsed Tier A rows from PDF: {len(tier_a_raw):,}")
    print(f"  Matched Tier A municipalities: {len(tier_a_keys):,}")
    print(f"  Total municipalities: {len(result):,}")
    print(f"  Saved -> {os.path.relpath(TIER_OUTPUT, PROJECT_ROOT)}")
    if os.path.exists(MS4_RESILIENCE):
        print(f"  Updated -> {os.path.relpath(MS4_RESILIENCE, PROJECT_ROOT)}")
    print("=" * 60)

    return outputs


if __name__ == "__main__":
    process_ms4_tiers()
