"""
process_ms4.py
--------------
Downloads NJDEP MS4 outfall inventory from the NJDEP MapServer REST API,
aggregates to municipality level, and computes stormwater infrastructure
density as a resilience metric (replacing FEMA CRS scores).

Source: NJDEP MS4 Map — Outfall layer (73k+ records, NJ statewide)
  https://gisdata-njdep.opendata.arcgis.com/datasets/njdep::outfalls-in-new-jersey-njdep-ms4-inventory-and-mapping/about
  REST: https://mapsdep.nj.gov/arcgis/rest/services/Applications/MS4_Map/MapServer/5

Output: data/data_cleaned/finance/nj_ms4_resilience.csv
  mun                  – canonical MUN key matching nj_municipality_list
  municipality         – standardized name for panel join
  ms4_outfall_count    – total mapped outfall points in municipality
  ms4_outfall_density  – outfalls per sq mile (infrastructure intensity)
  ms4_has_data         – 1 if any outfall recorded (0 for municipalities with none)
"""

import os, sys, time, re
import requests
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "finance")
MUN_LIST     = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "boundaries", "nj_municipality_list.csv")

MS4_QUERY_URL = "https://mapsdep.nj.gov/arcgis/rest/services/Applications/MS4_Map/MapServer/5/query"
PAGE_SIZE     = 2000   # server maximum


def fetch_all_outfalls() -> pd.DataFrame:
    """Paginate through the MS4 outfalls layer and return every record."""
    records = []
    offset  = 0

    print("  Downloading MS4 outfall records (73k+, ~37 pages)...")
    while True:
        params = {
            "where":             "1=1",
            "outFields":         "MUNICIPALITY,COUNTY",
            "resultOffset":      offset,
            "resultRecordCount": PAGE_SIZE,
            "f":                 "json",
        }
        r = requests.get(MS4_QUERY_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            records.append(feat.get("attributes", {}))

        offset += len(features)
        print(f"    {offset:,} records fetched...", end="\r")

        if len(features) < PAGE_SIZE:
            break

        time.sleep(0.05)

    print(f"\n  Total records: {len(records):,}")
    return pd.DataFrame(records)


def normalize_muni(raw: str) -> str:
    """
    Normalize MS4 municipality names to match our canonical MUN format.
    e.g. 'Aberdeen Township' -> 'ABERDEEN TWP'
         'City of Trenton'   -> 'TRENTON CITY'
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""

    name = raw.strip().upper()

    # 'CITY OF X', 'TOWNSHIP OF X' etc. -> 'X CITY', 'X TWP'
    m = re.match(r"^(CITY|BOROUGH|TOWNSHIP|TOWN|VILLAGE)\s+OF\s+(.+)$", name)
    if m:
        suffix_map = {"CITY": "CITY", "BOROUGH": "BORO", "TOWNSHIP": "TWP", "TOWN": "TOWN", "VILLAGE": "VILLAGE"}
        name = m.group(2).strip() + " " + suffix_map[m.group(1)]
        return name

    # 'X TOWNSHIP' -> 'X TWP', 'X BOROUGH' -> 'X BORO'
    replacements = [
        (r"\bTOWNSHIP$", "TWP"),
        (r"\bBOROUGH$",  "BORO"),
    ]
    for pattern, repl in replacements:
        name = re.sub(pattern, repl, name)

    return name.strip()


def process_ms4():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("MS4 OUTFALL RESILIENCE PROCESSING")
    print("=" * 60)

    # ── 1. Download ───────────────────────────────────────────────────────
    print("\n[1/3] Fetching outfall records from NJDEP MapServer...")
    try:
        df = fetch_all_outfalls()
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Could not reach NJDEP MapServer. Check network.")
        return []

    if df.empty:
        print("  ERROR: No records returned.")
        return []

    df["muni_norm"] = df["MUNICIPALITY"].apply(normalize_muni)
    df = df[df["muni_norm"] != ""]

    # ── 2. Aggregate to municipality ──────────────────────────────────────
    print("\n[2/3] Aggregating to municipality level...")
    counts = (
        df.groupby("muni_norm")
          .size()
          .reset_index(name="ms4_outfall_count")
    )

    # ── 3. Join to canonical municipality list ────────────────────────────
    print("\n[3/3] Joining to municipality list and computing density...")
    mun_list = pd.read_csv(MUN_LIST)
    mun_list["muni_norm"] = mun_list["MUN"].apply(normalize_muni)

    merged = mun_list[["MUN", "municipality", "SQ_MILES", "muni_norm"]].merge(
        counts, on="muni_norm", how="left"
    )

    # Fallback: direct string match on raw MUN if normalized join missed records
    unmatched_count = merged["ms4_outfall_count"].isna().sum()
    if unmatched_count > 0:
        print(f"  {unmatched_count} municipalities unmatched after normalization — checking direct MUN match...")
        direct = counts[~counts["muni_norm"].isin(merged.loc[merged["ms4_outfall_count"].notna(), "muni_norm"])]
        direct_match = direct[direct["muni_norm"].isin(mun_list["MUN"])]
        if not direct_match.empty:
            for _, row in direct_match.iterrows():
                merged.loc[merged["MUN"] == row["muni_norm"], "ms4_outfall_count"] = row["ms4_outfall_count"]

    merged["ms4_outfall_count"]   = merged["ms4_outfall_count"].fillna(0).astype(int)
    merged["ms4_outfall_density"] = (
        merged["ms4_outfall_count"] / merged["SQ_MILES"].replace(0, float("nan"))
    ).fillna(0.0).round(4)
    merged["ms4_has_data"] = (merged["ms4_outfall_count"] > 0).astype(int)

    result = merged[["MUN", "municipality", "ms4_outfall_count", "ms4_outfall_density", "ms4_has_data"]].rename(
        columns={"MUN": "mun"}
    )

    out_path = os.path.join(OUTPUT_DIR, "nj_ms4_resilience.csv")
    result.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print("MS4 RESILIENCE SUMMARY")
    print(f"  Municipalities total:             {len(result)}")
    print(f"  With mapped outfalls:             {result['ms4_has_data'].sum()}")
    print(f"  Total outfalls mapped statewide:  {result['ms4_outfall_count'].sum():,}")
    has_data = result[result["ms4_has_data"] == 1]
    print(f"  Avg density (municipalities w/ data): "
          f"{has_data['ms4_outfall_density'].mean():.2f} outfalls/sq mi")
    print(f"  ✓ Saved → {os.path.relpath(out_path, PROJECT_ROOT)}")
    print("=" * 60)

    return [out_path]


if __name__ == "__main__":
    process_ms4()
