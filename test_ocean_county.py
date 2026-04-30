"""
Ocean County Flood Risk Test Pipeline (5ft SLR) — Memory Efficient
====================================================================
Processes parcels in small chunks to stay within 8GB RAM.
Shows a real progress bar for the intersection step.

Gridcode key for 5ft SLR file:
    3    : Ocean-connected permanent inundation (high confidence)
    1    : Hydrologically unconnected low-lying areas (lower confidence)
    2260 : Large contiguous low-lying region (treated same as gridcode 1)

Requirements:
    pip3 install geopandas pandas numpy tqdm shapely

Usage:
    /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 test_ocean_county.py
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PARCEL_GDB_PATH    = "Statewide_Parcels_MODIV.gdb"
PARCEL_LAYER       = "Cad_parcel_mod4"
FLOOD_GEOJSON_PATH = "nj_slr_5ft_merged.geojson"
EQ_RATIO_CSV_PATH  = "nj_county_avg_ratio.csv"
OUTPUT_CSV_PATH    = "ocean_county_frm_scores.csv"

WEIGHT_CONNECTED   = 1.0
WEIGHT_LOWLYING    = 0.5
TAXABLE_CLASSES    = ["1", "2", "3A", "3B", "4A", "4B", "4C"]
NJ_CRS             = "EPSG:3424"
GRIDCODE_CONNECTED = [3]
GRIDCODE_LOWLYING  = [1, 2260]
CHUNK_SIZE         = 5000   # parcels per batch — reduces memory pressure

SCORE_THRESHOLDS = {
    1: (0.00, 0.02),
    2: (0.02, 0.07),
    3: (0.07, 0.15),
    4: (0.15, 0.30),
    5: (0.30, 1.01),
}

# ---------------------------------------------------------------------------
# HELPER: compute flood overlap using STRtree spatial index + chunked batches
# ---------------------------------------------------------------------------

def compute_overlap(parcels_proj, flood_geoms_list, label):
    """
    For each parcel, computes what fraction of its area overlaps with
    flood polygons. Uses Shapely's STRtree spatial index for fast lookup
    and processes parcels in chunks to keep memory usage low.

    Args:
        parcels_proj    : GeoDataFrame of parcels in NJ_CRS
        flood_geoms_list: list of Shapely geometries for the flood tier
        label           : string label for progress bar

    Returns:
        pd.Series of pct_flooded values (0-1), indexed like parcels_proj
    """
    if not flood_geoms_list:
        print(f"  Skipping {label} — no features in this area.")
        return pd.Series(0.0, index=parcels_proj.index)

    print(f"\n  Building spatial index for {label} ({len(flood_geoms_list):,} polygons)...")
    tree = STRtree(flood_geoms_list)

    parcel_geoms  = parcels_proj.geometry.values
    parcel_areas  = parcels_proj.geometry.area.values
    pct_flooded   = np.zeros(len(parcels_proj), dtype=np.float64)

    n_chunks = int(np.ceil(len(parcel_geoms) / CHUNK_SIZE))

    print(f"  Intersecting {len(parcels_proj):,} parcels in {n_chunks} chunks of {CHUNK_SIZE:,}...")
    for i in tqdm(range(n_chunks), desc=f"  {label}", unit="chunk"):
        start = i * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, len(parcel_geoms))

        for j in range(start, end):
            parcel_geom = parcel_geoms[j]
            if parcel_geom is None or parcel_geom.is_empty:
                continue

            # Find candidate flood polygons that might intersect this parcel
            candidate_idxs = tree.query(parcel_geom)
            if len(candidate_idxs) == 0:
                continue

            # Compute actual intersection area
            flooded_area = 0.0
            for idx in candidate_idxs:
                flood_geom = flood_geoms_list[idx]
                if parcel_geom.intersects(flood_geom):
                    flooded_area += parcel_geom.intersection(flood_geom).area

            if parcel_areas[j] > 0:
                pct_flooded[j] = min(flooded_area / parcel_areas[j], 1.0)

    result = pd.Series(pct_flooded, index=parcels_proj.index)
    n_exposed = (result > 0).sum()
    print(f"  Parcels exposed: {n_exposed:,} ({n_exposed/len(parcels_proj)*100:.1f}%)")
    return result


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    # -----------------------------------------------------------------------
    # STEP 1: Load equalization ratios
    # -----------------------------------------------------------------------
    print("=" * 55)
    print("STEP 1: Loading equalization ratios")
    print("=" * 55)
    eq_ratios = pd.read_csv(EQ_RATIO_CSV_PATH)
    eq_ratios.columns = ["COUNTY_NAME", "EQ_RATIO"]
    eq_ratios = eq_ratios[eq_ratios["COUNTY_NAME"] != "STATE TOTALS"].copy()
    eq_ratios["EQ_RATIO"] = eq_ratios["EQ_RATIO"] / 100
    eq_ratios["COUNTY_NAME"] = (
        eq_ratios["COUNTY_NAME"]
        .str.replace(" COUNTY", "", regex=False)
        .str.strip()
    )
    print(f"  Ratios loaded for {len(eq_ratios)} counties.")

    # -----------------------------------------------------------------------
    # STEP 2: Load Ocean County parcels
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 2: Loading Ocean County parcels")
    print("=" * 55)
    print("  Reading GDB (this may take a minute)...")
    all_parcels = gpd.read_file(
        PARCEL_GDB_PATH, layer=PARCEL_LAYER, engine="pyogrio"
    )
    parcels = all_parcels[all_parcels["COUNTY"] == "OCEAN"].copy()
    del all_parcels  # free memory immediately
    print(f"  Ocean County parcels loaded: {len(parcels):,}")

    # -----------------------------------------------------------------------
    # STEP 3: Load flood layer
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 3: Loading 5ft SLR flood layer")
    print("=" * 55)
    flood = gpd.read_file(FLOOD_GEOJSON_PATH)
    print(f"  Total flood features: {len(flood):,}")
    print(f"  Gridcode distribution:\n{flood['gridcode'].value_counts().to_string()}")

    # -----------------------------------------------------------------------
    # STEP 4: Clip flood layer to Ocean County bbox
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 4: Clipping flood layer to Ocean County bbox")
    print("=" * 55)
    parcels_wgs84 = parcels.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = parcels_wgs84.total_bounds
    print(f"  Ocean County bbox (WGS84): {minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}")

    county_box = box(minx, miny, maxx, maxy)
    flood_connected_clip = flood[
        flood["gridcode"].isin(GRIDCODE_CONNECTED) &
        flood.geometry.intersects(county_box)
    ].copy()
    flood_lowlying_clip = flood[
        flood["gridcode"].isin(GRIDCODE_LOWLYING) &
        flood.geometry.intersects(county_box)
    ].copy()
    del flood  # free memory

    print(f"  Ocean-connected (gridcode 3) in bbox:  {len(flood_connected_clip):,}")
    print(f"  Low-lying (gridcode 1, 2260) in bbox:  {len(flood_lowlying_clip):,}")

    if len(flood_connected_clip) == 0:
        print("  NOTE: No ocean-connected polygons in Ocean County — using low-lying only.")

    # -----------------------------------------------------------------------
    # STEP 5: Reproject everything to NJ State Plane
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 5: Reprojecting to NJ State Plane")
    print("=" * 55)
    parcels_proj = parcels_wgs84.to_crs(NJ_CRS)
    del parcels_wgs84  # free memory

    flood_connected_proj = (
        flood_connected_clip.to_crs(NJ_CRS)
        if len(flood_connected_clip) > 0 else None
    )
    flood_lowlying_proj = (
        flood_lowlying_clip.to_crs(NJ_CRS)
        if len(flood_lowlying_clip) > 0 else None
    )
    print(f"  Done.")

    # -----------------------------------------------------------------------
    # STEP 6: Clean parcels + compute market value
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 6: Cleaning parcels")
    print("=" * 55)
    parcels_proj = parcels_proj[
        parcels_proj["PROP_CLASS"].astype(str).isin(TAXABLE_CLASSES)
    ].copy()
    parcels_proj = parcels_proj[
        parcels_proj["NET_VALUE"].notna() & (parcels_proj["NET_VALUE"] > 0)
    ].copy()
    print(f"  After filters: {len(parcels_proj):,} parcels")

    parcels_proj["COUNTY_JOIN"] = parcels_proj["COUNTY"].str.strip().str.upper()
    eq_ratios["COUNTY_JOIN"]    = eq_ratios["COUNTY_NAME"].str.strip().str.upper()
    parcels_proj = parcels_proj.merge(
        eq_ratios[["COUNTY_JOIN", "EQ_RATIO"]], on="COUNTY_JOIN", how="left"
    )
    parcels_proj["EQ_RATIO"]     = parcels_proj["EQ_RATIO"].fillna(1.0)
    parcels_proj["MARKET_VALUE"] = parcels_proj["NET_VALUE"] / parcels_proj["EQ_RATIO"]
    print(f"  Ocean County total market value: ${parcels_proj['MARKET_VALUE'].sum():,.0f}")

    # -----------------------------------------------------------------------
    # STEP 7: Spatial intersection (chunked STRtree)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 7: Spatial intersection")
    print("=" * 55)

    # Extract flood geometries as plain Shapely lists (lower memory than GDF)
    connected_geoms = (
        list(flood_connected_proj.geometry)
        if flood_connected_proj is not None else []
    )
    lowlying_geoms = (
        list(flood_lowlying_proj.geometry)
        if flood_lowlying_proj is not None else []
    )
    del flood_connected_proj, flood_lowlying_proj  # free memory

    pct_connected = compute_overlap(parcels_proj, connected_geoms, "ocean-connected")
    del connected_geoms  # free memory

    pct_lowlying  = compute_overlap(parcels_proj, lowlying_geoms,  "low-lying")
    del lowlying_geoms   # free memory

    combined_weight = (
        WEIGHT_CONNECTED * pct_connected + WEIGHT_LOWLYING * pct_lowlying
    ).clip(0, 1)

    parcels_proj["VALUE_AT_RISK"] = parcels_proj["MARKET_VALUE"] * combined_weight
    parcels_proj["PCT_CONNECTED"] = pct_connected
    parcels_proj["PCT_LOWLYING"]  = pct_lowlying
    parcels_proj["FLOOD_WEIGHT"]  = combined_weight

    total_at_risk = parcels_proj["VALUE_AT_RISK"].sum()
    total_market  = parcels_proj["MARKET_VALUE"].sum()
    print(f"\n  Ocean County value at risk: ${total_at_risk:,.0f}")
    print(f"  Ocean County FRM:           {total_at_risk/total_market*100:.2f}%")

    # -----------------------------------------------------------------------
    # STEP 8: Aggregate to municipality
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 8: Aggregating to municipality")
    print("=" * 55)
    mun = (
        parcels_proj
        .groupby(["CD_CODE", "MUN_NAME", "COUNTY"], as_index=False)
        .agg(
            TOTAL_MARKET_VALUE  = ("MARKET_VALUE",  "sum"),
            TOTAL_VALUE_AT_RISK = ("VALUE_AT_RISK",  "sum"),
            N_PARCELS           = ("MARKET_VALUE",   "count"),
            N_EXPOSED_PARCELS   = ("FLOOD_WEIGHT",   lambda x: (x > 0).sum()),
        )
    )
    mun["FRM_RATIO"] = (
        mun["TOTAL_VALUE_AT_RISK"] / mun["TOTAL_MARKET_VALUE"]
    ).fillna(0).clip(0, 1)
    print(f"  Municipalities computed: {len(mun):,}")

    # -----------------------------------------------------------------------
    # STEP 9: Score 1-5
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 9: Assigning 1-5 scores")
    print("=" * 55)

    def score(frm):
        for s, (lo, hi) in SCORE_THRESHOLDS.items():
            if lo <= frm < hi:
                return s
        return 5

    mun["FLOOD_RISK_SCORE"] = mun["FRM_RATIO"].apply(score)

    labels = {
        1: "Negligible (<2%)",
        2: "Low      (2-7%)",
        3: "Moderate (7-15%)",
        4: "High     (15-30%)",
        5: "Severe   (>30%)",
    }
    dist = mun["FLOOD_RISK_SCORE"].value_counts().sort_index()
    for score_val, count in dist.items():
        print(f"  Score {score_val} [{labels[score_val]}]: {count} municipalities")

    # -----------------------------------------------------------------------
    # STEP 10: Output
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("STEP 10: Saving output")
    print("=" * 55)
    mun.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"  Saved to: {OUTPUT_CSV_PATH}")
    print("\nOcean County municipalities by flood risk (highest first):")
    print(
        mun.sort_values("FRM_RATIO", ascending=False)
        [["MUN_NAME", "FRM_RATIO", "FLOOD_RISK_SCORE", "N_PARCELS", "N_EXPOSED_PARCELS"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()