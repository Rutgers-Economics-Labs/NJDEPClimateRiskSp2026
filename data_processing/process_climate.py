"""
process_climate.py
------------------
Locates NOAA Sea Level Rise GeoDatabase files across the project,
extracts the slr_2ft, slr_5ft, and slr_7ft layers from each region
(Northern, Middle, Southern NJ), merges them, and exports clean
GeoJSON files to data_cleaned/climate/.
"""

import os
import sys
import glob
import geopandas as gpd
import fiona

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "climate")
SLR_LEVELS = [2, 5, 7]  # feet of sea-level rise to extract
REGIONS = ["Northern", "Middle", "Southern"]


def find_gdb_files():
    """Search the data_raw directory for NOAA SLR .gdb directories."""
    pattern = os.path.join(DATA_RAW_DIR, "**", "*.gdb")
    gdbs = glob.glob(pattern, recursive=True)
    # Filter to only SLR-related GDBs
    slr_gdbs = [g for g in gdbs if "slr" in os.path.basename(g).lower()]
    return slr_gdbs


def identify_region(gdb_path):
    """Determine which NJ region a GDB belongs to."""
    basename = os.path.basename(gdb_path).lower()
    for region in REGIONS:
        if region.lower() in basename:
            return region
    return None


def extract_slr_layer(gdb_path, region, level):
    """
    Read a specific SLR layer from a GDB.
    Layer naming convention: NJ_{Region}_slr_{N}ft
    """
    target_layer = f"NJ_{region}_slr_{level}ft"
    available_layers = fiona.listlayers(gdb_path)

    if target_layer in available_layers:
        print(f"  Reading layer: {target_layer}")
        gdf = gpd.read_file(gdb_path, layer=target_layer)
        gdf["source_region"] = region
        gdf["slr_feet"] = level
        return gdf
    else:
        # Try partial match
        matches = [l for l in available_layers if f"slr_{level}ft" in l.lower()]
        if matches:
            print(f"  Reading layer: {matches[0]} (partial match)")
            gdf = gpd.read_file(gdb_path, layer=matches[0])
            gdf["source_region"] = region
            gdf["slr_feet"] = level
            return gdf
        else:
            print(f"  WARNING: No layer found for slr_{level}ft in {os.path.basename(gdb_path)}")
            print(f"  Available layers: {available_layers}")
            return None


def process_climate():
    """Main processing pipeline for climate/SLR data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("CLIMATE / SLR DATA PROCESSING")
    print("=" * 60)

    # Find all GDB files
    gdb_files = find_gdb_files()
    if not gdb_files:
        print("ERROR: No SLR .gdb files found in project tree!")
        print(f"Searched under: {DATA_RAW_DIR}")
        return []

    print(f"\nFound {len(gdb_files)} SLR GeoDatabase(s):")
    for g in gdb_files:
        print(f"  • {os.path.relpath(g, PROJECT_ROOT)}")

    # Map GDBs to their regions
    region_gdbs = {}
    for gdb_path in gdb_files:
        region = identify_region(gdb_path)
        if region:
            region_gdbs[region] = gdb_path

    print(f"\nRegion mapping: {list(region_gdbs.keys())}")

    # Process each SLR level
    output_files = []
    for level in SLR_LEVELS:
        print(f"\n--- Processing SLR {level}ft ---")
        frames = []

        for region in REGIONS:
            if region not in region_gdbs:
                print(f"  WARNING: No GDB found for {region} region")
                continue

            gdf = extract_slr_layer(region_gdbs[region], region, level)
            if gdf is not None:
                frames.append(gdf)

        if frames:
            # Merge all regions for this SLR level
            merged = gpd.pd.concat(frames, ignore_index=True)

            # Ensure consistent CRS (WGS84)
            if merged.crs and merged.crs.to_epsg() != 4326:
                print(f"  Reprojecting from {merged.crs} to EPSG:4326")
                merged = merged.to_crs(epsg=4326)

            # Export
            out_path = os.path.join(OUTPUT_DIR, f"nj_slr_{level}ft_merged.geojson")
            merged.to_file(out_path, driver="GeoJSON")
            print(f"  ✓ Exported: {os.path.relpath(out_path, PROJECT_ROOT)}")
            print(f"    Features: {len(merged)}, CRS: {merged.crs}")
            output_files.append(out_path)
        else:
            print(f"  ERROR: No data found for SLR {level}ft across any region")

    return output_files


if __name__ == "__main__":
    output_files = process_climate()
    print(f"\n{'=' * 60}")
    print(f"Climate processing complete. {len(output_files)} file(s) created.")
