"""
process_boundaries.py
---------------------
Locates the NJ Municipal Boundaries shapefile in the project,
reprojects to WGS84 (EPSG:4326) for compatibility with SLR data,
and exports as GeoJSON to data_cleaned/boundaries/.
"""

import os
import glob
import geopandas as gpd

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "boundaries")


def find_boundary_shapefile():
    """Search the data_raw directory for NJ Municipal Boundaries shapefile."""
    pattern = os.path.join(DATA_RAW_DIR, "**", "NJ_Municipal_Boundaries*.shp")
    files = glob.glob(pattern, recursive=True)
    # Exclude .shp.xml files
    shp_files = [f for f in files if f.endswith(".shp")]
    return shp_files


def process_boundaries():
    """Main processing pipeline for municipal boundaries."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("MUNICIPAL BOUNDARIES PROCESSING")
    print("=" * 60)

    shp_files = find_boundary_shapefile()
    if not shp_files:
        print("ERROR: No NJ Municipal Boundaries shapefile found!")
        print(f"Searched under: {PROJECT_ROOT}")
        return []

    shp_path = shp_files[0]
    print(f"\nUsing shapefile: {os.path.relpath(shp_path, PROJECT_ROOT)}")

    # Read shapefile
    gdf = gpd.read_file(shp_path)
    print(f"  Original CRS: {gdf.crs}")
    print(f"  Features: {len(gdf)}")
    print(f"  Columns: {list(gdf.columns)}")

    # Reproject to WGS84 for compatibility with NOAA SLR data
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        print(f"  Reprojecting to EPSG:4326 (WGS84)...")
        gdf = gdf.to_crs(epsg=4326)

    # Standardize municipality name column if present
    name_cols = [c for c in gdf.columns if "name" in c.lower() or "mun" in c.lower()]
    if name_cols:
        print(f"  Name column(s) found: {name_cols}")
        primary_name_col = name_cols[0]
        gdf["municipality"] = gdf[primary_name_col].str.strip().str.upper()
    else:
        print("  WARNING: No name column found in boundaries shapefile")

    # Export as GeoJSON
    out_path = os.path.join(OUTPUT_DIR, "nj_municipal_boundaries.geojson")
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"\n  ✓ Exported: {os.path.relpath(out_path, PROJECT_ROOT)}")
    print(f"    Features: {len(gdf)}, CRS: {gdf.crs}")

    # Also export a simple CSV with municipality names (for reference)
    if "municipality" in gdf.columns:
        csv_path = os.path.join(OUTPUT_DIR, "nj_municipality_list.csv")
        name_df = gdf.drop(columns=["geometry"])
        name_df.to_csv(csv_path, index=False)
        print(f"  ✓ Municipality list: {os.path.relpath(csv_path, PROJECT_ROOT)}")

    return [out_path]


if __name__ == "__main__":
    output_files = process_boundaries()
    print(f"\n{'=' * 60}")
    print(f"Boundaries processing complete. {len(output_files)} file(s) created.")
