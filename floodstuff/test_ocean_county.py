import os
import gc
import pyogrio
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree
from tqdm import tqdm
import concurrent.futures
from functools import partial
import warnings

warnings.filterwarnings("ignore")
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PARCEL_GDB_PATH   = "Statewide_Parcels_MODIV.gdb"
PARCEL_LAYER      = "Cad_parcel_mod4"
EQ_RATIO_CSV_PATH = "nj_county_avg_ratio.csv"
OUTPUT_CSV_PATH   = "nj_statewide_frm_scores.csv"

# Flood files - update paths as needed
FLOOD_PATHS = {
    2: "nj_slr_2ft_merged.geojson",
    5: "nj_slr_5ft_merged.geojson",
    7: "nj_slr_7ft_merged.geojson"
}

NJ_CRS = "EPSG:3424"
TAXABLE_CLASSES = ["1", "2", "3A", "3B", "4A", "4B", "4C"]

NJ_COUNTIES = [
    "ATLANTIC", "BERGEN", "BURLINGTON", "CAMDEN", "CAPE MAY", 
    "CUMBERLAND", "ESSEX", "GLOUCESTER", "HUDSON", "HUNTERDON", 
    "MERCER", "MIDDLESEX", "MONMOUTH", "MORRIS", "OCEAN", 
    "PASSAIC", "SALEM", "SOMERSET", "SUSSEX", "UNION", "WARREN"
]

# ---------------------------------------------------------------------------
# MATH FUNCTIONS
# ---------------------------------------------------------------------------

def calculate_damage_fraction(pct_flooded):
    """
    Maps % area flooded to % market value lost.
    Using sqrt(x): 10% area flooded = ~31.6% value lost (retains ~68.4%).
    100% area flooded = 100% value lost.
    """
    return np.sqrt(pct_flooded)

# ---------------------------------------------------------------------------
# WORKER FUNCTIONS
# ---------------------------------------------------------------------------

def compute_chunk_overlap(parcel_geoms, parcel_areas, tree, flood_geoms):
    pct_flooded = np.zeros(len(parcel_geoms), dtype=np.float64)
    for i in range(len(parcel_geoms)):
        geom = parcel_geoms[i]
        if geom is None or geom.is_empty:
            continue
            
        candidate_idxs = tree.query(geom)
        if len(candidate_idxs) == 0:
            continue

        flooded_area = 0.0
        for idx in candidate_idxs:
            f_geom = flood_geoms[idx]
            if geom.intersects(f_geom):
                if f_geom.contains(geom):
                    flooded_area = parcel_areas[i]
                    break
                # Ensure valid geometries to prevent Shapely TopologyExceptions
                flooded_area += geom.intersection(f_geom.buffer(0)).area

        if parcel_areas[i] > 0:
            pct_flooded[i] = min(flooded_area / parcel_areas[i], 1.0)
            
    return pct_flooded

def compute_overlap_parallel(parcels_proj, flood_geoms_list):
    if not flood_geoms_list:
        return pd.Series(0.0, index=parcels_proj.index)

    tree = STRtree(flood_geoms_list)
    parcel_geoms = parcels_proj.geometry.values
    parcel_areas = parcels_proj.geometry.area.values
    
    # ---------------------------------------------------------
    # WORKSTATION OPTIMIZATION: Max cores, massive chunks
    # ---------------------------------------------------------
    # Use all cores except 2 to keep the OS responsive
    num_workers = max(1, os.cpu_count() - 2)
    
    # Massive chunks to fully utilize 64GB RAM and reduce IPC overhead
    chunk_size = int(np.ceil(len(parcel_geoms) / num_workers))
    
    chunks = [(parcel_geoms[i:i+chunk_size], parcel_areas[i:i+chunk_size]) 
              for i in range(0, len(parcel_geoms), chunk_size)]
              
    worker_func = partial(compute_chunk_overlap, tree=tree, flood_geoms=flood_geoms_list)
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {executor.submit(worker_func, g, a): i for i, (g, a) in enumerate(chunks)}
        for future in concurrent.futures.as_completed(future_to_chunk):
            results.append((future_to_chunk[future], future.result()))

    results.sort(key=lambda x: x[0])
    return pd.Series(np.concatenate([r[1] for r in results]), index=parcels_proj.index)


# ---------------------------------------------------------------------------
# MAIN STATEWIDE PIPELINE
# ---------------------------------------------------------------------------

def main():
    print("Loading Equalization Ratios...")
    eq_ratios = pd.read_csv(EQ_RATIO_CSV_PATH)
    eq_ratios.columns = ["COUNTY_NAME", "EQ_RATIO"]
    eq_ratios["EQ_RATIO"] = eq_ratios["EQ_RATIO"] / 100
    eq_ratios["COUNTY_JOIN"] = eq_ratios["COUNTY_NAME"].str.replace(" COUNTY", "").str.strip().str.upper()

    print("Loading Global Flood Layers (WGS84)...")
    flood_layers = {}
    for depth, path in FLOOD_PATHS.items():
        if os.path.exists(path):
            print(f"  Loading {depth}ft SLR...")
            fl_gdf = gpd.read_file(path)
            fl_gdf.geometry = fl_gdf.geometry.buffer(0) 
            flood_layers[depth] = fl_gdf
        else:
            print(f"  WARNING: {path} not found. Skipping {depth}ft.")

    statewide_municipal_results = []

    for county in NJ_COUNTIES:
        print(f"\n{'='*50}\nProcessing County: {county}\n{'='*50}")
        
        try:
            parcels = pyogrio.read_dataframe(PARCEL_GDB_PATH, layer=PARCEL_LAYER, where=f"COUNTY = '{county}'")
        except Exception as e:
            print(f"  Could not load parcels for {county}. Skipping.")
            continue
            
        if len(parcels) == 0:
            print(f"  No parcels found for {county}. Skipping.")
            continue

        parcels = parcels[parcels["PROP_CLASS"].astype(str).isin(TAXABLE_CLASSES)]
        parcels = parcels[parcels["NET_VALUE"].notna() & (parcels["NET_VALUE"] > 0)].copy()
        
        parcels["COUNTY_JOIN"] = parcels["COUNTY"].str.strip().str.upper()
        parcels = parcels.merge(eq_ratios[["COUNTY_JOIN", "EQ_RATIO"]], on="COUNTY_JOIN", how="left")
        parcels["EQ_RATIO"] = parcels["EQ_RATIO"].fillna(1.0)
        parcels["MARKET_VALUE"] = parcels["NET_VALUE"] / parcels["EQ_RATIO"]

        parcels_wgs84 = parcels.to_crs("EPSG:4326")
        county_bbox = box(*parcels_wgs84.total_bounds)
        parcels_proj = parcels_wgs84.to_crs(NJ_CRS)
        del parcels_wgs84
        gc.collect()

        for depth, flood_gdf in flood_layers.items():
            print(f"  Intersecting {depth}ft SLR...")
            
            flood_clip = flood_gdf[flood_gdf.geometry.intersects(county_bbox)].copy()
            if len(flood_clip) == 0:
                parcels_proj[f"PCT_{depth}FT"] = 0.0
                continue
                
            flood_proj = flood_clip.to_crs(NJ_CRS)
            flood_geoms_list = list(flood_proj.geometry)
            
            parcels_proj[f"PCT_{depth}FT"] = compute_overlap_parallel(parcels_proj, flood_geoms_list)
            
            del flood_clip, flood_proj, flood_geoms_list
            gc.collect()

        if "PCT_2FT" in parcels_proj and "PCT_5FT" in parcels_proj:
            print("  Interpolating 4ft SLR...")
            pct_4ft = parcels_proj["PCT_2FT"] + (parcels_proj["PCT_5FT"] - parcels_proj["PCT_2FT"]) * (2.0 / 3.0)
            parcels_proj["PCT_4FT"] = pct_4ft.clip(0, 1)

        depths_to_calc = [d for d in [2, 4, 5, 7] if f"PCT_{d}FT" in parcels_proj]
        
        for d in depths_to_calc:
            col_pct = f"PCT_{d}FT"
            col_risk = f"RISK_VALUE_{d}FT"
            
            damage_fraction = calculate_damage_fraction(parcels_proj[col_pct])
            parcels_proj[col_risk] = parcels_proj["MARKET_VALUE"] * damage_fraction

        print("  Aggregating to municipality...")
        agg_dict = {
            "N_PARCELS": ("MARKET_VALUE", "count"),
            "TOTAL_MARKET_VALUE": ("MARKET_VALUE", "sum")
        }
        
        for d in depths_to_calc:
            agg_dict[f"VALUE_AT_RISK_{d}FT"] = (f"RISK_VALUE_{d}FT", "sum")
            agg_dict[f"EXPOSED_PARCELS_{d}FT"] = (f"PCT_{d}FT", lambda x: (x > 0).sum())

        mun = parcels_proj.groupby(["CD_CODE", "MUN_NAME", "COUNTY"], as_index=False).agg(**agg_dict)
        
        for d in depths_to_calc:
            mun[f"FRM_RATIO_{d}FT"] = (mun[f"VALUE_AT_RISK_{d}FT"] / mun["TOTAL_MARKET_VALUE"]).fillna(0).clip(0, 1)

        statewide_municipal_results.append(mun)

        del parcels, parcels_proj, mun
        gc.collect()

    print(f"\n{'='*50}\nMerging and Saving Statewide Data\n{'='*50}")
    statewide_df = pd.concat(statewide_municipal_results, ignore_index=True)
    statewide_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Success! Data saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()