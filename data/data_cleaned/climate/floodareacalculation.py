import os

# Remove GeoJSON object size limits so very large flood features can be read.
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

import geopandas as gpd
import pandas as pd

# Path to the municipality boundary layer.
MUNICIPALITIES_FILE = "NJ_Municipal_Boundaries_3424_6549928991592885940.geojson"
# Output CSV that will store the flood-area summary by municipality.
OUTPUT_FILE = "nj_municipality_flood_area_summary.csv"
# Keep only the identifying columns needed in the final summary plus geometry.
AREA_COLUMNS = ["MUN", "MUN_LABEL", "COUNTY", "geometry"]

# Map each sea-level-rise scenario label to its source GeoJSON file.
SCENARIOS = {
    "2ft": "nj_slr_2ft_merged.geojson",
    "5ft": "nj_slr_5ft_merged.geojson",
    "7ft": "nj_slr_7ft_merged.geojson",
}

# Conversion factor from square meters to square miles.
SQ_METERS_PER_SQ_MILE = 2_589_988.110336
# Conversion factor from square meters to acres.
SQ_METERS_PER_ACRE = 4_046.8564224


def load_municipalities():
    # Show progress because spatial reads can take a little time.
    print("Loading municipalities...")
    # Read the municipal boundaries and keep only the columns needed for analysis/output.
    municipalities = gpd.read_file(MUNICIPALITIES_FILE)[AREA_COLUMNS].copy()
    # Stop early if the source data has no coordinate reference system.
    if municipalities.crs is None:
        raise ValueError("Municipality layer is missing a CRS.")
    # Pick a local projected CRS so area is measured in meters instead of degrees.
    projected_crs = municipalities.estimate_utm_crs()
    # Reproject the municipality polygons into that projected CRS.
    municipalities = municipalities.to_crs(projected_crs)
    # Repair invalid polygon geometry before intersection/area operations.
    municipalities["geometry"] = municipalities.geometry.make_valid()
    # Compute each municipality's total area in square meters.
    municipalities["municipality_area_sq_m"] = municipalities.geometry.area
    # Return both the prepared GeoDataFrame and the projected CRS for flood layers.
    return municipalities, projected_crs


def load_flood_union(path, projected_crs):
    # Show which flood scenario file is being processed.
    print(f"Loading flood layer: {path}")
    # Read only geometry from the flood layer and project it to the same CRS as municipalities.
    flood_layer = gpd.read_file(path, columns=[]).to_crs(projected_crs)
    # Repair invalid flood polygons before merging or intersecting them.
    flood_layer["geometry"] = flood_layer.geometry.make_valid()
    # Drop any null or empty geometries that would interfere with later operations.
    flood_layer = flood_layer[flood_layer.geometry.notna() & ~flood_layer.geometry.is_empty].copy()
    # Show progress because merging many polygons can take time.
    print(f"Merging flood geometries: {path}")
    # Merge all flood polygons for this scenario into one combined geometry.
    return flood_layer.geometry.union_all()


def add_scenario_area(municipalities, scenario_name, flood_union):
    # Show which scenario's intersections are being calculated.
    print(f"Calculating municipality intersections for {scenario_name}...")
    # Intersect each municipality polygon with the full flood geometry for this scenario.
    flooded_geometries = municipalities.geometry.intersection(flood_union)
    # Measure the flooded portion of each municipality in square meters.
    flooded_area_sq_m = flooded_geometries.area.fillna(0)
    # Store flooded area in square meters for this scenario.
    municipalities[f"flooded_area_{scenario_name}_sq_m"] = flooded_area_sq_m
    # Convert flooded area to square miles for easier reporting.
    municipalities[f"flooded_area_{scenario_name}_sq_mi"] = flooded_area_sq_m / SQ_METERS_PER_SQ_MILE
    # Convert flooded area to acres as an additional unit.
    municipalities[f"flooded_area_{scenario_name}_acres"] = flooded_area_sq_m / SQ_METERS_PER_ACRE
    # Calculate what percent of the municipality's total area is flooded.
    municipalities[f"flooded_pct_{scenario_name}"] = (
        flooded_area_sq_m / municipalities["municipality_area_sq_m"] * 100
    ).fillna(0)


def main():
    # Load municipality boundaries and get the projected CRS used for area calculations.
    municipalities, projected_crs = load_municipalities()

    # Convert each municipality's total area from square meters to square miles.
    municipalities["municipality_area_sq_mi"] = (
        municipalities["municipality_area_sq_m"] / SQ_METERS_PER_SQ_MILE
    )
    # Convert each municipality's total area from square meters to acres.
    municipalities["municipality_area_acres"] = (
        municipalities["municipality_area_sq_m"] / SQ_METERS_PER_ACRE
    )

    # Process each flood scenario one at a time.
    for scenario_name, path in SCENARIOS.items():
        # Print the scenario label so the progress output is easy to follow.
        print(f"Starting scenario: {scenario_name}")
        # Load and merge all flood polygons for the current scenario.
        flood_union = load_flood_union(path, projected_crs)
        # Calculate flooded area columns for the current scenario.
        add_scenario_area(municipalities, scenario_name, flood_union)
        # Mark the end of this scenario in the progress output.
        print(f"Finished scenario: {scenario_name}")

    # Start the list of columns that will be written to the final CSV.
    output_columns = [
        "MUN",
        "MUN_LABEL",
        "COUNTY",
        "municipality_area_sq_mi",
        "municipality_area_acres",
    ]

    # Add the output columns for each flood scenario in a consistent order.
    for scenario_name in SCENARIOS:
        output_columns.extend(
            [
                f"flooded_area_{scenario_name}_sq_mi",
                f"flooded_area_{scenario_name}_acres",
                f"flooded_pct_{scenario_name}",
            ]
        )

    # Build a clean tabular summary sorted by county and municipality name.
    summary = municipalities[output_columns].sort_values(["COUNTY", "MUN_LABEL"]).reset_index(drop=True)
    # Write the municipality summary table to CSV.
    summary.to_csv(OUTPUT_FILE, index=False)

    # Build a short statewide totals summary for the terminal output.
    statewide = {
        "municipality_area_sq_mi": summary["municipality_area_sq_mi"].sum(),
        "municipality_area_acres": summary["municipality_area_acres"].sum(),
    }
    # Add flooded-area totals for each scenario to the statewide summary.
    for scenario_name in SCENARIOS:
        statewide[f"flooded_area_{scenario_name}_sq_mi"] = summary[f"flooded_area_{scenario_name}_sq_mi"].sum()
        statewide[f"flooded_area_{scenario_name}_acres"] = summary[f"flooded_area_{scenario_name}_acres"].sum()

    # Confirm where the CSV was written.
    print(f"Saved {OUTPUT_FILE}")
    # Print statewide totals rounded for readability.
    print(pd.Series(statewide).round(2).to_string())


# Run the script only when this file is executed directly.
if __name__ == "__main__":
    # Enter the main workflow.
    main()
