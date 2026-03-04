import os

os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

import folium
import geopandas as gpd
import matplotlib.pyplot as plt

MUNICIPALITIES_FILE = "NJ_Municipal_Boundaries_3424_6549928991592885940.geojson"
OUTPUT_FILE = "nj_slr_interactive_map.html"
SIMPLIFY_PROJECTION = 3857
MUNICIPALITY_SIMPLIFY_METERS = 60
FLOOD_SIMPLIFY_METERS = 250
BOUNDARY_OVERLAY_FILE = "nj_municipal_boundaries_overlay.png"

layers = {
    "2ft": ("nj_slr_2ft_merged.geojson", "#f4d35e", "nj_slr_2ft_overlay.png"),
    "5ft": ("nj_slr_5ft_merged.geojson", "#ee964b", "nj_slr_5ft_overlay.png"),
    "7ft": ("nj_slr_7ft_merged.geojson", "#d1495b", "nj_slr_7ft_overlay.png"),
}


def load_geometry_only(path):
    geodataframe = gpd.read_file(path, columns=[])
    if geodataframe.crs is None:
        raise ValueError(f"{path} is missing a CRS.")
    return geodataframe[["geometry"]].copy()


def simplify_geometries(geodataframe, tolerance_meters, merge=False):
    working = geodataframe.to_crs(epsg=SIMPLIFY_PROJECTION)
    working["geometry"] = working.geometry.simplify(
        tolerance=tolerance_meters,
        preserve_topology=True,
    )
    working = working[working.geometry.notna() & ~working.geometry.is_empty].copy()

    if merge:
        working = gpd.GeoDataFrame(
            geometry=[working.geometry.union_all()],
            crs=working.crs,
        )

    return working.to_crs(epsg=4326)


def save_overlay_image(geodataframe, output_path, bounds, color, alpha, boundaries_only=False):
    minx, miny, maxx, maxy = bounds
    width = 10
    height = width * ((maxy - miny) / (maxx - minx))
    fig, ax = plt.subplots(figsize=(width, height))

    fig.patch.set_alpha(0)
    ax.set_facecolor((1, 1, 1, 0))
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()

    if boundaries_only:
        geodataframe.boundary.plot(ax=ax, color=color, linewidth=0.4)
    else:
        geodataframe.plot(ax=ax, color=color, alpha=alpha, edgecolor="none")

    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


municipalities = simplify_geometries(
    load_geometry_only(MUNICIPALITIES_FILE),
    tolerance_meters=MUNICIPALITY_SIMPLIFY_METERS,
)
bounds = municipalities.total_bounds
minx, miny, maxx, maxy = bounds

save_overlay_image(
    municipalities,
    BOUNDARY_OVERLAY_FILE,
    bounds=bounds,
    color="black",
    alpha=1.0,
    boundaries_only=True,
)

map_object = folium.Map(
    location=[(miny + maxy) / 2, (minx + maxx) / 2],
    zoom_start=8,
    tiles="CartoDB positron",
)
map_object.fit_bounds([[miny, minx], [maxy, maxx]])

folium.raster_layers.ImageOverlay(
    image=BOUNDARY_OVERLAY_FILE,
    bounds=[[miny, minx], [maxy, maxx]],
    name="Municipal boundaries",
    opacity=1.0,
    interactive=False,
    zindex=1,
).add_to(map_object)

for name, (filename, color, overlay_file) in layers.items():
    flood_layer = simplify_geometries(
        load_geometry_only(filename),
        tolerance_meters=FLOOD_SIMPLIFY_METERS,
        merge=True,
    )
    save_overlay_image(
        flood_layer,
        overlay_file,
        bounds=bounds,
        color=color,
        alpha=0.55,
    )

    folium.raster_layers.ImageOverlay(
        image=overlay_file,
        bounds=[[miny, minx], [maxy, maxx]],
        name=f"{name} flood extent",
        opacity=0.75,
        interactive=True,
        zindex=2,
        show=name == "2ft",
    ).add_to(map_object)

folium.LayerControl(collapsed=False).add_to(map_object)

title_html = """
<div style="
    position: fixed;
    top: 12px;
    left: 50px;
    z-index: 9999;
    background: rgba(255, 255, 255, 0.92);
    padding: 10px 14px;
    border: 1px solid #444;
    border-radius: 6px;
    font-family: Arial, sans-serif;
    font-size: 16px;
    font-weight: bold;
">
    New Jersey Municipal Boundaries and Sea Level Rise
</div>
"""

map_object.get_root().html.add_child(folium.Element(title_html))
map_object.save(OUTPUT_FILE)

print(f"Saved {OUTPUT_FILE}")
