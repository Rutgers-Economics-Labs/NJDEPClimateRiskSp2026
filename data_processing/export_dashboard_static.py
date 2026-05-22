import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dashboard.backend import main  # noqa: E402


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dashboard", "frontend", "public", "data")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_json(filename, payload):
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, ensure_ascii=False, separators=(",", ":"))
    return output_path


def get_level_names(level):
    if level == "municipality":
        if main.lookup_df.empty:
            return []
        return (
            main.lookup_df[main.lookup_df["level"] == "municipality"]["name"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
    if level == "county":
        if main.lookup_df.empty:
            return []
        return (
            main.lookup_df[main.lookup_df["level"] == "county"]["name"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
    return ["NEW JERSEY"]


def export_summaries():
    payload = {}
    for level in ["municipality", "county", "state"]:
        payload[level] = {}
        for view in ["weekly", "monthly"]:
            payload[level][view] = {}
            for name in get_level_names(level):
                summary = main.build_summary(level, None if level == "state" else name, view)
                payload[level][view][name] = summary
    return write_json("summaries.json", payload)


def export_timeseries():
    output_paths = []
    for level in ["municipality", "county", "state"]:
        for view in ["weekly", "monthly"]:
            payload = {}
            for name in get_level_names(level):
                series = main.get_timeseries(
                    level=level,
                    name=None if level == "state" else name,
                    metric="spread_bps",
                    view=view,
                )
                payload[name] = series
            output_paths.append(write_json(f"timeseries_{level}_{view}.json", payload))
    return output_paths


def main_export():
    main.ensure_loaded()
    main.ensure_map_loaded()
    ensure_output_dir()

    outputs = [
        write_json("options.json", main.get_options()),
        write_json("map.json", main.get_map()),
        export_summaries(),
        *export_timeseries(),
    ]

    nojekyll_path = os.path.join(PROJECT_ROOT, "dashboard", "frontend", "public", ".nojekyll")
    with open(nojekyll_path, "w", encoding="utf-8") as outfile:
        outfile.write("")
    outputs.append(nojekyll_path)

    print("Exported static dashboard data:")
    for output in outputs:
        print(f"  - {os.path.relpath(output, PROJECT_ROOT)}")


if __name__ == "__main__":
    main_export()
