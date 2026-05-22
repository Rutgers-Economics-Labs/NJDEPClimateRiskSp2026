import os
import json
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned")
MUNI_TS_FILE = os.path.join(DATA_CLEANED_DIR, "premium_timeseries_muni.csv")
COUNTY_TS_FILE = os.path.join(DATA_CLEANED_DIR, "premium_timeseries_county.csv")
STATE_TS_FILE = os.path.join(DATA_CLEANED_DIR, "premium_timeseries_state.csv")
LOOKUP_FILE = os.path.join(DATA_CLEANED_DIR, "premium_lookup.csv")
MUNI_CHARS_FILE = os.path.join(DATA_CLEANED_DIR, "nj_municipality_characteristics.csv")
BOUNDARIES_FILE = os.path.join(DATA_CLEANED_DIR, "boundaries", "nj_municipal_boundaries.geojson")

muni_ts = pd.DataFrame()
county_ts = pd.DataFrame()
state_ts = pd.DataFrame()
lookup_df = pd.DataFrame()
muni_chars = pd.DataFrame()
boundary_geojson = {}


def load_csv(filepath, parse_dates=None):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    return pd.read_csv(filepath, parse_dates=parse_dates, low_memory=False)


def aggregate_timeseries_frame(analytics_df, level, geography_columns):
    frames = []
    views = {
        "weekly": "date_bucket_weekly",
        "monthly": "date_bucket_monthly",
    }

    for view_name, bucket_column in views.items():
        grouped = (
            analytics_df.groupby([bucket_column, *geography_columns], dropna=False)
            .agg(
                trade_count=("cusip", "size"),
                cusip_count=("cusip", "nunique"),
                avg_yield=("yield", "mean"),
                avg_synthetic_aaa=("avg_synthetic_aaa", "mean"),
                avg_spread_bps=("spread_bps", "mean"),
            )
            .reset_index()
            .rename(columns={bucket_column: "date_bucket"})
        )
        grouped["level"] = level
        grouped["view"] = view_name
        grouped = grouped.sort_values([*geography_columns, "date_bucket"])
        if geography_columns:
            grouped["spread_bps_rolling_4w"] = (
                grouped.groupby(geography_columns)["avg_spread_bps"]
                .transform(lambda values: values.rolling(window=4, min_periods=1).mean())
            )
        else:
            grouped["spread_bps_rolling_4w"] = grouped["avg_spread_bps"].rolling(window=4, min_periods=1).mean()
        frames.append(grouped)

    return pd.concat(frames, ignore_index=True)


def build_lookup_frame(muni_frame, county_frame):
    municipalities = (
        muni_frame[muni_frame["view"] == "weekly"][["mun", "mun_label", "county"]]
        .dropna(subset=["mun"])
        .drop_duplicates()
        .sort_values(["mun_label", "mun"])
        .reset_index(drop=True)
    )
    municipalities["level"] = "municipality"
    municipalities["name"] = municipalities["mun"]
    municipalities["label"] = municipalities["mun_label"].fillna(municipalities["mun"])

    counties = (
        county_frame[county_frame["view"] == "weekly"][["county"]]
        .dropna(subset=["county"])
        .drop_duplicates()
        .sort_values("county")
        .reset_index(drop=True)
    )
    counties["level"] = "county"
    counties["name"] = counties["county"]
    counties["label"] = counties["county"]
    counties["mun"] = None
    counties["mun_label"] = None

    state = pd.DataFrame(
        [{"level": "state", "name": "NEW JERSEY", "label": "New Jersey", "county": None, "mun": None, "mun_label": None}]
    )

    columns = ["level", "name", "label", "county", "mun", "mun_label"]
    return pd.concat([state[columns], counties[columns], municipalities[columns]], ignore_index=True)


def build_dashboard_frames_from_full_trades(full_trades_df, muni_chars_df):
    chars_subset = muni_chars_df.copy()
    chars_subset["mun"] = chars_subset["mun"].astype(str).str.upper().str.strip()
    chars_subset = chars_subset.dropna(subset=["mun"]).drop_duplicates(subset=["mun"]).copy()

    trades = full_trades_df.copy()
    trades.columns = [column.lower() for column in trades.columns]
    trades["trade_date"] = pd.to_datetime(trades["trade_date_dt"], errors="coerce")
    trades["yield"] = pd.to_numeric(trades["yield"], errors="coerce")
    trades["avg_synthetic_aaa"] = pd.to_numeric(trades["benchmark_aaa_yield"], errors="coerce")
    trades["spread_bps"] = pd.to_numeric(trades["spread_bps"], errors="coerce")
    if "mun" in trades.columns:
        trades["mun"] = trades["mun"].astype(str).str.upper().str.strip()
    else:
        trades["mun"] = None
    if "mun_label" in trades.columns:
        trades["mun_label"] = trades["mun_label"].astype(str).str.strip()
    else:
        trades["mun_label"] = None
    if "county" in trades.columns:
        trades["county"] = trades["county"].astype(str).str.upper().str.strip()
    else:
        trades["county"] = None

    trades = trades.dropna(subset=["trade_date", "mun", "yield", "avg_synthetic_aaa", "spread_bps"]).copy()
    trades = trades[trades["mun"] != ""].copy()

    chars_join = chars_subset[
        [
            "mun",
            "mun_label",
            "county",
            "is_resilient",
            "flooded_pct_2ft",
            "flooded_pct_5ft",
            "flooded_pct_7ft",
            "crs_class",
            "crs_discount_pct",
            "median_household_income",
        ]
    ].drop_duplicates(subset=["mun"])

    trades = trades.merge(chars_join, on="mun", how="left", suffixes=("", "_chars"))
    trades["mun_label"] = trades["mun_label"].where(trades["mun_label"].notna(), trades["mun_label_chars"])
    trades["county"] = trades["county"].where(trades["county"].notna(), trades["county_chars"])
    for column in [
        "is_resilient",
        "flooded_pct_2ft",
        "flooded_pct_5ft",
        "flooded_pct_7ft",
        "crs_class",
        "crs_discount_pct",
        "median_household_income",
    ]:
        chars_column = f"{column}_chars"
        if chars_column in trades.columns:
            trades[column] = trades[column].where(trades[column].notna(), trades[chars_column])

    drop_columns = [column for column in trades.columns if column.endswith("_chars")]
    if drop_columns:
        trades = trades.drop(columns=drop_columns)
    trades["date_bucket_weekly"] = trades["trade_date"].dt.to_period("W-SUN").dt.start_time
    trades["date_bucket_monthly"] = trades["trade_date"].dt.to_period("M").dt.start_time

    muni_frame = aggregate_timeseries_frame(
        trades.dropna(subset=["mun", "county"]).copy(),
        "municipality",
        ["mun", "mun_label", "county"],
    )
    county_frame = aggregate_timeseries_frame(
        trades.dropna(subset=["county"]).copy(),
        "county",
        ["county"],
    )
    state_base = trades.copy()
    state_base["name"] = "NEW JERSEY"
    state_frame = aggregate_timeseries_frame(state_base, "state", ["name"])
    lookup_frame = build_lookup_frame(muni_frame, county_frame)

    return muni_frame, county_frame, state_frame, lookup_frame


def load_data():
    global muni_ts, county_ts, state_ts, lookup_df, muni_chars, boundary_geojson

    muni_ts = load_csv(MUNI_TS_FILE, parse_dates=["date_bucket"])
    county_ts = load_csv(COUNTY_TS_FILE, parse_dates=["date_bucket"])
    state_ts = load_csv(STATE_TS_FILE, parse_dates=["date_bucket"])
    lookup_df = load_csv(LOOKUP_FILE)
    muni_chars = load_csv(MUNI_CHARS_FILE)

    if not muni_chars.empty:
        muni_chars.columns = [column.lower() for column in muni_chars.columns]
        muni_chars = muni_chars.loc[:, ~muni_chars.columns.duplicated()].copy()
        if "is_resilient" in muni_chars.columns:
            muni_chars["is_resilient"] = muni_chars["is_resilient"].fillna(False).astype(bool)

    if os.path.exists(BOUNDARIES_FILE):
        with open(BOUNDARIES_FILE, "r", encoding="utf-8") as infile:
            boundary_geojson = json.load(infile)
    else:
        boundary_geojson = {}


def ensure_loaded():
    if muni_ts.empty and county_ts.empty and state_ts.empty:
        raise HTTPException(
            status_code=503,
            detail="Dashboard data not found. Please run 'make process-all'.",
        )


def ensure_map_loaded():
    ensure_loaded()
    if not boundary_geojson:
        raise HTTPException(status_code=503, detail="Boundary geometry not found.")


def get_dataset(level, view):
    datasets = {
        "municipality": muni_ts,
        "county": county_ts,
        "state": state_ts,
    }
    if level not in datasets:
        raise HTTPException(status_code=400, detail="Unsupported level.")

    dataset = datasets[level]
    if dataset.empty:
        raise HTTPException(status_code=404, detail=f"No dataset available for level '{level}'.")

    filtered = dataset[dataset["view"] == view].copy()
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"No '{view}' data available for level '{level}'.")

    return filtered


def get_name_column(level):
    return {
        "municipality": "mun",
        "county": "county",
        "state": "name",
    }[level]


def build_map_geojson():
    ensure_map_loaded()

    premium_lookup = {}
    if not muni_ts.empty:
        muni_weekly = muni_ts[muni_ts["view"] == "weekly"].copy()
        if not muni_weekly.empty:
            premium_lookup = (
                muni_weekly.groupby("mun", dropna=False)["avg_spread_bps"]
                .mean()
                .round(2)
                .to_dict()
            )

    chars_lookup = {}
    if not muni_chars.empty:
        chars_subset = muni_chars.dropna(subset=["mun"]).drop_duplicates(subset=["mun"]).copy()
        chars_lookup = (
            chars_subset.set_index("mun")[
                [
                    "mun_label",
                    "county",
                    "is_resilient",
                    "crs_class",
                    "crs_discount_pct",
                    "flooded_pct_2ft",
                    "flooded_pct_5ft",
                    "flooded_pct_7ft",
                ]
            ]
            .to_dict(orient="index")
        )

    features = []
    for feature in boundary_geojson.get("features", []):
        properties = feature.get("properties", {})
        mun = properties.get("MUN")
        merged = chars_lookup.get(mun, {})
        feature_properties = {
            "mun": mun,
            "mun_label": merged.get("mun_label") or properties.get("MUN_LABEL") or properties.get("NAME") or mun,
            "county": merged.get("county") or properties.get("COUNTY"),
            "is_resilient": None if pd.isna(merged.get("is_resilient")) else bool(merged.get("is_resilient")) if "is_resilient" in merged else None,
            "crs_class": coerce_int(merged.get("crs_class")),
            "crs_discount_pct": coerce_float(merged.get("crs_discount_pct")),
            "flooded_pct_2ft": coerce_float(merged.get("flooded_pct_2ft")),
            "flooded_pct_5ft": coerce_float(merged.get("flooded_pct_5ft")),
            "flooded_pct_7ft": coerce_float(merged.get("flooded_pct_7ft")),
            "avg_premium_bps": coerce_float(premium_lookup.get(mun)),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": feature.get("geometry"),
                "properties": feature_properties,
            }
        )

    return {"type": "FeatureCollection", "features": features}


def filter_dataset(dataset, level, name=None):
    if level == "state":
        return dataset.copy(), "NEW JERSEY"

    if not name:
        raise HTTPException(status_code=400, detail="A name query parameter is required for this level.")

    name_column = get_name_column(level)
    filtered = dataset[dataset[name_column] == name].copy()
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"No data found for '{name}'.")

    return filtered, name


def coerce_float(value):
    if pd.isna(value):
        return None
    return float(round(float(value), 2))


def coerce_int(value):
    if pd.isna(value):
        return None
    return int(value)


def safe_mean(series):
    if series is None:
        return None
    valid = pd.Series(series).dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def safe_median(series):
    if series is None:
        return None
    valid = pd.Series(series).dropna()
    if valid.empty:
        return None
    return float(valid.median())


def aggregate_flood_share(scope, scenario):
    area_column = "municipality_area_sq_mi"
    flooded_column = f"flooded_area_{scenario}_sq_mi"
    if area_column not in scope.columns or flooded_column not in scope.columns:
        return None

    valid = scope[[area_column, flooded_column]].dropna()
    if valid.empty:
        return None

    total_area = valid[area_column].sum()
    if not total_area:
        return None

    return float(valid[flooded_column].sum() / total_area * 100)


def get_context_metrics(level, resolved_name):
    if muni_chars.empty:
        return {}

    if level == "municipality":
        scope = muni_chars[muni_chars["mun"] == resolved_name].copy()
    elif level == "county":
        scope = muni_chars[muni_chars["county"] == resolved_name].copy()
    else:
        scope = muni_chars.copy()

    if scope.empty:
        return {}

    municipality_count = int(scope["mun"].dropna().nunique()) if "mun" in scope.columns else len(scope)
    resilient_count = int(scope["is_resilient"].fillna(False).astype(bool).sum()) if "is_resilient" in scope.columns else 0

    metrics = {
        "municipality_count": municipality_count,
        "resilient_muni_count": resilient_count,
        "resilient_share_pct": coerce_float((resilient_count / municipality_count) * 100 if municipality_count else None),
        "avg_crs_class": coerce_float(safe_mean(scope["crs_class"])) if "crs_class" in scope.columns else None,
        "avg_crs_discount_pct": coerce_float(safe_mean(scope["crs_discount_pct"])) if "crs_discount_pct" in scope.columns else None,
        "avg_flooded_pct_2ft": coerce_float(safe_mean(scope["flooded_pct_2ft"])) if "flooded_pct_2ft" in scope.columns else None,
        "avg_flooded_pct_5ft": coerce_float(safe_mean(scope["flooded_pct_5ft"])) if "flooded_pct_5ft" in scope.columns else None,
        "avg_flooded_pct_7ft": coerce_float(safe_mean(scope["flooded_pct_7ft"])) if "flooded_pct_7ft" in scope.columns else None,
        "flooded_pct_2ft": coerce_float(aggregate_flood_share(scope, "2ft")),
        "flooded_pct_5ft": coerce_float(aggregate_flood_share(scope, "5ft")),
        "flooded_pct_7ft": coerce_float(aggregate_flood_share(scope, "7ft")),
        "median_income_median": coerce_float(safe_median(scope["median_household_income"])) if "median_household_income" in scope.columns else None,
    }

    if level == "municipality":
        row = scope.iloc[0]
        metrics.update(
            {
                "label": row.get("mun_label", resolved_name),
                "county": row.get("county"),
                "is_resilient": None if pd.isna(row.get("is_resilient")) else bool(row.get("is_resilient")),
                "crs_class": coerce_int(row.get("crs_class")),
                "crs_discount_pct": coerce_float(row.get("crs_discount_pct")),
                "median_household_income": coerce_float(row.get("median_household_income")),
            }
        )
    elif level == "county":
        metrics.update({"county": resolved_name})
    else:
        metrics.update({"label": "New Jersey"})

    return metrics


def build_summary(level, name, view):
    dataset = get_dataset(level, view)
    filtered, resolved_name = filter_dataset(dataset, level, name)
    filtered = filtered.sort_values("date_bucket")
    latest = filtered.iloc[-1]

    response = {
        "level": level,
        "name": resolved_name,
        "view": view,
        "latest_spread_bps": coerce_float(latest["avg_spread_bps"]),
        "latest_spread_bps_rolling_4w": coerce_float(latest["spread_bps_rolling_4w"]),
        "avg_spread_bps": coerce_float(filtered["avg_spread_bps"].mean()),
        "peak_spread_bps": coerce_float(filtered["avg_spread_bps"].max()),
        "trade_count": coerce_int(filtered["trade_count"].sum()),
        "cusip_count": coerce_int(filtered["cusip_count"].sum()),
        "date_min": filtered["date_bucket"].min().strftime("%Y-%m-%d"),
        "date_max": filtered["date_bucket"].max().strftime("%Y-%m-%d"),
    }

    response.update(get_context_metrics(level, resolved_name))

    return response


load_data()


@app.get("/api/options")
def get_options():
    ensure_loaded()

    municipalities = []
    counties = []
    default_municipality = None
    default_county = None
    if not lookup_df.empty:
        municipalities = (
            lookup_df[lookup_df["level"] == "municipality"]["name"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        counties = (
            lookup_df[lookup_df["level"] == "county"]["name"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

    if not muni_ts.empty:
        muni_weekly = muni_ts[muni_ts["view"] == "weekly"].copy()
        if not muni_weekly.empty:
            muni_rank = (
                muni_weekly.groupby("mun", dropna=False)["trade_count"]
                .sum()
                .sort_values(ascending=False)
            )
            if not muni_rank.empty:
                default_municipality = muni_rank.index[0]

    if not county_ts.empty:
        county_weekly = county_ts[county_ts["view"] == "weekly"].copy()
        if not county_weekly.empty:
            county_rank = (
                county_weekly.groupby("county", dropna=False)["trade_count"]
                .sum()
                .sort_values(ascending=False)
            )
            if not county_rank.empty:
                default_county = county_rank.index[0]

    date_frames = [frame for frame in [muni_ts, county_ts, state_ts] if not frame.empty]
    date_min = min(frame["date_bucket"].min() for frame in date_frames).strftime("%Y-%m-%d")
    date_max = max(frame["date_bucket"].max() for frame in date_frames).strftime("%Y-%m-%d")

    return {
        "municipalities": municipalities,
        "counties": counties,
        "default_municipality": default_municipality,
        "default_county": default_county,
        "date_min": date_min,
        "date_max": date_max,
    }


@app.get("/api/summary")
def get_summary(
    level: str = Query("municipality", pattern="^(municipality|county|state)$"),
    name: Optional[str] = None,
    view: str = Query("weekly", pattern="^(weekly|monthly)$"),
):
    ensure_loaded()
    return build_summary(level, name, view)


@app.get("/api/timeseries")
def get_timeseries(
    level: str = Query("municipality", pattern="^(municipality|county|state)$"),
    name: Optional[str] = None,
    metric: str = Query("spread_bps", pattern="^(spread_bps)$"),
    view: str = Query("weekly", pattern="^(weekly|monthly)$"),
):
    ensure_loaded()

    dataset = get_dataset(level, view)
    filtered, resolved_name = filter_dataset(dataset, level, name)
    filtered = filtered.sort_values("date_bucket")

    if metric != "spread_bps":
        raise HTTPException(status_code=400, detail="Only spread_bps is supported.")

    series = []
    for _, row in filtered.iterrows():
        series.append(
            {
                "date_bucket": row["date_bucket"].strftime("%Y-%m-%d"),
                "avg_spread_bps": coerce_float(row["avg_spread_bps"]),
                "spread_bps_rolling_4w": coerce_float(row["spread_bps_rolling_4w"]),
                "trade_count": coerce_int(row["trade_count"]),
                "cusip_count": coerce_int(row["cusip_count"]),
                "avg_yield": coerce_float(row["avg_yield"]),
                "avg_synthetic_aaa": coerce_float(row["avg_synthetic_aaa"]),
            }
        )

    return {
        "level": level,
        "name": resolved_name,
        "metric": metric,
        "view": view,
        "series": series,
    }


@app.get("/api/cross_section")
def get_cross_section(
    date: str,
    level: str = Query(..., pattern="^(municipality|county)$"),
    view: str = Query("weekly", pattern="^(weekly|monthly)$"),
):
    ensure_loaded()

    dataset = get_dataset(level, view)
    selected_date = pd.to_datetime(date, errors="coerce")
    if pd.isna(selected_date):
        raise HTTPException(status_code=400, detail="Invalid date.")

    filtered = dataset[dataset["date_bucket"] == selected_date].copy()
    if filtered.empty:
        raise HTTPException(status_code=404, detail="No data found for that date.")

    name_column = get_name_column(level)
    label_column = "mun_label" if level == "municipality" else "county"
    records = []
    for _, row in filtered.sort_values(name_column).iterrows():
        records.append(
            {
                "name": row[name_column],
                "label": row.get(label_column, row[name_column]),
                "date_bucket": row["date_bucket"].strftime("%Y-%m-%d"),
                "avg_spread_bps": coerce_float(row["avg_spread_bps"]),
                "spread_bps_rolling_4w": coerce_float(row["spread_bps_rolling_4w"]),
                "trade_count": coerce_int(row["trade_count"]),
                "cusip_count": coerce_int(row["cusip_count"]),
            }
        )

    return {
        "level": level,
        "view": view,
        "date": selected_date.strftime("%Y-%m-%d"),
        "records": records,
    }


@app.get("/api/map")
def get_map():
    return build_map_geojson()


@app.get("/api/municipalities")
def get_legacy_municipalities():
    ensure_loaded()
    summary = build_summary("state", None, "weekly")
    default_name = None
    if not lookup_df.empty:
        municipal_names = lookup_df[lookup_df["level"] == "municipality"]["name"].dropna()
        if not municipal_names.empty:
            default_name = municipal_names.sort_values().iloc[0]

    default_series = []
    if default_name:
        default_series = get_timeseries(level="municipality", name=default_name, view="weekly")["series"][:24]

    return {
        "macro": {
            "avg_resilient_spread": summary["avg_spread_bps"],
            "avg_non_resilient_spread": summary["peak_spread_bps"],
            "resilience_premium": summary["latest_spread_bps_rolling_4w"],
            "unique_bonds": summary["cusip_count"],
            "total_trades": summary["trade_count"],
        },
        "details_sample": default_series,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
