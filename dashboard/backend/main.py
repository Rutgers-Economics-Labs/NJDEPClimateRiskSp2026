import os

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

muni_ts = pd.DataFrame()
county_ts = pd.DataFrame()
state_ts = pd.DataFrame()
lookup_df = pd.DataFrame()
muni_chars = pd.DataFrame()


def load_csv(filepath, parse_dates=None):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    return pd.read_csv(filepath, parse_dates=parse_dates, low_memory=False)


def load_data():
    global muni_ts, county_ts, state_ts, lookup_df, muni_chars

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


def ensure_loaded():
    if muni_ts.empty and county_ts.empty and state_ts.empty:
        raise HTTPException(
            status_code=503,
            detail="Dashboard data not found. Please run 'make process-all'.",
        )


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
                "flooded_pct_2ft": coerce_float(row.get("flooded_pct_2ft")),
                "flooded_pct_5ft": coerce_float(row.get("flooded_pct_5ft")),
                "flooded_pct_7ft": coerce_float(row.get("flooded_pct_7ft")),
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
    name: str | None = None,
    view: str = Query("weekly", pattern="^(weekly|monthly)$"),
):
    ensure_loaded()
    return build_summary(level, name, view)


@app.get("/api/timeseries")
def get_timeseries(
    level: str = Query("municipality", pattern="^(municipality|county|state)$"),
    name: str | None = None,
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
