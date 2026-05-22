import glob
import os
import re
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw", "wrds")
DATA_CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned")
MUNI_CHARS_FILE = os.path.join(DATA_CLEANED_DIR, "nj_municipality_characteristics.csv")
ANALYTICS_OUTPUT = os.path.join(DATA_CLEANED_DIR, "nj_bonds_analytics.csv")
MASTER_OUTPUT = os.path.join(DATA_CLEANED_DIR, "nj_bonds_master.csv")
MUNI_TS_OUTPUT = os.path.join(DATA_CLEANED_DIR, "premium_timeseries_muni.csv")
COUNTY_TS_OUTPUT = os.path.join(DATA_CLEANED_DIR, "premium_timeseries_county.csv")
STATE_TS_OUTPUT = os.path.join(DATA_CLEANED_DIR, "premium_timeseries_state.csv")
LOOKUP_OUTPUT = os.path.join(DATA_CLEANED_DIR, "premium_lookup.csv")
UNMATCHED_OUTPUT = os.path.join(DATA_CLEANED_DIR, "premium_unmatched_diagnostics.csv")
MISSING_OUTPUT = os.path.join(DATA_CLEANED_DIR, "missing_municipalities_diagnostics.csv")

DETERMINISTIC_SEED = 20260402
NJ_PATTERN = r"\bNEW\s+JERSEY\b|(?:\s|^)(?:N\s*\.?\s*J\.?|NJ)(?:\s|,|$)"
GO_PATTERN = r"\b(?:GO|OBLIG|GEN OBL|GENERAL OBLIGATION|GENL OBLIG|G O)\b"

EXCLUSION_PATTERNS = [
    r"\bAUTH(?:ORITY)?\b",
    r"\bPORT AUTH(?:ORITY)?\b",
    r"\bIMPROVEMENT AUTH(?:ORITY)?\b",
    r"\bCOUNTY IMPROVEMENT\b",
    r"\bUTIL(?:ITY|ITIES)?\b",
    r"\bSEWER(?:AGE)?\b",
    r"\bWATER\b",
    r"\bREV(?:ENUE)?\b",
    r"\bTURNPIKE\b",
    r"\bTRANSPORT\b",
    r"\bHOUSING\b",
    r"\bPARKING\b",
    r"\bREDEVELOPMENT\b",
    r"\bMUA\b",
    r"\bPORT AUTH\b",
    r"\bPA & NJ\b",
    r"\bPENNSYLVANIA\b",
]

SUFFIX_ALIASES = {
    "TWP": "TOWNSHIP",
    "BORO": "BOROUGH",
    "BOROUGH": "BOROUGH",
    "CITY": "CITY",
    "TOWN": "TOWN",
    "VILLAGE": "VILLAGE",
}


def generate_synthetic_benchmark():
    """Create a deterministic daily AAA proxy for 2015-2025."""
    rng = np.random.default_rng(DETERMINISTIC_SEED)
    dates = pd.date_range(start="2015-01-01", end="2025-12-31", freq="D")
    n_dates = len(dates)

    trend = np.linspace(2.1, 3.9, n_dates)
    seasonal = np.sin(np.linspace(0, 18, n_dates)) * 0.35
    short_cycle = np.sin(np.linspace(0, 120, n_dates)) * 0.06
    noise = rng.normal(0, 0.025, n_dates)

    benchmark = pd.DataFrame(
        {
            "trade_date": dates,
            "synthetic_aaa": trend + seasonal + short_cycle + noise,
        }
    )
    return benchmark


def normalize_text(value):
    text = str(value).upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_suffix(text):
    return re.sub(r"\b(TOWNSHIP|BOROUGH|CITY|TOWN|VILLAGE|TWP|BORO)\b$", "", text).strip()


def build_alias_map(muni_chars):
    alias_map = {}
    exclusion_terms = {normalize_text(col) for col in ["NEW JERSEY", "STATE OF NEW JERSEY"]}

    for _, row in muni_chars.iterrows():
        canonical = normalize_text(row["mun"])
        if not canonical or canonical in exclusion_terms:
            continue

        tokens = canonical.split()
        base = strip_suffix(canonical)
        suffix = tokens[-1] if tokens and tokens[-1] in SUFFIX_ALIASES else None

        aliases = {canonical, base}
        if suffix:
            aliases.add(f"{base} {suffix}")
            aliases.add(f"{base} {SUFFIX_ALIASES[suffix]}")

        label = row.get("mun_label")
        if pd.notna(label):
            aliases.add(normalize_text(label))

        for alias in list(aliases):
            cleaned = alias.strip()
            if cleaned:
                alias_map[cleaned] = canonical

    # Prefer longer, more specific aliases first during matching.
    ordered_aliases = sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True)
    return ordered_aliases


def find_municipality_match(description, ordered_aliases):
    normalized = normalize_text(description)
    padded = f" {normalized} "

    if any(re.search(pattern, normalized) for pattern in EXCLUSION_PATTERNS):
        return None, normalized

    for alias, canonical in ordered_aliases:
        if not alias:
            continue

        padded_alias = f" {alias} "
        if padded_alias in padded:
            return canonical, normalized

    return None, normalized


def build_trade_level_analytics(master_df, muni_chars):
    benchmark = generate_synthetic_benchmark()

    master_df["trade_date"] = pd.to_datetime(master_df["trade_date"], errors="coerce")
    master_df["dated_date"] = pd.to_datetime(master_df["dated_date"], errors="coerce")
    master_df["maturity_date"] = pd.to_datetime(master_df["maturity_date"], errors="coerce")
    master_df["yield"] = pd.to_numeric(master_df["yield"], errors="coerce")
    master_df["coupon"] = pd.to_numeric(master_df["coupon"], errors="coerce")
    master_df = master_df.dropna(subset=["trade_date", "yield", "matched_town"]).copy()

    master_df = master_df.merge(benchmark, on="trade_date", how="left")
    master_df["spread_bps"] = (master_df["yield"] - master_df["synthetic_aaa"]) * 100

    char_columns = [
        "mun",
        "county",
        "mun_label",
        "is_resilient",
        "flooded_pct_2ft",
        "flooded_pct_5ft",
        "flooded_pct_7ft",
        "crs_class",
        "median_household_income",
    ]
    chars_subset = muni_chars[char_columns].drop_duplicates().copy()

    master_df = master_df.merge(
        chars_subset,
        left_on="matched_town",
        right_on="mun",
        how="left",
    )

    master_df["trade_year"] = master_df["trade_date"].dt.year.astype(int)
    master_df["date_bucket_weekly"] = master_df["trade_date"].dt.to_period("W-SUN").dt.start_time
    master_df["date_bucket_monthly"] = master_df["trade_date"].dt.to_period("M").dt.start_time

    ordered_columns = [
        "cusip",
        "trade_date",
        "trade_year",
        "security_description",
        "yield",
        "coupon",
        "dated_date",
        "maturity_date",
        "synthetic_aaa",
        "spread_bps",
        "matched_town",
        "mun",
        "mun_label",
        "county",
        "is_resilient",
        "flooded_pct_2ft",
        "flooded_pct_5ft",
        "flooded_pct_7ft",
        "crs_class",
        "median_household_income",
        "date_bucket_weekly",
        "date_bucket_monthly",
    ]
    return master_df[ordered_columns].sort_values(["trade_date", "matched_town", "cusip"])


def aggregate_timeseries(analytics_df, level, geography_columns):
    frames = []
    views = {
        "weekly": "date_bucket_weekly",
        "monthly": "date_bucket_monthly",
    }

    for view_name, bucket_column in views.items():
        group_columns = [bucket_column, *geography_columns]
        grouped = (
            analytics_df.groupby(group_columns, dropna=False)
            .agg(
                trade_count=("cusip", "size"),
                cusip_count=("cusip", "nunique"),
                avg_yield=("yield", "mean"),
                avg_synthetic_aaa=("synthetic_aaa", "mean"),
                avg_spread_bps=("spread_bps", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.rename(columns={bucket_column: "date_bucket"})
        grouped["date_bucket"] = pd.to_datetime(grouped["date_bucket"], errors="coerce")
        grouped["level"] = level
        grouped["view"] = view_name

        sort_columns = [*geography_columns, "date_bucket"]
        grouped = grouped.sort_values(sort_columns)
        if geography_columns:
            grouped["spread_bps_rolling_4w"] = (
                grouped.groupby(geography_columns)["avg_spread_bps"]
                .transform(lambda values: values.rolling(window=4, min_periods=1).mean())
            )
        else:
            grouped["spread_bps_rolling_4w"] = grouped["avg_spread_bps"].rolling(window=4, min_periods=1).mean()

        grouped["date_bucket"] = grouped["date_bucket"].dt.strftime("%Y-%m-%d")
        frames.append(grouped)

    return pd.concat(frames, ignore_index=True)


def build_lookup(muni_ts, county_ts):
    municipalities = (
        muni_ts[muni_ts["view"] == "weekly"][["mun", "mun_label", "county"]]
        .dropna(subset=["mun"])
        .drop_duplicates()
        .sort_values(["mun_label", "mun"])
        .reset_index(drop=True)
    )
    municipalities["level"] = "municipality"
    municipalities["name"] = municipalities["mun"]
    municipalities["label"] = municipalities["mun_label"].fillna(municipalities["mun"])

    counties = (
        county_ts[county_ts["view"] == "weekly"][["county"]]
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


def process_wrds_data():
    print("=" * 60)
    print("WRDS BOND EXTRACTION — SYNTHETIC AAA PREMIUM DASHBOARD")
    print("=" * 60)

    muni_chars = pd.read_csv(MUNI_CHARS_FILE, low_memory=False)
    muni_chars.columns = [column.lower() for column in muni_chars.columns]
    muni_chars = muni_chars.loc[:, ~muni_chars.columns.duplicated()].copy()
    ordered_aliases = build_alias_map(muni_chars)

    gz_files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, "*.csv.gz")))
    if not gz_files:
        # Fallback to look at regular CSVs if gz are not present, helpful for preview testing
        gz_files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, "*.csv")))
    
    all_matches = []
    unmatched_counts = {}
    required_cols = {"security_description", "trade_date", "dated_date", "yield", "coupon", "maturity_date", "cusip"}

    for gz_file in gz_files:
        year_str = os.path.basename(gz_file).split("_")[0].split(".")[0]
        print(f"Processing Year: {year_str}...")

        try:
            year_int = int(year_str)
            dt_format = "%Y-%m-%d" if year_int > 2021 else "%Y%m%d"
        except ValueError:
            dt_format = None

        chunks = pd.read_csv(
            gz_file, 
            chunksize=100000, 
            compression="gzip" if gz_file.endswith(".gz") else None, 
            low_memory=False,
            usecols=lambda c: c.lower() in required_cols
        )

        for chunk in chunks:
            chunk.columns = chunk.columns.str.lower()
            if not required_cols.issubset(chunk.columns):
                continue

            # Optimize: Check for strings first before running expensive date parsing
            is_nj = chunk["security_description"].str.contains(NJ_PATTERN, case=False, na=False, regex=True)
            is_go = chunk["security_description"].str.contains(GO_PATTERN, case=False, na=False, regex=True)

            candidate_chunk = chunk[is_nj & is_go].copy()
            if candidate_chunk.empty:
                continue

            if dt_format:
                candidate_chunk["dated_date"] = pd.to_datetime(candidate_chunk["dated_date"], format=dt_format, errors="coerce")
            else:
                candidate_chunk["dated_date"] = pd.to_datetime(candidate_chunk["dated_date"], errors="coerce")
                
            is_date_valid = (candidate_chunk["dated_date"] >= "2015-01-01") & (candidate_chunk["dated_date"] <= "2025-12-31")
            candidate_chunk = candidate_chunk[is_date_valid].copy()
            
            if candidate_chunk.empty:
                continue

            matched_info = candidate_chunk["security_description"].apply(
                lambda description: find_municipality_match(description, ordered_aliases)
            )
            candidate_chunk["matched_town"] = matched_info.str[0]
            candidate_chunk["normalized_description"] = matched_info.str[1]

            unmatched = candidate_chunk[candidate_chunk["matched_town"].isna()]
            for description, count in unmatched["normalized_description"].value_counts().head(250).items():
                unmatched_counts[description] = unmatched_counts.get(description, 0) + int(count)

            matched = candidate_chunk[candidate_chunk["matched_town"].notna()].copy()
            if matched.empty:
                continue

            all_matches.append(
                matched[
                    [
                        "cusip",
                        "trade_date",
                        "security_description",
                        "yield",
                        "coupon",
                        "maturity_date",
                        "dated_date",
                        "matched_town",
                    ]
                ]
            )

    if not all_matches:
        print("No matched municipal trades were found.")
        return

    matched_df = pd.concat(all_matches, ignore_index=True)
    analytics_df = build_trade_level_analytics(matched_df, muni_chars)
    with tempfile.TemporaryDirectory(prefix="wrds_premium_") as temp_dir:
        analytics_temp = os.path.join(temp_dir, os.path.basename(ANALYTICS_OUTPUT))
        master_temp = os.path.join(temp_dir, os.path.basename(MASTER_OUTPUT))
        muni_ts_temp = os.path.join(temp_dir, os.path.basename(MUNI_TS_OUTPUT))
        county_ts_temp = os.path.join(temp_dir, os.path.basename(COUNTY_TS_OUTPUT))
        state_ts_temp = os.path.join(temp_dir, os.path.basename(STATE_TS_OUTPUT))
        lookup_temp = os.path.join(temp_dir, os.path.basename(LOOKUP_OUTPUT))
        unmatched_temp = os.path.join(temp_dir, os.path.basename(UNMATCHED_OUTPUT))

        analytics_df.to_csv(analytics_temp, index=False)
        analytics_df.to_csv(master_temp, index=False)

        muni_ts = aggregate_timeseries(
            analytics_df.dropna(subset=["mun", "county"]).copy(),
            "municipality",
            ["mun", "mun_label", "county"],
        )
        county_ts = aggregate_timeseries(
            analytics_df.dropna(subset=["county"]).copy(),
            "county",
            ["county"],
        )
        state_base = analytics_df.copy()
        state_base["state_name"] = "NEW JERSEY"
        state_ts = aggregate_timeseries(state_base, "state", ["state_name"]).rename(columns={"state_name": "name"})

        muni_ts.to_csv(muni_ts_temp, index=False)
        county_ts.to_csv(county_ts_temp, index=False)
        state_ts.to_csv(state_ts_temp, index=False)
        build_lookup(muni_ts, county_ts).to_csv(lookup_temp, index=False)

        unmatched_df = (
            pd.DataFrame(
                [{"normalized_description": description, "count": count} for description, count in unmatched_counts.items()]
            )
            .sort_values(["count", "normalized_description"], ascending=[False, True])
            .head(500)
        )
        unmatched_df.to_csv(unmatched_temp, index=False)

        known_municipalities = set(muni_chars["mun"].unique())
        matched_municipalities = set(analytics_df["mun"].dropna().unique())
        missing_municipalities = sorted(known_municipalities - matched_municipalities)
        
        missing_df = pd.DataFrame({"missing_mun": missing_municipalities})
        missing_temp = os.path.join(temp_dir, os.path.basename(MISSING_OUTPUT))
        missing_df.to_csv(missing_temp, index=False)

        os.replace(analytics_temp, ANALYTICS_OUTPUT)
        os.replace(master_temp, MASTER_OUTPUT)
        os.replace(muni_ts_temp, MUNI_TS_OUTPUT)
        os.replace(county_ts_temp, COUNTY_TS_OUTPUT)
        os.replace(state_ts_temp, STATE_TS_OUTPUT)
        os.replace(lookup_temp, LOOKUP_OUTPUT)
        os.replace(unmatched_temp, UNMATCHED_OUTPUT)
        os.replace(missing_temp, MISSING_OUTPUT)

    print()
    print("EXTRACTION SUCCESSFUL.")
    print(f"Trade-level analytics rows:            {len(analytics_df):,}")
    print(f"Unique municipality matches:           {analytics_df['mun'].nunique():,}")
    print(f"Unique county matches:                 {analytics_df['county'].nunique():,}")
    print(f"Weekly/monthly muni series rows:       {len(muni_ts):,}")
    print(f"Weekly/monthly county series rows:     {len(county_ts):,}")
    print(f"Weekly/monthly statewide series rows:  {len(state_ts):,}")


if __name__ == "__main__":
    process_wrds_data()
