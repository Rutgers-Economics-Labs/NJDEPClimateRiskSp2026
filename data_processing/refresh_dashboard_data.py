import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dashboard.backend.main import (
    LOOKUP_FILE,
    MUNI_CHARS_FILE,
    MUNI_TS_FILE,
    COUNTY_TS_FILE,
    STATE_TS_FILE,
    build_dashboard_frames_from_full_trades,
)

FULL_TRADES_FILE = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "nj_muni_spreads_full_trades.csv")

def main():
    full_trades = pd.read_csv(FULL_TRADES_FILE, low_memory=False)
    muni_chars = pd.read_csv(MUNI_CHARS_FILE, low_memory=False)
    muni_chars.columns = [column.lower() for column in muni_chars.columns]
    muni_chars = muni_chars.loc[:, ~muni_chars.columns.duplicated()].copy()
    if "is_resilient" in muni_chars.columns:
        muni_chars["is_resilient"] = muni_chars["is_resilient"].fillna(False).astype(bool)

    muni_ts, county_ts, state_ts, lookup_df = build_dashboard_frames_from_full_trades(full_trades, muni_chars)

    for frame in [muni_ts, county_ts, state_ts]:
        if "date_bucket" in frame.columns:
            frame["date_bucket"] = pd.to_datetime(frame["date_bucket"]).dt.strftime("%Y-%m-%d")

    muni_ts.to_csv(MUNI_TS_FILE, index=False)
    county_ts.to_csv(COUNTY_TS_FILE, index=False)
    state_ts.to_csv(STATE_TS_FILE, index=False)
    lookup_df.to_csv(LOOKUP_FILE, index=False)

    print(f"Wrote {len(muni_ts)} rows to {MUNI_TS_FILE}")
    print(f"Wrote {len(county_ts)} rows to {COUNTY_TS_FILE}")
    print(f"Wrote {len(state_ts)} rows to {STATE_TS_FILE}")
    print(f"Wrote {len(lookup_df)} rows to {LOOKUP_FILE}")


if __name__ == "__main__":
    main()
