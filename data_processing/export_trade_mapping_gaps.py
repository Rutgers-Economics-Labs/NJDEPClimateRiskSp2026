import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned")
TRADES_FILE = os.path.join(DATA_CLEANED_DIR, "nj_muni_spreads_full_trades.csv")
MUNI_CHARS_FILE = os.path.join(DATA_CLEANED_DIR, "nj_municipality_characteristics.csv")
MISSING_OUTPUT = os.path.join(DATA_CLEANED_DIR, "missing_canonical_munis.csv")
UNMAPPED_OUTPUT = os.path.join(DATA_CLEANED_DIR, "unmapped_trade_names.csv")


def main():
    chars = pd.read_csv(MUNI_CHARS_FILE, low_memory=False)
    chars.columns = [column.lower() for column in chars.columns]
    chars = chars.loc[:, ~chars.columns.duplicated()].copy()
    chars["mun"] = chars["mun"].astype(str).str.upper().str.strip()
    chars = chars.dropna(subset=["mun"]).drop_duplicates(subset=["mun"]).copy()

    trades = pd.read_csv(TRADES_FILE, low_memory=False, usecols=["MUN", "MATCHED_MUNI"])
    trades["mun"] = trades["MUN"].astype(str).str.upper().str.strip()
    trades["raw_match"] = trades["MATCHED_MUNI"].astype(str).str.upper().str.strip()
    trades = trades[trades["mun"] != ""].copy()

    mapped_unique = set(trades["mun"].dropna())
    canonical_unique = set(chars["mun"])

    missing_from_trade_file = sorted(canonical_unique - mapped_unique)
    unmapped_trade_names = sorted(set(trades.loc[trades["mun"].isna() | (trades["mun"] == ""), "raw_match"].dropna()) - {""})

    missing_df = pd.DataFrame({"missing_canonical_muni": missing_from_trade_file})
    unmapped_df = pd.DataFrame({"unmapped_trade_name": unmapped_trade_names})

    missing_df.to_csv(MISSING_OUTPUT, index=False)
    unmapped_df.to_csv(UNMAPPED_OUTPUT, index=False)

    print(f"Wrote {len(missing_df)} rows to {MISSING_OUTPUT}")
    print(f"Wrote {len(unmapped_df)} rows to {UNMAPPED_OUTPUT}")


if __name__ == "__main__":
    main()
