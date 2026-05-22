import glob
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRDS_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_raw", "wrds")
WRDS_PREVIEW_DIR = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "wrds_preview")
PREVIEW_ROWS = 200


def preview_wrds_files():
    os.makedirs(WRDS_PREVIEW_DIR, exist_ok=True)

    raw_files = sorted(glob.glob(os.path.join(WRDS_RAW_DIR, "*.csv.gz")))
    if not raw_files:
        print("No WRDS yearly files were found.")
        return

    print("=" * 60)
    print("WRDS YEARLY PREVIEW EXPORT")
    print("=" * 60)

    for raw_file in raw_files:
        year_name = os.path.basename(raw_file).replace(".csv.gz", "")
        output_file = os.path.join(WRDS_PREVIEW_DIR, f"{year_name}_preview.csv")

        preview_df = pd.read_csv(raw_file, compression="gzip", nrows=PREVIEW_ROWS, low_memory=False)
        preview_df.to_csv(output_file, index=False)

        print(f"{year_name}: wrote {len(preview_df):,} rows to {output_file}")


if __name__ == "__main__":
    preview_wrds_files()
