import json
import os
import statsmodels.formula.api as smf
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "final_panel_master.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dashboard", "frontend", "public", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(INPUT_FILE)
df['issue_date'] = pd.to_datetime(df['issue_date'])
df['year'] = df['issue_date'].dt.year

formula_slr_int = (
    "spread_bps ~ is_resilient:is_post_2022"
    " + is_post_2022:slr_exposure_pct"
    " + is_resilient:is_post_2022:slr_exposure_pct"
    " + debt_to_gdp + time_to_maturity"
    " + C(muni_name) + C(year)"
)

print("Running Triple-DiD regression...")
m3 = smf.ols(formula_slr_int, data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['muni_name']}
)

coefs = {
    "is_resilient_is_post_2022": float(m3.params.get('is_resilient:is_post_2022', 0)),
    "is_post_2022_slr_exposure_pct": float(m3.params.get('is_post_2022:slr_exposure_pct', 0)),
    "is_resilient_is_post_2022_slr_exposure_pct": float(m3.params.get('is_resilient:is_post_2022:slr_exposure_pct', 0)),
    "term_premium": float(m3.params.get('time_to_maturity', 0))
}

out_path = os.path.join(OUTPUT_DIR, "did_coefficients.json")
with open(out_path, "w") as f:
    json.dump(coefs, f, indent=2)

print(f"Exported Triple-DiD coefficients to {out_path}")
