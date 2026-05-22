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
if "debt_to_gdp" in df.columns and "debt_to_av" not in df.columns:
    df = df.rename(columns={"debt_to_gdp": "debt_to_av"})
required = ["ever_champ", "time_to_maturity", "spread_bps", "is_post_2022", "slr_exposure_pct"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in {INPUT_FILE}: {missing}")

support = df.groupby("muni_name")["is_post_2022"].agg(
    pre=lambda s: int((s == 0).sum()),
    post=lambda s: int((s == 1).sum()),
)
both_period_munis = support[(support["pre"] > 0) & (support["post"] > 0)].index
df = df[df["muni_name"].isin(both_period_munis)].copy()
low, high = df["spread_bps"].quantile([0.01, 0.99])
df["spread_bps_winsor"] = df["spread_bps"].clip(lower=low, upper=high)

formula_slr_int = (
    "spread_bps_winsor ~ ever_champ:is_post_2022"
    " + is_post_2022:slr_exposure_pct"
    " + ever_champ:is_post_2022:slr_exposure_pct"
    " + time_to_maturity"
    " + C(muni_name) + C(year)"
)

print("Running Triple-DiD regression...")
m3 = smf.ols(formula_slr_int, data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['muni_name']}
)

coefs = {
    "ever_champ_is_post_2022": float(m3.params.get('ever_champ:is_post_2022', 0)),
    "is_post_2022_slr_exposure_pct": float(m3.params.get('is_post_2022:slr_exposure_pct', 0)),
    "ever_champ_is_post_2022_slr_exposure_pct": float(m3.params.get('ever_champ:is_post_2022:slr_exposure_pct', 0)),
    "term_premium": float(m3.params.get('time_to_maturity', 0)),
    "spread_winsor_p1": float(low),
    "spread_winsor_p99": float(high),
    "n_obs": int(m3.nobs),
    "note": "Exploratory secondary-trade association using synthetic AAA benchmark, winsorized spread, and towns with both pre/post observations. Model C excludes debt-to-AV."
}

out_path = os.path.join(OUTPUT_DIR, "did_coefficients.json")
with open(out_path, "w") as f:
    json.dump(coefs, f, indent=2)

print(f"Exported Triple-DiD coefficients to {out_path}")
