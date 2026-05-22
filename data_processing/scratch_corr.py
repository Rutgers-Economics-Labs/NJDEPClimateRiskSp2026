import pandas as pd
df = pd.read_csv('data/data_cleaned/final_panel_master.csv', low_memory=False)
cols = ['spread_bps', 'slr_exposure_pct', 'is_resilient', 'ms4_outfall_density', 'debt_to_av', 'median_income']
print(df[cols].corr().to_string())
