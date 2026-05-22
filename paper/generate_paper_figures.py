"""
Generate paper-ready tables and figures for the NJDEP climate risk paper.

Outputs are written into the paper/ directory as both CSVs and PNGs so they can
be inserted directly into the final report or referenced from the text draft.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "paper"
PANEL_FILE = PROJECT_ROOT / "data" / "data_cleaned" / "final_panel_master.csv"


def load_trade_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_FILE, low_memory=False)
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    df["year"] = df["issue_date"].dt.year

    support = df.groupby("muni_name")["is_post_2022"].agg(
        pre=lambda s: int((s == 0).sum()),
        post=lambda s: int((s == 1).sum()),
    )
    both_period_munis = support[(support["pre"] > 0) & (support["post"] > 0)].index
    df = df[df["muni_name"].isin(both_period_munis)].copy()

    low, high = df["spread_bps"].quantile([0.01, 0.99])
    df["spread_bps_winsor"] = df["spread_bps"].clip(lower=low, upper=high)
    df["median_income_10k"] = df["median_income"] / 10_000
    df["ever_champ"] = df["ever_champ"].fillna(0).astype(int)
    return df


def build_muni_panel(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[df["is_post_2022"] == 0].groupby("muni_name").agg(
        spread_pre=("spread_bps_winsor", "mean"),
        n_trades_pre=("spread_bps_winsor", "count"),
    )
    post = df[df["is_post_2022"] == 1].groupby("muni_name").agg(
        spread_post=("spread_bps_winsor", "mean"),
        n_trades_post=("spread_bps_winsor", "count"),
    )
    chars = df.groupby("muni_name").agg(
        ever_champ=("ever_champ", "first"),
        slr_exposure_pct=("slr_exposure_pct", "first"),
        debt_to_av=("debt_to_av", "first"),
        median_income=("median_income", "first"),
    )
    muni = pre.join(post).join(chars).dropna().copy()
    muni["delta_spread"] = muni["spread_post"] - muni["spread_pre"]
    muni["median_income_10k"] = muni["median_income"] / 10_000
    return muni


def save_table(df: pd.DataFrame, stem: str, title: str) -> None:
    csv_path = PAPER_DIR / f"{stem}.csv"
    png_path = PAPER_DIR / f"{stem}.png"
    df.to_csv(csv_path, index=False)

    rows, cols = df.shape
    fig_width = max(8, cols * 1.7)
    fig_height = max(2.5, rows * 0.42 + 1.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#234a6f")
        elif r % 2 == 0:
            cell.set_facecolor("#eef3f8")
    plt.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_sample_counts(df: pd.DataFrame) -> None:
    table = (
        df.groupby(["year", "ever_champ"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "Non-CHAMP Trades", 1: "CHAMP Trades"})
        .reset_index()
    )
    table["Total Trades"] = table["Non-CHAMP Trades"] + table["CHAMP Trades"]
    save_table(table, "figure_1_sample_counts", "Figure 1. Sample Coverage by Year and CHAMP Cohort")


def figure_descriptive_stats(df: pd.DataFrame, muni: pd.DataFrame) -> None:
    stats = pd.DataFrame(
        [
            ["Trade-level", "Winsorized spread (bps)", df["spread_bps_winsor"].mean(), df["spread_bps_winsor"].std(), df["spread_bps_winsor"].median()],
            ["Trade-level", "Time to maturity (years)", df["time_to_maturity"].mean(), df["time_to_maturity"].std(), df["time_to_maturity"].median()],
            ["Municipality-level", "SLR exposure (%)", muni["slr_exposure_pct"].mean(), muni["slr_exposure_pct"].std(), muni["slr_exposure_pct"].median()],
            ["Municipality-level", "Debt to AV", muni["debt_to_av"].mean(), muni["debt_to_av"].std(), muni["debt_to_av"].median()],
            ["Municipality-level", "Median income ($)", muni["median_income"].mean(), muni["median_income"].std(), muni["median_income"].median()],
            ["Municipality-level", "Pre-period spread (bps)", muni["spread_pre"].mean(), muni["spread_pre"].std(), muni["spread_pre"].median()],
            ["Municipality-level", "Post-period spread (bps)", muni["spread_post"].mean(), muni["spread_post"].std(), muni["spread_post"].median()],
        ],
        columns=["Level", "Variable", "Mean", "Std. Dev.", "Median"],
    )
    for col in ["Mean", "Std. Dev.", "Median"]:
        stats[col] = stats[col].round(3)
    save_table(stats, "figure_2_descriptive_statistics", "Figure 2. Descriptive Statistics Used in the Paper")


def figure_pretrends(df: pd.DataFrame) -> None:
    trend = (
        df.groupby(["year", "ever_champ"])["spread_bps"]
        .mean()
        .reset_index()
    )
    labels = {0: "Non-CHAMP cohort", 1: "CHAMP cohort"}
    colors = {0: "#234a6f", 1: "#b05a2b"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for champ in [0, 1]:
        temp = trend[trend["ever_champ"] == champ]
        ax.plot(
            temp["year"],
            temp["spread_bps"],
            marker="o",
            linewidth=2.4,
            color=colors[champ],
            label=labels[champ],
        )
    ax.axvline(2022, color="#666666", linestyle="--", linewidth=1.5, label="2022 policy pivot")
    ax.set_title("Figure 3. Mean Raw Bond Spreads by Cohort and Year", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean spread (bps)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(PAPER_DIR / "figure_3_pretrend_spreads.png", dpi=220, bbox_inches="tight")
    trend.to_csv(PAPER_DIR / "figure_3_pretrend_spreads.csv", index=False)
    plt.close(fig)


def fit_models(df: pd.DataFrame):
    formula_pooled = (
        "spread_bps_winsor ~ ever_champ + slr_exposure_pct"
        " + debt_to_av + median_income_10k + time_to_maturity"
        " + is_post_2022"
    )
    formula_did = (
        "spread_bps_winsor ~ ever_champ:is_post_2022"
        " + time_to_maturity"
        " + C(muni_name) + C(year)"
    )
    formula_ddd = (
        "spread_bps_winsor ~ ever_champ:is_post_2022"
        " + is_post_2022:slr_exposure_pct"
        " + ever_champ:is_post_2022:slr_exposure_pct"
        " + time_to_maturity"
        " + C(muni_name) + C(year)"
    )

    pre = df[df["is_post_2022"] == 0].copy()
    pre["pre_year_index"] = pre["year"] - pre["year"].min()
    pretrend_model = smf.ols(
        "spread_bps_winsor ~ ever_champ:pre_year_index + time_to_maturity + C(muni_name) + C(year)",
        data=pre,
    ).fit(cov_type="cluster", cov_kwds={"groups": pre["muni_name"]})

    es_yr_coverage = df.groupby("muni_name")["year"].nunique()
    es_munis = es_yr_coverage[es_yr_coverage >= 3].index
    df_es = df[df["muni_name"].isin(es_munis)].copy()
    muni_means = df_es.groupby("muni_name")[["spread_bps_winsor", "time_to_maturity"]].transform("mean")
    df_es["spread_dm"] = df_es["spread_bps_winsor"] - muni_means["spread_bps_winsor"]
    df_es["ttm_dm"] = df_es["time_to_maturity"] - muni_means["time_to_maturity"]
    ref_year = 2021
    event_years = [y for y in sorted(df_es["year"].unique()) if y != ref_year]
    for y in event_years:
        df_es[f"champ_x_{y}"] = (df_es["ever_champ"] * (df_es["year"] == y)).astype(int)
    champ_year_terms = " + ".join(f"champ_x_{y}" for y in event_years)
    year_terms = " + ".join(f"I(year == {y})" for y in event_years)
    formula_event = f"spread_dm ~ {champ_year_terms} + ttm_dm + {year_terms}"

    m1 = smf.ols(formula_pooled, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["muni_name"]})
    m2 = smf.ols(formula_did, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["muni_name"]})
    m3 = smf.ols(formula_ddd, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["muni_name"]})
    m4 = smf.ols(formula_event, data=df_es).fit(cov_type="cluster", cov_kwds={"groups": df_es["muni_name"]})

    return m1, m2, m3, m4, pretrend_model, event_years


def star(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def figure_regression_summary(df: pd.DataFrame) -> tuple:
    m1, m2, m3, m4, pretrend_model, event_years = fit_models(df)
    rows = [
        ["Model A", "ever_champ", m1.params["ever_champ"], m1.bse["ever_champ"], m1.pvalues["ever_champ"]],
        ["Model A", "is_post_2022", m1.params["is_post_2022"], m1.bse["is_post_2022"], m1.pvalues["is_post_2022"]],
        ["Model A", "slr_exposure_pct", m1.params["slr_exposure_pct"], m1.bse["slr_exposure_pct"], m1.pvalues["slr_exposure_pct"]],
        ["Model A", "debt_to_av", m1.params["debt_to_av"], m1.bse["debt_to_av"], m1.pvalues["debt_to_av"]],
        ["Model A", "median_income_10k", m1.params["median_income_10k"], m1.bse["median_income_10k"], m1.pvalues["median_income_10k"]],
        ["Model A", "time_to_maturity", m1.params["time_to_maturity"], m1.bse["time_to_maturity"], m1.pvalues["time_to_maturity"]],
        ["Model B", "ever_champ x post_2022", m2.params["ever_champ:is_post_2022"], m2.bse["ever_champ:is_post_2022"], m2.pvalues["ever_champ:is_post_2022"]],
        ["Model B", "time_to_maturity", m2.params["time_to_maturity"], m2.bse["time_to_maturity"], m2.pvalues["time_to_maturity"]],
        ["Model C", "ever_champ x post_2022", m3.params["ever_champ:is_post_2022"], m3.bse["ever_champ:is_post_2022"], m3.pvalues["ever_champ:is_post_2022"]],
        ["Model C", "post_2022 x slr_exposure_pct", m3.params["is_post_2022:slr_exposure_pct"], m3.bse["is_post_2022:slr_exposure_pct"], m3.pvalues["is_post_2022:slr_exposure_pct"]],
        ["Model C", "ever_champ x post_2022 x slr_exposure_pct", m3.params["ever_champ:is_post_2022:slr_exposure_pct"], m3.bse["ever_champ:is_post_2022:slr_exposure_pct"], m3.pvalues["ever_champ:is_post_2022:slr_exposure_pct"]],
        ["Pre-trend", "ever_champ x pre_year_index", pretrend_model.params["ever_champ:pre_year_index"], pretrend_model.bse["ever_champ:pre_year_index"], pretrend_model.pvalues["ever_champ:pre_year_index"]],
    ]
    table = pd.DataFrame(rows, columns=["Model", "Term", "Coefficient", "Std. Error", "p-value"])
    table["Coefficient"] = table["Coefficient"].round(3)
    table["Std. Error"] = table["Std. Error"].round(3)
    table["p-value"] = table["p-value"].round(4)
    table["Sig."] = table["p-value"].apply(star)
    save_table(table, "figure_4_regression_summary", "Figure 4. Core Regression Results")
    return m4, event_years


def figure_event_study(m4, event_years) -> None:
    rows = []
    for year in event_years:
        term = f"champ_x_{year}"
        coef = m4.params[term]
        se = m4.bse[term]
        rows.append(
            {
                "year": year,
                "coefficient": coef,
                "std_error": se,
                "lower_95": coef - 1.96 * se,
                "upper_95": coef + 1.96 * se,
                "p_value": m4.pvalues[term],
            }
        )
    event_df = pd.DataFrame(rows)
    event_df.to_csv(PAPER_DIR / "figure_5_event_study.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(
        event_df["year"],
        event_df["coefficient"],
        yerr=1.96 * event_df["std_error"],
        fmt="o-",
        color="#234a6f",
        linewidth=2.2,
        capsize=4,
    )
    ax.axhline(0, color="#666666", linestyle="--", linewidth=1.3)
    ax.axvline(2021, color="#b05a2b", linestyle=":", linewidth=1.5)
    ax.set_title("Figure 5. Event-Study Style CHAMP-by-Year Coefficients", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year (reference year omitted: 2021)")
    ax.set_ylabel("Within-municipality spread difference (bps)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(PAPER_DIR / "figure_5_event_study.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_balance_and_overlap(muni: pd.DataFrame) -> None:
    covariates = ["slr_exposure_pct", "debt_to_av", "median_income_10k", "spread_pre"]
    X = muni[covariates].values
    D = muni["ever_champ"].values
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0] = 1
    Xz = (X - x_mean) / x_std

    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr.fit(Xz, D)
    p_score = np.clip(lr.predict_proba(Xz)[:, 1], 0.01, 0.99)
    muni = muni.copy()
    muni["p_score"] = p_score

    rows = []
    for cov in covariates:
        treated = muni.loc[muni["ever_champ"] == 1, cov]
        control = muni.loc[muni["ever_champ"] == 0, cov]
        pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
        raw_std_diff = (treated.mean() - control.mean()) / pooled_sd if pooled_sd else 0

        w1 = 1.0 / muni.loc[muni["ever_champ"] == 1, "p_score"]
        w0 = 1.0 / (1.0 - muni.loc[muni["ever_champ"] == 0, "p_score"])
        weighted_diff = np.average(treated, weights=w1) - np.average(control, weights=w0)
        ipw_std_diff = weighted_diff / pooled_sd if pooled_sd else 0
        rows.append(
            {
                "Variable": cov,
                "Mean (CHAMP)": round(treated.mean(), 4),
                "Mean (Non-CHAMP)": round(control.mean(), 4),
                "Raw Std Diff": round(raw_std_diff, 4),
                "IPW Std Diff": round(ipw_std_diff, 4),
            }
        )

    balance_df = pd.DataFrame(rows)
    save_table(balance_df, "figure_6_covariate_balance", "Figure 6. Covariate Balance in the AIPW-DiD Setup")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.hist(
        muni.loc[muni["ever_champ"] == 0, "p_score"],
        bins=18,
        alpha=0.65,
        color="#234a6f",
        label="Non-CHAMP municipalities",
    )
    ax.hist(
        muni.loc[muni["ever_champ"] == 1, "p_score"],
        bins=18,
        alpha=0.65,
        color="#b05a2b",
        label="CHAMP municipalities",
    )
    ax.set_title("Figure 7. Propensity Score Overlap", fontsize=13, fontweight="bold")
    ax.set_xlabel("Estimated propensity score")
    ax.set_ylabel("Municipality count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(PAPER_DIR / "figure_7_propensity_overlap.png", dpi=220, bbox_inches="tight")
    muni[["ever_champ", "slr_exposure_pct", "debt_to_av", "median_income_10k", "spread_pre", "p_score"]].to_csv(
        PAPER_DIR / "figure_7_propensity_overlap.csv",
        index=True,
    )
    plt.close(fig)


def main() -> None:
    PAPER_DIR.mkdir(exist_ok=True)
    df = load_trade_panel()
    muni = build_muni_panel(df)
    figure_sample_counts(df)
    figure_descriptive_stats(df, muni)
    figure_pretrends(df)
    m4, event_years = figure_regression_summary(df)
    figure_event_study(m4, event_years)
    figure_balance_and_overlap(muni)
    print(f"Wrote figures and tables to {PAPER_DIR}")


if __name__ == "__main__":
    main()
