"""
export_paper_tables.py
----------------------
Creates slide-ready PNG tables for the current regression results and data
dictionary. The styling intentionally mirrors img1.png and img2.png in the
project root.

Usage:
    python3 data_processing/export_paper_tables.py
"""

import os
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_FILE = os.path.join(PROJECT_ROOT, "data", "data_cleaned", "final_panel_master.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLUE = "#0f6fb5"
GRID = "#202020"
TEXT = "#222222"


def prepare_panel():
    df = pd.read_csv(PANEL_FILE, low_memory=False)
    if "debt_to_gdp" in df.columns and "debt_to_av" not in df.columns:
        df = df.rename(columns={"debt_to_gdp": "debt_to_av"})

    df["issue_date"] = pd.to_datetime(df["issue_date"])
    df["year"] = df["issue_date"].dt.year
    df["median_income_10k"] = df["median_income"] / 10_000
    df["ever_champ"] = df["ever_champ"].fillna(0).astype(int)

    support = df.groupby("muni_name")["is_post_2023"].agg(
        pre=lambda s: int((s == 0).sum()),
        post=lambda s: int((s == 1).sum()),
    )
    both_period_munis = support[(support["pre"] > 0) & (support["post"] > 0)].index
    df = df[df["muni_name"].isin(both_period_munis)].copy()

    low, high = df["spread_bps"].quantile([0.01, 0.99])
    df["spread_bps_winsor"] = df["spread_bps"].clip(lower=low, upper=high)
    return df, low, high


def fit_models(df):
    formulas = {
        "A": (
            "spread_bps_winsor ~ ever_champ + slr_exposure_pct"
            " + debt_to_av + median_income_10k + time_to_maturity"
            " + is_post_2023"
        ),
        "B": (
            "spread_bps_winsor ~ ever_champ:is_post_2023"
            " + time_to_maturity + C(muni_name) + C(year)"
        ),
        "C": (
            "spread_bps_winsor ~ ever_champ:is_post_2023"
            " + is_post_2023:slr_exposure_pct"
            " + ever_champ:is_post_2023:slr_exposure_pct"
            " + time_to_maturity + C(muni_name) + C(year)"
        ),
        "D": (
            "spread_bps_winsor ~ ever_champ:C(year)"
            " + debt_to_av + time_to_maturity + C(muni_name) + C(year)"
        ),
    }
    return {
        name: smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["muni_name"]}
        )
        for name, formula in formulas.items()
    }


def stars(p_value):
    if p_value < 0.01:
        return " ***"
    if p_value < 0.05:
        return " **"
    if p_value < 0.1:
        return " *"
    return ""


def coefficient(model, term):
    if term not in model.params:
        return ""
    return f"{model.params[term]:+.2f}{stars(model.pvalues[term])}"


def wrap_rows(rows, widths):
    wrapped = []
    for row in rows:
        wrapped.append(
            [
                textwrap.fill(str(value), width=width, break_long_words=False)
                if value not in ("", "N/A")
                else str(value)
                for value, width in zip(row, widths)
            ]
        )
    return wrapped


def draw_table(
    title,
    columns,
    rows,
    output_path,
    widths,
    font_size=24,
    title_size=58,
    row_height=0.105,
    wrap_widths=None,
):
    if wrap_widths:
        rows = wrap_rows(rows, wrap_widths)
        columns = [
            textwrap.fill(str(column), width=width, break_long_words=False)
            for column, width in zip(columns, wrap_widths)
        ]

    fig_height = max(8.0, 1.3 + row_height * len(rows) * 10)
    fig, ax = plt.subplots(figsize=(18, fig_height), dpi=200)
    ax.axis("off")

    top = 0.86
    if title and title_size > 0:
        ax.text(
            0.5,
            0.96,
            title,
            ha="center",
            va="top",
            fontsize=title_size,
            fontweight="bold",
            transform=ax.transAxes,
        )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colLoc="left",
        cellLoc="left",
        colWidths=widths,
        bbox=[0.02, 0.035, 0.96, top - 0.035],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(1.4)
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(0.09)
        else:
            cell.set_facecolor("white")
            cell.get_text().set_color(TEXT)
            if c == 0:
                cell.get_text().set_color(BLUE)
                cell.get_text().set_fontweight("bold")

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_regression_table(models, df, low, high):
    rows = [
        [
            "CHAMP Cohort\n(ever_champ)",
            coefficient(models["A"], "ever_champ"),
            "Absorbed by FE",
            "Absorbed by FE",
            "Event terms",
        ],
        [
            "Post-2023",
            coefficient(models["A"], "is_post_2023"),
            "Absorbed by year FE",
            "Absorbed by year FE",
            "Absorbed by year FE",
        ],
        [
            "SLR Exposure",
            coefficient(models["A"], "slr_exposure_pct"),
            "Absorbed by muni FE",
            "Absorbed by muni FE",
            "Absorbed by muni FE",
        ],
        [
            "Policy Premium\n(CHAMP x Post-2023)",
            "N/A",
            coefficient(models["B"], "ever_champ:is_post_2023"),
            coefficient(models["C"], "ever_champ:is_post_2023"),
            "N/A",
        ],
        [
            "Exposure Premium\n(SLR x Post-2023)",
            "N/A",
            "N/A",
            coefficient(models["C"], "is_post_2023:slr_exposure_pct"),
            "N/A",
        ],
        [
            "Coastal Resilience\n(CHAMP x Post x SLR)",
            "N/A",
            "N/A",
            coefficient(models["C"], "ever_champ:is_post_2023:slr_exposure_pct"),
            "N/A",
        ],
        [
            "Debt-to-AV Ratio",
            coefficient(models["A"], "debt_to_av"),
            "Excluded",
            "Excluded",
            coefficient(models["D"], "debt_to_av"),
        ],
        [
            "Median Income ($10k)",
            coefficient(models["A"], "median_income_10k"),
            "Excluded",
            "Excluded",
            "Excluded",
        ],
        [
            "Time to Maturity",
            coefficient(models["A"], "time_to_maturity"),
            coefficient(models["B"], "time_to_maturity"),
            coefficient(models["C"], "time_to_maturity"),
            coefficient(models["D"], "time_to_maturity"),
        ],
        [
            "Municipality FE",
            "No",
            "Yes",
            "Yes",
            "Yes",
        ],
        [
            "Year FE",
            "No",
            "Yes",
            "Yes",
            "Yes",
        ],
        [
            "N",
            f"{int(models['A'].nobs):,}",
            f"{int(models['B'].nobs):,}",
            f"{int(models['C'].nobs):,}",
            f"{int(models['D'].nobs):,}",
        ],
        [
            "R-squared",
            f"{models['A'].rsquared:.3f}",
            f"{models['B'].rsquared:.3f}",
            f"{models['C'].rsquared:.3f}",
            f"{models['D'].rsquared:.3f}",
        ],
    ]

    output_path = os.path.join(OUTPUT_DIR, "regression_results_current.png")
    draw_table(
        "Regression Results",
        [
            "Variable",
            "(A) Baseline\nPooled OLS",
            "(B) Policy Pivot\nTWFE DiD",
            "(C) Exposure Impact\nTWFE DDD",
            "(D) Pre-Trend\nEvent Study",
        ],
        rows,
        output_path,
        widths=[0.32, 0.17, 0.17, 0.17, 0.17],
        font_size=15,
        title_size=50,
        row_height=0.12,
        wrap_widths=[24, 16, 16, 16, 16],
    )
    return output_path


def export_data_dictionary(df, low, high):
    rows = [
        [
            "spread_bps_winsor",
            f"Winsorized secondary-trade spread, p1={low:.1f} to p99={high:.1f} bps; dependent variable",
            "WRDS + synthetic AAA proxy",
        ],
        [
            "spread_bps",
            "Raw secondary-trade yield spread over synthetic AAA benchmark, in basis points",
            "WRDS + synthetic AAA proxy",
        ],
        [
            "slr_exposure_pct",
            "% of municipal market value exposed under the 4-foot SLR scenario",
            "NJ parcel/SLR exposure join",
        ],
        [
            "ever_champ",
            "1 if municipality appears in available parsed CHAMP applicant PDFs",
            "NJ CHAMP PDFs",
        ],
        [
            "is_post_2023",
            "1 if trade date is on or after 2023-01-01, the CHAMP period cutoff",
            "Engineered from trade date",
        ],
        [
            "time_to_maturity",
            "Years from trade date to bond maturity",
            "WRDS",
        ],
        [
            "debt_to_av",
            "Municipal debt-to-assessed-value ratio; excluded from Models B and C",
            "UFB / municipal finance",
        ],
        [
            "median_income_10k",
            "Median household income divided by $10,000",
            "US Census ACS",
        ],
        [
            "crs_resilient",
            "Preliminary CRS-derived resilience flag retained for context, not in main regressions",
            "FEMA CRS",
        ],
        [
            "ms4_outfall_density",
            "Stormwater outfalls per square mile; retained for context, not in main regressions",
            "NJ MS4 processing",
        ],
    ]

    output_path = os.path.join(OUTPUT_DIR, "data_dictionary_current.png")
    draw_table(
        "Data Used",
        ["Variable", "Description", "Source"],
        rows,
        output_path,
        widths=[0.21, 0.57, 0.22],
        font_size=16,
        title_size=0,
        row_height=0.105,
        wrap_widths=[20, 58, 24],
    )
    return output_path


def main():
    df, low, high = prepare_panel()
    models = fit_models(df)
    reg_path = export_regression_table(models, df, low, high)
    dict_path = export_data_dictionary(df, low, high)
    print(f"Saved {os.path.relpath(reg_path, PROJECT_ROOT)}")
    print(f"Saved {os.path.relpath(dict_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
