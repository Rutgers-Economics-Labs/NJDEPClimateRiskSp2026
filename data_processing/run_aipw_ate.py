"""
run_aipw_ate.py
===============
Augmented Inverse Probability Weighting (AIPW) estimator for the
Average Treatment Effect of CHAMP cohort membership on municipal
bond spreads.

Uses the doubly-robust AIPW-DiD framework (Sant'Anna & Zhao 2020):
  1. Collapse trade-level data to municipality-level pre/post averages
  2. Estimate propensity score P(CHAMP=1 | X) via logistic regression
  3. Estimate outcome regression E[ΔSpread | X, CHAMP] via OLS
  4. Combine into AIPW ATE with bootstrap standard errors

This addresses the pre-trend concern from the TWFE event study by:
  - Operating on within-municipality changes (ΔSpread)
  - Reweighting by propensity scores to balance covariates
  - Being doubly robust: consistent if EITHER the propensity score
    OR the outcome model is correctly specified

Usage:
    python3 data_processing/run_aipw_ate.py
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEANED = os.path.join(PROJECT_ROOT, "data", "data_cleaned")
PANEL_FILE   = os.path.join(DATA_CLEANED, "final_panel_master.csv")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BOOTSTRAP = 2000
RANDOM_SEED = 42


def load_and_prepare():
    """Load trade-level panel → municipality-level first differences."""
    df = pd.read_csv(PANEL_FILE, low_memory=False)
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    df['year'] = df['issue_date'].dt.year
    if "debt_to_gdp" in df.columns and "debt_to_av" not in df.columns:
        df = df.rename(columns={"debt_to_gdp": "debt_to_av"})

    df["ever_champ"] = df["ever_champ"].fillna(0).astype(int)

    # Winsorize at 1/99 (same as DiD script)
    low, high = df["spread_bps"].quantile([0.01, 0.99])
    df["spread_bps_w"] = df["spread_bps"].clip(lower=low, upper=high)

    # Require both-period municipalities
    support = df.groupby('muni_name')['is_post_2022'].agg(
        pre=lambda s: (s == 0).sum(), post=lambda s: (s == 1).sum()
    )
    both = support[(support['pre'] > 0) & (support['post'] > 0)].index
    df = df[df['muni_name'].isin(both)].copy()

    # ── Collapse to municipality level ──────────────────────────────────
    pre = df[df['is_post_2022'] == 0].groupby('muni_name').agg(
        spread_pre   = ('spread_bps_w', 'mean'),
        ttm_pre      = ('time_to_maturity', 'mean'),
        n_trades_pre = ('spread_bps_w', 'count'),
    )
    post = df[df['is_post_2022'] == 1].groupby('muni_name').agg(
        spread_post   = ('spread_bps_w', 'mean'),
        ttm_post      = ('time_to_maturity', 'mean'),
        n_trades_post = ('spread_bps_w', 'count'),
    )

    # Municipality-level covariates (time-invariant or pre-period)
    muni_chars = df.groupby('muni_name').agg(
        ever_champ      = ('ever_champ', 'first'),
        slr_exposure_pct= ('slr_exposure_pct', 'first'),
        debt_to_av      = ('debt_to_av', 'first'),
        median_income   = ('median_income', 'first'),
    )

    muni = pre.join(post).join(muni_chars).dropna()
    muni['delta_spread'] = muni['spread_post'] - muni['spread_pre']
    muni['delta_ttm']    = muni['ttm_post'] - muni['ttm_pre']
    muni['median_income_10k'] = muni['median_income'] / 10_000
    muni['log_n_trades_pre'] = np.log1p(muni['n_trades_pre'])

    return muni, df


def estimate_propensity(muni, covariates):
    """Logistic regression propensity score with diagnostics."""
    X = muni[covariates].values
    D = muni['ever_champ'].values

    # Standardize for numerical stability
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1
    X_z = (X - X_mean) / X_std

    lr = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs',
        max_iter=5000, random_state=RANDOM_SEED
    )
    lr.fit(X_z, D)
    p_score = lr.predict_proba(X_z)[:, 1]

    # Clip for stability (avoid extreme weights)
    p_score = np.clip(p_score, 0.01, 0.99)

    return p_score, lr, X_mean, X_std


def outcome_regression(muni, covariates):
    """OLS outcome models for treated and control groups separately."""
    X = muni[covariates].values
    Y = muni['delta_spread'].values
    D = muni['ever_champ'].values

    # Fit separate outcome models
    from sklearn.linear_model import LinearRegression

    # μ₁(x) = E[ΔY | X=x, D=1]
    m1 = LinearRegression().fit(X[D == 1], Y[D == 1])
    mu1_all = m1.predict(X)  # predict for everyone

    # μ₀(x) = E[ΔY | X=x, D=0]
    m0 = LinearRegression().fit(X[D == 0], Y[D == 0])
    mu0_all = m0.predict(X)

    return mu1_all, mu0_all, m1, m0


def aipw_ate(Y, D, p_score, mu1, mu0):
    """
    Doubly-robust AIPW ATE estimator.

    τ_AIPW = (1/N) Σ [ D*(Y - μ₁(X))/p(X) - (1-D)*(Y - μ₀(X))/(1-p(X))
                        + μ₁(X) - μ₀(X) ]
    """
    n = len(Y)
    # Influence function components
    treated_term = D * (Y - mu1) / p_score
    control_term = (1 - D) * (Y - mu0) / (1 - p_score)
    regression_term = mu1 - mu0

    psi = treated_term - control_term + regression_term
    tau = psi.mean()
    return tau, psi


def bootstrap_inference(muni, covariates, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Non-parametric bootstrap for AIPW ATE standard errors and CI."""
    rng = np.random.RandomState(seed)
    n = len(muni)
    boot_taus = []
    ipw_taus = []
    reg_taus = []

    Y = muni['delta_spread'].values
    D = muni['ever_champ'].values

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        muni_b = muni.iloc[idx].copy()

        try:
            p_b, _, _, _ = estimate_propensity(muni_b, covariates)
            mu1_b, mu0_b, _, _ = outcome_regression(muni_b, covariates)

            Y_b = muni_b['delta_spread'].values
            D_b = muni_b['ever_champ'].values

            # AIPW
            tau_b, _ = aipw_ate(Y_b, D_b, p_b, mu1_b, mu0_b)
            boot_taus.append(tau_b)

            # IPW only (for comparison)
            ipw_tau = (D_b * Y_b / p_b).sum() / (D_b / p_b).sum() - \
                      ((1-D_b) * Y_b / (1-p_b)).sum() / ((1-D_b) / (1-p_b)).sum()
            ipw_taus.append(ipw_tau)

            # Regression only
            reg_tau = (mu1_b - mu0_b).mean()
            reg_taus.append(reg_tau)

        except Exception:
            continue

    return np.array(boot_taus), np.array(ipw_taus), np.array(reg_taus)


def covariate_balance(muni, covariates, p_score):
    """Check covariate balance: raw and IPW-weighted standardized differences."""
    D = muni['ever_champ'].values
    X = muni[covariates].values
    n1, n0 = D.sum(), (1-D).sum()

    # IPW weights
    w1 = 1.0 / p_score       # for treated
    w0 = 1.0 / (1 - p_score) # for control

    rows = []
    for j, var in enumerate(covariates):
        x = X[:, j]
        # Raw
        raw_diff = x[D == 1].mean() - x[D == 0].mean()
        pooled_sd = np.sqrt((x[D == 1].var() + x[D == 0].var()) / 2)
        raw_std_diff = raw_diff / pooled_sd if pooled_sd > 0 else 0

        # IPW weighted
        wm1 = np.average(x[D == 1], weights=w1[D == 1])
        wm0 = np.average(x[D == 0], weights=w0[D == 0])
        ipw_diff = wm1 - wm0
        ipw_std_diff = ipw_diff / pooled_sd if pooled_sd > 0 else 0

        rows.append({
            'Variable': var,
            'Mean (CHAMP)': x[D == 1].mean(),
            'Mean (Non-CHAMP)': x[D == 0].mean(),
            'Raw Std Diff': raw_std_diff,
            'IPW Std Diff': ipw_std_diff,
        })

    return pd.DataFrame(rows)


def run_aipw():
    print("=" * 70)
    print("AIPW-DiD: Doubly-Robust ATE of CHAMP on Municipal Bond Spreads")
    print("=" * 70)

    muni, trade_df = load_and_prepare()

    n_champ = muni['ever_champ'].sum()
    n_ctrl  = len(muni) - n_champ
    print(f"\n  Municipality panel: {len(muni)} towns "
          f"({n_champ} CHAMP, {n_ctrl} non-CHAMP)")
    print(f"  ΔSpread (CHAMP):     {muni.loc[muni['ever_champ']==1, 'delta_spread'].mean():+.2f} bps")
    print(f"  ΔSpread (non-CHAMP): {muni.loc[muni['ever_champ']==0, 'delta_spread'].mean():+.2f} bps")
    print(f"  Naive DiM:           {muni.loc[muni['ever_champ']==1, 'delta_spread'].mean() - muni.loc[muni['ever_champ']==0, 'delta_spread'].mean():+.2f} bps")

    # ── Covariates for propensity + outcome models ─────────────────────
    # Parsimonious: only 4 covariates for 15 treated units
    covariates = ['slr_exposure_pct', 'debt_to_av', 'median_income_10k', 'spread_pre']

    print(f"\n  Covariates: {covariates}")

    # ── Step 1: Propensity Score ───────────────────────────────────────
    print("\n" + "-" * 70)
    print("STEP 1: Propensity Score Estimation (Logistic Regression)")
    print("-" * 70)

    p_score, lr_model, X_mean, X_std = estimate_propensity(muni, covariates)
    muni['p_score'] = p_score

    print(f"\n  Propensity score distribution:")
    print(f"    CHAMP towns:     min={p_score[muni['ever_champ']==1].min():.4f}  "
          f"mean={p_score[muni['ever_champ']==1].mean():.4f}  "
          f"max={p_score[muni['ever_champ']==1].max():.4f}")
    print(f"    Non-CHAMP towns: min={p_score[muni['ever_champ']==0].min():.4f}  "
          f"mean={p_score[muni['ever_champ']==0].mean():.4f}  "
          f"max={p_score[muni['ever_champ']==0].max():.4f}")

    # Overlap diagnostic
    overlap_min = max(p_score[muni['ever_champ']==1].min(),
                      p_score[muni['ever_champ']==0].min())
    overlap_max = min(p_score[muni['ever_champ']==1].max(),
                      p_score[muni['ever_champ']==0].max())
    n_overlap = ((p_score >= overlap_min) & (p_score <= overlap_max)).sum()
    print(f"\n  Overlap region: [{overlap_min:.4f}, {overlap_max:.4f}]")
    print(f"  Towns in overlap: {n_overlap}/{len(muni)}")

    # Effective sample size
    D = muni['ever_champ'].values
    ess_treated = (D / p_score).sum()**2 / (D / p_score**2).sum()
    ess_control = ((1-D) / (1-p_score)).sum()**2 / ((1-D) / (1-p_score)**2).sum()
    print(f"  Effective sample size: treated={ess_treated:.1f}, control={ess_control:.1f}")

    # ── Covariate Balance ──────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("COVARIATE BALANCE (Raw vs IPW-Weighted)")
    print("-" * 70)
    balance = covariate_balance(muni, covariates, p_score)
    print(balance.to_string(index=False, float_format='{:.4f}'.format))
    print("\n  Rule of thumb: |Std Diff| < 0.10 is good balance")

    # ── Step 2: Outcome Regression ─────────────────────────────────────
    print("\n" + "-" * 70)
    print("STEP 2: Outcome Regression E[ΔSpread | X, CHAMP]")
    print("-" * 70)

    mu1_all, mu0_all, m1, m0 = outcome_regression(muni, covariates)
    Y = muni['delta_spread'].values

    print(f"  Outcome model R² (treated):  {m1.score(muni[covariates].values[D==1], Y[D==1]):.4f}")
    print(f"  Outcome model R² (control):  {m0.score(muni[covariates].values[D==0], Y[D==0]):.4f}")
    print(f"  Regression-only ATE:         {(mu1_all - mu0_all).mean():+.2f} bps")

    # ── Step 3: AIPW Estimator ─────────────────────────────────────────
    print("\n" + "-" * 70)
    print("STEP 3: AIPW ATE (Doubly Robust)")
    print("-" * 70)

    tau_aipw, psi = aipw_ate(Y, D, p_score, mu1_all, mu0_all)

    # Analytic SE from influence function
    se_analytic = np.sqrt(np.var(psi, ddof=1) / len(psi))
    z_analytic = tau_aipw / se_analytic
    p_analytic = 2 * (1 - stats.norm.cdf(abs(z_analytic)))

    print(f"\n  AIPW ATE (point estimate): {tau_aipw:+.3f} bps")
    print(f"  Analytic SE:              {se_analytic:.3f}")
    print(f"  z-stat:                   {z_analytic:.3f}")
    print(f"  p-value:                  {p_analytic:.4f}")
    print(f"  95% CI:                   [{tau_aipw - 1.96*se_analytic:+.2f}, {tau_aipw + 1.96*se_analytic:+.2f}]")

    # ── Step 4: Bootstrap Inference ────────────────────────────────────
    print(f"\n  Running {N_BOOTSTRAP} bootstrap replications...")
    boot_aipw, boot_ipw, boot_reg = bootstrap_inference(muni, covariates)

    se_boot = boot_aipw.std()
    ci_low, ci_high = np.percentile(boot_aipw, [2.5, 97.5])
    p_boot = 2 * min(
        (boot_aipw >= 0).mean(),
        (boot_aipw <= 0).mean()
    )

    # IPW-only for comparison
    ipw_ate = np.median(boot_ipw)
    ipw_se = boot_ipw.std()
    ipw_ci_low, ipw_ci_high = np.percentile(boot_ipw, [2.5, 97.5])

    # Regression-only
    reg_ate = np.median(boot_reg)
    reg_se = boot_reg.std()
    reg_ci_low, reg_ci_high = np.percentile(boot_reg, [2.5, 97.5])

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Estimator':<25} {'ATE (bps)':>12} {'SE':>10} {'95% CI':>25} {'p-value':>10}")
    print("-" * 82)

    # Naive DiM
    dim = muni.loc[D==1, 'delta_spread'].mean() - muni.loc[D==0, 'delta_spread'].mean()
    dim_se_t = muni.loc[D==1, 'delta_spread'].std() / np.sqrt(n_champ)
    dim_se_c = muni.loc[D==0, 'delta_spread'].std() / np.sqrt(n_ctrl)
    dim_se = np.sqrt(dim_se_t**2 + dim_se_c**2)
    dim_p = 2 * (1 - stats.norm.cdf(abs(dim / dim_se)))
    print(f"{'Naive DiM':<25} {dim:>+12.2f} {dim_se:>10.2f} {'['+f'{dim-1.96*dim_se:+.2f}, {dim+1.96*dim_se:+.2f}'+']':>25} {dim_p:>10.4f}")

    print(f"{'IPW (bootstrap)':<25} {ipw_ate:>+12.2f} {ipw_se:>10.2f} {'['+f'{ipw_ci_low:+.2f}, {ipw_ci_high:+.2f}'+']':>25} {'':>10}")
    print(f"{'Regression (bootstrap)':<25} {reg_ate:>+12.2f} {reg_se:>10.2f} {'['+f'{reg_ci_low:+.2f}, {reg_ci_high:+.2f}'+']':>25} {'':>10}")
    print(f"{'AIPW (analytic)':<25} {tau_aipw:>+12.2f} {se_analytic:>10.2f} {'['+f'{tau_aipw-1.96*se_analytic:+.2f}, {tau_aipw+1.96*se_analytic:+.2f}'+']':>25} {p_analytic:>10.4f}")
    print(f"{'AIPW (bootstrap)':<25} {tau_aipw:>+12.2f} {se_boot:>10.2f} {'['+f'{ci_low:+.2f}, {ci_high:+.2f}'+']':>25} {p_boot:>10.4f}")
    print(f"{'TWFE DiD (Model B)':<25} {'-32.62':>12} {'7.40':>10} {'[-47.13, -18.12]':>25} {'<0.001':>10}")

    # ── Sensitivity: trimmed propensity scores ─────────────────────────
    print("\n" + "-" * 70)
    print("SENSITIVITY: Trimmed Propensity Score (drop extreme weights)")
    print("-" * 70)
    for trim in [0.05, 0.10, 0.15]:
        mask = (p_score >= trim) & (p_score <= 1 - trim)
        if mask.sum() < 10 or D[mask].sum() < 3:
            print(f"  Trim [{trim:.2f}, {1-trim:.2f}]: too few obs after trimming")
            continue
        muni_trim = muni[mask].copy()
        p_trim = p_score[mask]
        mu1_t = mu1_all[mask]
        mu0_t = mu0_all[mask]
        Y_t = Y[mask]
        D_t = D[mask]
        tau_t, _ = aipw_ate(Y_t, D_t, p_trim, mu1_t, mu0_t)
        print(f"  Trim [{trim:.2f}, {1-trim:.2f}]: N={mask.sum()}, "
              f"N_champ={D_t.sum()}, ATE={tau_t:+.2f} bps")

    # ── Can we make causal claims? ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("CAUSAL INFERENCE ASSESSMENT")
    print("=" * 70)

    # Assess overlap
    max_weight = max(1/p_score[D==1].min(), 1/(1-p_score[D==0].min()))
    overlap_ok = overlap_max > overlap_min and max_weight < 20

    # Assess balance
    max_raw_imbal = balance['Raw Std Diff'].abs().max()
    max_ipw_imbal = balance['IPW Std Diff'].abs().max()
    balance_ok = max_ipw_imbal < 0.25  # relaxed for small sample

    # Assess significance
    sig_ok = p_boot < 0.05

    # Assess consistency across estimators
    estimates = [tau_aipw, ipw_ate, reg_ate, dim]
    sign_consistent = all(e < 0 for e in estimates) or all(e > 0 for e in estimates)

    print(f"\n  ✓ Overlap:           {'PASS' if overlap_ok else 'CONCERN'} (max weight = {max_weight:.1f})")
    print(f"  ✓ Covariate balance: {'PASS' if balance_ok else 'CONCERN'} (max IPW std diff = {max_ipw_imbal:.3f})")
    print(f"  ✓ Significance:      {'PASS' if sig_ok else 'FAIL'} (bootstrap p = {p_boot:.4f})")
    print(f"  ✓ Sign consistency:  {'PASS' if sign_consistent else 'MIXED'} (all estimators same sign)")
    print(f"  ✓ Small sample:      CONCERN (only {n_champ} treated units)")

    if sig_ok and sign_consistent and overlap_ok:
        print("\n  VERDICT: The AIPW estimate supports a causal interpretation that")
        print(f"  CHAMP cohort municipalities experienced a {tau_aipw:+.1f} bps ATE on")
        print(f"  spread changes, with doubly-robust 95% CI [{ci_low:+.1f}, {ci_high:+.1f}].")
        print("  However, with only 15 treated units, external validity is limited.")
    else:
        reasons = []
        if not sig_ok:
            reasons.append("statistical insignificance")
        if not overlap_ok:
            reasons.append("poor propensity score overlap")
        if not sign_consistent:
            reasons.append("inconsistent signs across estimators")
        print(f"\n  VERDICT: Causal claims are weakened by: {', '.join(reasons)}.")
        print("  Results are suggestive but not conclusive.")

    # ── Save results ───────────────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, "aipw_results.txt")
    with open(results_path, 'w') as f:
        f.write("AIPW-DiD: Doubly-Robust ATE of CHAMP on Municipal Bond Spreads\n")
        f.write("=" * 70 + "\n\n")
        f.write("Framework: Sant'Anna & Zhao (2020) style AIPW-DiD\n")
        f.write("Treatment: ever_champ (binary, municipality-level)\n")
        f.write("Outcome: ΔSpread = mean(spread_post) - mean(spread_pre)\n")
        f.write(f"Covariates: {covariates}\n")
        f.write(f"Panel: {len(muni)} municipalities ({n_champ} CHAMP, {n_ctrl} non-CHAMP)\n")
        f.write(f"Bootstrap replications: {N_BOOTSTRAP}\n\n")

        f.write("-" * 82 + "\n")
        f.write(f"{'Estimator':<25} {'ATE (bps)':>12} {'SE':>10} {'95% CI':>25} {'p-value':>10}\n")
        f.write("-" * 82 + "\n")
        f.write(f"{'Naive DiM':<25} {dim:>+12.2f} {dim_se:>10.2f} {'['+f'{dim-1.96*dim_se:+.2f}, {dim+1.96*dim_se:+.2f}'+']':>25} {dim_p:>10.4f}\n")
        f.write(f"{'IPW (bootstrap)':<25} {ipw_ate:>+12.2f} {ipw_se:>10.2f} {'['+f'{ipw_ci_low:+.2f}, {ipw_ci_high:+.2f}'+']':>25}\n")
        f.write(f"{'Regression (bootstrap)':<25} {reg_ate:>+12.2f} {reg_se:>10.2f} {'['+f'{reg_ci_low:+.2f}, {reg_ci_high:+.2f}'+']':>25}\n")
        f.write(f"{'AIPW (analytic)':<25} {tau_aipw:>+12.2f} {se_analytic:>10.2f} {'['+f'{tau_aipw-1.96*se_analytic:+.2f}, {tau_aipw+1.96*se_analytic:+.2f}'+']':>25} {p_analytic:>10.4f}\n")
        f.write(f"{'AIPW (bootstrap)':<25} {tau_aipw:>+12.2f} {se_boot:>10.2f} {'['+f'{ci_low:+.2f}, {ci_high:+.2f}'+']':>25} {p_boot:>10.4f}\n")
        f.write(f"{'TWFE DiD (Model B)':<25} {'-32.62':>12} {'7.40':>10} {'[-47.13, -18.12]':>25} {'<0.001':>10}\n")
        f.write("-" * 82 + "\n")

        f.write("\n\nCOVARIATE BALANCE\n")
        f.write(balance.to_string(index=False, float_format='{:.4f}'.format))

        f.write("\n\nPROPENSITY SCORE DIAGNOSTICS\n")
        f.write(f"  CHAMP:     [{p_score[D==1].min():.4f}, {p_score[D==1].max():.4f}]\n")
        f.write(f"  Non-CHAMP: [{p_score[D==0].min():.4f}, {p_score[D==0].max():.4f}]\n")
        f.write(f"  Overlap:   [{overlap_min:.4f}, {overlap_max:.4f}]\n")
        f.write(f"  ESS treated: {ess_treated:.1f}, ESS control: {ess_control:.1f}\n")

        f.write(f"\n\nDOUBLY-ROBUST PROPERTY\n")
        f.write("The AIPW estimator is consistent if EITHER:\n")
        f.write("  (a) The propensity score model is correctly specified, OR\n")
        f.write("  (b) The outcome regression model is correctly specified.\n")
        f.write("This provides insurance against misspecification of either model.\n")

        f.write(f"\n\nCAUSAL INTERPRETATION\n")
        f.write(f"  Overlap:           {'PASS' if overlap_ok else 'CONCERN'}\n")
        f.write(f"  Covariate balance: {'PASS' if balance_ok else 'CONCERN'}\n")
        f.write(f"  Significance:      {'PASS' if sig_ok else 'FAIL'}\n")
        f.write(f"  Sign consistency:  {'PASS' if sign_consistent else 'MIXED'}\n")
        f.write(f"  Small sample:      CONCERN ({n_champ} treated units)\n")

    print(f"\n  Results saved → {results_path}")


if __name__ == '__main__':
    run_aipw()
