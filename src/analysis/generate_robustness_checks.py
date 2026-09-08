"""
Robustness checks for the Section 4.3 regression results (Table 1).

WHY THIS SCRIPT EXISTS
-----------------------
generate_regression_table.py fits three OLS specifications with classical
(homoskedastic, independent-errors) standard errors. Section 4.2 of the paper
independently documents strong, statistically significant positive spatial
autocorrelation in both the outcome and the key regressor (Moran's I = 0.5959
for price, 0.5797 for dem_share, both p = 0.001 — see
results/morans_i_results.json). Positive spatial autocorrelation in the
residuals of a model that does not explicitly account for it means the
classical OLS standard errors in Table 1 are very likely understated, which
overstates the statistical significance of the dem_share coefficient. This
script does not change Table 1; it produces the supplementary checks needed
to show the headline result is not an artifact of that violation, and to show
it survives more granular fixed effects and outlier treatment than Model 3
currently uses.

WHAT IT PRODUCES
-----------------
results/robustness_checks.json — machine-readable, all four checks below.
tables/table_robustness.tex     — a LaTeX table formatted to match the style
                                   of table1_ols_regression.tex, ready to drop
                                   into a new "Robustness" subsection.

CHECKS
------
1. Cluster-robust standard errors on Model 3 (paper's own control set +
   borough FE), clustered by CounDist (51 clusters) — the natural clustering
   unit given Moran's I was computed at the council-district level.
2. Model 4: replace the 5 borough dummies with 51 council-district dummies
   (CounDist FE) — a much more granular within-district comparison; also
   reported with district-clustered SEs.
3. Model 5: Model 3 + sale-year fixed effects (sale_year: 2017/2021/2025 in
   this canonical build) — the paper's own "temporal ideology shift" figure
   shows dem_share moves over time, so a pooled cross-section without year FE
   risks conflating within-district ideological change with citywide price
   appreciation over the same years.
4. Outlier sensitivity: Model 3 re-estimated after (a) winsorizing
   target_log_price at the 1st/99th percentile and (b) dropping observations
   with Cook's distance > 4/n, to confirm the dem_share coefficient's sign,
   magnitude, and significance are not driven by a small number of extreme
   sales.

HOW TO RUN
----------
    python src/analysis/generate_robustness_checks.py

Runtime: under a minute (OLS on ~470k rows x up to 56 columns; Cook's
distance computed via the projection-matrix diagonal, not
statsmodels' get_influence(), which is infeasible at this n).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "results" / "robustness_checks.json"
OUT_TEX = ROOT / "tables" / "table_robustness.tex"

CONTROLS = ["yearbuilt", "numfloors", "landuse", "commfar"]
REQUIRED = ["target_log_price", "dem_share", "BOROUGH", "CounDist", "sale_year"] + CONTROLS


def load_data():
    df = pd.read_parquet(ROOT / "data" / "canonical" / "modeling_dataset_canonical_v2.parquet")
    df = df.dropna(subset=REQUIRED)
    return df


def fit_model3(df):
    """Reproduce Model 3 from generate_regression_table.py exactly (controls + borough FE)."""
    d = pd.get_dummies(df, columns=["BOROUGH"], drop_first=True, dtype=float)
    boro_cols = [c for c in d.columns if c.startswith("BOROUGH_")]
    X = sm.add_constant(d[["dem_share"] + CONTROLS + boro_cols])
    y = df["target_log_price"]
    return sm.OLS(y, X).fit(), X, y, d


def summarize(fit, label, coef="dem_share"):
    return dict(
        label=label,
        n=int(fit.nobs),
        dem_share_coef=float(fit.params[coef]),
        dem_share_se=float(fit.bse[coef]),
        dem_share_p=float(fit.pvalues[coef]),
        r_squared=float(fit.rsquared),
    )


def main():
    df = load_data()
    results = {"n_obs_used": int(len(df))}

    # ---- Baseline Model 3 (classical SE), for comparison -----------------
    m3, X3, y3, d3 = fit_model3(df)
    results["model3_classical_se"] = summarize(m3, "Model 3 (paper) — classical SE")

    # ---- Check 1: cluster-robust SE on Model 3, clustered by CounDist ----
    m3_cl = sm.OLS(y3, X3).fit(cov_type="cluster", cov_kwds={"groups": df["CounDist"]})
    results["check1_model3_clustered_by_council_district"] = summarize(
        m3_cl, "Model 3 — SE clustered by CounDist (51 clusters)"
    )
    results["check1_n_clusters"] = int(df["CounDist"].nunique())

    # ---- Check 2: council-district FE instead of borough FE --------------
    d4 = pd.get_dummies(df, columns=["CounDist"], drop_first=True, dtype=float)
    cd_cols = [c for c in d4.columns if c.startswith("CounDist_")]
    X4 = sm.add_constant(d4[["dem_share"] + CONTROLS + cd_cols])
    m4 = sm.OLS(y3, X4).fit()
    results["check2_model4_council_district_fe_classical_se"] = summarize(
        m4, "Model 4 — 51 council-district FE, classical SE"
    )
    m4_cl = sm.OLS(y3, X4).fit(cov_type="cluster", cov_kwds={"groups": df["CounDist"]})
    results["check2_model4_council_district_fe_clustered_se"] = summarize(
        m4_cl, "Model 4 — 51 council-district FE, SE clustered by CounDist"
    )

    # ---- Check 3: Model 3 + sale-year FE ----------------------------------
    d5 = pd.get_dummies(df, columns=["BOROUGH", "sale_year"], drop_first=True, dtype=float)
    extra_cols = [c for c in d5.columns if c.startswith("BOROUGH_") or c.startswith("sale_year_")]
    X5 = sm.add_constant(d5[["dem_share"] + CONTROLS + extra_cols])
    m5 = sm.OLS(y3, X5).fit()
    results["check3_model5_borough_and_year_fe"] = summarize(
        m5, "Model 5 — Model 3 + sale-year FE, classical SE"
    )
    m5_cl = sm.OLS(y3, X5).fit(cov_type="cluster", cov_kwds={"groups": df["CounDist"]})
    results["check3_model5_borough_and_year_fe_clustered_se"] = summarize(
        m5_cl, "Model 5 — Model 3 + sale-year FE, SE clustered by CounDist"
    )
    results["check3_sale_years_present"] = sorted(df["sale_year"].dropna().unique().tolist())

    # ---- Check 4a: winsorized outcome -------------------------------------
    lo, hi = df["target_log_price"].quantile([0.01, 0.99])
    y_wz = df["target_log_price"].clip(lo, hi)
    m3w = sm.OLS(y_wz, X3).fit()
    results["check4a_winsorized_1_99pct"] = summarize(m3w, "Model 3 — target_log_price winsorized at 1st/99th pct")
    results["check4a_winsor_bounds"] = dict(lower=float(lo), upper=float(hi))

    # ---- Check 4b: drop high-leverage points via Cook's distance ---------
    # Exact Cook's distance via the hat-matrix diagonal is O(n^2) and
    # infeasible at n~470k with 56 columns via statsmodels' get_influence().
    # We use the standard closed-form approximation:
    #   leverage h_ii from the diagonal of X(X'X)^-1 X' computed in blocks,
    #   Cook's D_i = (e_i^2 / (p * MSE)) * (h_ii / (1-h_ii)^2)
    Xmat = X3.to_numpy(dtype=float)
    beta = m3.params.to_numpy()
    resid = m3.resid.to_numpy()
    mse = float(np.sum(resid ** 2) / (len(resid) - Xmat.shape[1]))
    XtX_inv = np.linalg.inv(Xmat.T @ Xmat)
    # Block leverage computation to bound memory: h_ii = x_i (X'X)^-1 x_i'
    block = 50_000
    h = np.empty(len(resid))
    for start in range(0, len(resid), block):
        chunk = Xmat[start:start + block]
        h[start:start + block] = np.einsum("ij,jk,ik->i", chunk, XtX_inv, chunk)
    p = Xmat.shape[1]
    cooks_d = (resid ** 2 / (p * mse)) * (h / (1 - h) ** 2)
    threshold = 4.0 / len(resid)
    keep = cooks_d <= threshold
    results["check4b_cooks_distance_threshold"] = float(threshold)
    results["check4b_n_dropped"] = int((~keep).sum())
    results["check4b_pct_dropped"] = float((~keep).sum() / len(resid) * 100)

    X3_trim = X3.loc[keep]
    y3_trim = y3.loc[keep]
    m3_trim = sm.OLS(y3_trim, X3_trim).fit()
    results["check4b_model3_cooks_trimmed"] = summarize(m3_trim, "Model 3 — high-Cook's-D observations dropped")

    # ---- Headline interpretation ------------------------------------------
    coefs = [
        results["model3_classical_se"]["dem_share_coef"],
        results["check1_model3_clustered_by_council_district"]["dem_share_coef"],
        results["check2_model4_council_district_fe_classical_se"]["dem_share_coef"],
        results["check3_model5_borough_and_year_fe"]["dem_share_coef"],
        results["check4a_winsorized_1_99pct"]["dem_share_coef"],
        results["check4b_model3_cooks_trimmed"]["dem_share_coef"],
    ]
    same_sign = all((c > 0) == (coefs[0] > 0) for c in coefs)
    results["headline"] = {
        "dem_share_coef_range": [float(min(coefs)), float(max(coefs))],
        "sign_stable_across_all_checks": bool(same_sign),
        "note": (
            "Clustering by CounDist (check 1) is the check that matters most given the "
            "Moran's I evidence in Section 4.2 — compare its p-value to the classical-SE "
            "p-value in model3_classical_se directly; a material increase in the p-value "
            "under clustering is the expected, honest signature of previously-understated "
            "significance, not a sign of a coding error."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")

    # ---- LaTeX table --------------------------------------------------
    def row(key, label):
        r = results[key]
        stars = "***" if r["dem_share_p"] < 0.001 else "**" if r["dem_share_p"] < 0.01 else "*" if r["dem_share_p"] < 0.05 else ""
        return f"{label} & {r['dem_share_coef']:.4f}{stars} & ({r['dem_share_se']:.4f}) & {r['r_squared']:.4f} & {r['n']:,} \\\\"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Robustness of the dem\_share coefficient across alternative specifications}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Specification & Coef. (dem\_share) & (SE) & $R^2$ & $N$ \\",
        r"\midrule",
        row("model3_classical_se", "Model 3 (paper), classical SE"),
        row("check1_model3_clustered_by_council_district", "Model 3, SE clustered by district"),
        row("check2_model4_council_district_fe_classical_se", "Model 4, 51 district FE"),
        row("check3_model5_borough_and_year_fe", "Model 5, borough + sale-year FE"),
        row("check4a_winsorized_1_99pct", "Model 3, price winsorized 1/99pct"),
        row("check4b_model3_cooks_trimmed", "Model 3, high-Cook's-D dropped"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{0.95\textwidth}",
        r"\footnotesize Standard errors in parentheses. * p$<$0.05, ** p$<$0.01, *** p$<$0.001. "
        r"District-clustered rows use 51 clusters (CounDist). See "
        r"\texttt{results/robustness\_checks.json} for full output.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(tex_lines))
    print(f"Wrote {OUT_TEX.relative_to(ROOT)}")

    print("\n=== HEADLINE ===")
    print(json.dumps(results["headline"], indent=2))


if __name__ == "__main__":
    main()
