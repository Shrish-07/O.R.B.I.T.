
"""City of Yes DiD (Round 6, Task 2) — runs on data already on disk only (no scraping/requests).

TREATMENT FRAMING (declared per Round 6 rules)
  Nothing on record in this repo distinguishes the City of Yes provisions into
  universal (city-wide) vs geography-specific components. Searched:
  src/data_pipeline/build_city_of_yes_roll_calls.py, README.md ("Where the
  paper stands" — condensed from the merged ACTION_PLAN_FOR_SUBMISSION.md),
  and VERIFICATION_REPORT.md (incl. its Appendix A, the merged data-refresh
  report). The only reference to a
  geographically-targeted City of Yes action on file is Economic Opportunity (June
  2024), named in ACTION_PLAN_FOR_SUBMISSION.md as an example of spatially
  targeted policy; its member-level roll call is NOT available in this repo
  (Legistar API unavailable — see build_city_of_yes_roll_calls.py docstring).

  Per the Round-6 brief, the design therefore uses the IDEOLOGY/PREFERENCE-PROXY
  reading: treatment district = the district whose member voted Yes on City of
  Yes for Housing Opportunity (2024-12-05). This is NOT a policy-application DiD
  — Housing Opportunity is a city-wide text amendment; the Yes/No split measures
  a district's revealed pro-/anti-development political stance. Every record and
  the Round-6 report carries this caveat.

DESIGN
  Pre-period: sale years 2016-2023; transition year: 2024 (adopted 2024-12-05,
  almost entirely pre-treatment); post-period: 2025.

  Pre-trend event study (mandatory; reported regardless of outcome):
    target_log_price ~ sum_bins Yes_district x period_bin + CounDist FE +
                       sale_year FE + controls
    pre-period bins 2016-17, 2018-19, 2020-21 against reference bin 2022-23, plus
    separate 2024 and 2025 bins. A joint Wald test on the three pre-period
    interaction terms decides whether pre-trends look parallel; the verdict is
    computed programmatically from the Wald p-value (parallel iff p > 0.05).

  Main DiD: target_log_price ~ Yes_district x Post + CounDist FE + sale_year FE +
            controls (the permanent 18/19-feature spec; Round 7: BBL_pluto
            blacklisted, was 19/20), Post = 1(sale_year >=
            2025), 2024 transition-year rows excluded (documented).
  Sensitivity: same spec with Post = 1(sale_year >= 2024), 2024 rows included.

  Estimator: statsmodels OLS with covariance clustered by council district (51
  clusters) — the conventional DiD clustering at the treatment-assignment level.

Outputs are written regardless of the pre-trend verdict; a failed pre-trend check
is reported, not hidden. The stage exits 0 either way.

Output: results/causal/city_of_yes_did_results.json

Run:
    python src/causal/city_of_yes_did.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "canonical" / "modeling_dataset_canonical_v2.parquet"
ROLL = ROOT / "data" / "processed" / "city_of_yes_roll_calls.parquet"
OUT = ROOT / "results" / "causal" / "city_of_yes_did_results.json"

CONTROLS = ["yearbuilt", "numfloors", "landuse", "commfar",
            "RESIDENTIAL UNITS", "COMMERCIAL UNITS", "TOTAL UNITS"]
REQUIRED = ["target_log_price", "CounDist", "sale_year"] + CONTROLS


TASK0_FINDING = (
    "No record in this repo distinguishes City of Yes provisions into universal "
    "(city-wide) vs geography-specific components. Searched: "
    "build_city_of_yes_roll_calls.py, README.md ('Where the paper stands' — the "
    "merged ACTION_PLAN_FOR_SUBMISSION.md), and VERIFICATION_REPORT.md (incl. its "
    "Appendix A, the merged DATA_REFRESH_IMPACT_REPORT.md). The only "
    "geographically-targeted City of Yes action on record is Economic Opportunity "
    "(June 2024), cited in the merged action plan as an example of "
    "spatially-targeted policy; its roll call is not in this repo (Legistar not "
    "retrievable — see build_city_of_yes_roll_calls.py). Housing Opportunity "
    "(Dec 2024) is documented as a city-wide text amendment. Proceeding under the "
    "ideology/preference-proxy framing per the Round-6 brief."
)

TREATMENT_FRAMING = (
    "ideology/preference-proxy DiD, NOT a policy-application DiD — the underlying "
    "zoning text amendment applies city-wide regardless of how the member voted; "
    "this design tests whether a district's revealed pro-/anti-development "
    "political stance predicts differential price trends, not whether the "
    "specific policy provisions differentially applied"
)

PERIOD_BINS = ["2016-17", "2018-19", "2020-21", "2022-23", "2024", "2025"]
REF_BIN = "2022-23"
PRE_BINS = ["2016-17", "2018-19", "2020-21"]


def bin_for_year(y):
    if y in (2016, 2017):
        return "2016-17"
    if y in (2018, 2019):
        return "2018-19"
    if y in (2020, 2021):
        return "2020-21"
    if y in (2022, 2023):
        return "2022-23"
    if y == 2024:
        return "2024"
    if y == 2025:
        return "2025"
    return None


def prepare(df, roll):
    d = df.copy()
    d = d.merge(roll[["council_district", "member_vote"]],
                left_on="CounDist", right_on="council_district", how="inner")
    d["yes_district"] = (d["member_vote"] == "Yes").astype(float)
    d["period_bin"] = d["sale_year"].astype(int).map(bin_for_year)
    return d


def design_matrix(d, extra_cols):
    d2 = d.copy()
    d2["CounDist"] = d2["CounDist"].astype(int)
    d2["sale_year"] = d2["sale_year"].astype(int)
    dist_fe = pd.get_dummies(d2["CounDist"], prefix="district", drop_first=True, dtype=float)
    year_fe = pd.get_dummies(d2["sale_year"], prefix="sale_year", drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=d2.index, name="const"),
                   d2[CONTROLS].astype(float), d2[extra_cols], dist_fe, year_fe], axis=1)
    return X.astype(float), d2["target_log_price"].astype(float), d2["CounDist"]


def fit_ols(X, y, groups):
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})



def run_event_study(d):
    d = d.copy()
    vins = []
    for b in PERIOD_BINS:
        if b == REF_BIN:
            continue
        name = "yes_x_" + b
        d[name] = d["yes_district"] * (d["period_bin"] == b).astype(float)
        vins.append(name)
    X, y, groups = design_matrix(d, vins)
    res = fit_ols(X, y, groups)
    n_clusters = int(pd.Series(groups).nunique())
    pre_terms = [c for c in vins if c not in ("yes_x_2024", "yes_x_2025")]
    pre_i = [X.columns.get_loc(c) for c in pre_terms]
    w = res.wald_test(np.eye(X.shape[1])[pre_i], scalar=True) if pre_terms else None
    coef_tbl = {c: {"coef": float(res.params[c]), "se": float(res.bse[c]),
                    "p": float(res.pvalues[c])} for c in vins}
    # Round 12 bug fix: res.wald_test(..., scalar=True) on a
    # cluster-robust covariance returns a WALD CHI-SQUARE(q) statistic
    # (q = 3 pre-period terms), NOT an F. The old field name
    # pre_trend_joint_wald_f mislabeled the chi2 as an F. The statistic is
    # unchanged; it is now stored under its correct name, with the implied
    # F = chi2/q and its F(q, G-1) p-value added alongside.
    q = len(pre_terms) if pre_terms else None
    chi2 = float(w.statistic) if w is not None else None
    chi2_p = float(w.pvalue) if w is not None else None
    wald_f = chi2 / q if (chi2 is not None and q) else None
    from scipy import stats as _sps
    wald_f_p = float(_sps.f.sf(wald_f, q, n_clusters - 1)) \
        if (wald_f is not None and q) else None
    return {
        "n": int(res.nobs),
        "n_clusters_G": n_clusters,
        "period_terms": coef_tbl,
        "reference_bin": REF_BIN,
        "pre_period_bins_tested": PRE_BINS,
        "pre_trend_wald_chi2": chi2,
        "pre_trend_wald_chi2_p": chi2_p,
        "pre_trend_wald_q": q,
        "pre_trend_wald_f": wald_f,
        "pre_trend_wald_f_p": wald_f_p,
        "r_squared": float(res.rsquared),
        "estimator": "statsmodels OLS, cluster-robust SE by council district",
    }


def run_did(d, post_years, exclude_transition_year=True):
    d = d.copy()
    if exclude_transition_year:
        d = d[d["sale_year"] != 2024]
    d["post"] = d["sale_year"].isin(post_years).astype(float)
    d["yes_x_post"] = d["yes_district"] * d["post"]
    X, y, groups = design_matrix(d, ["yes_x_post"])
    res = fit_ols(X, y, groups)
    return {
        "n": int(res.nobs),
        "r_squared": float(res.rsquared),
        "post_definition": "sale_year in " + str(sorted(post_years)),
        "excluded_transition_year_2024": bool(exclude_transition_year),
        "did_coef": float(res.params["yes_x_post"]),
        "did_se": float(res.bse["yes_x_post"]),
        "did_p": float(res.pvalues["yes_x_post"]),
        "estimator": "statsmodels OLS, cluster-robust SE by council district",
    }



def main():
    if not CANON.exists() or not ROLL.exists():
        raise SystemExit(
            "Required inputs missing; run the data pipeline and "
            "build_city_of_yes_roll_calls.py first")
    df = pd.read_parquet(CANON).dropna(subset=REQUIRED)
    roll = pd.read_parquet(ROLL)
    d = prepare(df, roll)

    ev = run_event_study(d)
    did_main = run_did(d, [2025], exclude_transition_year=True)
    did_sens = run_did(d, [2024, 2025], exclude_transition_year=False)

    wald_p = ev["pre_trend_wald_chi2_p"]
    pre_flat = bool(wald_p is not None and wald_p > 0.05)
    pre_verdict = (
        ("pre-trends consistent with parallel trends "
         "(joint Wald chi2(3) p=%.4f > 0.05)" % wald_p)
        if pre_flat else
        "pre-trends NOT consistent with parallel trends "
        "(joint Wald chi2(3) p=%.4f <= 0.05) — Yes/No districts were already "
        "diverging before Dec 2024" % wald_p
    )
    main_verdict = (
        "pre-trend check: " + ("PASS" if pre_flat else "FAIL") + "; did coef " +
        ("%.4f" % did_main["did_coef"]) + " (SE " + ("%.4f" % did_main["did_se"]) +
        ", p=" + ("%.4f" % did_main["did_p"]) + ") — reported under the "
        "ideology/preference-proxy framing (not a policy-application effect); "
        "treat the causal reading with that gravity"
    )

    results = {
        "treatment_framing": TREATMENT_FRAMING,
        "task0_finding": TASK0_FINDING,
        "outcome": "target_log_price",
        "treatment_definition": ("district voted Yes on City of Yes for "
                                 "Housing Opportunity (2024-12-05)"),
        "controls": CONTROLS,
        "fixed_effects": ["council district FE", "sale year FE"],
        "clustering": "SEs clustered by council district (51 clusters)",
        "event_study_pre_trends": ev,
        "did_main": did_main,
        "did_sensitivity_2024_included": did_sens,
        "pre_trend_verdict": pre_verdict,
        "main_verdict": main_verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print("Wrote", OUT.relative_to(ROOT))
    print("")
    print("=== Event study (reference bin: 2022-23) ===")
    for c, t in ev["period_terms"].items():
        print("  %-12s coef %+.4f (SE %.4f, p=%.4f)" % (c, t["coef"], t["se"], t["p"]))
    print("  joint pre-trend Wald: chi2(3)=%.3f (p=%.4f); implied F(3,G-1)=%.3f (p=%.4f)"
          % (ev["pre_trend_wald_chi2"], ev["pre_trend_wald_chi2_p"],
             ev["pre_trend_wald_f"], ev["pre_trend_wald_f_p"]))
    print("  pre-trend verdict:", pre_verdict)
    print("")
    print("=== Main DiD (post=2025, 2024 excluded) ===")
    print("  yes_x_post coef %+.4f (SE %.4f, p=%.4f), n=%d" % (did_main["did_coef"], did_main["did_se"], did_main["did_p"], did_main["n"]))
    print("=== Sensitivity (post includes 2024) ===")
    print("  yes_x_post coef %+.4f (SE %.4f, p=%.4f), n=%d" % (did_sens["did_coef"], did_sens["did_se"], did_sens["did_p"], did_sens["n"]))
    print("")
    print("Framing reminder:", TREATMENT_FRAMING)

    # ---- Round 12 (Task F): v2 specs -> separate output ------------------
    v2, _ = run_v2_specs(df, roll)
    OUT_V2.write_text(json.dumps(v2, indent=2), encoding="utf-8")
    print("")
    print("Wrote", OUT_V2.relative_to(ROOT))
    mw = v2["main_wild_inference"]
    print("=== Main DiD, full inference ===")
    print("  coef %+.4f (cluster SE %.4f, p_t(G-1)=%.4f, "
          "p_wild=%.4f), N=%s, G=%s" % (
              mw["coef"], mw["se_cluster"], mw["p_t_G_minus_1"],
              mw["p_wild_cluster_bootstrap_t"], mw["n"], mw["n_clusters_G"]))
    for tag in ["window_2022_2025_incl_2024_as_pre",
                "window_2022_2025_excl_2024",
                "district_specific_linear_trends"]:
        r = v2[tag]
        print("  %-38s coef %+.4f (SE %.4f, p=%.4f), n=%s" % (
            tag, r["did_coef"], r["did_se"], r["did_p"], r["n"]))
    r = v2["continuous_treatment_dem_share_pres_x_post"]
    print("  %-38s coef %+.4f (SE %.4f, p=%.4f), n=%s" % (
        "dem_share_pres x post", r["interaction_coef"], r["interaction_se"],
        r["interaction_p"], r["n"]))
    print("  leave-one-borough-out:", {
        k: round(v["did_coef"], 4) for k, v in v2["leave_one_borough_out"].items()})


# ---------------------------------------------------------------------------
# Round 12 (Task F): v2 outputs — wild-bootstrap inference for the main
# estimate plus the new specifications. Written to a SEPARATE file
# (results/causal/city_of_yes_did_results_v2.json); nothing above changes
# beyond the Wald-field bug fix.
# ---------------------------------------------------------------------------
OUT_V2 = ROOT / "results" / "causal" / "city_of_yes_did_results_v2.json"
REPS_V2 = 9999
SEED_V2 = 42


def _wild_report_for_did(d, extra_cols, focus_col):
    """Full inference (incl. wild-cluster-bootstrap-t) for one DiD design."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src" / "analysis"))
    from wild_cluster_inference import WildClusterOLS
    d2 = d.copy()
    d2["CounDist"] = d2["CounDist"].astype(int)
    d2["sale_year"] = d2["sale_year"].astype(int)
    dist_fe = pd.get_dummies(d2["CounDist"], prefix="district",
                             drop_first=True, dtype=float)
    year_fe = pd.get_dummies(d2["sale_year"], prefix="sale_year",
                             drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=d2.index, name="const"),
                   d2[CONTROLS].astype(float),
                   d2[extra_cols].astype(float), dist_fe, year_fe], axis=1)
    keep = [c for c in X.columns
            if c == "const" or X[c].nunique(dropna=False) > 1]
    X = X[keep]
    fi = list(X.columns).index(focus_col)
    ols = WildClusterOLS(d2["target_log_price"].astype(float).to_numpy(),
                         X.to_numpy(dtype=np.float64),
                         d2["CounDist"].to_numpy(), fi).fit()
    return ols.report(reps=REPS_V2, seed=SEED_V2)


def run_v2_specs(df, roll):
    """New DiD specifications (Task F). All clustered by council district."""
    d = prepare(df, roll)
    v2 = {"outcome": "target_log_price",
          "controls": CONTROLS,
          "fixed_effects": ["council district FE", "sale year FE"],
          "clustering": "SEs clustered by council district",
          "wild_bootstrap": {"weights": "webb6", "reps": REPS_V2,
                             "seed": SEED_V2}}

    # (0) main estimate with full inference (wild p + CI)
    dm = d[d["sale_year"] != 2024].copy()
    dm["post"] = (dm["sale_year"] >= 2025).astype(float)
    dm["yes_x_post"] = dm["yes_district"] * dm["post"]
    v2["main_wild_inference"] = _wild_report_for_did(dm, ["yes_x_post"],
                                                      "yes_x_post")

    # (1) post-2025 vs 2022-23 only, sample 2022-2025
    dw = d[d["sale_year"].between(2022, 2025)].copy()
    dw_strict = dw[dw["sale_year"] != 2024].copy()
    for tag, dd in [("window_2022_2025_incl_2024_as_pre", dw),
                    ("window_2022_2025_excl_2024", dw_strict)]:
        dd = dd.copy()
        dd["post"] = (dd["sale_year"] >= 2025).astype(float)
        dd["yes_x_post"] = dd["yes_district"] * dd["post"]
        X, y, groups = design_matrix(dd, ["yes_x_post"])
        res = fit_ols(X, y, groups)
        v2[tag] = {
            "post_definition": "sale_year == 2025 (pre = 2022-23"
                               + (" and 2024" if "incl" in tag else "") + ")",
            "n": int(res.nobs),
            "n_clusters_G": int(pd.Series(groups).nunique()),
            "did_coef": float(res.params["yes_x_post"]),
            "did_se": float(res.bse["yes_x_post"]),
            "did_p": float(res.pvalues["yes_x_post"]),
            "note": ("2024 is pre-adoption for ~11 of 12 months; the incl "
                     "variant bins it with the pre-period, the excl variant "
                     "drops it like the main spec.")}
    # (2) district-specific linear trends added to the main design
    dt = dm.copy()
    dist_fe = pd.get_dummies(dt["CounDist"], prefix="district",
                             drop_first=True, dtype=float)
    t = (dt["sale_year"].astype(float) - 2016.0)
    dt = pd.concat([dt, dist_fe.mul(t.values, axis=0)
                    .add_suffix("_xtrend")], axis=1)
    trend_cols = [c for c in dt.columns if c.endswith("_xtrend")]
    year_fe = pd.get_dummies(dt["sale_year"].astype(int), prefix="sale_year",
                             drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=dt.index, name="const"),
                   dt[CONTROLS].astype(float),
                   dt[["yes_x_post"]].astype(float),
                   dist_fe, year_fe, dt[trend_cols].astype(float)], axis=1)
    keep = [c for c in X.columns
            if c == "const" or X[c].nunique(dropna=False) > 1]
    X = X[keep]
    res = sm.OLS(dt["target_log_price"].astype(float), X.astype(float)) \
        .fit(cov_type="cluster", cov_kwds={"groups": dt["CounDist"]})
    v2["district_specific_linear_trends"] = {
        "did_coef": float(res.params["yes_x_post"]),
        "did_se": float(res.bse["yes_x_post"]),
        "did_p": float(res.pvalues["yes_x_post"]),
        "n": int(res.nobs),
        "n_clusters_G": int(dt["CounDist"].nunique()),
        "note": ("main design + district FE x (sale_year - 2016) linear "
                 "trends + sale-year FE")}

    # (3) continuous treatment: dem_share_pres x post
    dc = d.dropna(subset=["dem_share_pres"]).copy()
    dc = dc[dc["sale_year"] != 2024]
    dc["post"] = (dc["sale_year"] >= 2025).astype(float)
    dc["dsp_x_post"] = dc["dem_share_pres"].astype(float) * dc["post"]
    X, y, groups = design_matrix(dc, ["dem_share_pres", "dsp_x_post"])
    res = fit_ols(X, y, groups)
    v2["continuous_treatment_dem_share_pres_x_post"] = {
        "treatment": "dem_share_pres (continuous) x post(2025)",
        "dem_share_pres_coef": float(res.params["dem_share_pres"]),
        "dem_share_pres_p": float(res.pvalues["dem_share_pres"]),
        "interaction_coef": float(res.params["dsp_x_post"]),
        "interaction_se": float(res.bse["dsp_x_post"]),
        "interaction_p": float(res.pvalues["dsp_x_post"]),
        "n": int(res.nobs),
        "n_clusters_G": int(pd.Series(groups).nunique()),
        "note": ("rows with missing dem_share_pres dropped; district FE + "
                 "sale-year FE + controls")}

    # (4) leave-one-borough-out for the main estimate
    boro_names = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens",
                  5: "Staten Island"}
    lobo = {}
    for b in sorted(d["BOROUGH"].dropna().unique()):
        db = dm[dm["BOROUGH"] != b].copy()
        X, y, groups = design_matrix(db, ["yes_x_post"])
        res = fit_ols(X, y, groups)
        lobo[boro_names.get(int(b), str(int(b)))] = {
            "excluded_borough": boro_names.get(int(b), str(int(b))),
            "did_coef": float(res.params["yes_x_post"]),
            "did_se": float(res.bse["yes_x_post"]),
            "did_p": float(res.pvalues["yes_x_post"]),
            "n": int(res.nobs),
            "n_clusters_G": int(pd.Series(groups).nunique())}
    v2["leave_one_borough_out"] = lobo
    return v2, d


if __name__ == "__main__":
    main()

