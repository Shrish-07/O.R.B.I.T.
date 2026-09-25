
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
    pre_terms = [c for c in vins if c not in ("yes_x_2024", "yes_x_2025")]
    pre_i = [X.columns.get_loc(c) for c in pre_terms]
    w = res.wald_test(np.eye(X.shape[1])[pre_i], scalar=True) if pre_terms else None
    coef_tbl = {c: {"coef": float(res.params[c]), "se": float(res.bse[c]),
                    "p": float(res.pvalues[c])} for c in vins}
    return {
        "n": int(res.nobs),
        "period_terms": coef_tbl,
        "reference_bin": REF_BIN,
        "pre_period_bins_tested": PRE_BINS,
        "pre_trend_joint_wald_f": float(w.statistic) if w is not None else None,
        "pre_trend_joint_wald_p": float(w.pvalue) if w is not None else None,
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

    wald_p = ev["pre_trend_joint_wald_p"]
    pre_flat = bool(wald_p is not None and wald_p > 0.05)
    pre_verdict = (
        ("pre-trends consistent with parallel trends "
         "(joint Wald p=%.4f > 0.05)" % wald_p)
        if pre_flat else
        "pre-trends NOT consistent with parallel trends "
        "(joint Wald p=%.4f <= 0.05) — Yes/No districts were already "
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
    print("  joint pre-trend Wald: F=%.3f, p=%.4f" % (ev["pre_trend_joint_wald_f"], ev["pre_trend_joint_wald_p"]))
    print("  pre-trend verdict:", pre_verdict)
    print("")
    print("=== Main DiD (post=2025, 2024 excluded) ===")
    print("  yes_x_post coef %+.4f (SE %.4f, p=%.4f), n=%d" % (did_main["did_coef"], did_main["did_se"], did_main["did_p"], did_main["n"]))
    print("=== Sensitivity (post includes 2024) ===")
    print("  yes_x_post coef %+.4f (SE %.4f, p=%.4f), n=%d" % (did_sens["did_coef"], did_sens["did_se"], did_sens["did_p"], did_sens["n"]))
    print("")
    print("Framing reminder:", TREATMENT_FRAMING)


if __name__ == "__main__":
    main()

