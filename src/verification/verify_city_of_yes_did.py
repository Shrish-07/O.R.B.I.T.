"""Independent verification of the City of Yes DiD (Round 6, verify stage).

Same pattern as verify_regression_ols.py: re-derive from
data/canonical/modeling_dataset_canonical_v2.parquet +
data/processed/city_of_yes_roll_calls.parquet WITHOUT importing
src/causal/city_of_yes_did.py, then compare against
results/causal/city_of_yes_did_results.json to 4 decimal places.

Re-derived: sample (same dropna), Yes flag, period bins, event-study OLS with
district/sale-year FE + controls + clustered-by-district covariance (every
period-term coef/SE/p + joint pre-trend Wald F/p), main DiD (post=2025, 2024
excluded) and sensitivity (post in {2024,2025}). Discrepancies printed and
JSON-recorded, never silently forced.

Writes results/causal/city_of_yes_did_verification.json.

Run: python src/verification/verify_city_of_yes_did.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "canonical" / "modeling_dataset_canonical_v2.parquet"
ROLL = ROOT / "data" / "processed" / "city_of_yes_roll_calls.parquet"
PIPELINE_JSON = ROOT / "results" / "causal" / "city_of_yes_did_results.json"
OUT = ROOT / "results" / "causal" / "city_of_yes_did_verification.json"
CONTROLS = ["yearbuilt", "numfloors", "landuse", "commfar",
            "RESIDENTIAL UNITS", "COMMERCIAL UNITS", "TOTAL UNITS"]
REQUIRED = ["target_log_price", "CounDist", "sale_year"] + CONTROLS
PERIOD_BINS = ["2016-17", "2018-19", "2020-21", "2022-23", "2024", "2025"]
REF_BIN = "2022-23"
TOL = 5e-5

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


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
    dist_fe = pd.get_dummies(d2["CounDist"], prefix="district", drop_first=True,
                             dtype=float)
    year_fe = pd.get_dummies(d2["sale_year"], prefix="sale_year", drop_first=True,
                             dtype=float)
    X = pd.concat([pd.Series(1.0, index=d2.index, name="const"),
                   d2[CONTROLS].astype(float), d2[extra_cols],
                   dist_fe, year_fe], axis=1)
    return X.astype(float), d2["target_log_price"].astype(float), d2["CounDist"]


def close(a, b, tol=TOL):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _fit(X, y, groups):
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def recompute_all(df, roll):
    d = prepare(df, roll)
    de = d.copy()
    vins = []
    for b in PERIOD_BINS:
        if b == REF_BIN:
            continue
        name = "yes_x_" + b
        de[name] = de["yes_district"] * (de["period_bin"] == b).astype(float)
        vins.append(name)
    Xe, ye, ge = design_matrix(de, vins)
    rese = _fit(Xe, ye, ge)
    pre_terms = [c for c in vins if c not in ("yes_x_2024", "yes_x_2025")]
    w = rese.wald_test(
        np.eye(Xe.shape[1])[[Xe.columns.get_loc(c) for c in pre_terms]],
        scalar=True)
    ev = {c: {"coef": float(rese.params[c]), "se": float(rese.bse[c]),
              "p": float(rese.pvalues[c])} for c in vins}
    ev_meta = {"n": int(rese.nobs), "r_squared": float(rese.rsquared),
               "pre_trend_wald_chi2": float(w.statistic),
               "pre_trend_wald_chi2_p": float(w.pvalue)}
    # Round 12: implied F = chi2/q with its F(q, G-1) p-value
    q = 3
    G = int(pd.Series(ge).nunique())
    from scipy import stats as _sps
    ev_meta["pre_trend_wald_f"] = ev_meta["pre_trend_wald_chi2"] / q
    ev_meta["pre_trend_wald_f_p"] = float(
        _sps.f.sf(ev_meta["pre_trend_wald_f"], q, G - 1))

    dm = d[d["sale_year"] != 2024].copy()
    dm["yes_x_post"] = (dm["yes_district"] * (dm["sale_year"] >= 2025)).astype(float)
    Xm, ym, gm = design_matrix(dm, ["yes_x_post"])
    resm = _fit(Xm, ym, gm)
    main_out = {"n": int(resm.nobs), "r_squared": float(resm.rsquared),
                "did_coef": float(resm.params["yes_x_post"]),
                "did_se": float(resm.bse["yes_x_post"]),
                "did_p": float(resm.pvalues["yes_x_post"])}

    ds = d.copy()
    ds["yes_x_post"] = (ds["yes_district"] *
                        ds["sale_year"].isin([2024, 2025])).astype(float)
    Xs, ys, gs = design_matrix(ds, ["yes_x_post"])
    ress = _fit(Xs, ys, gs)
    sens = {"n": int(ress.nobs), "r_squared": float(ress.rsquared),
            "did_coef": float(ress.params["yes_x_post"]),
            "did_se": float(ress.bse["yes_x_post"]),
            "did_p": float(ress.pvalues["yes_x_post"])}
    return ev, ev_meta, main_out, sens




def main():
    if not PIPELINE_JSON.exists():
        raise SystemExit(
            "results/causal/city_of_yes_did_results.json not found — run "
            "src/causal/city_of_yes_did.py first")
    df = pd.read_parquet(CANON).dropna(subset=REQUIRED)
    roll = pd.read_parquet(ROLL)
    pipeline = json.loads(PIPELINE_JSON.read_text())
    ev, ev_meta, main_r, sens_r = recompute_all(df, roll)

    checks = []

    def cmp(name, a, b):
        checks.append({"field": name, "match": bool(close(a, b)),
                       "verified": a, "pipeline": b})

    rec_ev = pipeline["event_study_pre_trends"]
    for c in ev:
        cmp(f"ev.{c}.coef", ev[c]["coef"], rec_ev["period_terms"][c]["coef"])
        cmp(f"ev.{c}.se", ev[c]["se"], rec_ev["period_terms"][c]["se"])
        cmp(f"ev.{c}.p", ev[c]["p"], rec_ev["period_terms"][c]["p"])
    cmp("ev.n", ev_meta["n"], rec_ev["n"])
    cmp("ev.r_squared", ev_meta["r_squared"], rec_ev["r_squared"])
    cmp("ev.wald_chi2", ev_meta["pre_trend_wald_chi2"],
        rec_ev["pre_trend_wald_chi2"])
    cmp("ev.wald_chi2_p", ev_meta["pre_trend_wald_chi2_p"],
        rec_ev["pre_trend_wald_chi2_p"])
    cmp("ev.wald_f_implied", ev_meta["pre_trend_wald_f"],
        rec_ev["pre_trend_wald_f"])
    cmp("ev.wald_f_p", ev_meta["pre_trend_wald_f_p"],
        rec_ev["pre_trend_wald_f_p"])

    for tag, mine, rec in [("main", main_r, pipeline["did_main"]),
                           ("sens", sens_r,
                            pipeline["did_sensitivity_2024_included"])]:
        cmp(f"{tag}.n", mine["n"], rec["n"])
        cmp(f"{tag}.r_squared", mine["r_squared"], rec["r_squared"])
        cmp(f"{tag}.did_coef", mine["did_coef"], rec["did_coef"])
        cmp(f"{tag}.did_se", mine["did_se"], rec["did_se"])
        cmp(f"{tag}.did_p", mine["did_p"], rec["did_p"])

    n_match = sum(1 for c in checks if c["match"])
    all_match = all(c["match"] for c in checks)
    print(f"City of Yes DiD verification: {n_match}/{len(checks)} matched (4dp)")
    for c in checks:
        if not c["match"]:
            print("  DISCREPANCY:", c)
    out = {
        "source_reverified": "results/causal/city_of_yes_did_results.json",
        "n_checks": len(checks), "n_matched": n_match,
        "all_match": bool(all_match), "checks": checks,
        "recomputed_main_did": main_r,
        "recomputed_sensitivity_did": sens_r,
        "recomputed_pre_trend_wald": ev_meta,
        "note": "Matches are on the re-derived values, incl. all event-study "
                "terms, the joint pre-trend Wald test, and both DiD variants.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print("WROTE", OUT.relative_to(ROOT))

    # Round 11: machine-countable status (the internal 5e-5 4dp tolerance
    # per field lives in close() above)
    sys.path.insert(0, str(ROOT / 'src' / 'verification'))
    from _shared import record_status  # noqa: E402
    record_status('verify_city_of_yes_did',
                  [{'check': c['field'], 'expected': c['pipeline'],
                    'actual': c['verified'], 'tolerance': TOL,
                    'match': c['match']} for c in checks],
                  note='Independent DiD/event-study re-derivation vs '
                       'results/causal/city_of_yes_did_results.json.')

    if not all_match:
        sys.exit(1)


if __name__ == "__main__":
    main()