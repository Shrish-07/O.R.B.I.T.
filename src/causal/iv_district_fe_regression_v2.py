"""IV v2 additions (Task G): matched OLS benchmark, cluster counts, RD spec.

The 2SLS second stage in src/causal/iv_district_fe_regression.py uses
district FE + sale-year FE + RESIDENTIAL/COMMERCIAL UNITS controls on the
close-race sample, but the JSON's `model4_benchmark` points at Table 2
Model 4 (district FE only, no units, no year FE, N=520,365 on the full
regression sample) — a spec/sample-mismatched comparison. This stage adds
the CORRECT benchmark without touching the original file or its 12
recorded specification arms:

  * matched_ols_benchmark: for EVERY ideology measure (legacy dem_share,
    allcand, pres + vintage variants) and EVERY bandwidth, an OLS of
    target_log_price on the measure + the identical exog (const + the same
    controls + district FE + sale-year FE) on the identical close-race
    sample, clustered by council district. `benchmark_pointer` names this
    section as the comparison the 2SLS estimates should be read against;
    the old model4_benchmark is retained in the original file, marked
    superseded here.
  * per-bandwidth cluster bookkeeping: G (number of council districts in
    the instrumented sample), treated (Z=1) and control (Z=0) cell counts,
    and an explicit G < 30 flag.
  * a local-linear RD-style specification at bandwidths 10-40pp: margin_pct
    as the running variable with a linear term interacted with Z on each
    side (instruments: Z and Z x margin; exog: margin, controls, district
    FE, sale-year FE), reporting the first-stage joint partial F of the
    excluded instruments. Round 13 (R4): the fit now uses
    cov_type='clustered' with clusters=CounDist; a singular clustered
    covariance marks the spec 'not estimable' with no unadjusted fallback.
    The JSON also carries an explicit council_district_vintage_mismatch
    flag (prior-cycle winners elected under the pre-2022 plan mapped onto
    current-district sales rows).

Output: results/causal/iv_district_fe_results_v2.json (the original
results/causal/iv_district_fe_results.json is NOT modified).

Run: python src/causal/iv_district_fe_regression_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "canonical" / "modeling_dataset_canonical_v2.parquet"
MARGINS = ROOT / "data" / "processed" / "council_election_margins.parquet"
ORIG = ROOT / "results" / "causal" / "iv_district_fe_results.json"
OUT = ROOT / "results" / "causal" / "iv_district_fe_results_v2.json"

BANDWIDTHS_PP = [5, 10, 15, 20]
RD_BANDWIDTHS = [10, 15, 20, 25, 30, 35, 40]
G_MIN = 30
CONTROLS = ["yearbuilt", "numfloors", "landuse", "commfar",
            "RESIDENTIAL UNITS", "COMMERCIAL UNITS"]
REQUIRED = ["target_log_price", "dem_share", "CounDist", "sale_year"] + CONTROLS
MEASURES = ["dem_share", "dem_share_mayor_allcand", "dem_share_pres",
            "dem_share_vintage", "dem_share_mayor_allcand_vintage",
            "dem_share_pres_vintage"]

def build_maps(margins, bandwidth):
    """(council_district, prior_cycle) -> (Z, margin_pct) for close races."""
    zmap, mmap = {}, {}
    close = margins[margins["margin_pct"] <= bandwidth]
    for r in close.itertuples():
        if r.winner_party == "Democratic":
            zmap[(int(r.council_district), int(r.year))] = 1.0
            mmap[(int(r.council_district), int(r.year))] = float(r.margin_pct)
        elif r.winner_party == "Republican":
            zmap[(int(r.council_district), int(r.year))] = 0.0
            mmap[(int(r.council_district), int(r.year))] = float(r.margin_pct)
    return zmap, mmap


def build_sample(df, zmap, mmap):
    d = df.copy()
    d["prior_cycle"] = d["election_year"].astype(int) - 4
    keys = list(zip(d["CounDist"].astype(int), d["prior_cycle"]))
    d["Z"] = [zmap.get(k, np.nan) for k in keys]
    d["margin"] = [mmap.get(k, np.nan) for k in keys]
    return d.dropna(subset=["Z"])


def exog_for(d):
    d2 = d.copy()
    d2["CounDist"] = d2["CounDist"].astype(int)
    d2["sale_year"] = d2["sale_year"].astype(int)
    return pd.concat([
        pd.Series(1.0, index=d2.index, name="const"),
        d2[CONTROLS].astype(float),
        pd.get_dummies(d2["CounDist"], prefix="CounDist", drop_first=True,
                        dtype=float),
        pd.get_dummies(d2["sale_year"], prefix="sale_year", drop_first=True,
                       dtype=float),
    ], axis=1)


def matched_ols(d, measure):
    """OLS with the IDENTICAL spec/sample as the 2SLS second stage."""
    import statsmodels.api as sm
    dd = d.dropna(subset=[measure])
    exog = pd.concat([exog_for(dd), dd[[measure]].astype(float)], axis=1)
    res = sm.OLS(dd["target_log_price"].astype(float), exog) \
        .fit(cov_type="cluster", cov_kwds={"groups": dd["CounDist"].astype(int)})
    return {
        "spec": "target_log_price ~ measure + controls + district FE + "
                "sale-year FE, clustered by council district",
        "coef": float(res.params[measure]),
        "se": float(res.bse[measure]),
        "p": float(res.pvalues[measure]),
        "n": int(res.nobs),
        "n_clusters_G": int(dd["CounDist"].nunique()),
    }

def rd_local_linear(d):
    """2SLS with instruments Z and Z x margin (linear slope each side),
    margin as an exogenous running-variable control.

    Round 13 (R4) fix: fitted with cov_type='clustered', clusters =
    CounDist (district-cycle level clustering). The previous default
    heteroskedasticity-robust covariance produced first-stage joint F
    values of 3,000-77,000 and second-stage SEs that ignored the
    district clustering. If the clustered covariance is singular /
    non-finite, the spec is marked 'not estimable'; there is NO
    unadjusted fallback."""
    dd = d.copy()
    dd["Zxmargin"] = dd["Z"] * dd["margin"]
    exog = pd.concat([exog_for(dd), dd[["margin"]].astype(float)], axis=1)
    try:
        res = IV2SLS(dd["target_log_price"].astype(float), exog,
                     dd["dem_share"].astype(float),
                     dd[["Z", "Zxmargin"]].astype(float)).fit(
                         cov_type="clustered",
                         clusters=dd["CounDist"].astype(int))
        fs = res.first_stage.diagnostics.loc["dem_share"]
        f_stat = float(fs["f.stat"])
        cov_diag = np.asarray(res.cov, dtype=float).diagonal()
        cov_ok = (np.isfinite(f_stat) and np.isfinite(cov_diag).all()
                  and (cov_diag > 0).all()
                  and np.isfinite(float(res.std_errors["dem_share"])))
        if not cov_ok:
            return {"identified": False,
                    "status": "not estimable",
                    "reason": ("clustered covariance singular or non-finite "
                               "at this bandwidth; no unadjusted fallback "
                               "is applied"),
                    "n": int(len(dd)),
                    "n_clusters_G": int(dd["CounDist"].nunique())}
        return {
            "identified": True,
            "status": "estimated (covariance clustered by CounDist)",
            "first_stage_joint_f_excluded_instruments": f_stat,
            "first_stage_joint_f_pvalue": float(fs["f.pval"]),
            "dem_share_coef": float(res.params["dem_share"]),
            "dem_share_se": float(res.std_errors["dem_share"]),
            "dem_share_p": float(res.pvalues["dem_share"]),
            "n": int(res.nobs),
            "n_clusters_G": int(dd["CounDist"].nunique()),
            "covariance": ("clustered by council district (CounDist), "
                           "district-cycle level clustering; Round 13 (R4) "
                           "fix — the pre-fix default "
                           "heteroskedasticity-robust fit reported "
                           "first-stage F of 3,000-77,000 and SEs that "
                           "ignored the clustering"),
            "spec": ("instruments: Z, Z x margin_pct (linear each side); "
                     "exog: margin_pct + controls + district FE + "
                     "sale-year FE; sample: close-race cells within the "
                     "bandwidth"),
        }
    except Exception as e:
        return {"identified": False, "status": "not estimable",
                "reason": f"clustered fit failed: {e}; no unadjusted "
                          "fallback is applied",
                "n": int(len(dd)),
                "n_clusters_G": int(dd["CounDist"].nunique())}


def main():
    if not (CANON.exists() and MARGINS.exists() and ORIG.exists()):
        raise SystemExit("Missing inputs — run the data pipeline and "
                         "src/causal/iv_district_fe_regression.py first.")
    orig = json.loads(ORIG.read_text())
    df = pd.read_parquet(CANON).dropna(subset=REQUIRED)
    margins = pd.read_parquet(MARGINS)

    results = {
        "purpose": ("Task G additions: spec-matched OLS benchmark per "
                    "ideology measure, cluster/cell bookkeeping, and a "
                    "local-linear RD-style specification. The original "
                    "results/causal/iv_district_fe_results.json and its 12 "
                    "recorded specification arms are unchanged."),
        "benchmark_pointer": "matched_ols_benchmark",
        "council_district_vintage_mismatch": {
            "flag": True,
            "finding": ("The close-election instrument maps prior-cycle "
                        "(2013/2017/2021) winners, elected under the "
                        "PRE-2022 council district plan, onto sales rows "
                        "keyed to CURRENT (post-2022) council districts; "
                        "district identities do not line up across "
                        "vintages wherever an old-plan district does not "
                        "map cleanly onto its 2023-plan successor."),
            "see_also": ("results/crosswalk_audit.json "
                         "council_district_vintage_note; "
                         "data/processed/council_2013_to_current.parquet "
                         "(areal correspondence built by "
                         "build_legacy_ed_crosswalk.py)"),
            "original_12_arms_untouched": (
                "results/causal/iv_district_fe_results.json is unchanged; "
                "this flag documents a known limitation of the instrument "
                "shared by both files."),
        },
        "superseded_benchmark_note": (
            "The original file's model4_benchmark (Table 2 Model 4: "
            "district FE only, no unit controls, no sale-year FE, "
            "N=520,365 full-sample OLS) does NOT match the 2SLS second "
            "stage (district FE + sale-year FE + RESIDENTIAL/COMMERCIAL "
            "UNITS on the close-race sample). Compare the 2SLS estimates "
            "to matched_ols_benchmark instead."),
        "controls": CONTROLS,
        "fixed_effects": ["CounDist", "sale_year"],
        "g_min_threshold": G_MIN,
        "bandwidths": {},
        "rd_local_linear": {},
    }

    for bw in BANDWIDTHS_PP:
        zmap, mmap = build_maps(margins, bw)
        d = build_sample(df, zmap, mmap)
        cells = d[["CounDist", "election_year", "Z"]].drop_duplicates()
        treated = int((cells["Z"] == 1.0).sum())
        control = int((cells["Z"] == 0.0).sum())
        G = int(d["CounDist"].nunique())
        rec = {
            "n_obs": int(len(d)),
            "n_clusters_G": G,
            "treated_cells_Z1": treated,
            "control_cells_Z0": control,
            "g_below_30_flag": bool(G < G_MIN),
            "matched_ols_by_measure": {},
        }
        for m in MEASURES:
            rec["matched_ols_by_measure"][m] = matched_ols(d, m)
        results["bandwidths"][f"bw_{bw}pp"] = rec
        print(f"  bw {bw:>2}pp: N={len(d):,} G={G} "
              f"{'(<30 FLAGGED)' if G < G_MIN else ''} "
              f"treated/control cells={treated}/{control}; "
              f"legacy matched OLS coef="
              f"{rec['matched_ols_by_measure']['dem_share']['coef']:+.4f}")

    for bw in RD_BANDWIDTHS:
        zmap, mmap = build_maps(margins, bw)
        d = build_sample(df, zmap, mmap)
        results["rd_local_linear"][f"bw_{bw}pp"] = rd_local_linear(d)
        r = results["rd_local_linear"][f"bw_{bw}pp"]
        print(f"  RD bw {bw:>2}pp: identified={r.get('identified')} "
              f"F={r.get('first_stage_joint_f_excluded_instruments')} "
              f"G={r.get('n_clusters_G')} "
              f"N={r.get('n'):,}" if r.get("identified")
              else f"  RD bw {bw:>2}pp: NOT ESTIMABLE "
                   f"({str(r.get('reason', ''))[:70]})")

    # convenience view: the matched benchmark at the widest primary bandwidth
    results["matched_ols_benchmark"] = {
        m: results["bandwidths"]["bw_20pp"]["matched_ols_by_measure"][m]
        for m in MEASURES}
    results["matched_ols_benchmark_note"] = (
        "Shown at the widest primary bandwidth (±20pp); per-bandwidth "
        "matched OLS for every measure is under bandwidths.bw_*pp."
        ".matched_ols_by_measure.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()


