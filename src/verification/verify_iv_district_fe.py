"""Independent verification of the IV / instrumented district-FE 2SLS results
(Round 5, verify stage).

Follows the pattern of verify_spatial_regression.py / verify_regression_ols.py:
the analysis is RE-DERIVED here from the canonical parquet and the cleaned
council-election margins parquet, WITHOUT importing anything from
src/causal/iv_district_fe_regression.py, then compared against
results/causal/iv_district_fe_results.json to 4 decimal places. Discrepancies
are printed and recorded, never silently forced to match.

Re-derived per bandwidth, fresh:
  * the close-race instrument map (council_district, prior cycle) -> Z in {0,1},
  * the retained sample (n_obs), #close cells, #districts, #Z cells,
  * the design-matrix rank of [const + controls + district FE + sale_year FE]
    with dem_share (regressor block) and with Z (instrument block) appended —
    these decide whether the spec is identified at all,
  * if (and only if) a spec is identified: the 2SLS first-stage F-stat /
    coefficient and the second-stage dem_share coefficient/SE/p are re-fit with
    linearmodels.iv.IV2SLS and compared to the JSON.

Writes results/causal/iv_district_fe_verification.json.

Run:
    python src/verification/verify_iv_district_fe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "canonical" / "modeling_dataset_canonical_v2.parquet"
MARGINS = ROOT / "data" / "processed" / "council_election_margins.parquet"
PIPELINE_JSON = ROOT / "results" / "causal" / "iv_district_fe_results.json"
OUT = ROOT / "results" / "causal" / "iv_district_fe_verification.json"

BANDWIDTHS_PP = [5, 10, 15, 20]
CONTROLS = ["yearbuilt", "numfloors", "landuse", "commfar",
            "RESIDENTIAL UNITS", "COMMERCIAL UNITS"]
REQUIRED = ["target_log_price", "dem_share", "CounDist", "sale_year"] + CONTROLS
TOL = 5e-5  # 4-decimal-place agreement

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def zmap_for(margins, bw):
    out = {}
    for r in margins[margins["margin_pct"] <= bw].itertuples():
        if r.winner_party == "Democratic":
            out[(int(r.council_district), int(r.year))] = 1.0
        elif r.winner_party == "Republican":
            out[(int(r.council_district), int(r.year))] = 0.0
    return out


def sample_for(df, zmap):
    d = df.copy()
    prior = d["election_year"].astype(int) - 4
    d["Z"] = [zmap.get((int(c), int(p)), np.nan)
              for c, p in zip(d["CounDist"], prior)]
    return d.dropna(subset=["Z"])


def close(a, b, tol=TOL):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def describe_spec(d):
    """Sample counts + identification rank diagnostics (fresh from d)."""
    rec = {
        "n_obs": int(len(d)),
        "n_instrument_cells": int(
            d[["CounDist", "election_year"]].drop_duplicates().shape[0]),
        "n_districts": int(d["CounDist"].astype(int).nunique()),
    }
    if len(d) == 0:
        rec["identified"] = False
        return rec, None
    d2 = d.copy()
    d2["CounDist"] = d2["CounDist"].astype(int)
    d2["sale_year"] = d2["sale_year"].astype(int)
    dfe = pd.get_dummies(d2["CounDist"], prefix="CounDist", drop_first=True,
                         dtype=float)
    yfe = pd.get_dummies(d2["sale_year"], prefix="sale_year", drop_first=True,
                         dtype=float)
    exog = pd.concat([pd.Series(1.0, index=d2.index, name="const"),
                      d2[CONTROLS].astype(float), dfe, yfe], axis=1)
    X = exog.to_numpy(dtype=float)
    rank_x = int(np.linalg.matrix_rank(
        np.column_stack([X, d2["dem_share"].to_numpy(float).reshape(-1, 1)])))
    rank_xz = int(np.linalg.matrix_rank(
        np.column_stack([X, d2["Z"].to_numpy(float).reshape(-1, 1)])))
    rec.update({
        "design_params": int(X.shape[1] + 1),
        "design_rank": rank_x,
        "instrument_design_rank": rank_xz,
        "identified": bool(rank_x == X.shape[1] + 1 and rank_xz == X.shape[1] + 1),
    })
    return rec, (exog, d2)


def refit_2sls(exog, d2):
    """Independent re-fit of the identified spec (fresh construction)."""
    dep = d2["target_log_price"].astype(float)
    endog = d2["dem_share"].astype(float)
    instr = d2["Z"].rename("z_close_dem").astype(float)
    try:
        res = IV2SLS(dep, exog, endog, instr).fit(
            cov_type="clustered", clusters=d2["CounDist"])
        cov_used = "clustered"
    except Exception:
        res = IV2SLS(dep, exog, endog, instr).fit(cov_type="unadjusted")
        cov_used = "unadjusted (clustered covariance singular at this bandwidth)"
    fs = res.first_stage
    return {
        "cov_type_used": cov_used,
        "first_stage": {
            "instrument_coef": float(fs.params["z_close_dem"]),
            "instrument_se": float(fs.std_errors["z_close_dem"]),
            "instrument_p": float(fs.pvalues["z_close_dem"]),
            "f_stat": float(fs.diagnostics.loc["dem_share"]["f.stat"]),
            "f_pval": float(fs.diagnostics.loc["dem_share"]["f.pval"]),
            "partial_r2_excluded_instrument":
                float(fs.diagnostics.loc["dem_share"]["partial.rsquared"]),
        },
        "second_stage": {
            "dem_share_coef": float(res.params["dem_share"]),
            "dem_share_se": float(res.std_errors["dem_share"]),
            "dem_share_p": float(res.pvalues["dem_share"]),
            "r_squared": float(res.rsquared),
            "n": int(res.nobs),
        },
    }


def main():
    if not PIPELINE_JSON.exists():
        raise SystemExit(
            "results/causal/iv_district_fe_results.json not found — run "
            "src/causal/iv_district_fe_regression.py first")
    df = pd.read_parquet(CANON).dropna(subset=REQUIRED)
    margins = pd.read_parquet(MARGINS)
    pipeline = json.loads(PIPELINE_JSON.read_text())

    out = {"source_reverified": "results/causal/iv_district_fe_results.json",
           "bandwidths_pp": BANDWIDTHS_PP,
           "per_bandwidth": {}}
    all_match = True

    for bw in BANDWIDTHS_PP:
        key = f"bw_{bw}pp"
        recorded = pipeline.get(key, {})
        zmap = zmap_for(margins, bw)
        d = sample_for(df, zmap)
        mine, mats = describe_spec(d)

        checks = {}
        dis = []

        def cmp(name, a, b):
            ok = bool(a == b) if isinstance(a, (int, bool)) else close(a, b)
            checks[name] = ok
            if not ok:
                dis.append(f"{key}.{name}: pipeline={b} verified={a}")
            return ok

        cmp("n_instrument_z_cells", int(len(zmap)),
            recorded.get("n_instrument_z_cells"))
        cmp("n_obs", mine["n_obs"], recorded.get("n_obs"))
        cmp("n_instrument_cells", mine["n_instrument_cells"],
            recorded.get("n_instrument_cells"))
        cmp("n_districts", mine["n_districts"], recorded.get("n_districts"))
        cmp("identified", bool(mine["identified"]),
            bool(recorded.get("identified")))
        if mine["identified"]:
            cmp("design_rank", mine["design_rank"], recorded.get("design_rank"))
            cmp("instrument_design_rank", mine["instrument_design_rank"],
                recorded.get("instrument_design_rank"))

        # Only if BOTH agree the spec is identified do we re-fit and compare
        # coefficients to 4 dp. When every spec is not identified (the current
        # NYC data reality) this block is correctly skipped for all bandwidths.
        if bool(recorded.get("identified")) and bool(mine["identified"]):
            rebuilt = refit_2sls(*mats)
            cmp("cov_type_used",
                rebuilt["cov_type_used"], recorded.get("cov_type_used"))
            fs_rec, ss_rec = recorded["first_stage"], recorded["second_stage"]
            cmp("f_stat", rebuilt["first_stage"]["f_stat"], fs_rec["f_stat"])
            cmp("instrument_coef", rebuilt["first_stage"]["instrument_coef"],
                fs_rec["instrument_coef"])
            cmp("dem_share_coef", rebuilt["second_stage"]["dem_share_coef"],
                ss_rec["dem_share_coef"])
            cmp("dem_share_se", rebuilt["second_stage"]["dem_share_se"],
                ss_rec["dem_share_se"])
            cmp("dem_share_p", rebuilt["second_stage"]["dem_share_p"],
                ss_rec["dem_share_p"])

        bw_ok = all(checks.values())
        all_match = all_match and bw_ok
        out["per_bandwidth"][key] = {
            "matches_pipeline": bool(bw_ok),
            "checks": checks,
            "discrepancies": dis,
            "recorded_verdict": recorded.get("verdict"),
        }
        print(f"\n=== bandwidth +/-{bw} pp ===")
        print(f"  identified recorded/re-derived: "
              f"{recorded.get('identified')} / {mine['identified']}")
        print(f"  cells recorded/re-derived: "
              f"{recorded.get('n_instrument_cells')} / {mine['n_instrument_cells']}")
        print(f"  n_obs recorded/re-derived: "
              f"{recorded.get('n_obs')} / {mine['n_obs']}")
        print(f"  matches_pipeline: {bw_ok}")
        for dd in dis:
            print(f"    DISCREPANCY: {dd}")

    out["all_bandwidths_match"] = bool(all_match)
    out["note"] = ("When a bandwidth is not identified, the honest match is the "
                   "concordance of the identification verdict and the sample/"
                   "cell counts — there is no coefficient to compare.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWROTE {OUT.relative_to(ROOT)}")
    print(f"all_bandwidths_match: {all_match}")


if __name__ == "__main__":
    main()
