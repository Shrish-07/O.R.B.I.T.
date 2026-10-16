"""Independent verification of the IV rescue runs (Round 6, verify stage).

Same pattern as verify_iv_district_fe.py: re-derive EVERYTHING from the
canonical parquet + margins parquet WITHOUT importing
src/causal/iv_district_fe_rescue.py, then compare against
results/causal/iv_district_fe_rescue_results.json to 4 decimal places.
Discrepancies are printed and recorded, never silently forced to match.

For every record in both rescue arms (1a borough FE; 1b wider bandwidth with
district FE) we re-derive: the instrument map, the excluded-sample size, the
counts, and both design-matrix rank checks. Where (and only where) the record
is identified, we re-fit the 2SLS fresh (same construction, new code) and
compare the first-stage F/coef and second-stage dem_share coef/SE/p.

Writes results/causal/iv_district_fe_rescue_verification.json.

Run:
    python src/verification/verify_iv_district_fe_rescue.py
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
PIPELINE_JSON = ROOT / "results" / "causal" / "iv_district_fe_rescue_results.json"
OUT = ROOT / "results" / "causal" / "iv_district_fe_rescue_verification.json"

CONTROLS = ["yearbuilt", "numfloors", "landuse", "commfar",
            "RESIDENTIAL UNITS", "COMMERCIAL UNITS"]
REQUIRED = (["target_log_price", "dem_share", "CounDist", "BOROUGH", "sale_year"]
            + CONTROLS)
ARMS = {"rescue_1a_borough_fe_variant": "BOROUGH",
        "rescue_1b_wider_bandwidth_district_fe": "CounDist"}
TOL = 5e-5

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


def build_design(d, fe_col):
    d2 = d.copy()
    d2["CounDist"] = d2["CounDist"].astype(int)
    d2["sale_year"] = d2["sale_year"].astype(int)
    unit_fe = pd.get_dummies(d2[fe_col], prefix=fe_col, drop_first=True, dtype=float)
    year_fe = pd.get_dummies(d2["sale_year"], prefix="sale_year", drop_first=True,
                             dtype=float)
    exog = pd.concat([pd.Series(1.0, index=d2.index, name="const"),
                      d2[CONTROLS].astype(float), unit_fe, year_fe], axis=1)
    return exog, d2["target_log_price"].astype(float), d2["dem_share"].astype(float), \
        d2["Z"].rename("z_close_dem").astype(float)


def close(a, b, tol=TOL):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def verify_record(df, margins, arm_key, fe_col, rec):
    bw = rec["bandwidth_pp"]
    zmap = zmap_for(margins, bw)
    d = sample_for(df, zmap)
    exog, dep, endog, instr = build_design(d, fe_col)
    X = exog.to_numpy(dtype=float)
    rx = int(np.linalg.matrix_rank(np.column_stack(
        [X, endog.to_numpy(dtype=float).reshape(-1, 1)])))
    rz = int(np.linalg.matrix_rank(np.column_stack(
        [X, instr.to_numpy(dtype=float).reshape(-1, 1)])))
    mine = {
        "n_obs": int(len(d)),
        "n_close_cells": int(d[["CounDist", "election_year"]].drop_duplicates().shape[0]),
        "n_districts": int(d["CounDist"].nunique()),
        "n_z_cells": int(len(zmap)),
        "design_params": int(X.shape[1] + 1),
        "design_rank": rx,
        "instrument_design_rank": rz,
        "identified": bool(rx == X.shape[1] + 1 and rz == X.shape[1] + 1),
    }
    checks, dis = {}, []

    def cmp(name, a, b):
        ok = (a == b) if isinstance(a, (int, bool, str)) else close(a, b)
        checks[name] = ok
        if not ok:
            dis.append(f"{arm_key}/{bw}pp.{name}: pipeline={b} verified={a}")
        return ok

    for k in ["n_obs", "n_close_cells", "n_districts", "n_z_cells",
              "design_params", "design_rank", "instrument_design_rank"]:
        cmp(k, mine[k], rec.get(k))
    cmp("identified", mine["identified"], rec.get("identified"))

    if bool(rec.get("identified")) and bool(mine["identified"]):
        try:
            res = IV2SLS(dep, exog, endog, instr).fit(cov_type="clustered",
                                                      clusters=d["CounDist"])
            cov = "clustered"
        except Exception:
            res = IV2SLS(dep, exog, endog, instr).fit(cov_type="unadjusted")
            cov = "unadjusted (clustered covariance singular at this bandwidth)"
        ind = res.first_stage.individual["dem_share"]
        fsd = res.first_stage.diagnostics
        fsrec = rec["first_stage"]; ssrec = rec["second_stage"]
        cmp("cov_type_used", cov, rec.get("cov_type_used"))
        cmp("f_stat", float(fsd.loc["dem_share"]["f.stat"]), fsrec["f_stat"])
        cmp("instrument_coef", float(ind.params["z_close_dem"]), fsrec["instrument_coef"])
        cmp("dem_share_coef", float(res.params["dem_share"]), ssrec["dem_share_coef"])
        cmp("dem_share_se", float(res.std_errors["dem_share"]), ssrec["dem_share_se"])
        cmp("dem_share_p", float(res.pvalues["dem_share"]), ssrec["dem_share_p"])
    return checks, dis, rec.get("verdict")


def main():
    if not PIPELINE_JSON.exists():
        raise SystemExit("results/causal/iv_district_fe_rescue_results.json not found — "
                         "run src/causal/iv_district_fe_rescue.py first")
    df = pd.read_parquet(CANON).dropna(subset=REQUIRED)
    margins = pd.read_parquet(MARGINS)
    pipeline = json.loads(PIPELINE_JSON.read_text())
    out = {"source_reverified": str(PIPELINE_JSON.relative_to(ROOT)).replace(chr(92), "/"),
           "arms": {}}
    all_match = True
    for arm_key, fe_col in ARMS.items():
        arm_out = {"records": []}
        for rec in pipeline.get(arm_key, []):
            checks, dis, verdict = verify_record(df, margins, arm_key, fe_col, rec)
            ok = all(checks.values())
            all_match = all_match and ok
            arm_out["records"].append({
                "bandwidth_pp": rec["bandwidth_pp"],
                "identified": rec.get("identified"),
                "matches_pipeline": ok,
                "checks": checks,
                "discrepancies": dis,
                "verdict": verdict,
            })
            print(f"\n=== {arm_key} +/-{rec['bandwidth_pp']}pp "
                  f"(identified={rec.get('identified')}) ===")
            print(f"  matches_pipeline: {ok}")
            for dd in dis:
                print(f"    DISCREPANCY: {dd}")
        arm_out["all_match"] = all(r["matches_pipeline"] for r in arm_out["records"])
        out["arms"][arm_key] = arm_out
    out["all_arms_match"] = bool(all_match)
    out["note"] = ("For identified specs we re-fit fresh and compare 4dp; for "
                   "not-identified specs the honest match is the concordant "
                   "verdict and counts. A rescue that passes an F threshold is independently "
                   "re-derived here before it can be trusted.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWROTE {OUT.relative_to(ROOT)}")
    print(f"all_arms_match: {all_match}")

    # Round 11: machine-countable status
    sys.path.insert(0, str(ROOT / 'src' / 'verification'))
    from _shared import record_status  # noqa: E402
    _checks = []
    for arm_key, arm in out["arms"].items():
        for rec in arm["records"]:
            for name, ok in rec["checks"].items():
                _checks.append({'check': f"{arm_key}/{rec['bandwidth_pp']}pp.{name}",
                                'expected': 'match',
                                'actual': 'match' if ok else 'MISMATCH',
                                'tolerance': TOL, 'match': bool(ok)})
    record_status('verify_iv_district_fe_rescue', _checks,
                  note='Independent rescue-arm re-derivation vs results/causal/'
                       'iv_district_fe_rescue_results.json.')
    if not all_match:
        sys.exit(1)


if __name__ == "__main__":
    main()
