"""IV rescue attempts (Round 6, Task 1) — a specification search over the two
directions the Round-5 negative result left open, using only data already on
disk; nothing is fetched, scraped, or requested.

Round 5 (src/causal/iv_district_fe_regression.py -> results/causal/
iv_district_fe_results.json) established that the primary design — CounDist FE
+ sale_year FE with a prior-cycle close-race instrument Z varying only at the
(district, prior-cycle) cell level — is NOT IDENTIFIED at any of +/-5/10/15/20 pp:
Z lies exactly in the column space of the district + sale_year FE block
(residual ~1e-11). This rescue script runs, records, and prints ALL rescue
attempts, identified or not — the output is the full specification trail, not
a filtered "what worked" list:

  1a. BOROUGH-FE VARIANT. Identical instrument construction, sample exclusion,
      control set, rank checks and IV2SLS machinery as Round 5; only the 51
      district dummies are replaced with the 5 borough dummies (drop_first).
      Bandwidths 5/10/15/20 pp.

  1b. WIDER-BANDWIDTH SWEEP on the original district-FE design. Bandwidths
      20/25/30/35/40 pp — 20 repeated as the anchor point (Round 5 found the
      instrument block one rank short there, 24 of 25). For each bandwidth the
      SAME rank check runs first; the full IV2SLS is fitted only if the design
      reaches full rank at that bandwidth.

Honesty obligations (carried into the JSON and the run log):
  * every bandwidth, every variant appears in the output, identified or not;
  * every record carries its own plain-language credibility note;
  * a looser FE or a wider bandwidth that manages to identify is a lower-
    confidence design than the Round-5 primary, and its record says so itself.

The Round-5 primary file results/causal/iv_district_fe_results.json is NOT
modified.

Output: results/causal/iv_district_fe_rescue_results.json

Run:
    python src/causal/iv_district_fe_rescue.py
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
OUT = ROOT / "results" / "causal" / "iv_district_fe_rescue_results.json"

BANDWIDTHS_RESCUE_A = [5, 10, 15, 20]
BANDWIDTHS_RESCUE_B = [20, 25, 30, 35, 40]
WEAK_F_THRESHOLD = 10.0

CONTROLS = ["yearbuilt", "numfloors", "landuse", "commfar",
            "RESIDENTIAL UNITS", "COMMERCIAL UNITS"]
REQUIRED = (["target_log_price", "dem_share", "CounDist", "BOROUGH", "sale_year"]
            + CONTROLS)

CRED_BOROUGH = ("borough FE controls only for 5 region-level confounders, not "
                "the 51 district-level ones the district-FE design (and Model 4) "
                "control for — a materially weaker design even where it "
                "identifies")
CRED_BANDWIDTH = ("as bandwidth widens beyond the original ±20pp cap, races "
                  "included are decreasingly 'close' and the as-good-as-random "
                  "assumption underlying the instrument gets progressively weaker "
                  "— treat any identified result here as a lower-confidence "
                  "robustness probe, not a primary estimate")


def build_instrument_map(margins, bandwidth):
    """(council_district, prior_cycle_year) -> Z in {0,1}; identical to Round 5."""
    zmap = {}
    close = margins[margins["margin_pct"] <= bandwidth]
    for r in close.itertuples():
        if r.winner_party == "Democratic":
            zmap[(int(r.council_district), int(r.year))] = 1.0
        elif r.winner_party == "Republican":
            zmap[(int(r.council_district), int(r.year))] = 0.0
    return zmap


def build_sample(df, zmap):
    d = df.copy()
    d["prior_cycle"] = d["election_year"].astype(int) - 4
    d["Z"] = [zmap.get((cd, pc), np.nan)
              for cd, pc in zip(d["CounDist"].astype(int), d["prior_cycle"])]
    return d.dropna(subset=["Z"])


def build_design(d, fe_col):
    """exog (const + controls + fe_col dummies + sale_year dummies), endog
    (dem_share), instrument (Z), dependent (target_log_price), cluster key
    (CounDist). fe_col is "CounDist" (Round-5 structure) or "BOROUGH" (rescue 1a).
    The verifier replicates this construction exactly."""
    d2 = d.copy()
    d2["CounDist"] = d2["CounDist"].astype(int)
    d2["sale_year"] = d2["sale_year"].astype(int)
    unit_fe = pd.get_dummies(d2[fe_col], prefix=fe_col, drop_first=True, dtype=float)
    year_fe = pd.get_dummies(d2["sale_year"], prefix="sale_year", drop_first=True,
                             dtype=float)
    exog = pd.concat([pd.Series(1.0, index=d2.index, name="const"),
                      d2[CONTROLS].astype(float), unit_fe, year_fe], axis=1)
    return exog, d2["target_log_price"].astype(float), d2["dem_share"].astype(float), \
        d2["Z"].rename("z_close_dem").astype(float), d2["CounDist"]


def rank_check(exog, endog, instr):
    n_reg = exog.shape[1] + 1
    rx = int(np.linalg.matrix_rank(np.column_stack(
        [exog.to_numpy(dtype=float), endog.to_numpy(dtype=float).reshape(-1, 1)])))
    rz = int(np.linalg.matrix_rank(np.column_stack(
        [exog.to_numpy(dtype=float), instr.to_numpy(dtype=float).reshape(-1, 1)])))
    return {"design_params": n_reg, "design_rank": rx,
            "instrument_design_rank": rz,
            "identified": bool(rx == n_reg and rz == n_reg)}


def fit_one(exog, dep, endog, instr, clusters):
    cov_type_used = "clustered"
    try:
        res = IV2SLS(dep, exog, endog, instr).fit(cov_type="clustered", clusters=clusters)
    except Exception:
        res = IV2SLS(dep, exog, endog, instr).fit(cov_type="unadjusted")
        cov_type_used = "unadjusted (clustered covariance singular at this bandwidth)"
    fs = res.first_stage
    fsd = fs.diagnostics
    f_stat_val = fsd.loc["dem_share"]["f.stat"] if "dem_share" in fsd.index else fsd.iloc[0]["f.stat"]
    f_stat = float(f_stat_val)
    ind = fs.individual["dem_share"]
    out = {
        "cov_type_used": cov_type_used,
        "first_stage": {
            "equation": "dem_share ~ Z + FE + sale_year_FE + controls",
            "instrument_coef": float(ind.params["z_close_dem"]),
            "instrument_se": float(ind.std_errors["z_close_dem"]),
            "instrument_p": float(ind.pvalues["z_close_dem"]),
            "f_stat": f_stat,
        },
        "second_stage": {
            "equation": "target_log_price ~ dem_share_hat + FE + sale_year_FE + controls",
            "dem_share_coef": float(res.params["dem_share"]),
            "dem_share_se": float(res.std_errors["dem_share"]),
            "dem_share_p": float(res.pvalues["dem_share"]),
            "r_squared": float(res.rsquared),
            "n": int(res.nobs),
        },
    }
    out["verdict"] = ("instrument appears strong enough to interpret" if f_stat > WEAK_F_THRESHOLD
                      else "instrument is weak at this bandwidth — treat as suggestive, not identified")
    return out


def run_variant(df, margins, bandwidths, fe_col, credibility_note):
    recs = []
    for bw in bandwidths:
        zmap = build_instrument_map(margins, bw)
        d = build_sample(df, zmap)
        exog, dep, endog, instr, clusters = build_design(d, fe_col)
        rk = rank_check(exog, endog, instr)
        rec = {
            "bandwidth_pp": bw,
            "fe_variant": fe_col,
            "credibility_note": credibility_note,
            "n_obs": int(len(d)),
            "n_close_cells": int(d[["CounDist", "election_year"]].drop_duplicates().shape[0]),
            "n_districts": int(d["CounDist"].nunique()),
            "n_z_cells": len(zmap),
            "design_params": rk["design_params"],
            "design_rank": rk["design_rank"],
            "instrument_design_rank": rk["instrument_design_rank"],
            "identified": rk["identified"],
        }
        if rk["identified"]:
            rec.update(fit_one(exog, dep, endog, instr, clusters))
        else:
            rec["second_stage"] = None
            rec["first_stage"] = None
            rec["verdict"] = ("NOT IDENTIFIED at this bandwidth — design matrix "
                              "rank-deficient; treating as suggestive at most")
        recs.append(rec)
    return recs


def main():
    if not CANON.exists() or not MARGINS.exists():
        raise SystemExit("Required inputs missing; run the data pipeline and "
                         "build_council_election_margins.py first")
    df = pd.read_parquet(CANON).dropna(subset=REQUIRED)
    margins = pd.read_parquet(MARGINS)
    results = {
        "purpose": ("Round 6 rescue: specification search over the two openings the "
                    "Round-5 not-identified finding left open, recorded in full "
                    "(identified or not) with per-record credibility notes."),
        "controls": CONTROLS,
        "rescue_1a_borough_fe_variant": run_variant(df, margins, BANDWIDTHS_RESCUE_A, "BOROUGH", CRED_BOROUGH),
        "rescue_1b_wider_bandwidth_district_fe": run_variant(df, margins, BANDWIDTHS_RESCUE_B, "CounDist", CRED_BANDWIDTH),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT.relative_to(ROOT)}\n")
    print("=== Rescue 1a: borough FE (weaker confounder control — only 5 regions) ===")
    for r in results["rescue_1a_borough_fe_variant"]:
        s = (f"  +/-{r['bandwidth_pp']:>2d}pp: identified={r['identified']} "
             f"rank={r['design_rank']}/{r['design_params']} (instr {r['instrument_design_rank']}) ")
        if r["identified"]:
            fs, ss = r["first_stage"], r["second_stage"]
            s += (f"F={fs['f_stat']:.3f} dem_share={ss['dem_share_coef']:.4f} "
                  f"(SE {ss['dem_share_se']:.4f}, p={ss['dem_share_p']:.3g})")
        print(s)
        print(f"         verdict: {r['verdict']}")
    print("\n=== Rescue 1b: wider bandwidth, original district FE ===")
    fr = [r for r in results["rescue_1b_wider_bandwidth_district_fe"] if r["identified"]]
    for r in results["rescue_1b_wider_bandwidth_district_fe"]:
        s = (f"  +/-{r['bandwidth_pp']:>2d}pp: identified={r['identified']} "
             f"rank={r['design_rank']}/{r['design_params']} (instr {r['instrument_design_rank']}) "
             f"cells={r['n_close_cells']}")
        if r["identified"]:
            fs, ss = r["first_stage"], r["second_stage"]
            s += (f" F={fs['f_stat']:.3f} dem_share={ss['dem_share_coef']:.4f} "
                  f"(SE {ss['dem_share_se']:.4f}, p={ss['dem_share_p']:.3g})")
        print(s)
        print(f"         verdict: {r['verdict']}")
    if fr:
        print(f"\nNOTE: first full-rank bandwidth in district-FE sweep: +/-{fr[0]['bandwidth_pp']}pp "
              "— treat as a lower-confidence robustness probe (see credibility notes).")
    else:
        print("\nNOTE: no bandwidth in the wider district-FE sweep reached full rank; "
              "the Round-5 not-identified finding holds at every wider bandwidth tried.")
    print("\nBoth rescue arms exit 0 regardless of identification — a failed rescue is a "
          "finding, not a pipeline failure.")


if __name__ == "__main__":
    main()
