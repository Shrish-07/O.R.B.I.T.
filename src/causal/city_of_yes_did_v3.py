"""City of Yes DiD v3 additions (Round 13, R7).

Adds, WITHOUT touching results/causal/city_of_yes_did_results.json or
city_of_yes_did_results_v2.json:

  * wild-cluster-bootstrap-t p and CI (Webb 6-point weights, 9,999 reps,
    seed 42) for the 2022-2025 window specifications — incl. 2024 as pre
    and excl. 2024 — and for the district-specific-linear-trends
    specification;
  * a continuous-treatment arm: dem_share_pres_vintage_aw x post in the
    2022-2025 window, both excl. 2024 and incl. 2024 as pre (full wild
    inference on the interaction).

All designs reuse city_of_yes_did.py's prepare()/design_matrix() and the
wild engine, reading modeling_dataset_canonical_v3.parquet for the _aw
measure. Clustered by council district throughout.

Output: results/causal/city_of_yes_did_results_v3.json

Run: python src/causal/city_of_yes_did_v3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import city_of_yes_did as did  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "canonical" / "modeling_dataset_canonical_v3.parquet"
ROLL = ROOT / "data" / "processed" / "city_of_yes_roll_calls.parquet"
OUT = ROOT / "results" / "causal" / "city_of_yes_did_results_v3.json"

REPS = 9999
SEED = 42
AW_MEASURE = "dem_share_pres_vintage_aw"


def wild_report(d, extra_cols, focus_col):
    """Full inference (incl. wild-cluster-bootstrap-t) for one DiD design."""
    d2 = d.copy()
    d2["CounDist"] = d2["CounDist"].astype(int)
    d2["sale_year"] = d2["sale_year"].astype(int)
    dist_fe = pd.get_dummies(d2["CounDist"], prefix="district",
                             drop_first=True, dtype=float)
    year_fe = pd.get_dummies(d2["sale_year"], prefix="sale_year",
                             drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=d2.index, name="const"),
                   d2[did.CONTROLS].astype(float),
                   d2[extra_cols].astype(float), dist_fe, year_fe], axis=1)
    keep = [c for c in X.columns
            if c == "const" or X[c].nunique(dropna=False) > 1]
    X = X[keep]
    fi = list(X.columns).index(focus_col)
    sys.path.insert(0, str(ROOT / "src" / "analysis"))
    from wild_cluster_inference import WildClusterOLS
    ols = WildClusterOLS(d2["target_log_price"].astype(float).to_numpy(),
                         X.to_numpy(dtype=np.float64),
                         d2["CounDist"].to_numpy(), fi).fit()
    return ols.report(reps=REPS, seed=SEED)


def main():
    for p in [CANON, ROLL]:
        if not p.exists():
            raise SystemExit(f"MISSING INPUT: {p}")
    df = pd.read_parquet(CANON)
    df = df.dropna(subset=did.REQUIRED)
    roll = pd.read_parquet(ROLL)
    d = did.prepare(df, roll)

    out = {
        "purpose": ("Round 13 (R7): wild-bootstrap inference for the "
                    "2022-25 window specs and the district-trends spec, "
                    "plus the continuous-treatment dem_share_pres_vintage_aw "
                    "x post arm in the 2022-25 window. The original "
                    "city_of_yes_did_results.json and _v2.json are "
                    "unchanged."),
        "outcome": "target_log_price",
        "controls": did.CONTROLS,
        "fixed_effects": ["council district FE", "sale year FE"],
        "clustering": "SEs clustered by council district",
        "wild_bootstrap": {"weights": "webb6", "reps": REPS, "seed": SEED},
    }

    # ---- (1) 2022-25 window specs with wild p + CI --------------------------
    dw = d[d["sale_year"].between(2022, 2025)].copy()
    dw_strict = dw[dw["sale_year"] != 2024].copy()
    for tag, dd in [("window_2022_2025_incl_2024_as_pre", dw),
                    ("window_2022_2025_excl_2024", dw_strict)]:
        dd = dd.copy()
        dd["post"] = (dd["sale_year"] >= 2025).astype(float)
        dd["yes_x_post"] = dd["yes_district"] * dd["post"]
        rep = wild_report(dd, ["yes_x_post"], "yes_x_post")
        rep["post_definition"] = ("sale_year == 2025 (pre = 2022-23"
                                  + (" and 2024" if "incl" in tag else "")
                                  + ")")
        out[tag] = rep
        print(f"  {tag}: coef={rep['coef']:+.5f} p_wild="
              f"{rep['p_wild_cluster_bootstrap_t']:.4f} N={rep['n']:,} "
              f"G={rep['n_clusters_G']}", flush=True)

    # ---- (2) district-specific linear trends with wild p + CI ----------------
    dm = d[d["sale_year"] != 2024].copy()
    dm["post"] = (dm["sale_year"] >= 2025).astype(float)
    dm["yes_x_post"] = dm["yes_district"] * dm["post"]
    dist_fe = pd.get_dummies(dm["CounDist"], prefix="district",
                             drop_first=True, dtype=float)
    t = (dm["sale_year"].astype(float) - 2016.0)
    dm = pd.concat([dm, dist_fe.mul(t.values, axis=0).add_suffix("_xtrend")],
                   axis=1)
    trend_cols = [c for c in dm.columns if c.endswith("_xtrend")]
    rep = wild_report(dm, ["yes_x_post"] + trend_cols, "yes_x_post")
    rep["note"] = ("main design + district FE x (sale_year - 2016) linear "
                   "trends + sale-year FE, sample excl. 2024")
    out["district_specific_linear_trends"] = rep
    print(f"  district_trends: coef={rep['coef']:+.5f} p_wild="
          f"{rep['p_wild_cluster_bootstrap_t']:.4f} N={rep['n']:,} "
          f"G={rep['n_clusters_G']}", flush=True)

    # ---- (3) continuous treatment: dem_share_pres_vintage_aw x post ----------
    dc_all = d.dropna(subset=[AW_MEASURE]).copy()
    dc_all = dc_all[dc_all["sale_year"].between(2022, 2025)]
    arms = {}
    for tag, keep_2024 in [("excl_2024", False), ("incl_2024_as_pre", True)]:
        dc = dc_all.copy() if keep_2024 else dc_all[dc_all["sale_year"] != 2024].copy()
        dc["post"] = (dc["sale_year"] >= 2025).astype(float)
        dc["dsp_x_post"] = dc[AW_MEASURE].astype(float) * dc["post"]
        rep = wild_report(dc, [AW_MEASURE, "dsp_x_post"], "dsp_x_post")
        arms[tag] = {
            "treatment": f"{AW_MEASURE} (continuous) x post(2025), "
                         "2022-25 window" + ("" if keep_2024
                                             else ", 2024 excluded"),
            "interaction_wild_inference": rep,
            "n": int(rep["n"]), "n_clusters_G": int(rep["n_clusters_G"]),
        }
        print(f"  continuous_aw/{tag}: interaction coef={rep['coef']:+.5f} "
              f"p_wild={rep['p_wild_cluster_bootstrap_t']:.4f} "
              f"N={rep['n']:,}", flush=True)
    out["continuous_treatment_dem_share_pres_vintage_aw_x_post"] = arms
    out["continuous_treatment_note"] = (
        "dem_share_pres_vintage_aw is the Round-13 area-weighted, "
        "vintage-corrected presidential two-party share; the interaction "
        "tests whether more Democratic districts show differential 2025 "
        "price trends in the City of Yes window. Both 2024-excluded and "
        "2024-as-pre variants are reported.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

