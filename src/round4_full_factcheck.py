"""Round 4 master fact-check script.

Recomputes EVERY quantitative claim in the manuscript fresh from raw/canonical data
or raw prediction files (Tier 1), falling back to existing artifacts (Tier 2), and
stating Tier 3 when a check is not feasible in this repo.

Outputs results/round4_full_factcheck.json (machine-readable), consumed by the report writer.
"""
from __future__ import annotations
import json, math, re, subprocess
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CANON_V2 = ROOT / "data" / "canonical" / "modeling_dataset_canonical_v2.parquet"
CANON_V1 = ROOT / "data" / "canonical" / "modeling_dataset_canonical.parquet"
CANON_SCHEMA = ROOT / "models" / "artifacts" / "canonical_schema.json"
IDEO_BY_COUNCIL = ROOT / "data" / "processed" / "ideology_by_council.parquet"
DISTRICT_IDEOLOGY = ROOT / "data" / "processed" / "district_ideology.parquet"
CROSSWALK = ROOT / "data" / "processed" / "ed_to_council_crosswalk.parquet"
TRAIN_SPLIT = ROOT / "data" / "splits" / "all_years_train.parquet"
TEST_SPLIT = ROOT / "data" / "splits" / "all_years_test.parquet"
MORAN_JSON = ROOT / "results" / "morans_i_results.json"
REG_TABLE = ROOT / "results" / "regression_table.csv"
REG_FULL = ROOT / "results" / "regression_model3_full_summary.txt"
MODEL_COMPARE = ROOT / "docs" / "model_comparison_table.csv"
MODEL_DIR = ROOT / "models"
ARTIFACTS = ROOT / "models" / "artifacts"
POL_SHAP = ARTIFACTS / "lgbm_all_years_political_shap_summary.json"
BASE_SHAP = ARTIFACTS / "lgbm_all_years_base_shap_summary.json"
SCEN_CSV = ROOT / "results" / "scenario_comparison_clean.csv"
CD_BOROUGH_MAP = ROOT / "results" / "council_district_borough_map.csv"
SHAP_TOP10_PNG = ROOT / "docs" / "figures" / "shap_top10.png"
PREDS_DIR = ROOT / "experiments" / "predictions"
OUT = ROOT / "results" / "round4_full_factcheck.json"


def load_canon():
    return CANON_V2 if CANON_V2.exists() else CANON_V1


CANON = load_canon()


def num(x, nd=6):
    return None if x is None else round(float(x), nd)

PAPER_A1 = """District,Borough,Liberal,Conservative,MixedGov
1,Manhattan,6.85,-2.94,5.69
2,Manhattan,4.97,0.77,-2.87
3,Manhattan,7.13,-0.56,-13.03
4,Manhattan,2.54,-1.31,12.73
5,Manhattan,1.90,-2.23,13.32
6,Manhattan,2.83,-3.18,-5.55
7,Manhattan,4.11,-0.82,-11.15
8,Manhattan,8.15,-0.38,-12.38
9,Manhattan,6.57,-0.45,-11.25
10,Manhattan,2.33,-7.29,-16.46
11,Bronx,6.27,-0.56,1.49
12,Bronx,0.53,0.11,-5.02
13,Bronx,4.73,-0.16,-2.00
14,Bronx,11.28,-0.42,-4.43
15,Bronx,6.42,0.70,-7.14
16,Bronx,5.13,1.05,-7.66
17,Bronx,7.49,0.17,-13.72
18,Bronx,4.49,0.12,-4.25
19,Queens,11.33,0.09,-0.64
20,Queens,6.91,-0.80,1.20
21,Queens,7.29,-0.90,-0.85
22,Queens,4.04,0.08,3.50
23,Queens,8.53,0.02,-0.88
24,Queens,6.99,-0.33,0.27
25,Queens,3.22,-4.98,-4.62
26,Queens,0.35,-0.45,0.42
27,Queens,9.81,-0.17,-3.09
28,Queens,10.16,-0.16,-5.63
29,Queens,5.27,-2.53,-6.17
30,Queens,2.39,-0.51,-7.59
31,Queens,30.77,0.33,-0.40
32,Queens,18.33,0.20,0.02
33,Brooklyn,1.60,-0.77,-16.73
34,Brooklyn,0.41,-0.44,-3.55
35,Brooklyn,1.16,-1.41,-11.29
36,Brooklyn,0.76,-0.85,-8.83
37,Brooklyn,0.54,-0.44,-2.88
38,Brooklyn,2.59,-0.09,9.04
39,Brooklyn,1.27,-1.16,-16.04
40,Brooklyn,9.89,-2.18,-9.56
41,Brooklyn,-1.06,-1.38,-2.10
42,Brooklyn,-1.21,-0.95,-3.98
43,Brooklyn,6.21,-1.06,1.23
44,Brooklyn,5.97,-1.79,-1.12
45,Brooklyn,9.37,0.42,6.88
46,Brooklyn,16.35,0.93,-7.52
47,Brooklyn,6.85,-0.59,2.16
48,Brooklyn,8.68,-0.68,-0.70
49,StatenIsland,27.65,0.08,-0.08
50,StatenIsland,30.02,0.08,0.25
51,StatenIsland,32.44,0.14,-0.01
"""


def check_abstract(df, results):
    n = int(len(df))
    results["A_transaction_count"] = {"paper": 514618, "computed": n, "match": n == 514618, "tier": 1, "source": str(CANON.relative_to(ROOT))}
    cd_col = "CounDist" if "CounDist" in df.columns else next((c for c in df.columns if "counc" in c.lower()), None)
    n_cd = int(df[cd_col].nunique()) if cd_col else None
    results["A_council_districts"] = {"paper": 51, "computed": n_cd, "match": n_cd == 51, "tier": 1, "source": str(CANON.relative_to(ROOT))}
    ey = sorted([int(x) for x in df["election_year"].dropna().unique().tolist()])
    results["A_election_years"] = {"paper": [2017, 2021, 2025], "computed": ey, "match": ey == [2017, 2021, 2025], "tier": 1, "source": str(CANON.relative_to(ROOT))}
    rt = pd.read_csv(REG_TABLE)
    bio = rt[rt["Model"] == "Bivariate"]
    results["A_bivariate_beta"] = {"paper": 0.889, "computed": num(bio["Coef"].iloc[0]) if len(bio) else None, "match": len(bio) and abs(float(bio["Coef"].iloc[0]) - 0.889) < 1e-3, "tier": 2, "source": str(REG_TABLE.relative_to(ROOT))}
    results["A_bivariate_p"] = {"paper": "<0.001", "computed": float(bio["p"].iloc[0]) if len(bio) else None, "match": len(bio) and float(bio["p"].iloc[0]) < 0.001, "tier": 2, "source": str(REG_TABLE.relative_to(ROOT))}
    m3 = rt[rt["Model"] == "+ Borough FE"]
    dem_m3 = m3[m3["Variable"] == "dem_share"]
    beta_m3 = float(dem_m3["Coef"].iloc[0]) if len(dem_m3) else None
    shrink = (0.889 - beta_m3) / 0.889 * 100 if beta_m3 else None
    results["A_controlled_boroughFE_beta"] = {"paper": 0.184, "computed": num(beta_m3, 4), "match": beta_m3 is not None and abs(beta_m3 - 0.184) < 1e-3, "tier": 2, "source": str(REG_TABLE.relative_to(ROOT))}
    results["A_shrinkage_80pct"] = {"paper": "shrinking 80 percent", "computed": num(shrink, 2), "match": shrink is not None and 79 <= shrink <= 81, "tier": 1, "source": "arithmetic on beta values"}
    mc = pd.read_csv(MODEL_COMPARE)
    base = mc[mc["name"] == "lgbm_all_years_base"].iloc[0]
    pol = mc[mc["name"] == "lgbm_all_years_political"].iloc[0]
    results["A_lgbm_base_mae"] = {"paper": 0.428972, "computed": num(base["mae"], 6), "match": abs(float(base["mae"]) - 0.428972) < 1e-6, "tier": 2, "source": str(MODEL_COMPARE.relative_to(ROOT))}
    results["A_lgbm_political_mae"] = {"paper": 0.429342, "computed": num(pol["mae"], 6), "match": abs(float(pol["mae"]) - 0.429342) < 1e-6, "tier": 2, "source": str(MODEL_COMPARE.relative_to(ROOT))}
    shap = json.loads(POL_SHAP.read_text())["summary"]
    feats = [(e["feature"], float(e["mean_abs_shap"])) for e in shap]
    feats_sorted = sorted(feats, key=lambda x: -x[1])
    rank = next(i for i, (f, v) in enumerate(feats_sorted, 1) if f == "dem_share")
    results["A_shap_dem_share_rank"] = {"paper": "12th of 17", "computed": f"{rank}th of 17", "match": rank == 12, "tier": 2, "source": str(POL_SHAP.relative_to(ROOT))}
    mj = json.loads(MORAN_JSON.read_text())
    results["A_morans_rounding_price"] = {"paper": 0.596, "computed": round(mj["moran_i_price"], 3), "match": round(mj["moran_i_price"], 3) == 0.596, "tier": 2, "source": str(MORAN_JSON.relative_to(ROOT))}
    results["A_morans_rounding_dem"] = {"paper": 0.580, "computed": round(mj["moran_i_dem_share"], 3), "match": round(mj["moran_i_dem_share"], 3) == 0.580, "tier": 2, "source": str(MORAN_JSON.relative_to(ROOT))}


def check_intro_ranges(df, results):
    cd_col = "CounDist"
    dg = df.groupby(cd_col)["dem_share"].mean()
    results["B_dem_share_range"] = {"paper": "below 20% to exceeding 95%", "computed": [num(dg.min(), 4), num(dg.max(), 4)], "computed_min_pct": num(dg.min() * 100, 2), "computed_max_pct": num(dg.max() * 100, 2), "match": dg.min() < 0.21 and dg.max() > 0.95, "tier": 1, "source": "groupby CounDist dem_share mean on canonical"}
    pg = df.groupby(cd_col)["target_log_price"].median()
    results["B_logprice_range"] = {"paper": "near 12.8 to above 14.4", "computed": [num(pg.min(), 4), num(pg.max(), 4)], "min_below_12.81": bool(pg.min() < 12.81), "max_above_14.4": bool(pg.max() > 14.4), "match": bool(pg.min() < 12.81) and bool(pg.max() > 14.4), "tier": 1, "source": "groupby CounDist target_log_price median on canonical"}


def check_ideology(df, results):
    ideo = pd.read_parquet(IDEO_BY_COUNCIL)
    cd1 = ideo[(ideo["CounDist"] == 1) & (ideo["election_year"] == 2017)]
    cd1_ds = float(cd1["dem_share"].iloc[0]) if len(cd1) else None
    results["C_CD1_dem_share_2017"] = {"paper": 0.715908, "computed": num(cd1_ds, 6), "match": cd1_ds is not None and abs(cd1_ds - 0.715908) < 1e-6, "tier": 1, "source": str(IDEO_BY_COUNCIL.relative_to(ROOT))}
    di = pd.read_parquet(DISTRICT_IDEOLOGY)
    dcol = "CounDist" if "CounDist" in di.columns else next((c for c in di.columns if "coun" in c.lower()), None)
    cd1_di = di[di[dcol] == 1] if dcol else di.iloc[0:0]
    di_val = None
    if len(cd1_di) > 0:
        for c in ["district_ideology", "ideology"]:
            if c in cd1_di.columns:
                di_val = float(cd1_di.iloc[0][c]); break
    results["C_CD1_district_ideology_2017"] = {"paper": 0.711309, "computed": num(di_val, 6), "match": di_val is not None and abs(di_val - 0.711309) < 1e-3, "tier": 1, "source": str(DISTRICT_IDEOLOGY.relative_to(ROOT))}
    nmiss = int(df["dem_share"].isna().sum()); total = int(len(df)); coverage = (total - nmiss) / total
    results["C_dem_share_coverage"] = {"paper": "97.56% (12,558 of 514,618 missing = 2.44%)", "computed_missing": nmiss, "computed_coverage_pct": num(coverage * 100, 2), "computed_missing_pct": num(nmiss / total * 100, 2), "match": nmiss == 12558 and abs(coverage * 100 - 97.56) < 0.01, "tier": 1, "source": "df['dem_share'].isna() on canonical"}
    cw = pd.read_parquet(CROSSWALK)
    n_cw = int(len(cw)); n_distinct = int(cw["ElectDist"].nunique()); vc = cw["ElectDist"].value_counts(); n_gt1 = int((vc > 1).sum())
    straddle = vc[vc > 1].index.tolist(); cws = cw[cw["ElectDist"].isin(straddle)].copy()
    area_col = "area" if "area" in cws.columns else next((c for c in cws.columns if "area" in c.lower() or "share" in c.lower()), None)
    max_min = 0.0
    for ed, grp in cws.groupby("ElectDist"):
        areas = grp[area_col].values.astype(float); tot = areas.sum(); dom = areas.max(); ms = (tot - dom) / tot if tot > 0 else 0.0
        if ms > max_min: max_min = float(ms)
    results["C_crosswalk"] = {"paper": "5,783 rows, 4,264 distinct EDs, 1,383 in >1 row (32.4%), max minority-area share <=0.11%", "computed_rows": n_cw, "computed_distinct_eds": n_distinct, "computed_ed_gt1": n_gt1, "computed_gt1_pct": num(n_gt1 / n_distinct * 100, 1), "computed_max_minority_share": num(max_min, 6), "match": (n_cw == 5783 and n_distinct == 4264 and n_gt1 == 1383 and abs(n_gt1 / n_distinct * 100 - 32.4) < 0.1 and max_min <= 0.0011), "tier": 1, "source": str(CROSSWALK.relative_to(ROOT))}


def check_canonical_structure(df, results):
    schema = json.loads(CANON_SCHEMA.read_text())
    results["D_canonical_shape"] = {"paper": "514,618 rows x 53 columns", "computed": [int(schema["shape"][0]), int(schema["shape"][1])], "computed_df_shape": [int(len(df)), int(df.shape[1])], "match": schema["shape"] == [514618, 53], "tier": 2, "source": str(CANON_SCHEMA.relative_to(ROOT))}
    base_header = (MODEL_DIR / "lgbm_all_years_base.txt").read_text()
    pol_header = (MODEL_DIR / "lgbm_all_years_political.txt").read_text()
    def feat_names(header):
        m = re.search(r"feature_names=(.+)", header)
        return m.group(1).split() if m else None
    bn = feat_names(base_header)
    pn = feat_names(pol_header)
    results["D_base_feature_count"] = {"paper": 16, "computed": len(bn) if bn else None, "match": bn is not None and len(bn) == 16, "tier": 1, "source": "models/lgbm_all_years_base.txt feature_names= line"}
    results["D_political_feature_count"] = {"paper": 17, "computed": len(pn) if pn else None, "match": pn is not None and len(pn) == 17, "tier": 1, "source": "models/lgbm_all_years_political.txt feature_names= line"}
    both = df[["YEAR BUILT", "yearbuilt"]].dropna()
    n_both = int(len(both)); agree = int((both["YEAR BUILT"] == both["yearbuilt"]).sum()); disagree = n_both - agree
    max_diff = int((both["YEAR BUILT"] - both["yearbuilt"]).abs().max()); corr = float(both["YEAR BUILT"].corr(both["yearbuilt"]))
    results["D_yearbuilt_vs_YEARBUILT"] = {"paper": "both 464204; agree 410164 (88.4%); disagree 54040 (11.6%); max diff 2025; corr 0.146", "computed_both": n_both, "computed_agree": agree, "computed_agree_pct": num(agree / n_both * 100, 1), "computed_disagree": disagree, "computed_disagree_pct": num(disagree / n_both * 100, 1), "computed_max_diff": max_diff, "computed_corr": num(corr, 3), "match": (n_both == 464204 and agree == 410164 and disagree == 54040 and max_diff == 2025 and abs(corr - 0.146) < 1e-3), "tier": 1, "source": "canonical df[['YEAR BUILT','yearbuilt']].dropna()"}

def check_splits(df, results):
    tr = pd.read_parquet(TRAIN_SPLIT); te = pd.read_parquet(TEST_SPLIT)
    n_tr = int(len(tr)); n_te = int(len(te))
    tr_sy = tr["sale_year"] if "sale_year" in tr.columns else None
    te_sy = te["sale_year"] if "sale_year" in te.columns else None
    results["E_train_test_counts"] = {"paper": "Train 121,622 (sale year <2018); Test 56,817 (sale year ==2018)", "computed_train": n_tr, "computed_test": n_te, "train_saleyear_minmax": ([int(tr_sy.min()), int(tr_sy.max())] if tr_sy is not None else None), "test_saleyear_minmax": ([int(te_sy.min()), int(te_sy.max())] if te_sy is not None else None), "match": n_tr == 121622 and n_te == 56817, "tier": 1, "source": "data/splits/all_years_train.parquet, all_years_test.parquet"}
    vc = df["election_year"].value_counts().sort_index()
    d = {int(k): int(v) for k, v in vc.items()}
    results["E_election_year_dist"] = {"paper": "2017:178,439 (34.7%), 2021:285,617 (55.5%), 2025:50,562 (9.8%)", "computed": d, "computed_pct": {int(k): num(v / len(df) * 100, 1) for k, v in vc.items()}, "match": (d == {2017: 178439, 2021: 285617, 2025: 50562}), "tier": 1, "source": "df['election_year'].value_counts() on canonical"}
    n_2017 = int((df["election_year"] == 2017).sum())
    results["E_train_test_equals_2017"] = {"paper": "121,622 + 56,817 = 178,439 == election_year==2017", "computed_train_plus_test": n_tr + n_te, "computed_2017_subset": n_2017, "match": (n_tr + n_te) == 178439 and n_2017 == 178439, "tier": 1, "source": "TRAIN+TEST counts vs df[df['election_year']==2017] row count"}
    sub2017 = df[df["election_year"] == 2017]; sy_in_2017 = sub2017["sale_year"].value_counts().sort_index()
    results["E_2017_saleyear_breakdown"] = {"computed": {int(k): int(v) for k, v in sy_in_2017.items()}, "note": "election_year==2017 rows by sale_year; train picks sale_year<2018, test picks sale_year==2018", "tier": 1, "source": "df[df['election_year']==2017]['sale_year'].value_counts()"}


def check_spatial_descriptive(df, results):
    cd_col = "CounDist"
    g = df.groupby(cd_col).agg(mean_dem_share=("dem_share", "mean"), median_log_price=("target_log_price", "median"))
    r = float(np.corrcoef(g["mean_dem_share"], g["median_log_price"])[0, 1])
    results["F_bivariate_corr_51"] = {"paper": "r = 0.163", "computed": num(r, 3), "match": abs(r - 0.163) < 1e-3, "tier": 1, "source": "np.corrcoef on 51-district mean dem_share vs median log price"}
    d1 = g.loc[1] if 1 in g.index else None
    results["F_district1_range"] = {"paper": "dem_share 0.75-0.85, log price 14.4-14.6", "computed_dem_share": num(d1["mean_dem_share"], 4) if d1 is not None else None, "computed_log_price": num(d1["median_log_price"], 4) if d1 is not None else None, "match": (d1 is not None and 0.75 <= d1["mean_dem_share"] <= 0.85 and 14.4 <= d1["median_log_price"] <= 14.6), "tier": 1, "source": "groupby CounDist mean dem_share, median log price"}
    names = [11, 16, 18, 27]; sub = g.loc[names]
    vals = {int(i): (num(sub.loc[i, "mean_dem_share"], 4), num(sub.loc[i, "median_log_price"], 4)) for i in names}
    dem_ok = all(sub.loc[i, "mean_dem_share"] > 0.80 for i in names); price_ok = all(12.6 <= sub.loc[i, "median_log_price"] <= 13.2 for i in names)
    results["F_districts_11_16_18_27"] = {"paper": "dem_share above 0.80, log price 12.6-13.2", "computed": vals, "match": dem_ok and price_ok, "tier": 1, "source": "groupby CounDist mean dem_share, median log price"}
    ideo = pd.read_parquet(IDEO_BY_COUNCIL)
    i17 = ideo[ideo["election_year"] == 2017].set_index("CounDist")["dem_share"]
    i25 = ideo[ideo["election_year"] == 2025].set_index("CounDist")["dem_share"]
    diff = (i25 - i17).dropna()
    n_pos = int((diff > 0).sum()); n_neg = int((diff < 0).sum()); neg_districts = sorted([int(x) for x in diff[diff < 0].index.tolist()])
    results["F_49_of_51_shifted"] = {"paper": "49 of 51 shifted higher; only two opposite (16 and 17)", "computed_positive": n_pos, "computed_negative": n_neg, "computed_negative_districts": neg_districts, "match": n_pos == 49 and n_neg == 2 and neg_districts == [16, 17], "tier": 1, "source": "ideology_by_council.parquet 2025-2017 diff per CounDist"}


def check_morans(results):
    mj = json.loads(MORAN_JSON.read_text())
    results["G_morans_price"] = {"paper": "I=0.5959 (rounded 0.596), z=7.58, p=0.001", "computed_I": num(mj["moran_i_price"], 7), "computed_z": num(mj["z_price"], 2), "computed_p": mj["p_sim_price"], "match_I": bool(abs(mj["moran_i_price"] - 0.5959) < 1e-4), "match": bool(abs(mj["moran_i_price"] - 0.5959) < 1e-4 and abs(mj["p_sim_price"] - 0.001) <= 1e-6), "z_known_gap": "z differs (got 7.46 vs paper 7.58) - documented", "tier": 1, "source": "results/morans_i_results.json (re-run of src/compute_morans_i.py)"}
    results["G_morans_dem"] = {"paper": "I=0.5797 (rounded 0.580), z=7.69, p=0.001", "computed_I": num(mj["moran_i_dem_share"], 7), "computed_z": num(mj["z_dem_share"], 2), "computed_p": mj["p_sim_dem_share"], "match_I": bool(abs(mj["moran_i_dem_share"] - 0.5797) < 1e-4), "match": bool(abs(mj["moran_i_dem_share"] - 0.5797) < 1e-4 and abs(mj["p_sim_dem_share"] - 0.001) <= 1e-6), "z_known_gap": "z differs (got 7.99 vs paper 7.69) - documented", "tier": 1, "source": "results/morans_i_results.json (re-run of src/compute_morans_i.py)"}


def check_ols(results):
    rt = pd.read_csv(REG_TABLE)
    expected = {("Bivariate", "dem_share"): (0.889, 0.007), ("+ Controls", "dem_share"): (0.5741, 0.0071), ("+ Controls", "yearbuilt"): (-0.0001, 0.0), ("+ Controls", "numfloors"): (0.0043, 0.0002), ("+ Controls", "landuse"): (0.0778, 0.0013), ("+ Controls", "commfar"): (0.0585, 0.0006), ("+ Borough FE", "dem_share"): (0.1843, 0.0093), ("+ Borough FE", "yearbuilt"): (0.0000, 0.0), ("+ Borough FE", "numfloors"): (-0.0035, 0.0002), ("+ Borough FE", "landuse"): (0.0307, 0.0013), ("+ Borough FE", "commfar"): (0.0494, 0.0006)}
    cells = {}; all_match = True
    for (m, v), (ec, ese) in expected.items():
        row = rt[(rt["Model"] == m) & (rt["Variable"] == v)]
        if len(row) == 0:
            cells[f"{m}|{v}"] = {"paper": [ec, ese], "computed": None, "match": False}; all_match = False; continue
        cc = float(row["Coef"].iloc[0]); se = float(row["SE"].iloc[0]); match_c = abs(cc - ec) < 1e-4; match_s = abs(se - ese) < 1e-4
        cells[f"{m}|{v}"] = {"paper": [ec, ese], "computed": [num(cc, 4), num(se, 5)], "match": match_c and match_s}; all_match = all_match and match_c and match_s
    r2_expected = {"Bivariate": (0.0336, 470535), "+ Controls": (0.0932, 470535), "+ Borough FE": (0.1478, 470535)}
    r2_cells = {}
    for m, (er2, en) in r2_expected.items():
        row = rt[rt["Model"] == m].iloc[0]; r2_cells[m] = {"paper": [er2, en], "computed": [num(row["R2"], 4), int(row["N"])], "match": abs(float(row["R2"]) - er2) < 1e-4 and int(row["N"]) == en}; all_match = all_match and r2_cells[m]["match"]
    results["H_ols_table1_cells"] = {"computed_cells": cells, "computed_r2_n": r2_cells, "match": all_match, "tier": 2, "source": str(REG_TABLE.relative_to(ROOT))}
    full = REG_FULL.read_text()
    borough_expected = {"BOROUGH_2": (-0.8127, -138.285, "<0.001"), "BOROUGH_3": (-0.2850, -64.342, "<0.001"), "BOROUGH_4": (-0.6609, -142.580, "<0.001"), "BOROUGH_5": (-0.6583, -86.587, "<0.001")}
    bcells = {}; bmatch = True
    for term, (ec, et, ep) in borough_expected.items():
        m = re.search(rf"{term}\s+(-?[\d.]+)\s+[\d.\-e]+\s+(-?[\d.]+)\s+([\d.]+)", full)
        if m:
            cc = float(m.group(1)); tt = float(m.group(2)); pv = float(m.group(3))
            bcells[term] = {"paper": [ec, et, ep], "computed": [num(cc, 4), num(tt, 3), pv], "match": abs(cc - ec) < 1e-4 and abs(tt - et) < 0.01 and pv < 0.001}; bmatch = bmatch and bcells[term]["match"]
        else:
            bcells[term] = {"paper": [ec, et, ep], "computed": None, "match": False}; bmatch = False
    results["H_borough_dummies"] = {"computed": bcells, "match": bmatch, "tier": 2, "source": str(REG_FULL.relative_to(ROOT))}
    a2_expected = {"R2": 0.148, "Adj_R2": 0.148, "F_stat": 9067, "Prob_F": 0.0, "N": 470535, "Omnibus": 50100.014, "Prob_Omnibus": 0.0, "Jarque_Bera": 352014.397, "Prob_JB": 0.0, "Skew": 0.264, "Kurtosis": 7.204, "Condition_No": 3.64e4, "const": 13.7219, "dem_share_t": 19.859, "yearbuilt_coef": 1.848e-05, "yearbuilt_t": 1.559, "yearbuilt_p": 0.119, "numfloors_t": -19.215, "landuse_t": 23.812, "commfar_t": 79.687}
    def grab(pat):
        m = re.search(pat, full); return m.group(1) if m else None
    a2_computed = {"R2": grab(r"R-squared:\s+([\d.]+)"), "Adj_R2": grab(r"Adj. R-squared:\s+([\d.]+)"), "F_stat": grab(r"F-statistic:\s+([\d.]+)"), "Prob_F": grab(r"Prob \(F-statistic\):\s+([\d.]+)"), "N": grab(r"No. Observations:\s+(\d+)"), "Omnibus": grab(r"Omnibus:\s+([\d.]+)"), "Prob_Omnibus": grab(r"Prob\(Omnibus\):\s+([\d.]+)"), "Jarque_Bera": grab(r"Jarque-Bera \(JB\):\s+([\d.]+)"), "Prob_JB": grab(r"Prob\(JB\):\s+([\d.]+)"), "Skew": grab(r"Skew:\s+([\d.\-]+)"), "Kurtosis": grab(r"Kurtosis:\s+([\d.]+)"), "Condition_No": grab(r"Cond. No.\s+([\d.e+\-]+)")}
    def rowvals(term):
        m = re.search(rf"{term}\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+(-?[\d.\-]+)\s+([\d.]+)", full); return (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))) if m else None
    const = rowvals("const"); ds = rowvals("dem_share"); yb = rowvals("yearbuilt"); nf = rowvals("numfloors"); lu = rowvals("landuse"); cf = rowvals("commfar")
    a2_computed["const"] = const[0] if const else None; a2_computed["dem_share_t"] = ds[2] if ds else None; a2_computed["yearbuilt_coef"] = yb[0] if yb else None; a2_computed["yearbuilt_t"] = yb[2] if yb else None; a2_computed["yearbuilt_p"] = yb[3] if yb else None; a2_computed["numfloors_t"] = nf[2] if nf else None; a2_computed["landuse_t"] = lu[2] if lu else None; a2_computed["commfar_t"] = cf[2] if cf else None
    results["H_appendix_a2"] = {"paper": a2_expected, "computed": a2_computed, "tier": 2, "source": str(REG_FULL.relative_to(ROOT))}
    df = pd.read_parquet(CANON)
    farf = ["yearbuilt", "commfar", "residfar", "facilfar"]; mask_farf = df[farf[0]].isna()
    for c in farf[1:]: mask_farf = mask_farf & df[c].isna()
    n_farf = int(mask_farf.sum()); n_num_only = int((df["numfloors"].isna() & ~mask_farf).sum()); n_dem_miss = int(df["dem_share"].isna().sum()); dem_farf_overlap = int((df["dem_share"].isna() & mask_farf).sum()); dem_unique = n_dem_miss - dem_farf_overlap
    reg_vars = ["target_log_price", "dem_share", "yearbuilt", "numfloors", "landuse", "BOROUGH", "commfar"]; avail = [v for v in reg_vars if v in df.columns]; ma = df[avail[0]].isna()
    for v in avail[1:]: ma = ma | df[v].isna()
    n_union = int(ma.sum()); n_after = int((~ma).sum()); pct_shrink = (0.889 - 0.184) / 0.889 * 100
    results["H_missingness"] = {"paper": {"N_reduction": "514,618 -> 470,535 (44,083 rows)", "farf_group": 31692, "numfloors_only": 11918, "dem_share_missing": 12558, "dem_overlap_farf": 12424, "dem_unique": 134}, "computed": {"canonical_total": int(len(df)), "n_after_dropna": n_after, "union_removed": n_union, "farf_group": n_farf, "numfloors_only": n_num_only, "dem_share_missing": n_dem_miss, "dem_overlap_farf": dem_farf_overlap, "dem_unique": dem_unique}, "numfloors_known_gap": "11,930 vs paper 11,918 (Delta +12) - documented", "match_except_numfloors": (n_union == 44083 and n_after == 470535 and n_farf == 31692 and n_dem_miss == 12558 and dem_farf_overlap == 12424 and dem_unique == 134), "tier": 1, "source": "canonical recomputation"}
    beta_x = 0.1843 * 0.10; pct_exp = (math.exp(0.01843) - 1) * 100
    results["H_counterfactual_arithmetic"] = {"paper": "0.10 increase in dem_share -> 0.0184 log-point (~1.9%)", "computed_0.1843_x_0.10": num(0.1843 * 0.10, 6), "computed_exp_minus1_pct": num(pct_exp, 2), "match": abs(0.1843 * 0.10 - 0.01843) < 1e-5 and abs(pct_exp - 1.86) < 0.1, "tier": 1, "source": "arithmetic on 0.1843"}


def check_models(df, results):
    mc = pd.read_csv(MODEL_COMPARE)
    expected = {"lgbm_all_years_base": (0.428972, 0.541772), "lgbm_all_years_political": (0.429342, 0.541345), "rf_all_years_political_n200": (0.43705, 0.50663), "xgb_all_years_political_lr0.05_md6": (0.43922, 0.52869), "cat_all_years_political_lr0.05_d6": (0.45391, 0.50574), "ridge_all_years_political": (0.65580, 0.11138), "elasticnet_all_years_political": (0.69037, -0.00965)}
    rows = {}
    for name, (pe, pr) in expected.items():
        r = mc[mc["name"] == name]
        if len(r):
            cmae = float(r.iloc[0]["mae"]); cr2 = float(r.iloc[0]["r2"])
            tol = 5e-5 if name in ("lgbm_all_years_base", "lgbm_all_years_political") else 1e-4
            rows[name] = {"paper": [pe, pr], "computed": [num(cmae, 6), num(cr2, 6)], "match": abs(cmae - pe) < tol and abs(cr2 - pr) < tol}
    results["I_model_comparison_table"] = {"computed": rows, "tier": 2, "source": str(MODEL_COMPARE.relative_to(ROOT))}
    b = mc[mc["name"] == "lgbm_all_years_base"].iloc[0]; p = mc[mc["name"] == "lgbm_all_years_political"].iloc[0]
    dmae = float(p["mae"]) - float(b["mae"]); dr2 = float(p["r2"]) - float(b["r2"])
    results["I_delta_base_to_political"] = {"paper": "MAE +0.000370, R2 -0.000427", "computed_mae_delta": num(dmae, 6), "computed_r2_delta": num(dr2, 6), "match": abs(dmae - 0.000370) < 1e-6 and abs(dr2 - (-0.000427)) < 1e-6, "tier": 1, "source": "arithmetic on two model rows"}
    pol_preds_path = PREDS_DIR / "lgbm_all_years_political_test_preds.parquet"; base_preds_path = PREDS_DIR / "lgbm_all_years_base_prediction_intervals.parquet"
    spot = {}
    if pol_preds_path.exists():
        try:
            preds = pd.read_parquet(pol_preds_path); test = pd.read_parquet(TEST_SPLIT)
            if "BBL" in test.columns and "BBL" in preds.columns:
                t = test[["BBL", "target_log_price"]].copy(); pr = preds[["BBL", "pred_log_price"]].copy()
                merged = t.merge(pr, on="BBL", how="inner")
                if len(merged):
                    mae = float((merged["target_log_price"] - merged["pred_log_price"]).abs().mean()); spot["political_mae_recomputed"] = num(mae, 6); spot["political_n_rows_merged"] = int(len(merged)); spot["political_match_registry"] = abs(mae - 0.429342) < 1e-6
        except Exception as e:
            spot["political_recompute_error"] = str(e)
    else:
        spot["political_preds_file_missing"] = True
    results["I_mae_tier1_spotcheck"] = {"computed": spot, "tier": 1 if spot else 3, "source": "experiments/predictions/lgbm_all_years_political_test_preds.parquet joined to test split"}
    base_spot = {}
    if base_preds_path.exists():
        try:
            bp = pd.read_parquet(base_preds_path); base_spot["base_preds_cols"] = list(bp.columns); base_spot["base_preds_shape"] = list(bp.shape)
        except Exception as e:
            base_spot["error"] = str(e)
    results["I_base_preds_info"] = {"computed": base_spot, "tier": 3, "source": str(base_preds_path.relative_to(ROOT)) if base_preds_path.exists() else None}


def check_shap(results):
    shap = json.loads(POL_SHAP.read_text())["summary"]
    expected = {"Community Board": 0.27472, "landuse": 0.11688, "yearbuilt": 0.10600, "numfloors": 0.09722, "TAX CLASS AT TIME OF SALE": 0.08728, "BBL_pluto": 0.07654, "YEAR BUILT": 0.06552, "ZIP CODE": 0.05823, "Council District": 0.05419, "facilfar": 0.04773, "Census Tract 2020": 0.04500, "dem_share": 0.03473, "residfar": 0.02662, "commfar": 0.01983, "BOROUGH": 0.01099, "CounDist": 0.00446, "Election Year": 0.00000}
    rows = {}; all_match = True
    for e in shap:
        feat = e["feature"]; val = float(e["mean_abs_shap"]); paper_key = "Election Year" if feat == "election_year" else feat
        if paper_key in expected:
            pe = expected[paper_key]; m = abs(val - pe) < 1e-4; rows[feat] = {"paper": pe, "computed": num(val, 5), "match": m}; all_match = all_match and m
    results["J_shap_table3"] = {"paper_rows": 17, "computed_rows_in_file": len(shap), "computed_rows_matching_paper_17": len(rows), "computed": rows, "has_EASE-MENT_artifact": any(e["feature"] == "EASE-MENT" for e in shap), "match": all_match and len(rows) == 17, "tier": 2, "source": str(POL_SHAP.relative_to(ROOT))}
    bshap = json.loads(BASE_SHAP.read_text())["summary"]; cb_base = next(float(e["mean_abs_shap"]) for e in bshap if e["feature"] == "Community Board")
    results["J_base_community_board_shap"] = {"paper": 0.30101, "computed": num(cb_base, 5), "match": abs(cb_base - 0.30101) < 1e-3, "tier": 2, "source": str(BASE_SHAP.relative_to(ROOT))}


def check_counterfactual(results):
    src = (ROOT / "src" / "political_scenarios.py").read_text()
    mechanics = {"liberal_substr": "R3','R7'" in src, "conservative_substr": "R7','R3'" in src, "residfar_liberal_1.5": "* 1.5" in src, "residfar_conservative_0.7": "* 0.7" in src, "mixed_plus_0.15_clip": "+ 0.15).clip(0.0, 1.0)" in src, "filter_base_pred_ge_9": ">= 9.0" in src, "clip_100": "np.clip(out['pct_change'], -100.0, 100.0)" in src}
    results["K_mechanics"] = {"paper": {"liberal": "R3->R7 + residfar*1.5", "conservative": "R7->R3 + residfar*0.7", "mixed": "dem_share+0.15 clip[0,1]", "filter": "base_pred>=9.0 excluded", "clip": "[-100%,100%]"}, "computed": mechanics, "match": (mechanics["liberal_substr"] and mechanics["conservative_substr"] and mechanics["residfar_liberal_1.5"] and mechanics["residfar_conservative_0.7"] and mechanics["mixed_plus_0.15_clip"] and mechanics["filter_base_pred_ge_9"] and mechanics["clip_100"]), "tier": 1, "source": "src/political_scenarios.py (code reading)"}
    scen_dir = ROOT / "results" / "political_scenarios"; lib_all = pd.read_parquet(scen_dir / "liberal_policy_all_properties.parquet"); base_min = float(lib_all["base_pred"].min())
    lib_clip_upper = int((lib_all["pct_change"] >= 100.0).sum()); lib_clip_lower = int((lib_all["pct_change"] <= -100.0).sum())
    cons_all = pd.read_parquet(scen_dir / "conservative_policy_all_properties.parquet"); cons_upper = int((cons_all["pct_change"] >= 100.0).sum()); cons_lower = int((cons_all["pct_change"] <= -100.0).sum())
    mix_all = pd.read_parquet(scen_dir / "mixed_governance_all_properties.parquet"); mix_upper = int((mix_all["pct_change"] >= 100.0).sum()); mix_lower = int((mix_all["pct_change"] <= -100.0).sum())
    results["K_filter_and_clips"] = {"paper": "min base log-price ~9.56 above 9.0 threshold; liberal 109 clipped upper, 0 lower; conservative 9 upper, 0 lower; mixed 816 upper, 0 lower", "computed_min_base_pred": num(base_min, 4), "computed_lib_upper": lib_clip_upper, "computed_lib_lower": lib_clip_lower, "computed_cons_upper": cons_upper, "computed_cons_lower": cons_lower, "computed_mix_upper": mix_upper, "computed_mix_lower": mix_lower, "match_filter": bool(base_min >= 9.0), "match_clips": (lib_clip_upper == 109 and lib_clip_lower == 0 and cons_upper == 9 and cons_lower == 0 and mix_upper == 816 and mix_lower == 0), "tier": 1, "source": "results/political_scenarios/*_all_properties.parquet"}
    paper = pd.read_csv(pd.io.common.StringIO(PAPER_A1)); scen = pd.read_csv(SCEN_CSV)
    scen["Council District"] = scen["Council District"].astype(int); paper["District"] = paper["District"].astype(int)
    merged = paper.merge(scen, left_on="District", right_on="Council District", how="outer"); tol = 0.02; exceptions = []
    for _, row in merged.iterrows():
        d = int(row["District"])
        for pcol, scol in [("Liberal", "liberal_pct"), ("Conservative", "conservative_pct"), ("MixedGov", "mixed_gov_pct")]:
            pv = float(row[pcol]); sv = float(row[scol]); diff = abs(pv - sv)
            if diff > tol:
                exceptions.append({"district": d, "scenario": pcol, "paper": pv, "computed": num(sv, 4), "abs_diff": num(diff, 4)})
    lib_mean = float((paper["Liberal"]).mean()); lib_pos = int((paper["Liberal"] > 0).sum()); cons_mean = float((paper["Conservative"]).mean()); cons_neg = int((paper["Conservative"] < 0).sum()); mix_mean = float((paper["MixedGov"]).mean()); mix_pos = int((paper["MixedGov"] > 0).sum())
    lib_sorted = paper.sort_values("Liberal", ascending=False); lib_top5 = lib_sorted["District"].head(5).tolist(); lib_only_negative = lib_sorted[lib_sorted["Liberal"] < 0]["District"].tolist()
    cons_sorted = paper.sort_values("Conservative"); cons_two_most_neg = cons_sorted["District"].head(2).tolist()
    mix_sorted = paper.sort_values("MixedGov", ascending=False); mix_top5_pos = mix_sorted["District"].head(5).tolist(); mix_top5_neg = paper.sort_values("MixedGov")["District"].head(5).tolist()
    results["K_appendix_A1_diff"] = {"n_values_compared": 153, "n_exceptions": len(exceptions), "exceptions": exceptions, "tolerance_pp": tol, "summary_stats": {"liberal_mean_pct": num(lib_mean, 2), "liberal_positive": lib_pos, "conservative_mean_pct": num(cons_mean, 2), "conservative_negative": cons_neg, "mixed_mean_pct": num(mix_mean, 2), "mixed_positive": mix_pos}, "named_districts": {"liberal_top5": lib_top5, "liberal_only_negatives": lib_only_negative, "conservative_two_most_negative": cons_two_most_neg, "mixed_top5_positive": mix_top5_pos, "mixed_top5_negative": mix_top5_neg}, "paper_expected_named": {"liberal_top5": [51, 31, 50, 49, 32], "liberal_only_negatives": [41, 42], "conservative_two": [10, 25], "mixed_top5_positive": [5, 4, 38, 45, 1], "mixed_top5_negative": [33, 10, 39, 17, 3]}, "match_named_districts": (lib_top5 == [51, 31, 50, 49, 32] and lib_only_negative == [41, 42] and cons_two_most_neg == [10, 25] and mix_top5_pos == [5, 4, 38, 45, 1] and mix_top5_neg == [33, 10, 39, 17, 3]), "match_summary_stats": (abs(lib_mean - 7.42) < 0.05 and lib_pos == 49 and abs(cons_mean - (-0.78)) < 0.05 and cons_neg == 35 and abs(mix_mean - (-3.39)) < 0.05 and mix_pos == 14), "tier": 1, "source": "results/scenario_comparison_clean.csv vs paper table"}


def check_borough_map(df, results):
    mp = pd.read_csv(CD_BOROUGH_MAP); mp["CounDist"] = mp["CounDist"].astype(int)
    expected_ranges = {"Manhattan": (1, 10), "Bronx": (11, 18), "Queens": (19, 32), "Brooklyn": (33, 48), "Staten Island": (49, 51)}
    ranges = {}; all_match = True
    for boro, (lo, hi) in expected_ranges.items():
        sub = mp[mp["borough_name"] == boro]; ds = sorted(sub["CounDist"].tolist()); ranges[boro] = ds; m = len(ds) == (hi - lo + 1) and ds == list(range(lo, hi + 1)); ranges[boro + "_match"] = m; all_match = all_match and m
    canon_check = df.groupby("CounDist")["BOROUGH"].nunique(); canon_match = int((canon_check == 1).sum()) == 51
    results["L_borough_mapping"] = {"paper": expected_ranges, "computed_ranges": ranges, "canonical_one_borough_per_cddist": bool(canon_match), "match": all_match and canon_match, "tier": 1, "source": "results/council_district_borough_map.csv + canonical BOROUGH per CounDist"}


def check_sanity(results):
    bl_script = ROOT / "src" / "editorial_audit" / "verify_blacklist_fully.py"; bl_out = None
    if bl_script.exists():
        try:
            proc = subprocess.run(["python", str(bl_script)], cwd=str(ROOT), capture_output=True, text=True, timeout=300); bl_out = (proc.stdout + proc.stderr)
        except Exception as e:
            bl_out = f"ERROR: {e}"
    results["M_blacklist_verification"] = {"computed_output_tail": bl_out[-2500:] if bl_out else None, "ease_ment_in_trained_model": ("EASE-MENT in model feature_name(): False" in (bl_out or "")), "tier": 1, "source": str(bl_script.relative_to(ROOT)) if bl_script.exists() else None}
    try:
        from PIL import Image; im = Image.open(SHAP_TOP10_PNG); im.verify(); re_open = Image.open(SHAP_TOP10_PNG); results["M_shap_top10_png"] = {"opens_cleanly": True, "size": list(re_open.size), "mode": re_open.mode, "tier": 1, "source": str(SHAP_TOP10_PNG.relative_to(ROOT))}
    except Exception as e:
        results["M_shap_top10_png"] = {"opens_cleanly": False, "error": str(e), "tier": 1, "source": str(SHAP_TOP10_PNG.relative_to(ROOT))}
    tex = list(ROOT.rglob("*.tex")); docx = list(ROOT.rglob("*.docx"))
    results["M_manuscript_files"] = {"tex_files": [str(p.relative_to(ROOT)) for p in tex], "docx_files": [str(p.relative_to(ROOT)) for p in docx], "none_present": len(tex) == 0 and len(docx) == 0, "tier": 1, "source": "rglob *.tex, *.docx"}
    base_header = (MODEL_DIR / "lgbm_all_years_base.txt").read_text(); pol_header = (MODEL_DIR / "lgbm_all_years_political.txt").read_text()
    def feats(hdr):
        m = re.search(r"feature_names=(.+)", hdr); return m.group(1).split() if m else None
    bn = feats(base_header); pn = feats(pol_header)
    results["M_model_artifact_feature_counts"] = {"base_features_count": len(bn) if bn else None, "political_features_count": len(pn) if pn else None, "base_has_easement": any("EASE" in f.upper() for f in (bn or [])), "political_has_easement": any("EASE" in f.upper() for f in (pn or [])), "match": (bn is not None and pn is not None and len(bn) == 16 and len(pn) == 17 and not any("EASE" in f.upper() for f in bn) and not any("EASE" in f.upper() for f in pn)), "tier": 1, "source": "models/lgbm_all_years_base.txt, lgbm_all_years_political.txt headers"}


def main():
    results = {}; df = pd.read_parquet(CANON)
    print(f"Loaded canonical {CANON.name}, shape {df.shape}")
    check_abstract(df, results); check_intro_ranges(df, results); check_ideology(df, results); check_canonical_structure(df, results); check_splits(df, results); check_spatial_descriptive(df, results); check_morans(results); check_ols(results); check_models(df, results); check_shap(results); check_counterfactual(results); check_borough_map(df, results); check_sanity(results)
    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved results to {OUT}"); print("\n=== SUMMARY (match booleans) ===")
    for k, v in results.items():
        print(f"  {k}: {v.get('match', v.get('match_filter', v.get('match_named_districts', v.get('match_except_numfloors'))))}")


if __name__ == "__main__":
    main()
