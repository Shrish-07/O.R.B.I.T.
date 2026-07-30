"""Verify every unconfirmed number from the paper against real data.
Each check prints CONFIRMED or DISCREPANCY FOUND with actual numbers.
Do not skip any item.
"""
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CANON_V2 = Path("data/canonical/modeling_dataset_canonical_v2.parquet")
CANON_V1 = Path("data/canonical/modeling_dataset_canonical.parquet")
CROSSWALK = Path("data/processed/ed_to_council_crosswalk.parquet")
ED_IDEOLOGY = Path("data/processed/ed_ideology.parquet")
SCENARIO_DIR = Path("results/political_scenarios")
BASE_SHAP = Path("models/artifacts/lgbm_all_years_base_shap_summary.json")
POL_SHAP = Path("models/artifacts/lgbm_all_years_political_shap_summary.json")
POL_MODEL = Path("models/lgbm_all_years_political.txt")
POL_FEATURES = Path("models/artifacts/lgbm_all_years_political_features.json")

if not CANON_V2.exists():
    print(f"WARNING: canonical v2 not found ({CANON_V2}), trying v1")
    canon = CANON_V1
else:
    canon = CANON_V2

df = pd.read_parquet(canon)
print(f"\nCanonical dataset: {canon}, rows={len(df)}")

# =====================================================================
# 1. MISSINGNESS OVERLAP
# =====================================================================
print("\n=== 1. MISSINGNESS OVERLAP ===")

farf_group = ['yearbuilt', 'commfar', 'residfar', 'facilfar']
mask_farf = df[farf_group[0]].isna()
for c in farf_group[1:]:
    mask_farf = mask_farf & df[c].isna()
n_farf_missing = int(mask_farf.sum())
print(f"  farf (yearbuilt/commfar/residfar/facilfar) missing: {n_farf_missing} (paper says 31692)")
print(f"  STATUS: {'CONFIRMED' if n_farf_missing == 31692 else 'DISCREPANCY'}")

mask_numfloors = df['numfloors'].isna()
n_numfloors = int(mask_numfloors.sum())
n_numfloors_unique = int((mask_numfloors & ~mask_farf).sum())
print(f"\n  numfloors total missing: {n_numfloors}")
print(f"  numfloors additional (not in farf): {n_numfloors_unique} (paper says 11918)")
print(f"  STATUS: {'CONFIRMED' if n_numfloors_unique == 11918 else 'DISCREPANCY'}")

n_dem_miss = int(df['dem_share'].isna().sum())
overlap_dem_farf = int((df['dem_share'].isna() & mask_farf).sum())
print(f"\n  dem_share missing: {n_dem_miss}")
print(f"  dem_share overlap with farf: {overlap_dem_farf} (paper says 12424)")
print(f"  STATUS: {'CONFIRMED' if overlap_dem_farf == 12424 else 'DISCREPANCY'}")

reg_vars = ['numfloors', 'yearbuilt', 'commfar', 'residfar', 'facilfar', 'dem_share']
missing_any = df[reg_vars[0]].isna()
for v in reg_vars[1:]:
    missing_any = missing_any | df[v].isna()
n_union = int(missing_any.sum())
print(f"\n  Union across regression vars ({reg_vars}): {n_union} (paper says 44083)")
print(f"  STATUS: {'CONFIRMED' if n_union == 44083 else 'DISCREPANCY'}")

# =====================================================================
# 2. YEAR BUILT vs yearbuilt
# =====================================================================
print("\n=== 2. YEAR BUILT vs yearbuilt ===")
both = df[['YEAR BUILT', 'yearbuilt']].dropna()
n_both = len(both)
agree = int((both['YEAR BUILT'] == both['yearbuilt']).sum())
disagree = int((both['YEAR BUILT'] != both['yearbuilt']).sum())
pct_agree = agree / n_both * 100 if n_both > 0 else 0
pct_disagree = disagree / n_both * 100 if n_both > 0 else 0
max_diff = (both['YEAR BUILT'] - both['yearbuilt']).abs().max()
corr = both['YEAR BUILT'].corr(both['yearbuilt'])

print(f"  both populated: {n_both} (paper says 464204)")
print(f"STATUS: {'CONFIRMED' if n_both == 464204 else 'DISCREPANCY'}")
print(f"  agree: {agree} ({pct_agree:.1f}%) (paper says 410164/88.4%)")
print(f"  disagree: {disagree} ({pct_disagree:.1f}%) (paper says 54040/11.6%)")
print(f"  max disagreement: {max_diff:.0f} years (paper says 2025)")
print(f"  correlation: {corr:.4f} (paper says 0.146)")

status_agree = agree == 410164
status_diff = max_diff == 2025
status_corr = abs(corr - 0.146) < 0.001
all_ok = status_agree and status_diff and status_corr
print(f"STATUS: {'CONFIRMED' if all_ok else 'DISCREPANCY'}")

# =====================================================================
# 3. CROSSWALK
# =====================================================================
print("\n=== 3. CROSSWALK ===")
if CROSSWALK.exists():
    cw = pd.read_parquet(CROSSWALK)
    n_cw = len(cw)
    n_distinct_ed = cw['ElectDist'].nunique()
    ed_counts = cw['ElectDist'].value_counts()
    n_gt1 = int((ed_counts > 1).sum())
    print(f"  Crosswalk rows: {n_cw} (paper says 5783)")
    print(f"STATUS: {'CONFIRMED' if n_cw == 5783 else 'DISCREPANCY'}")
    print(f"  Distinct ElectDist: {n_distinct_ed} (paper says 4264)")
    print(f"STATUS: {'CONFIRMED' if n_distinct_ed == 4264 else 'DISCREPANCY'}")
    print(f"  ElectDist in >1 row: {n_gt1} (paper says 1383)")
    print(f"STATUS: {'CONFIRMED' if n_gt1 == 1383 else 'DISCREPANCY'}")

    area_col = next((c for c in cw.columns if 'area' in c.lower() or 'share' in c.lower()), None)
    if area_col:
        max_share = float(cw[area_col].max())
    else:
        non_key = [c for c in cw.columns if c not in ['ElectDist', 'CounDist', 'AD', 'ED']]
        ac = non_key[0] if non_key else cw.columns[-1]
        max_share = cw[ac].max()
        area_col = ac
    print(f"  Max {area_col}: {max_share:.6f} (paper says <= 0.0011)")
    print(f"STATUS: {'CONFIRMED' if max_share < 0.0012 else 'DISCREPANCY'}")
else:
    print("  SKIP: crosswalk not found at", CROSSWALK)

# =====================================================================
# 4. Council District 1, 2017
# =====================================================================
print("\n=== 4. Council District 1, 2017 ===")
if ED_IDEOLOGY.exists() and CROSSWALK.exists():
    ideo = pd.read_parquet(ED_IDEOLOGY)
    cw = pd.read_parquet(CROSSWALK)
    ideo['ElectDist'] = (ideo['AD'] * 1000 + ideo['ED']).astype(float)
    cw['ElectDist'] = cw['ElectDist'].astype(float)
    idea = ideo[['ElectDist', 'dem', 'rep']].dropna()
    idea['dem_share'] = idea['dem'] / (idea['dem'] + idea['rep'])
    merged = idea.merge(cw, on='ElectDist', how='inner')
    vote = merged.groupby('CounDist').agg(dem=('dem', 'sum'), rep=('rep', 'sum')).reset_index()
    vote['dem_share_sum'] = vote['dem'] / (vote['dem'] + vote['rep'])
    cd1 = vote[vote['CounDist'] == 1]
    if len(cd1) > 0:
        cd1_val = float(cd1['dem_share_sum'].iloc[0])
        print(f"  Council Dist 1 dem_share (vote-summed): {cd1_val:.6f} (paper says 0.711301)")
        print(f"STATUS: {'CONFIRMED' if abs(cd1_val - 0.711301) < 0.0001 else 'DISCREPANCY'}")

    # Try to load district_ideology from processed
    dist_ideo_path = Path("data/processed/district_ideology.parquet")
    if dist_ideo_path.exists():
        di = pd.read_parquet(dist_ideo_path)
        dist_col = 'CounDist' if 'CounDist' in di.columns else 'COUNCIL_DISTRICT'
        cd1_di = di[di[dist_col] == 1]
        if len(cd1_di) > 0:
            di_val = float(cd1_di.iloc[0]['district_ideology'] if 'district_ideology' in cd1_di.columns else cd1_di.iloc[0]['ideology'])
            print(f"  CD1 district_ideology (area-weighted): {di_val:.6f} (paper says 0.711309)")
            print(f"STATUS: {'CONFIRMED' if abs(di_val - 0.711309) < 0.001 else 'DISCREPANCY'}")
else:
    print("  Missing ed_ideology or crosswalk")

# =====================================================================
# 5. SCENARIO CLIPPING
# =====================================================================
print("\n=== 5. SCENARIO CLIPPING ===")
for name in ['liberal_policy', 'conservative_policy', 'mixed_governance']:
    path = SCENARIO_DIR / f"{name}_all_properties.parquet"
    if path.exists():
        sdf = pd.read_parquet(path)
        pct = sdf['pct_change']
        at_upper = int((pct >= 99.9).sum())
        at_lower = int((pct <= -99.9).sum())
        if name == 'liberal_policy':
            target_u = 109
        elif name == 'conservative_policy':
            target_u = 9
        else:
            target_u = 816
        st = 'CONFIRMED' if at_upper == target_u and at_lower == 0 else 'DISCREPANCY'
        print(f"  {name}: total={len(sdf)}, +100={at_upper}, -100={at_lower}, paper +100={target_u} => {st}")
    else:
        print(f"  {name}: FILE NOT FOUND ({path})")

# =====================================================================
# 6. BASE MODEL SHAP
# =====================================================================
print("\n=== 6. BASE MODEL SHAP ===")
if BASE_SHAP.exists():
    base_shap = json.loads(BASE_SHAP.read_text())
    print(f"  Full raw JSON keys: {list(base_shap.keys())}")
    print(f"  method: {base_shap.get('method', 'MISSING')}")
    # The summary key maps to a list of {feature, mean_abs_shap}
    summary = base_shap.get('summary', [])
    print(f"  Summary entries: {len(summary)}")
    for entry in summary:
        feat = entry.get('feature', entry)
        val = entry.get('mean_abs_shap', None)
        if feat == 'Community Board':
            print(f"  Community Board: mean_abs_shap={val} (paper says 0.30101)")
            if val is not None:
                cb_status = abs(val - 0.30101) < 0.001
                print(f"STATUS: {'CONFIRMED' if cb_status else 'DISCREPANCY'}")
        print(f"    Feature: {feat}, mean_abs_shap={val}")
else:
    print("  Base SHAP summary not found")

# =====================================================================
# 7. POLITICAL MODEL SHAP
# =====================================================================
print("\n=== 7. POLITICAL MODEL SHAP ===")
if POL_SHAP.exists():
    pol_shap = json.loads(POL_SHAP.read_text())
    print(f"  Full file contents: {json.dumps(pol_shap, indent=2)}")
    summary = pol_shap.get('summary', [])
    shap_names = [entry.get('feature', '') for entry in summary]
    print(f"\n  SHAP summary feature names ({len(shap_names)}): {shap_names}")
    has_easement = any('EASE' in n.upper() for n in shap_names)
    print(f"  EASE-MENT present in SHAP summary: {has_easement}")
else:
    print("  Political SHAP not found")
    shap_names = []
    pol_shap = {}
    has_easement = False
    summary = []

# 7b. Load actual trained model
print("\n=== 7b. TRAINED MODEL vs SHAP SUMMARY ===")
try:
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(POL_MODEL))
    model_features = list(booster.feature_name())
    print(f"  Model features ({len(model_features)}): {model_features}")
    has_ease_model = any('EASE' in f.upper() for f in model_features)
    print(f"  EASE-MENT in model feature_name(): {has_ease_model}")

    if POL_FEATURES.exists():
        feat_json = json.loads(POL_FEATURES.read_text())
        print(f"  Features JSON ({len(feat_json)}): {feat_json}")
        has_ease_json = any('EASE' in str(f).upper() for f in feat_json)
        print(f"  EASE-MENT in features JSON: {has_ease_json}")
    else:
        feat_json = []
        has_ease_json = False

    # Normalize comparison: SHAP summary uses spaces, model uses underscores
    shap_set = {n.replace(' ', '_') for n in shap_names}
    model_set = set(model_features)
    shap_not_model = shap_set - model_set
    model_not_shap = model_set - shap_set
    print(f"\n  SHAP summary features NOT in trained model: {shap_not_model}")
    print(f"  Model features NOT in SHAP summary: {model_not_shap}")

    # (a) Does every SHAP feature exist in the trained model?
    print(f"\n  (a) All SHAP features found in model: {len(shap_not_model) == 0}")

    # (b) EASE-MENT
    if has_easement:
        if has_ease_model:
            print("\n  (b) *** BLACKLIST BUG: EASE-MENT present in BOTH SHAP summary AND trained model feature list. ***")
        else:
            print("\n  (b) *** REPORT-GENERATION ARTIFACT: EASE-MENT in SHAP summary but NOT in trained model. ***")
            print("  The SHAP summary script iterated a wider column list than what was trained.")
            print("  EASE-MENT was never a trained feature (verified via booster.feature_name()).")
    else:
        if has_ease_model:
            print("\n  (b) *** EASE-MENT present in trained model but NOT in SHAP summary (unusual) ***")
        else:
            print("\n  (b) Neither SHAP summary nor trained model have EASE-MENT. Fine.")

except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== DONE - Verification Complete ===")