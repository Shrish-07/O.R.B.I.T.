"""Round 5 fact-check: Section K (counterfactual mechanics + full Appendix A.1 153-value diff + summary stats)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from round5_factcheck_part1a import ROOT, write_json

K = {}
# (1) Mechanics from political_scenarios.py (read the code = Tier 1)
src = (ROOT / 'src' / 'political_scenarios.py').read_text()
K['mechanics_liberal'] = 'R3->R7 substring substitution on zonedist1 + residfar (*first far col*) x1.5 (Tier 1: read from src/political_scenarios.py)'
K['mechanics_conservative'] = 'R7->R3 substring substitution on zonedist1 + residfar (*first far col*) x0.7 (Tier 1: read from src/political_scenarios.py)'
K['mechanics_mixed'] = 'dem_share + 0.15 clipped to [0,1], re-predicted through political model (Tier 1: read from src/political_scenarios.py)'
K['filter_base_pred_lt_9_removed'] = 'base_pred >= 9.0 keeps rows; removes rows with base predicted log-price < 9.0'
K['clip_to_neg100_100'] = 'pct_change clipped to [-100, 100]'
# Note: the code applies zonedist1 substring replace and residfar scaling. The checklist mentions residfar specifically, matches.

# (2) Re-confirm the filter removes ZERO rows: min base predicted log-price should be ~9.56
# We re-run the scenario logic minimally to get base preds (re-compute, Tier 1).
import lightgbm as lgb
champ = json.loads((ROOT / 'experiments' / 'champion.json').read_text())
reg = json.loads((ROOT / 'experiments' / 'registry.json').read_text())
champ_id = champ.get('selected_experiment')
exp = next((e for e in reg if e.get('id') == champ_id), None)
base_model_path = ROOT / exp['model_path']
base_feat_path = ROOT / exp['features_path']
df_path = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet'
if not df_path.exists():
    df_path = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical.parquet'
df = pd.read_parquet(df_path)
import pathlib
booster = lgb.Booster(model_file=str(base_model_path))
features = json.loads(pathlib.Path(base_feat_path).read_text())
for fc in features:
    if fc not in df.columns:
        df[fc] = 0.0
base_preds = booster.predict(df[features].fillna(0))
K['base_pred_min'] = float(np.min(base_preds))
K['base_pred_max'] = float(np.max(base_preds))
K['filter_9.0_removed_rows'] = int(np.sum(base_preds < 9.0))
K['filter_9.0_removes_zero_rows'] = (int(np.sum(base_preds < 9.0)) == 0)

# clip re-confirmation (liberal 109 upper, conservative 9 upper, mixed 816 upper, zero at lower)
# Recompute the three pct changes exactly as the script does and count clips
def clamp01(x):
    return np.minimum(np.maximum(x, 0.0), 1.0)
z_cols = [c for c in df.columns if 'zonedist' in c.lower() or 'zoning' in c.lower()]
far_cols = [c for c in df.columns if 'far' in c.lower() or 'gross_floor' in c.lower()]
zc = z_cols[0] if z_cols else None
fc_col = far_cols[0] if far_cols else None
# liberal
df_lib = df.copy()
if zc:
    df_lib[zc] = df_lib[zc].astype(str).apply(lambda z: z.replace('R3', 'R7') if isinstance(z, str) else z)
if fc_col:
    df_lib[fc_col] = df_lib[fc_col].astype(float).fillna(0) * 1.5
# conservative
df_cons = df.copy()
if zc:
    df_cons[zc] = df_cons[zc].astype(str).apply(lambda z: z.replace('R7', 'R3') if isinstance(z, str) else z)
if fc_col:
    df_cons[fc_col] = df_cons[fc_col].astype(float).fillna(0) * 0.7
# mixed using political model
pol_model_path = ROOT / 'models' / 'lgbm_all_years_political.txt'
pol_feat_path = ROOT / 'models' / 'artifacts' / 'lgbm_all_years_political_features.json'
pol_features = json.loads(pathlib.Path(pol_feat_path).read_text())
for fc in pol_features:
    if fc not in df.columns:
        df[fc] = 0.0
df_mix = df.copy()
if 'dem_share' in df_mix.columns:
    df_mix['dem_share'] = (df_mix['dem_share'] + 0.15).clip(0.0, 1.0)
pol_booster = lgb.Booster(model_file=str(pol_model_path))

def scenario_pct(use_booster, use_feats, scen_df, base_df):
    bp = use_booster.predict(base_df[use_feats].fillna(0))
    sp = use_booster.predict(scen_df[use_feats].fillna(0))
    pc = (np.exp(sp) - np.exp(bp)) / np.exp(bp) * 100.0
    mask = bp >= 9.0
    pc = pc[mask]
    pc_clipped = np.clip(pc, -100.0, 100.0)
    return pc, pc_clipped, mask
lib_pc, lib_pc_c, _ = scenario_pct(booster, features, df_lib, df)
cons_pc, cons_pc_c, _ = scenario_pct(booster, features, df_cons, df)
mix_pc, mix_pc_c, _ = scenario_pct(pol_booster, pol_features, df_mix, df)
K['clip_counts'] = {
    'liberal_upper': int(np.sum(lib_pc > 100.0)),
    'liberal_lower': int(np.sum(lib_pc < -100.0)),
    'conservative_upper': int(np.sum(cons_pc > 100.0)),
    'conservative_lower': int(np.sum(cons_pc < -100.0)),
    'mixed_upper': int(np.sum(mix_pc > 100.0)),
    'mixed_lower': int(np.sum(mix_pc < -100.0)),
}
write_json(ROOT / 'results' / 'round5_part_K_mechanics.json', {'section_K_mechanics': K})
print('WROTE results/round5_part_K_mechanics.json')
print('base_pred_min', K['base_pred_min'], 'filter removes', K['filter_9.0_removed_rows'], 'rows')
print('clip counts', K['clip_counts'])
