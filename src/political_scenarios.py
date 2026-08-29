
import json
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

# Load political model/features for mixed_governance
pol_model_path = Path('models/lgbm_all_years_political.txt')
pol_features_path = Path('models/artifacts/lgbm_all_years_political_features.json')
if pol_model_path.exists() and pol_features_path.exists():
    pol_booster = lgb.Booster(model_file=str(pol_model_path))
    pol_features = json.loads(pol_features_path.read_text())
else:
    pol_booster = None
    pol_features = None
import json
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

CHAMP = Path('experiments/champion.json')
REG = Path('experiments/registry.json')
CANON = Path('data/canonical/modeling_dataset_canonical_v2.parquet')
CANON_FALLBACK = Path('data/canonical/modeling_dataset_canonical.parquet')
OUT_DIR = Path('results/political_scenarios')
OUT_DIR.mkdir(parents=True, exist_ok=True)

champ = json.loads(CHAMP.read_text())
registry = json.loads(REG.read_text())
champ_id = champ.get('selected_experiment')
exp = next((e for e in registry if e.get('id') == champ_id), None)
if exp is None:
    print('Champion not found')
    raise SystemExit(1)

name = exp.get('name')
model_path = Path(exp.get('model_path'))
features_path = Path(exp.get('features_path'))
features = json.loads(features_path.read_text())

if CANON.exists():
    df = pd.read_parquet(CANON)
elif CANON_FALLBACK.exists():
    df = pd.read_parquet(CANON_FALLBACK)
else:
    # The old processed fallback (data/processed/modeling_dataset_fe_imputed.parquet)
    # was removed; there is no valid fallback anymore. Fail loudly instead of
    # silently loading a missing/legacy dataset.
    raise SystemExit(
        f"❌ No canonical dataset found. Tried:\n  {CANON}\n  {CANON_FALLBACK}\n"
        "Reproduce/copy the canonical dataset before running political scenarios."
    )

# detect council district column
council_candidates = [c for c in df.columns if 'council' in c.lower()]
council_col = council_candidates[0] if council_candidates else None

booster = lgb.Booster(model_file=str(model_path))

def clamp01(x):
    return np.minimum(np.maximum(x, 0.0), 1.0)

scenarios = {}

df_lib = df.copy(deep=True)
z_cols = [c for c in df.columns if 'zonedist' in c.lower() or 'zoning' in c.lower() or 'zonedist1' in c.lower()]
far_cols = [c for c in df.columns if 'far' in c.lower() or 'gross_floor' in c.lower()]
if z_cols:
    zc = z_cols[0]
    df_lib[zc] = df_lib[zc].astype(str).apply(lambda z: z.replace('R3','R7') if isinstance(z, str) else z)
if far_cols:
    df_lib[far_cols[0]] = df_lib[far_cols[0]].astype(float).fillna(0) * 1.5

scenarios['liberal_policy'] = df_lib

df_cons = df.copy(deep=True)
if z_cols:
    zc = z_cols[0]
    df_cons[zc] = df_cons[zc].astype(str).apply(lambda z: z.replace('R7','R3') if isinstance(z, str) else z)
if far_cols:
    df_cons[far_cols[0]] = df_cons[far_cols[0]].astype(float).fillna(0) * 0.7
scenarios['conservative_policy'] = df_cons

# mixed_governance: ideology shift +0.15
df_mix = df.copy(deep=True)
if 'dem_share' in df_mix.columns:
    df_mix['dem_share'] = (df_mix['dem_share'] + 0.15).clip(0.0, 1.0)
scenarios['mixed_governance'] = df_mix


for scen_name, scen_df in scenarios.items():
    if scen_name == 'mixed_governance':
        use_booster = pol_booster
        use_features = pol_features
    else:
        use_booster = booster
        use_features = features

    # fill missing features for both base and scenario
    for fc in use_features:
        if fc not in scen_df.columns:
            scen_df[fc] = 0
        if fc not in df.columns:
            df[fc] = 0

    if scen_name == 'mixed_governance' and 'dem_share' in scen_df.columns:
        pass

    base_preds = use_booster.predict(df[use_features].fillna(0))
    scen_preds = use_booster.predict(scen_df[use_features].fillna(0))
    delta = scen_preds - base_preds
    pct_change = (np.exp(scen_preds) - np.exp(base_preds)) / np.exp(base_preds) * 100.0
    out = scen_df[[council_col]] if council_col else pd.DataFrame({'idx': scen_df.index})
    out = out.copy()
    out['base_pred'] = base_preds
    out['scen_pred'] = scen_preds
    out['delta'] = delta
    out['pct_change'] = pct_change

    valid_mask = out['base_pred'] >= 9.0
    out = out[valid_mask].copy()
    out['pct_change'] = np.clip(out['pct_change'], -100.0, 100.0)

    if council_col:
        agg = out.groupby(council_col).agg(
            mean_delta=('delta', 'mean'),
            mean_pct_change=('pct_change', 'mean'),
            base_mean=('base_pred', 'mean'),
            scen_mean=('scen_pred', 'mean'),
            count=('pct_change', 'count')
        )
        agg.to_csv(OUT_DIR / f"{scen_name}_by_council.csv")
    out.to_parquet(OUT_DIR / f"{scen_name}_all_properties.parquet")
    print('Saved scenario', scen_name)

print('Political scenarios completed and saved to', OUT_DIR)
