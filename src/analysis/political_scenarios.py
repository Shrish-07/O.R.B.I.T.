
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

# The champion may be ANY model family in the registry (LightGBM .txt,
# sklearn .joblib, CatBoost .cbm, XGBoost .model) — refresh_champion picks
# on test MAE, not format. Load it family-agnostically (2026-09-24 fix:
# a RandomForest champion crashed this script at lgb.Booster on .joblib).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ops.model_io import load_model, predict_frame

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

booster = load_model(model_path)

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

    # FIX C16: these are LightGBM boosters, trained on raw splits with native
    # NaN handling - do NOT fillna. (The old fillna(0) pushed missing dem_share
    # to 0.0, below the observed minimum 0.155, making missing ideology rows look
    # maximally Republican in the counterfactuals.)
    base_preds = predict_frame(use_booster, df[use_features])
    scen_preds = predict_frame(use_booster, scen_df[use_features])
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

    # FIX A4 (disclosure only - the shock mechanics are unchanged): quantify how
    # much of the mixed-governance +0.15 dem_share shock runs outside the
    # political model's training support. LightGBM cannot extrapolate trends
    # past its largest training value; deltas for those rows are artifacts, not
    # identified responses. Written to results/political_scenarios/ and
    # re-derived independently by verify_scenario_mechanics.py.
    if scen_name == 'mixed_governance':
        _split = Path('data/splits/all_years_train.parquet')
        if _split.exists():
            _tr_max = float(pd.read_parquet(_split, columns=['dem_share'])['dem_share'].max())
        else:
            _tr_max = float(df['dem_share'].max())
        _base_ds = df['dem_share']
        _n_ds = int(_base_ds.notna().sum())
        _shocked = _base_ds + 0.15
        _n_clip = int((_shocked > 1.0).sum())
        _n_beyond = int((_shocked > _tr_max).sum())
        _diag = {
            "note": ("The mixed_governance shock (dem_share + 0.15 clipped to [0,1]) is a "
                     "model-level counterfactual. Rows counted below receive shocked dem_share "
                     "values at the [0,1] ceiling or beyond the political model's training "
                     "support; their deltas are extrapolation artifacts, not identified effects."),
            "n_rows_with_dem_share": _n_ds,
            "n_shock_clipped_at_1p0": _n_clip,
            "pct_shock_clipped_at_1p0": round(100.0 * _n_clip / _n_ds, 2),
            "train_dem_share_max_all_years_split": _tr_max,
            "n_shocked_beyond_train_support": _n_beyond,
            "pct_beyond_train_support": round(100.0 * _n_beyond / _n_ds, 2),
        }
        _diag_path = OUT_DIR / "mixed_governance_support_diag.json"
        _diag_path.write_text(json.dumps(_diag, indent=2))
        print("Mixed-governance support diagnostic:", json.dumps(_diag, indent=2))
        if _diag["pct_beyond_train_support"] > 0:
            print(("WARNING: {:.1f}% of rows receive a dem_share shock beyond the political "
                   "model's training support; treat mixed_governance magnitudes as "
                   "extrapolation.").format(_diag["pct_beyond_train_support"]))

print('Political scenarios completed and saved to', OUT_DIR)
