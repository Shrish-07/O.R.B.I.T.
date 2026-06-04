import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from pandas.api.types import is_numeric_dtype

import lightgbm as lgb

CHAMP = Path('experiments/champion.json')
REG = Path('experiments/registry.json')
SPLIT_DIR = Path('data/splits')
ARTIFACT_DIR = Path('models/artifacts')

if not CHAMP.exists() or not REG.exists():
    print('Champion or registry missing')
    raise SystemExit(1)

champ = json.loads(CHAMP.read_text())
registry = json.loads(REG.read_text())
exp = next((e for e in registry if e.get('id') == champ.get('selected_experiment')), None)
if exp is None:
    print('Champion not found in registry')
    raise SystemExit(1)

name = exp.get('name')
model_path = Path(exp.get('model_path'))
features = json.loads(Path(exp.get('features_path')).read_text())

# load shap summary for top features if available
shap_path = ARTIFACT_DIR / f"{name}_shap_summary.json"
if shap_path.exists():
    shap = json.loads(shap_path.read_text())
    shap_list = shap.get('summary', [])
    top_feats = [s.get('feature') for s in shap_list][:10]
else:
    # fallback to feature importance from model (if LGBM)
    top_feats = features[:10]

train = pd.read_parquet(SPLIT_DIR / 'all_years_train.parquet')
test = pd.read_parquet(SPLIT_DIR / 'all_years_test.parquet')
TARGET = 'target_log_price'

# load preproc if exists
preproc_path = ARTIFACT_DIR / f"{name}_preproc.joblib"
use_preproc = preproc_path.exists()
preproc = load(preproc_path) if use_preproc else None

booster = None
if model_path.suffix == '.txt':
    booster = lgb.Booster(model_file=str(model_path))
else:
    try:
        from joblib import load as jl
        booster = jl(str(model_path))
    except Exception:
        booster = lgb.Booster(model_file=str(model_path))

def predict_df(X_df):
    Xf = X_df[features].fillna(0)
    if use_preproc:
        try:
            Xp = preproc.transform(Xf)
        except Exception:
            Xp = Xf.values
    else:
        Xp = Xf.values
    if isinstance(booster, lgb.Booster):
        return booster.predict(Xp)
    else:
        return booster.predict(Xp)

base_preds = predict_df(test)
base_mean = base_preds.mean()

results = []
for feat in top_feats:
    if feat not in test.columns:
        continue
    if not is_numeric_dtype(test[feat].dtype):
        # skip non-numeric perturbations
        continue
    for pct in [-0.25, -0.10, 0.10, 0.25]:
        X_mod = test[features].copy()
        X_mod[feat] = X_mod[feat] * (1 + pct)
        preds_mod = predict_df(X_mod)
        delta = preds_mod - base_preds
        results.append({
            'feature': feat,
            'perturb': pct,
            'mean_delta': float(delta.mean()),
            'median_delta': float(np.median(delta)),
            'mean_pct_change': float(delta.mean() / base_mean) if base_mean != 0 else None,
        })

out_path = ARTIFACT_DIR / f"{name}_sensitivity_analysis.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print('Saved sensitivity analysis to', out_path)
