import json
from pathlib import Path
import numpy as np
import pandas as pd
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
features_path = Path(exp.get('features_path'))

# resolve relative paths
if not model_path.is_absolute():
    model_path = Path('.') / model_path
if not features_path.is_absolute():
    features_path = Path('.') / features_path

if not model_path.exists() or not features_path.exists():
    print('Missing model or features artifact')
    raise SystemExit(1)

features = json.loads(features_path.read_text())

# load test split
test = pd.read_parquet(SPLIT_DIR / 'all_years_test.parquet')

# load model
booster = lgb.Booster(model_file=str(model_path))

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
out_path = ARTIFACT_DIR / f"{name}_shap_summary.json"

try:
    import shap
    # Use first 10 features with <100 NA for sample selection
    na_counts = test[features].isna().sum()
    good_feats = [f for f in features if na_counts[f] < 100]
    if not good_feats:
        raise ValueError("No features with <100 NA for SHAP sample.")
    top_feats = good_feats[:10]
    sample = test.dropna(subset=top_feats)
    if len(sample) == 0:
        raise ValueError("No rows left after dropna on good features. Check data for missing values.")
    sample = sample[features].sample(n=min(2000, len(sample)), random_state=42)
    expl = shap.TreeExplainer(booster)
    shap_vals = expl.shap_values(sample)
    # shap_values for regression returns array (n_samples, n_features)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    summary = []
    for f, v in zip(features, mean_abs):
        summary.append({'feature': f, 'mean_abs_shap': float(v)})
    summary = sorted(summary, key=lambda x: x['mean_abs_shap'], reverse=True)
    payload = {'model': name, 'method': 'shap', 'sample_n': int(len(sample)), 'summary': summary}
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print('Saved SHAP summary to', out_path)
except Exception:
    # fallback: use feature importance from model
    names = booster.feature_name()
    imps = booster.feature_importance(importance_type='gain')
    summary = []
    for n, v in zip(names, imps):
        if n in features:
            summary.append({'feature': n, 'gain': int(v)})
    summary = sorted(summary, key=lambda x: x.get('gain', 0), reverse=True)
    payload = {'model': name, 'method': 'feature_importance_fallback', 'summary': summary}
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print('Saved fallback SHAP summary (feature importance) to', out_path)
