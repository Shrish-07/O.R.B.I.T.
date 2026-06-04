import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb

CHAMP = Path('experiments/champion.json')
REG = Path('experiments/registry.json')
SPLIT_DIR = Path('data/splits')
ARTIFACT_DIR = Path('models/artifacts')

if not CHAMP.exists():
    print('No champion file')
    raise SystemExit(1)
if not REG.exists():
    print('No registry file')
    raise SystemExit(1)

champ = json.loads(CHAMP.read_text())
registry = json.loads(REG.read_text())
champ_id = champ.get('selected_experiment')
exp = next((e for e in registry if e.get('id') == champ_id), None)
if exp is None:
    print('Champion entry not found in registry')
    raise SystemExit(1)

name = exp.get('name')
model_path = Path(exp.get('model_path'))
features_path = Path(exp.get('features_path'))

if not model_path.exists() or not features_path.exists():
    print('Missing model or features artifact')
    raise SystemExit(1)

features = json.loads(features_path.read_text())

# load test split
test = pd.read_parquet(SPLIT_DIR / 'all_years_test.parquet')

# load model (assume LightGBM champion)
booster = lgb.Booster(model_file=str(model_path))

# compute feature importances (gain)
names = booster.feature_name()
imps = booster.feature_importance(importance_type='gain')
fi = sorted(zip(names, imps), key=lambda x: x[1], reverse=True)
top_features = [f for f,_ in fi if f in features][:5]

out_summary = []
for feat in top_features:
    col = feat
    series = test[col].dropna()
    vmin = series.quantile(0.05)
    vmax = series.quantile(0.95)
    grid = np.linspace(vmin, vmax, 20)
    means = []
    # use a sample to speed up
    sample = test.sample(n=min(2000, len(test)), random_state=42).copy()
    for v in grid:
        sample_mod = sample.copy()
        sample_mod[col] = v
        preds = booster.predict(sample_mod[features])
        means.append(preds.mean())
    plt.figure(figsize=(6,4))
    plt.plot(grid, means, marker='o')
    plt.xlabel(col)
    plt.ylabel('Predicted log-price')
    plt.title(f'PDP: {name} — {col}')
    out_file = ARTIFACT_DIR / f"pdp_{name}_{col}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    out_summary.append({'feature': col, 'plot': str(out_file)})

with open(ARTIFACT_DIR / f"{name}_pdp_summary.json", 'w') as f:
    json.dump(out_summary, f, indent=2)

print('PDPs generated for', name)
