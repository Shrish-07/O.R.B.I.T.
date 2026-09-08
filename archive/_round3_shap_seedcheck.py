"""Round-3: is the old-vs-new political SHAP difference just SHAP-sample noise?
Recompute mean|SHAP| for the current 17-feature political model across 5
seeds and compare the spread against the paper/artifact difference."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

features = json.load(open('models/artifacts/lgbm_all_years_political_features.json'))
booster = lgb.Booster(model_file='models/lgbm_all_years_political.txt')
test = pd.read_parquet('data/splits/all_years_test.parquet')

na_counts = test[features].isna().sum()
good = [f for f in features if na_counts[f] < 100]
base = test.dropna(subset=good[:10])
print('rows after dropna on top-10 non-sparse features:', len(base))

runs = {}
for seed in [42, 0, 1, 7, 123]:
    sample = base[features].sample(n=min(2000, len(base)), random_state=seed)
    sv = shap.TreeExplainer(booster).shap_values(sample)
    runs[seed] = dict(zip(features, np.mean(np.abs(sv), axis=0)))

new = {r['feature']: r['mean_abs_shap']
       for r in json.load(open('models/artifacts/lgbm_all_years_political_shap_summary.json'))['summary']}

print(f"{'feature':28s} {'mean over seeds':>15s} {'minmax spread':>14s} {'artifact(17f)':>14s}")
for f in features:
    vals = np.array([runs[s][f] for s in runs])
    print(f"{f:28s} {vals.mean():15.6f} {vals.min():14.6f}-{vals.max():.6f} {new.get(f, float('nan')):14.6f}")
