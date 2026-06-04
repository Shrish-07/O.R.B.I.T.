import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

REG = Path('experiments/registry.json')
CHAMP = Path('experiments/champion.json')
SPLIT_DIR = Path('data/splits')
PRED_DIR = Path('experiments/predictions')
ARTIFACT_DIR = Path('models/artifacts')

PRED_DIR.mkdir(parents=True, exist_ok=True)

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

# load splits
train = pd.read_parquet(SPLIT_DIR / 'all_years_train.parquet')
test = pd.read_parquet(SPLIT_DIR / 'all_years_test.parquet')
TARGET = 'target_log_price'

# load model (assume lightgbm champion)
booster = lgb.Booster(model_file=str(model_path))

# base predictions
preds_test = booster.predict(test[features])
preds_train = booster.predict(train[features])

# residual quantiles on training set
residuals = train[TARGET] - preds_train
lower_q, upper_q = np.percentile(residuals, [2.5, 97.5])

intervals = pd.DataFrame({'pred': preds_test})
intervals['lower'] = intervals['pred'] + lower_q
intervals['upper'] = intervals['pred'] + upper_q
intervals_path = PRED_DIR / f"{name}_intervals.csv"
intervals.to_csv(intervals_path, index=False)

# sensitivity analysis for top features (use earlier PDP summary or feature importance)
fi = booster.feature_importance(importance_type='gain')
names = booster.feature_name()
feat_imp = sorted(zip(names, fi), key=lambda x: x[1], reverse=True)
top_features = [f for f,_ in feat_imp if f in features][:5]

summaries = []
base_mean = preds_test.mean()
for feat in top_features:
    for pct in [0.10, 0.25]:
        for sign in [1, -1]:
            factor = 1 + sign * pct
            X_mod = test[features].copy()
            if feat in X_mod.columns:
                X_mod[feat] = X_mod[feat] * factor
            preds_mod = booster.predict(X_mod)
            delta = preds_mod - preds_test
            summaries.append({
                'feature': feat,
                'perturb': f"{sign*pct*100:.0f}%",
                'mean_delta': float(delta.mean()),
                'median_delta': float(np.median(delta)),
                'mean_pct_change': float(delta.mean() / base_mean) if base_mean != 0 else None,
            })

with open(ARTIFACT_DIR / f"{name}_sensitivity.json", 'w') as f:
    json.dump(summaries, f, indent=2)

print('Saved intervals to', intervals_path)
print('Saved sensitivity summary to', ARTIFACT_DIR / f"{name}_sensitivity.json")
