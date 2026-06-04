import argparse
import json
from pathlib import Path
from datetime import datetime
import subprocess

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump, load

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['all_years','year2017'], required=True)
parser.add_argument('--mode', choices=['base','political'], required=True)
args = parser.parse_args()

REGISTRY = Path('experiments/registry.json')
RESULTS_DIR = Path('experiments/results')
MODEL_DIR = Path('models')
ARTIFACT_DIR = Path('models/artifacts')
SPLIT_DIR = Path('data/splits')

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not REGISTRY.exists():
    print('No registry found')
    raise SystemExit(1)

registry = json.loads(REGISTRY.read_text())

# pick top-3 non-tainted by MAE
candidates = []
for exp in registry:
    if exp.get('tainted'):
        continue
    metrics = exp.get('metrics', {})
    m = metrics.get('mae_log_price') or metrics.get('mae') or None
    if m is None:
        continue
    candidates.append((float(m), exp))

if len(candidates) < 3:
    print('Not enough candidates for ensemble')
    raise SystemExit(1)

candidates = sorted(candidates, key=lambda x: x[0])[:3]
top_exps = [c[1] for c in candidates]

print('Top 3 experiments for ensemble:')
for e in top_exps:
    print('-', e.get('name'), e.get('metrics'))

# load splits
train = pd.read_parquet(SPLIT_DIR / f"{args.variant}_train.parquet")
test = pd.read_parquet(SPLIT_DIR / f"{args.variant}_test.parquet")
TARGET = 'target_log_price'

def load_model_and_type(path_str):
    p = Path(path_str)
    if p.suffix == '.joblib':
        return ('joblib', load(str(p)))
    if p.suffix == '.model':
        booster = xgb.Booster()
        booster.load_model(str(p))
        return ('xgb', booster)
    if p.suffix == '.txt':
        booster = lgb.Booster(model_file=str(p))
        return ('lgb', booster)
    if p.suffix == '.cbm':
        m = CatBoostRegressor()
        m.load_model(str(p))
        return ('cat', m)
    # fallback
    return ('joblib', load(str(p)))

def predict_model(mtype, model_obj, X, feat_list, preproc_path=None):
    Xf = X[feat_list].copy()
    Xf = Xf.fillna(0)
    if preproc_path and Path(preproc_path).exists():
        try:
            p = load(preproc_path)
            Xp = p.transform(Xf)
        except Exception:
            Xp = Xf.values
    else:
        Xp = Xf.values

    if mtype == 'joblib':
        return model_obj.predict(Xp)
    if mtype == 'xgb':
        return model_obj.predict(xgb.DMatrix(Xp))
    if mtype == 'lgb':
        return model_obj.predict(Xp)
    if mtype == 'cat':
        return model_obj.predict(Xp)
    return model_obj.predict(Xp)

meta_train = pd.DataFrame(index=train.index)
meta_test = pd.DataFrame(index=test.index)

for i, exp in enumerate(top_exps):
    features_path = Path(exp.get('features_path'))
    model_path = Path(exp.get('model_path'))
    if not features_path.exists() or not model_path.exists():
        raise FileNotFoundError('Missing artifacts for ' + exp.get('name'))
    feats = json.loads(features_path.read_text())
    mtype, mobj = load_model_and_type(str(model_path))
    preproc_path = ARTIFACT_DIR / f"{exp.get('name')}_preproc.joblib"
    meta_train[f'pred_{i}'] = predict_model(mtype, mobj, train, feats, str(preproc_path))
    meta_test[f'pred_{i}'] = predict_model(mtype, mobj, test, feats, str(preproc_path))

y_train = train[TARGET]
y_test = test[TARGET]

meta_model = Ridge()
meta_model.fit(meta_train, y_train)
meta_preds = meta_model.predict(meta_test)

ens_r2 = r2_score(y_test, meta_preds)
ens_mae = mean_absolute_error(y_test, meta_preds)

name = f"ensemble_top3_{args.variant}_{args.mode}"
results = {
    'model': 'EnsembleRidge',
    'name': name,
    'variant': args.variant,
    'mode': args.mode,
    'n_train': int(len(train)),
    'n_test': int(len(test)),
    'n_base_models': len(top_exps),
    'r2': float(ens_r2),
    'mae': float(ens_mae),
}

with open(RESULTS_DIR / f"{name}.json", 'w') as f:
    json.dump(results, f, indent=2)

model_path = MODEL_DIR / f"{name}.joblib"
dump(meta_model, model_path)

with open(ARTIFACT_DIR / f"{name}_metrics.json", 'w') as f:
    json.dump({'r2': ens_r2, 'mae': ens_mae}, f, indent=2)

with open(ARTIFACT_DIR / f"{name}_features.json", 'w') as f:
    json.dump([e.get('name') for e in top_exps], f, indent=2)

print('Ensemble trained:', results)

# register
try:
    git = subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
except Exception:
    git = None

exp = {
    'id': f"exp-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
    'created_utc': datetime.utcnow().isoformat() + 'Z',
    'git_commit': git,
    'script': 'models/training/train_ensemble.py',
    'name': name,
    'mode': args.mode,
    'scope': '2017' if args.variant == 'year2017' else 'all',
    'hypothesis': 'Ensemble Ridge meta-learner on top-3 models',
    'reasoning': 'Meta-learner (Ridge) on model predictions',
    'metrics': {'r2': float(ens_r2), 'mae': float(ens_mae)},
    'results': results,
    'features_path': str(ARTIFACT_DIR / f"{name}_features.json"),
    'model_path': str(model_path),
}

registry = json.loads(REGISTRY.read_text())
registry.append(exp)
REGISTRY.write_text(json.dumps(registry, indent=2))
print('Registered ensemble experiment:', exp['id'])

try:
    subprocess.check_call([Path('src') / 'experiment_runner.py'])
except Exception:
    pass
