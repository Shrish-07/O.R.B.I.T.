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

# Helper to extract mae robustly
def extract_mae(exp):
    res = exp.get('results', {}) or {}
    metrics = exp.get('metrics', {}) or {}
    return float(res.get('mae') or metrics.get('mae') or metrics.get('mae_log_price') or float('inf'))

# Prefer these families
preferred_families = ['LightGBM', 'CatBoost', 'XGBoost']
family_map = {f: [] for f in preferred_families}
others = []
for exp in registry:
    if exp.get('tainted'):
        continue
    if exp.get('mode') != args.mode:
        continue
    model_family = (exp.get('results') or {}).get('model') or (exp.get('results') or {}).get('model')
    mae = extract_mae(exp)
    if model_family in family_map:
        family_map[model_family].append((mae, exp))
    else:
        others.append((mae, exp))

# pick best from each preferred family
selected = []
for fam in preferred_families:
    if family_map.get(fam):
        family_map[fam].sort(key=lambda x: x[0])
        selected.append(family_map[fam][0][1])

# if we couldn't find 3, fill with best remaining distinct families
if len(selected) < 3:
    pool = []
    for fam, lst in family_map.items():
        for m,e in lst:
            pool.append((m,e))
    pool += others
    pool = sorted(pool, key=lambda x: x[0])
    for m,e in pool:
        if e not in selected:
            selected.append(e)
        if len(selected) >= 3:
            break

if len(selected) < 3:
    print('Not enough distinct models to build fixed ensemble')
    raise SystemExit(1)

top_exps = selected[:3]
print('Selected base experiments for fixed ensemble:')
for e in top_exps:
    print('-', e.get('name'), 'family=', (e.get('results') or {}).get('model'), 'mae=', extract_mae(e))

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
        # XGBoost may require feature names; if we have a numpy array, restore columns
        if isinstance(Xp, (np.ndarray,)):
            try:
                Xpd = pd.DataFrame(Xp, columns=feat_list)
            except Exception:
                Xpd = Xp
            return model_obj.predict(xgb.DMatrix(Xpd))
        else:
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

name = f"ensemble_fixed_top3_{args.variant}_{args.mode}"
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

print('Fixed ensemble trained:', results)

# register
try:
    git = subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
except Exception:
    git = None

exp = {
    'id': f"exp-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
    'created_utc': datetime.utcnow().isoformat() + 'Z',
    'git_commit': git,
    'script': 'models/training/train_ensemble_fixed.py',
    'name': name,
    'mode': args.mode,
    'scope': '2017' if args.variant == 'year2017' else 'all',
    'hypothesis': 'Ensemble Ridge meta-learner on top-3 distinct families',
    'reasoning': 'Diverse base models (LGBM, CatBoost, XGBoost) to improve stacking robustness',
    'metrics': {'r2': float(ens_r2), 'mae': float(ens_mae)},
    'results': results,
    'features_path': str(ARTIFACT_DIR / f"{name}_features.json"),
    'model_path': str(model_path),
}

registry = json.loads(REGISTRY.read_text())
registry.append(exp)

# if ensemble is worse than best single model, mark tainted and log
best_single_mae = float('inf')
best_single_name = None
for e in registry:
    if e.get('tainted'):
        continue
    try:
        m = extract_mae(e)
    except Exception:
        continue
    if m < best_single_mae:
        best_single_mae = m
        best_single_name = e.get('name')

if ens_mae > best_single_mae:
    exp['tainted'] = True
    # append to actions.log
    log_entry = f"{datetime.utcnow().isoformat()}Z - ensemble {name} MAE {ens_mae:.6f} worse than best single {best_single_name} MAE {best_single_mae:.6f}. Marked tainted. Reason: stacking homogeneous models adds noise; prefer diverse families.\n"
    (Path('logs') / 'actions.log').parent.mkdir(parents=True, exist_ok=True)
    with open(Path('logs') / 'actions.log', 'a') as lf:
        lf.write(log_entry)
    print('Ensemble worse than best single model — marked tainted and logged')

REGISTRY.write_text(json.dumps(registry, indent=2))
print('Registered fixed ensemble experiment:', exp['id'])

try:
    subprocess.check_call([Path('src') / 'experiment_runner.py'])
except Exception:
    pass
