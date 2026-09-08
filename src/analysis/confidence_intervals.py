import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load

import lightgbm as lgb

CHAMP = Path('experiments/champion.json')
REG = Path('experiments/registry.json')
SPLIT_DIR = Path('data/splits')
PRED_DIR = Path('experiments/predictions')
ARTIFACT_DIR = Path('models/artifacts')

PRED_DIR.mkdir(parents=True, exist_ok=True)

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

# load data
train = pd.read_parquet(SPLIT_DIR / 'all_years_train.parquet')
test = pd.read_parquet(SPLIT_DIR / 'all_years_test.parquet')
TARGET = 'target_log_price'

# load model and preproc if present
preproc_path = ARTIFACT_DIR / f"{name}_preproc.joblib"
use_preproc = preproc_path.exists()
preproc = load(preproc_path) if use_preproc else None

booster = None
if model_path.suffix == '.txt':
    booster = lgb.Booster(model_file=str(model_path))
elif model_path.suffix == '.joblib':
    from joblib import load as jl
    booster = jl(str(model_path))
else:
    try:
        booster = lgb.Booster(model_file=str(model_path))
    except Exception:
        from joblib import load as jl
        booster = jl(str(model_path))

def predict(X_df):
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

# base preds
preds_train = predict(train)
preds_test = predict(test)

# residual bootstrap
residuals = train[TARGET] - preds_train
n_boot = 100
simulated = np.zeros((n_boot, len(test)))
for i in range(n_boot):
    res_sample = residuals.sample(frac=0.8, replace=True, random_state=42+i).values
    # draw residuals for each test row
    drawn = np.random.choice(res_sample, size=len(test), replace=True)
    simulated[i, :] = preds_test + drawn

# compute 10th and 90th percentiles
lower = np.percentile(simulated, 10, axis=0)
upper = np.percentile(simulated, 90, axis=0)

out = pd.DataFrame({'pred': preds_test, 'lower': lower, 'upper': upper}, index=test.index)
out_path = PRED_DIR / f"{name}_prediction_intervals.parquet"
out.to_parquet(out_path)
print('Saved intervals to', out_path)

# ---------------------------------------------------------------------------
# Paper's political model: the same bootstrap-style prediction-interval
# parquet (its historical copy had 56,798 of 56,817 rows — the same 19-row
# export gap as the sibling _test_preds.parquet / _intervals.csv; closed
# 2026-09-05 by regenerating from the full test split with fillna(0), exactly
# like the champion block above).
# ---------------------------------------------------------------------------
POL_NAME = 'lgbm_all_years_political'
pol = next((e for e in registry if e.get('name') == POL_NAME), None)
if pol is None:
    print(f'NOTE: {POL_NAME} not in registry; political prediction intervals not regenerated')
else:
    pol_features = json.loads(Path(pol.get('features_path')).read_text())
    pol_booster = lgb.Booster(model_file=str(Path(pol.get('model_path'))))
    pol_preds_test = pol_booster.predict(test[pol_features].fillna(0))
    pol_preds_train = pol_booster.predict(train[pol_features].fillna(0))
    pol_resid = train[TARGET] - pol_preds_train
    simulated = np.zeros((n_boot, len(test)))
    for i in range(n_boot):
        res_sample = pol_resid.sample(frac=0.8, replace=True, random_state=42+i).values
        drawn = np.random.choice(res_sample, size=len(test), replace=True)
        simulated[i, :] = pol_preds_test + drawn
    lower = np.percentile(simulated, 10, axis=0)
    upper = np.percentile(simulated, 90, axis=0)
    pol_out = pd.DataFrame({'pred': pol_preds_test, 'lower': lower, 'upper': upper},
                           index=test.index)
    assert len(pol_out) == len(test), f'{POL_NAME}: {len(pol_out)} rows vs {len(test)} test rows'
    pol_out_path = PRED_DIR / f"{POL_NAME}_prediction_intervals.parquet"
    pol_out.to_parquet(pol_out_path)
    print(f'Saved {len(pol_out)}-row political prediction intervals to', pol_out_path)

