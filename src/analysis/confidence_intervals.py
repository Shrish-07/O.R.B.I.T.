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

# FIX C12: the stale *_preproc.joblib bypass was removed (see
# sensitivity_analysis.py). Boosters predict on raw features with native NaN.

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
    return booster.predict(X_df[features])

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
# ---------------------------------------------------------------------------
# FIX B1 (structural): regenerate THIS artifact family for BOTH paper models
# on every run. The previous political-only / skip-if-champion logic let the
# non-champion model's artifacts go silently stale (a 9/14-era, pre-Round-7
# 19-feature summary survived while the base model was retrained).
# ---------------------------------------------------------------------------
for _mname in ('lgbm_all_years_political', 'lgbm_all_years_base'):
    _e = next((e for e in registry if e.get('name') == _mname), None)
    if _e is None:
        print(f'NOTE: {_mname} not in registry; not regenerated')
        continue
    if _e.get('id') == champ.get('selected_experiment'):
        continue  # the champion block above already wrote this model's intervals
    _feats = json.loads(Path(_e.get('features_path')).read_text())
    _bst = lgb.Booster(model_file=str(Path(_e.get('model_path'))))
    _preds_test = _bst.predict(test[_feats])
    _preds_train = _bst.predict(train[_feats])
    _resid = train[TARGET] - _preds_train
    _simulated = np.zeros((n_boot, len(test)))
    for i in range(n_boot):
        _res_sample = _resid.sample(frac=0.8, replace=True, random_state=42 + i).values
        _drawn = np.random.choice(_res_sample, size=len(test), replace=True)
        _simulated[i, :] = _preds_test + _drawn
    _lower = np.percentile(_simulated, 10, axis=0)
    _upper = np.percentile(_simulated, 90, axis=0)
    _out = pd.DataFrame({'pred': _preds_test, 'lower': _lower, 'upper': _upper},
                        index=test.index)
    assert len(_out) == len(test), f'{_mname}: {len(_out)} rows vs {len(test)} test rows'
    _out_path = PRED_DIR / f"{_mname}_prediction_intervals.parquet"
    _out.to_parquet(_out_path)
    print(f'Saved {len(_out)}-row prediction intervals to', _out_path)

