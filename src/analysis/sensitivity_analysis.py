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

# FIX C12: the stale *_preproc.joblib bypass was removed. Those artifacts were
# remnants of an abandoned linear-model stage; the booster is always evaluated on
# raw features with native NaN handling, consistent with how it was trained.

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
    return booster.predict(X_df[features])

def run_sensitivity(md_name, feature_list, top_feat_list, predict_fn):
    """Perturb each of top_feat_list by pm10/pm25% and record mean/median
    delta of predictions versus the unperturbed test predictions.
    predict_fn must map a DataFrame (raw feature columns) to predictions."""
    base_preds = predict_fn(test)
    base_mean = base_preds.mean()
    results = []
    for feat in top_feat_list:
        if feat not in test.columns:
            continue
        if not is_numeric_dtype(test[feat].dtype):
            # skip non-numeric perturbations
            continue
        for pct in [-0.25, -0.10, 0.10, 0.25]:
            X_mod = test[feature_list].copy()
            X_mod[feat] = X_mod[feat] * (1 + pct)
            preds_mod = predict_fn(X_mod)
            delta = preds_mod - base_preds
            results.append({
                'feature': feat,
                'perturb': pct,
                'mean_delta': float(delta.mean()),
                'median_delta': float(np.median(delta)),
                'mean_pct_change': float(delta.mean() / base_mean) if base_mean != 0 else None,
            })
    out_path = ARTIFACT_DIR / f"{md_name}_sensitivity_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved sensitivity analysis to', out_path)
    return results


results = run_sensitivity(name, features, top_feats, predict_df)

# ---------------------------------------------------------------------------
# Paper's political model (19-feature spec after the Round 7 BBL_pluto
# blacklist): regenerate its sensitivity
# analysis too. The previous lgbm_all_years_political_sensitivity_analysis.json
# was computed against a PRE-BLACKLIST-CLEANUP model (it referenced assesstot,
# exempttot, taxmap, 'Tax block', 'Tax lot', ... — nothing in the current
# canonical schema), exactly the staleness documented in the paper's
# Appendix A.4. Top features come from the political model's own SHAP summary
# (the Table 3 artifact), falling back to its first 10 features.
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
        continue  # the champion pass above already wrote this model's analysis
    _feats = json.loads(Path(_e.get('features_path')).read_text())
    _bst = lgb.Booster(model_file=str(_e.get('model_path')))
    _shap_path = ARTIFACT_DIR / f"{_mname}_shap_summary.json"
    if _shap_path.exists():
        _shap = json.loads(_shap_path.read_text())
        _top = [s.get('feature') for s in _shap.get('summary', [])][:10]
    else:
        _top = _feats[:10]
    run_sensitivity(_mname, _feats, _top,
                    lambda X_df, _b=_bst, _f=_feats: _b.predict(X_df[_f]))
