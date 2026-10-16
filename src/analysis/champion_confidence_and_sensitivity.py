import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

# The champion may be any model family (2026-09-24 fix: a RandomForest
# .joblib champion crashed this script at lgb.Booster). Family-agnostic
# loading / importance / prediction via src/ops/model_io.py.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ops.model_io import (
    load_model, gain_importance, predict_frame as _shared_predict,
)

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


def predict_frame(booster, feats, df):
    """Predict over ALL rows (no dropna) with family-correct NaN handling.

    LightGBM: native NaN (as trained). sklearn (e.g. the current RF
    champion): fillna(0), matching train_random_forest.py's fit inputs so
    predictions stay consistent with the recorded registry metrics.
    """
    return _shared_predict(booster, df[feats])


def write_test_preds(exp_name, preds, df):
    """{name}_test_preds.parquet: BBL + pred_log_price, index = test.index.

    NOTE (2026-09-06): the political model's historical test_preds.parquet
    carried 56,798 rows with a RESET 0..56,797 index (19 rows short of the
    56,817-row test split; the exporting script is no longer in the repo and
    the reset index means the missing rows are not even identifiable from the
    artifact). Regenerated here from the full split, with an assertion so the
    gap cannot silently recur.
    """
    out = pd.DataFrame({'BBL': df['BBL'].values, 'pred_log_price': preds},
                       index=df.index)
    assert len(out) == len(df), f'{exp_name}: {len(out)} rows vs {len(df)} test rows'
    path = PRED_DIR / f"{exp_name}_test_preds.parquet"
    out.to_parquet(path)
    print(f'Saved {len(out)}-row test preds to {path}')
    return path


def write_intervals(exp_name, preds, resid_q, n_test):
    """{name}_intervals.csv: pred/lower/upper via additive residual quantiles."""
    lower_q, upper_q = resid_q
    intervals = pd.DataFrame({'pred': preds})
    intervals['lower'] = intervals['pred'] + lower_q
    intervals['upper'] = intervals['pred'] + upper_q
    assert len(intervals) == n_test, f'{exp_name}: {len(intervals)} rows vs {n_test} test rows'
    path = PRED_DIR / f"{exp_name}_intervals.csv"
    intervals.to_csv(path, index=False)
    print(f'Saved {len(intervals)}-row intervals to {path}')
    return path

TARGET = 'target_log_price'

# load model — any family (.txt/.joblib/.cbm/.model)
booster = load_model(model_path)

# base predictions
preds_test = predict_frame(booster, features, test)
preds_train = predict_frame(booster, features, train)

# residual quantiles on training set
residuals = train[TARGET] - preds_train
lower_q, upper_q = np.percentile(residuals, [2.5, 97.5])

write_test_preds(name, preds_test, test)
intervals_path = write_intervals(name, preds_test, (lower_q, upper_q), len(test))

# sensitivity analysis for top features (use earlier PDP summary or feature importance)
feat_imp = gain_importance(booster, features)
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
            preds_mod = predict_frame(booster, feats=features, df=X_mod)
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

# ---------------------------------------------------------------------------
# Paper's political model: regenerate its prediction artifacts from the FULL
# 56,817-row test split (closes the 19-row gap recorded in
# results/round5_followup_resolution.json; the files were previously produced
# by a secondary export script that is no longer in this repo).
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
    if _e.get('id') == champ_id:
        continue  # the champion pass above already wrote this model's artifacts
    _feats = json.loads(Path(_e.get('features_path')).read_text())
    _bst = lgb.Booster(model_file=str(_e.get('model_path')))
    _preds_test = predict_frame(_bst, _feats, test)
    write_test_preds(_mname, _preds_test, test)
    _resid = train[TARGET] - predict_frame(_bst, _feats, train)
    write_intervals(_mname, _preds_test, tuple(np.percentile(_resid, [2.5, 97.5])), len(test))
    print(f'Regenerated {_mname} artifacts at full {len(test)}-row test coverage')

# ---------------------------------------------------------------------------
# Round 9 full-coverage pass: test preds, residual-quantile intervals and the
# +/-10/25% perturbation sensitivity for EVERY eligible ranked model
# (src/ops/champion_ranking.py), each scored via src/ops/predictors.py under
# the exact input convention its own trainer used. The champion (rank-1)
# block and the two classic LightGBM passes above are skipped by experiment
# id so nothing is computed twice.
# ---------------------------------------------------------------------------
from src.ops.champion_ranking import get_all_models, PAPER_MODEL_NAMES
from src.ops.predictors import build_predictor

_covered_ids = {champ_id}
_covered_ids |= {e.get('id') for e in registry if e.get('name') in PAPER_MODEL_NAMES}
for _ent in get_all_models(write_audit=False):
    if _ent['experiment_id'] in _covered_ids:
        continue
    _pf = build_predictor(_ent)
    _preds_test2 = _pf(test)
    _preds_train2 = _pf(train)
    write_test_preds(_ent['name'], _preds_test2, test)
    _resid2 = train[TARGET] - _preds_train2
    write_intervals(_ent['name'], _preds_test2,
                    tuple(np.percentile(_resid2, [2.5, 97.5])), len(test))
    _fi = gain_importance(_pf.model, _pf.features)
    _top5 = [f2 for f2, _v2 in _fi if f2 in _pf.features][:5]
    _summ = []
    _base_mean2 = _preds_test2.mean()
    for feat in _top5:
        for pct in [0.10, 0.25]:
            for sign in [1, -1]:
                factor = 1 + sign * pct
                # perturb in RAW feature space; the predictor then applies
                # the family's own convention (linear preproc / fillna(0) /
                # native NaN) -- the perturbation semantics match the model.
                X_mod = test[_pf.features].copy()
                if feat in X_mod.columns:
                    X_mod[feat] = X_mod[feat] * factor
                preds_mod = _pf(X_mod)
                delta = preds_mod - _preds_test2
                _summ.append({
                    'feature': feat,
                    'perturb': f'{sign*pct*100:.0f}%',
                    'mean_delta': float(delta.mean()),
                    'median_delta': float(np.median(delta)),
                    'mean_pct_change': float(delta.mean() / _base_mean2) if _base_mean2 != 0 else None,
                })
    with open(ARTIFACT_DIR / f"{_ent['name']}_sensitivity.json", 'w') as fp:
        json.dump(_summ, fp, indent=2)
    print(f"Full-coverage pass done for rank {_ent['rank']} "
          f"({_ent['name']}, {_ent['family']}): test_preds + intervals + sensitivity")
print('Confidence + sensitivity now covers all', len(get_all_models(write_audit=False)),
      'eligible ranked models')
