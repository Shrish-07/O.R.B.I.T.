import json
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from joblib import load
from catboost import CatBoostRegressor

REG = Path('experiments/registry.json')
CHAMP = Path('experiments/champion.json')

def _load_champion():
    champ = json.loads(CHAMP.read_text())
    registry = json.loads(REG.read_text())
    champ_id = champ.get('selected_experiment')
    exp = next((e for e in registry if e.get('id') == champ_id), None)
    if exp is None:
        raise RuntimeError('Champion not found')
    return exp

def _load_model(exp):
    model_path = Path(exp.get('model_path'))
    suffix = model_path.suffix
    if suffix == '.joblib':
        return load(str(model_path))
    if suffix == '.model':
        bst = xgb.Booster()
        bst.load_model(str(model_path))
        return bst
    if suffix == '.txt':
        return lgb.Booster(model_file=str(model_path))
    if suffix == '.cbm':
        m = CatBoostRegressor()
        m.load_model(str(model_path))
        return m
    # fallback
    return load(str(model_path))


def _Load_model_artifacts():
    """Deprecated compatibility shim."""
    return _load_model_artifacts()


def _load_model_artifacts():
    """Load champion experiment, model, feature list, and optional preprocessor."""
    exp = _load_champion()
    features_path = Path(exp.get('features_path'))
    features = json.loads(features_path.read_text())
    model = _load_model(exp)
    # try to load a serialized preprocessor located in models/artifacts/{model_stem}_preproc.joblib
    preproc = None
    preproc_path = Path('models') / 'artifacts' / f"{Path(exp.get('model_path')).stem}_preproc.joblib"
    if preproc_path.exists():
        try:
            preproc = load(str(preproc_path))
        except Exception:
            preproc = None
    return exp, model, features, preproc

def run_counterfactual(row, mutations: dict, model=None, features=None, preproc=None):
    """
    Compute a counterfactual for a single row.
    - `row`: pandas Series or dict representing a single property
    - `mutations`: dict mapping feature -> new value
    Optional preloaded `model`, `features`, and `preproc` can be passed to avoid repeated loads.
    Returns: dict with original_log_price, mutated_log_price, delta_log_price, pct_price_change
    """
    # load model artifacts if not provided
    if model is None or features is None:
        exp, model, features, preproc_local = _load_model_artifacts()
        if preproc is None:
            preproc = preproc_local
    # prepare row
    if isinstance(row, dict):
        s = pd.Series(row)
    else:
        s = row.copy()
    # ensure all features present
    for f in features:
        if f not in s.index:
            s[f] = 0

    X_orig = s[features].astype(float).fillna(0).to_frame().T
    # apply preprocessor if available
    if preproc is not None:
        try:
            Xp_orig = preproc.transform(X_orig)
        except Exception:
            Xp_orig = X_orig.values
    else:
        Xp_orig = X_orig.values

    # predict original
    if isinstance(model, lgb.Booster):
        orig = float(model.predict(Xp_orig)[0])
    elif isinstance(model, xgb.Booster):
        orig = float(model.predict(xgb.DMatrix(Xp_orig))[0])
    else:
        orig = float(model.predict(Xp_orig)[0])

    # apply mutations
    s2 = s.copy()
    for k, v in mutations.items():
        if k in s2.index:
            s2[k] = v

    X_new = s2[features].astype(float).fillna(0).to_frame().T
    if preproc is not None:
        try:
            Xp_new = preproc.transform(X_new)
        except Exception:
            Xp_new = X_new.values
    else:
        Xp_new = X_new.values

    if isinstance(model, lgb.Booster):
        new = float(model.predict(Xp_new)[0])
    elif isinstance(model, xgb.Booster):
        new = float(model.predict(xgb.DMatrix(Xp_new))[0])
    else:
        new = float(model.predict(Xp_new)[0])

    delta = new - orig
    pct = (np.exp(new) - np.exp(orig)) / np.exp(orig) if np.exp(orig) != 0 else None

    return {
        'original_log_price': orig,
        'mutated_log_price': new,
        'delta_log_price': delta,
        'pct_price_change': pct,
    }


def run_counterfactual_batch(df_rows, mutations: dict, model=None, features=None, preproc=None):
    """Apply the same mutations across multiple rows (DataFrame) and return a DataFrame of results."""
    if model is None or features is None:
        exp, model, features, preproc_local = _load_model_artifacts()
        if preproc is None:
            preproc = preproc_local

    if isinstance(df_rows, pd.Series):
        df = pd.DataFrame([df_rows])
    else:
        df = df_rows.copy()

    # ensure feature columns exist
    for f in features:
        if f not in df.columns:
            df[f] = 0

    X_orig = df[features].astype(float).fillna(0)
    X_new = X_orig.copy()
    for k, v in mutations.items():
        if k in X_new.columns:
            X_new.loc[:, k] = v

    # apply preproc
    if preproc is not None:
        try:
            Xp_orig = preproc.transform(X_orig)
            Xp_new = preproc.transform(X_new)
        except Exception:
            Xp_orig = X_orig.values
            Xp_new = X_new.values
    else:
        Xp_orig = X_orig.values
        Xp_new = X_new.values

    # predict
    if isinstance(model, lgb.Booster):
        orig_preds = model.predict(Xp_orig)
        new_preds = model.predict(Xp_new)
    elif isinstance(model, xgb.Booster):
        orig_preds = model.predict(xgb.DMatrix(Xp_orig))
        new_preds = model.predict(xgb.DMatrix(Xp_new))
    else:
        orig_preds = model.predict(Xp_orig)
        new_preds = model.predict(Xp_new)

    delta = new_preds - orig_preds
    pct = []
    for o, n in zip(orig_preds, new_preds):
        try:
            pct.append((np.exp(n) - np.exp(o)) / np.exp(o) if np.exp(o) != 0 else None)
        except Exception:
            pct.append(None)

    out = pd.DataFrame({
        'original_log_price': orig_preds,
        'mutated_log_price': new_preds,
        'delta_log_price': delta,
        'pct_price_change': pct,
    })
    return out
