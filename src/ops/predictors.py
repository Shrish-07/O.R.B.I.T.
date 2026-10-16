"""Family-optimized predictor factory (Round 9): every interpreted model is
scored under the exact input convention its own trainer used.

Training-time conventions (verified by reading each trainer script):
  * models/training/train_lgbm.py              -> LightGBM native NaN
  * models/training/train_xgboost.py           -> X = train[features].fillna(0)
  * models/training/train_catboost.py          -> X = train[features].fillna(0)
  * models/training/train_random_forest.py     -> X = train[features].fillna(0)
  * models/training/train_linear_baselines.py  -> X = preproc.fit_transform(
        train[features]) with median SimpleImputer + StandardScaler; the
        fitted preproc is saved as models/artifacts/{name}_preproc.joblib and
        MUST be applied before predict, or the model is scored in the wrong
        (raw) feature space.

Why a new module instead of editing model_io.py: model_io.predict_frame
routes CatBoost/XGBoost through native NaN handling and its docstring says
'natively, as trained' -- which is wrong for those two families (their
trainers fillna(0) first). model_io is left untouched because existing
consumers (and the legacy champion blocks in several analysis scripts)
depend on its current behavior; the factory below is the corrected
single-convention layer used by every new per-model pass. See
VERIFICATION_REPORT.md Round 9.
"""

import json
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_DIR = _REPO_ROOT / 'models' / 'artifacts'


def _as_path(p):
    """Resolve a registry/audit path: champion_ranking._entry now emits
    repo-relative POSIX paths in JSON outputs (Round 11); consumers re-anchor
    them here so stages can run from any cwd."""
    p = Path(str(p))
    return p if p.is_absolute() else (_REPO_ROOT / p)


def features_of(entry):
    """The trained feature list for a champion_ranking entry."""
    return json.loads(_as_path(entry['features_path']).read_text())


def _ensure_columns(X, features):
    """Frame containing exactly `features` (missing columns zero-filled --
    the pre-existing champion-pass convention), row order/index preserved."""
    X = X.copy()
    for f in features:
        if f not in X.columns:
            X[f] = 0.0
    return X[features]


def load_entry_model(entry):
    """Load (model, preproc) for a champion_ranking entry. preproc is None
    for every family except 'linear'."""
    from joblib import load as jl
    mp = _as_path(entry['model_path'])
    suffix = mp.suffix.lower()
    fam = entry.get('family')
    if fam == 'lightgbm' or suffix == '.txt':
        import lightgbm as lgb
        return lgb.Booster(model_file=str(mp)), None
    if fam == 'catboost' or suffix == '.cbm':
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(str(mp))
        return m, None
    if fam == 'xgboost' or suffix == '.model':
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(str(mp))
        return b, None
    if suffix == '.joblib':
        model = jl(str(mp))
        preproc = None
        preproc_path = entry.get('preproc_path') or str(
            _ARTIFACT_DIR / (str(entry.get('name')) + '_preproc.joblib'))
        _pp = _as_path(preproc_path)
        if _pp.exists():
            preproc = jl(str(_pp))
        return model, preproc
    raise ValueError('Unsupported model file for ' + str(entry.get('name')) +
                     ': ' + str(mp))


def predict_entry_with_features(entry, features, X, model=None, preproc=None):
    """Predict with the entry's own family convention. Returns a 1-D float
    numpy array aligned to X's row order."""
    if model is None:
        model, preproc = load_entry_model(entry)
    Xf = _ensure_columns(X, features)
    fam = entry.get('family')
    if fam == 'lightgbm':
        return model.predict(Xf)                  # native NaN, as trained
    if fam == 'catboost':
        return model.predict(Xf.fillna(0))        # trained on fillna(0)
    if fam == 'xgboost':
        import xgboost as xgb
        return model.predict(xgb.DMatrix(Xf.fillna(0),
                                         feature_names=list(Xf.columns)))
    if fam == 'linear':
        if preproc is not None:
            Xs = preproc.transform(Xf)            # as trained: median-impute
        else:                                     # + standardize
            import warnings
            warnings.warn('linear entry has no preproc artifact; falling '
                          'back to fillna(0) prediction inputs -- results for '
                          + str(entry.get('name')) + ' are suspect.')
            Xs = Xf.fillna(0).values
        return model.predict(Xs)
    # sklearn trees/forests: fillna(0), as trained
    return model.predict(Xf.fillna(0))


def predict_entry(entry, X, model=None, preproc=None):
    return predict_entry_with_features(entry, features_of(entry), X,
                                       model=model, preproc=preproc)


def build_predictor(entry):
    """Load model (+ preproc, features) exactly once; return fn(X_df).

    The returned callable carries .model, .preproc and .features so callers
    that also need the raw estimator (e.g. gain importance) do not reload it.
    """
    model, preproc = load_entry_model(entry)
    features = features_of(entry)

    def _fn(X):
        return predict_entry_with_features(entry, features, X,
                                           model=model, preproc=preproc)

    _fn.entry = entry
    _fn.model = model
    _fn.preproc = preproc
    _fn.features = features
    return _fn
