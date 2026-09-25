"""Model-format-agnostic loading / prediction / importance helpers.

Rationale (2026-09-24)
----------------------
`refresh_champion` (src/ops/experiment_runner.py) selects the lowest-MAE
registry entry regardless of model family. When the champion flipped from
LightGBM to a joblib-pickled RandomForest (rf_all_years_political_n200),
every script that loaded the champion via `lgb.Booster(model_file=...)`
crashed with "Unknown model format or submodel type in model file
...joblib", taking down 5 of 53 pipeline stages (political_scenarios,
generate_shap_summary, generate_pdp, champion_confidence_and_sensitivity,
verify_scenario_mechanics). The suffix-based dispatch below follows the
same pattern already used by src/analysis/counterfactual.py::_load_model
so any champion family loads and predicts through one code path:

    .txt    -> LightGBM native model file        (train_lgbm*.py)
    .joblib -> scikit-learn estimator (joblib)   (train_random_forest.py,
                                                  train_linear_baselines.py)
    .cbm    -> CatBoostRegressor                 (train_catboost.py)
    .model  -> xgboost.Booster                   (train_xgboost.py)

Convention notes
----------------
* sklearn models in this repo were FIT on `fillna(0)` inputs
  (train_random_forest.py / train_linear_baselines.py) and cannot consume
  NaN; predict_frame therefore fillna(0)s for the sklearn family so the
  predictions match the inputs behind the recorded registry metrics.
* LightGBM and CatBoost predict with their native NaN handling, exactly
  as they were trained; XGBoost's native Booster needs a DMatrix (NaN is
  handled natively by the trees).
"""

from pathlib import Path

import lightgbm as lgb
from joblib import load as _joblib_load


def load_model(path):
    """Load a persisted model of any family supported by this repo."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == '.txt':
        return lgb.Booster(model_file=str(p))
    if suffix == '.joblib':
        return _joblib_load(str(p))
    if suffix == '.cbm':
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(str(p))
        return m
    if suffix == '.model':
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(str(p))
        return bst
    # Unknown extension: LightGBM native format first, joblib as fallback
    # (mirrors the try/except fallback in sensitivity_analysis.py).
    try:
        return lgb.Booster(model_file=str(p))
    except Exception:
        return _joblib_load(str(p))


def model_family(model):
    """Return 'lightgbm' | 'xgboost' | 'catboost' | 'sklearn'."""
    mod = type(model).__module__
    name = type(model).__name__
    if isinstance(model, lgb.Booster):
        return 'lightgbm'
    if name == 'Booster' and 'xgboost' in mod:
        return 'xgboost'
    if 'catboost' in mod:
        return 'catboost'
    return 'sklearn'


def predict_frame(model, X):
    """Predict on a feature DataFrame X with family-correct NaN handling.

    sklearn -> fillna(0) (as trained); LightGBM / CatBoost / XGBoost ->
    native NaN handling (as trained).
    """
    fam = model_family(model)
    if fam == 'sklearn':
        return model.predict(X.fillna(0))
    if fam == 'xgboost':
        import xgboost as xgb
        return model.predict(xgb.DMatrix(X, feature_names=list(X.columns)))
    return model.predict(X)


def feature_names(model, features):
    """Feature names in model order; falls back to the features JSON list."""
    fam = model_family(model)
    if fam == 'lightgbm':
        return list(model.feature_name())
    if fam == 'catboost':
        return list(getattr(model, 'feature_names_', None) or features)
    if fam == 'xgboost':
        return list(model.feature_names or features)
    return list(features)  # sklearn: ordering comes from the training features JSON


def gain_importance(model, features):
    """[(feature, value), ...] sorted descending, lightgbm-gain semantics.

    LightGBM: gain; sklearn: feature_importances_ (|coef_| for linear
    models); CatBoost: PredictionValuesChange (CatBoost's default gain-like
    importance); XGBoost: gain.
    """
    fam = model_family(model)
    names = feature_names(model, features)
    if fam == 'lightgbm':
        vals = list(model.feature_importance(importance_type='gain'))
    elif fam == 'catboost':
        vals = list(model.get_feature_importance())
    elif fam == 'xgboost':
        score = model.get_score(importance_type='gain')
        vals = [float(score.get(n, 0.0)) for n in names]
    else:  # sklearn
        vals = getattr(model, 'feature_importances_', None)
        if vals is None:
            coef = getattr(model, 'coef_', [0.0] * len(names))
            vals = [abs(float(c)) for c in coef]
        vals = [float(v) for v in vals]
    pairs = list(zip(names, vals))
    pairs.sort(key=lambda t: t[1], reverse=True)
    return pairs
