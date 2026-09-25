import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

# The champion may be any model family (2026-09-24 fix: a RandomForest
# .joblib champion crashed this script at lgb.Booster). Family-agnostic
# loading / importance via src/ops/model_io.py.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ops.model_io import load_model, model_family, gain_importance

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
features_path = Path(exp.get('features_path'))

# resolve relative paths
if not model_path.is_absolute():
    model_path = Path('.') / model_path
if not features_path.is_absolute():
    features_path = Path('.') / features_path

if not model_path.exists() or not features_path.exists():
    print('Missing model or features artifact')
    raise SystemExit(1)

features = json.loads(features_path.read_text())

# load test split
test = pd.read_parquet(SPLIT_DIR / 'all_years_test.parquet')

# load model (any family — .txt/.joblib/.cbm/.model)
booster = load_model(model_path)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
def write_shap_summary(model_name, mdl_booster, model_features, test_df):
    """Compute mean|SHAP| on a sampled test slice and write
    models/artifacts/{model_name}_shap_summary.json. Falls back to LightGBM
    gain importance if shap is unavailable or yields no usable sample."""
    out_path = ARTIFACT_DIR / f"{model_name}_shap_summary.json"
    try:
        if model_family(mdl_booster) == 'sklearn':
            # sklearn forests in this repo are grown unpruned (max_depth=None,
            # min_samples_leaf=1 -> ~10^5 leaves per tree), which explodes
            # shap's per-tree complexity even at small samples (observed:
            # >40 CPU-min at n=2000; still unfinished at n=300 after >30
            # CPU-min). Skip straight to the documented importance fallback.
            # LightGBM (the paper-cited Table 3 models) keeps the real SHAP
            # path below unchanged.
            raise RuntimeError('TreeSHAP intractable for unpruned sklearn forest')
        import shap
        # Use first 10 features with <100 NA for sample selection
        na_counts = test_df[model_features].isna().sum()
        good_feats = [f for f in model_features if na_counts[f] < 100]
        if not good_feats:
            raise ValueError("No features with <100 NA for SHAP sample.")
        top_feats = good_feats[:10]
        sample = test_df.dropna(subset=top_feats)
        if len(sample) == 0:
            raise ValueError("No rows left after dropna on good features. Check data for missing values.")
        sample = sample[model_features].sample(n=min(2000, len(sample)), random_state=42)
        expl = shap.TreeExplainer(mdl_booster)
        shap_vals = expl.shap_values(sample)
        # shap_values for regression returns array (n_samples, n_features)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        summary = []
        for fname, v in zip(model_features, mean_abs):
            summary.append({'feature': fname, 'mean_abs_shap': float(v)})
        summary = sorted(summary, key=lambda x: x['mean_abs_shap'], reverse=True)
        payload = {'model': model_name, 'method': 'shap', 'sample_n': int(len(sample)), 'summary': summary}
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print('Saved SHAP summary to', out_path)
    except Exception:
        # fallback: use model feature importance, family-agnostic
        # (sklearn/CatBoost/XGBoost models have no LightGBM-style
        # feature_name()/feature_importance('gain') API).
        summary = []
        for n, v in gain_importance(mdl_booster, model_features):
            if n in model_features:
                # Emit both keys: downstream consumers (generate_figures.py,
                # generate_research_summary.py) plot the 'mean_abs_shap'
                # column unconditionally, so a 'gain'-only fallback entry
                # breaks them with KeyError (latent until a non-LightGBM
                # champion first used this fallback, 2026-09-24). The payload
                # 'method' field keeps the provenance label honest: these
                # values are importance scores, not SHAP values.
                summary.append({'feature': n, 'gain': float(v),
                                'mean_abs_shap': float(v)})
        summary = sorted(summary, key=lambda x: x.get('gain', 0), reverse=True)
        payload = {'model': model_name, 'method': 'feature_importance_fallback', 'summary': summary}
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print('Saved fallback SHAP summary (feature importance) to', out_path)


# Champion model (current champion.json selection).
write_shap_summary(name, booster, features, test)

# ---------------------------------------------------------------------------
# Explicitly regenerate the political model's SHAP summary too. The paper's
# Table 3 / Figure 4 cite lgbm_all_years_political (the dem_share-augmented
# spec), and the champion is usually the base model, so without this second
# pass the political file would never be refreshed and verify_shap_importance.py
# would only ever compare the paper against a frozen, stale artifact. Uses the
# same registry-lookup idiom as the political blocks in generate_pdp.py and
# sensitivity_analysis.py.
# ---------------------------------------------------------------------------
# FIX B1 (structural): ALWAYS regenerate the two paper models' artifacts on
# every run. The previous 'X IS the champion -> skip the other pass' shortcut
# let the non-champion model's artifacts go silently stale while verify
# scripts and the paper read them as if current.
for _mname in ('lgbm_all_years_political', 'lgbm_all_years_base'):
    _e = next((e for e in registry if e.get('name') == _mname), None)
    if _e is None:
        print(f'NOTE: {_mname} not in registry; artifacts not regenerated')
        continue
    if _e.get('id') == champ.get('selected_experiment'):
        continue  # the champion pass above already wrote this model's artifacts
    _feats = json.loads(Path(_e.get('features_path')).read_text())
    _bst = lgb.Booster(model_file=str(_e.get('model_path')))
    write_shap_summary(_mname, _bst, _feats, test)
