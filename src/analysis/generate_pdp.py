import json
from pathlib import Path
import numpy as np
import pandas as pd
# Headless backend BEFORE pyplot (matches src/figures/* convention). TkAgg
# teardown can abort the interpreter at exit ("Tcl_AsyncDelete: async
# handler deleted by the wrong thread") after the script otherwise
# completed, producing a spurious FAIL — observed 2026-09-24.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lightgbm as lgb

# The champion may be any model family (2026-09-24 fix: a RandomForest
# .joblib champion crashed this script at lgb.Booster). Family-agnostic
# loading / importance / prediction via src/ops/model_io.py.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ops.model_io import load_model, gain_importance, predict_frame

CHAMP = Path('experiments/champion.json')
REG = Path('experiments/registry.json')
SPLIT_DIR = Path('data/splits')
ARTIFACT_DIR = Path('models/artifacts')

if not CHAMP.exists():
    print('No champion file')
    raise SystemExit(1)
if not REG.exists():
    print('No registry file')
    raise SystemExit(1)

champ = json.loads(CHAMP.read_text())
registry = json.loads(REG.read_text())
champ_id = champ.get('selected_experiment')
exp = next((e for e in registry if e.get('id') == champ_id), None)
if exp is None:
    print('Champion entry not found in registry')
    raise SystemExit(1)

name = exp.get('name')
model_path = Path(exp.get('model_path'))
features_path = Path(exp.get('features_path'))

if not model_path.exists() or not features_path.exists():
    print('Missing model or features artifact')
    raise SystemExit(1)

features = json.loads(features_path.read_text())

# load test split
test = pd.read_parquet(SPLIT_DIR / 'all_years_test.parquet')

# load model (any family — .txt/.joblib/.cbm/.model)
booster = load_model(model_path)

def _generate_pdp(md_name, booster, feature_list):
    """One PDP panel per top-gain feature (top 5), writing
    models/artifacts/pdp_{name}_{feature}.png + {name}_pdp_summary.json."""
    fi = gain_importance(booster, feature_list)
    top_features = [f for f, _ in fi if f in feature_list][:5]

    out_summary = []
    for feat in top_features:
        col = feat
        series = test[col].dropna()
        vmin = series.quantile(0.05)
        vmax = series.quantile(0.95)
        grid = np.linspace(vmin, vmax, 20)
        means = []
        # use a sample to speed up
        sample = test.sample(n=min(2000, len(test)), random_state=42).copy()
        for v in grid:
            sample_mod = sample.copy()
            sample_mod[col] = v
            preds = predict_frame(booster, sample_mod[feature_list])
            means.append(preds.mean())
        plt.figure(figsize=(6,4))
        plt.plot(grid, means, marker='o')
        plt.xlabel(col)
        plt.ylabel('Predicted log-price')
        plt.title(f'PDP: {md_name} — {col}')
        out_file = ARTIFACT_DIR / f"pdp_{md_name}_{col}.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()
        # .as_posix(): str(Path) on Windows emits backslashes; keep the JSON
        # portable (see VERIFICATION_REPORT.md Rounds 1-2 for this bug class).
        out_summary.append({'feature': col, 'plot': out_file.as_posix()})

    with open(ARTIFACT_DIR / f"{md_name}_pdp_summary.json", 'w') as f:
        json.dump(out_summary, f, indent=2)
    print('PDPs generated for', md_name, '->', [c['feature'] for c in out_summary])


_generate_pdp(name, booster, features)

# ---------------------------------------------------------------------------
# Paper's political model (19-feature spec after the Round 7 BBL_pluto
# blacklist): regenerate its PDPs too. The
# previous lgbm_all_years_political_* PDP/sensitivity artifacts on disk were
# computed against a PRE-BLACKLIST-CLEANUP model (they referenced assesstot,
# exempttot, appbbl, taxmap, schooldist, ...) — exactly the staleness the
# paper's Appendix A.4 documents. This block makes the fresh, blacklist-clean
# political artifacts the only ones on disk.
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
    if _e.get('id') == champ_id:
        continue  # the champion pass above already wrote this model's artifacts
    _feats = json.loads(Path(_e.get('features_path')).read_text())
    _bst = lgb.Booster(model_file=str(_e.get('model_path')))
    _generate_pdp(_mname, _bst, _feats)
