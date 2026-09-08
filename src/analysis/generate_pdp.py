import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb

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

# load model (assume LightGBM champion)
booster = lgb.Booster(model_file=str(model_path))

def _generate_pdp(md_name, booster, feature_list):
    """One PDP panel per top-gain feature (top 5), writing
    models/artifacts/pdp_{name}_{feature}.png + {name}_pdp_summary.json."""
    names = booster.feature_name()
    imps = booster.feature_importance(importance_type='gain')
    fi = sorted(zip(names, imps), key=lambda x: x[1], reverse=True)
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
            preds = booster.predict(sample_mod[feature_list])
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
# Paper's political model (17-feature spec): regenerate its PDPs too. The
# previous lgbm_all_years_political_* PDP/sensitivity artifacts on disk were
# computed against a PRE-BLACKLIST-CLEANUP model (they referenced assesstot,
# exempttot, appbbl, taxmap, schooldist, ...) — exactly the staleness the
# paper's Appendix A.4 documents. This block makes the fresh, blacklist-clean
# political artifacts the only ones on disk.
# ---------------------------------------------------------------------------
POL_NAME = 'lgbm_all_years_political'
pol = next((e for e in registry if e.get('name') == POL_NAME), None)
if pol is None:
    print(f'NOTE: {POL_NAME} not in registry; political PDPs not regenerated')
elif pol.get('id') == champ_id:
    print(f'NOTE: {POL_NAME} IS the champion; PDPs already written above')
else:
    pol_features = json.loads(Path(pol.get('features_path')).read_text())
    pol_booster = lgb.Booster(model_file=str(pol.get('model_path')))
    _generate_pdp(POL_NAME, pol_booster, pol_features)
