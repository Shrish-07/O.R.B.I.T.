import json
from pathlib import Path
import numpy as np
import pandas as pd
# Headless backend BEFORE pyplot (matches src/figures/* convention). TkAgg
# teardown can produce a spurious FAIL after otherwise-successful completion.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Round 9: PDPs are generated for EVERY eligible ranked model (not just the
# champion plus the two paper models). Ranking via src/ops/champion_ranking.py
# (rank 1 = champion.json selection; ranks 2..N ascending test MAE); each
# model is predicted via src/ops/predictors.py under the exact input
# convention its trainer used, and ranked via src/ops/model_io.gain_importance.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ops.model_io import gain_importance
from src.ops.champion_ranking import get_all_models
from src.ops.predictors import (load_entry_model, features_of,
                                predict_entry_with_features)

SPLIT_DIR = Path('data/splits')
ARTIFACT_DIR = Path('models/artifacts')


def _generate_pdp(entry, model, preproc, feature_list, test):
    """One PDP panel per top-importance feature (top 5), writing
    models/artifacts/pdp_{name}_{feature}.png + {name}_pdp_summary.json.
    Importance and prediction are both family-aware (gain for tree models,
    |coef| for the linear baselines; the linear models are scored through
    their saved median-impute+scale preproc, exactly as trained)."""
    fi = gain_importance(model, feature_list)
    top_features = [f for f, _ in fi if f in feature_list][:5]

    out_summary = []
    for feat in top_features:
        col = feat
        series = pd.to_numeric(test[col], errors='coerce').dropna()
        if len(series) == 0:
            print('  PDP skipped for non-numeric feature', col, 'of', entry['name'])
            continue
        vmin = series.quantile(0.05)
        vmax = series.quantile(0.95)
        grid = np.linspace(vmin, vmax, 20)
        means = []
        # use a sample to speed up
        sample = test.sample(n=min(2000, len(test)), random_state=42).copy()
        for v in grid:
            sample_mod = sample.copy()
            sample_mod[col] = v
            # raw-space grid; the predictor applies each family's own
            # convention (preproc/fillna/native-NaN) afterwards
            preds = predict_entry_with_features(entry, feature_list, sample_mod,
                                                model=model, preproc=preproc)
            means.append(preds.mean())
        plt.figure(figsize=(6, 4))
        plt.plot(grid, means, marker='o')
        plt.xlabel(col)
        plt.ylabel('Predicted log-price')
        plt.title(f"PDP: {entry['name']} -- {col} (rank {entry['rank']})")
        out_file = ARTIFACT_DIR / f"pdp_{entry['name']}_{col}.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()
        # .as_posix(): str(Path) on Windows emits backslashes; keep the JSON
        # portable (see VERIFICATION_REPORT.md Rounds 1-2 for this bug class).
        out_summary.append({'feature': col, 'plot': out_file.as_posix()})

    with open(ARTIFACT_DIR / f"{entry['name']}_pdp_summary.json", 'w') as f:
        json.dump(out_summary, f, indent=2)
    print('PDPs generated for rank', entry['rank'], f"({entry['name']},"
          f" {entry['family']}) ->", [c['feature'] for c in out_summary])


ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
models = get_all_models()
test = pd.read_parquet(SPLIT_DIR / 'all_years_test.parquet')
for entry in models:
    model, preproc = load_entry_model(entry)
    _generate_pdp(entry, model, preproc, features_of(entry), test)
print('PDP passes completed for all', len(models), 'eligible ranked models')


# ---------------------------------------------------------------------------
# Round 11: prune stale PDP PNGs. Any models/artifacts/pdp_*.png whose
# (model, feature) pair is not in THIS run's PDP summary JSONs is a leftover
# from an obsolete feature set (e.g. the pre-blacklist pdp_<model>_BBL_pluto
# .png) and is DELETED. This run's *_pdp_summary.json files are the source of
# truth for what was regenerated.
# ---------------------------------------------------------------------------
current_pngs = set()
for _e in models:
    expected_pngs = json.loads(
        (ARTIFACT_DIR / f"{_e['name']}_pdp_summary.json").read_text())
    for rec in expected_pngs:
        current_pngs.add(f"pdp_{_e['name']}_{rec.get('feature', '')}.png")
        if rec.get('plot'):
            current_pngs.add(Path(rec['plot']).name)
removed = []
for png in ARTIFACT_DIR.glob('pdp_*.png'):
    if png.name not in current_pngs:
        removed.append(png.name)
        png.unlink()  # DELETE the stale artifact
        print('Deleted stale PDP artifact:', png.name)
print(f'PDP pruning: {len(removed)} stale PNG(s) deleted; '
      f'{len(current_pngs)} current panels across {len(models)} models.')
if removed:
    with open(ARTIFACT_DIR / 'pdp_prune_log.json', 'w') as _f:
        json.dump({'deleted_stale_pdps': removed}, _f, indent=2)

