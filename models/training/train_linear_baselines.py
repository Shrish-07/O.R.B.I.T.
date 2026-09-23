import sys

# Allow emoji/unicode prints (e.g. "📦 Artifacts written:") when this script is
# run directly on a cp1252 Windows console. When launched via main.py this is
# already handled (main.py sets PYTHONUTF8/PYTHONIOENCODING for stages).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path
import pandas as pd
import json
import yaml
import subprocess
import numpy as np
from datetime import datetime
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump

parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['all_years','year2017'], required=True)
parser.add_argument('--mode', choices=['base','political'], required=True)
args = parser.parse_args()

SPLIT_DIR = Path('data/splits')
MODEL_DIR = Path('models')
ARTIFACT_DIR = Path('models/artifacts')
RESULTS_DIR = Path('experiments/results')
REGISTRY = Path('experiments/registry.json')

MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'target_log_price'

TRAIN_PATH = SPLIT_DIR / f"{args.variant}_train.parquet"
TEST_PATH = SPLIT_DIR / f"{args.variant}_test.parquet"

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# load leakage blacklist
blacklist_path = Path('config/feature_blacklist.yaml')
if blacklist_path.exists():
    with open(blacklist_path) as f:
        blacklist = {c.lower() for c in yaml.safe_load(f).get('leakage_features', [])}
else:
    blacklist = set()

# select numeric features while excluding blacklist and target
features = [c for c in train.select_dtypes(include='number').columns if c != TARGET and c.lower() not in blacklist]

if args.mode == 'political':
    ideology_features = [c for c in ['dem_share','rep_share','turnout'] if c in train.columns]
    features += [c for c in ideology_features if c not in features]

X_train = train[features]
y_train = train[TARGET]
X_test = test[features]
y_test = test[TARGET]

# preprocessing pipeline
preproc = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

X_train_p = preproc.fit_transform(X_train)
X_test_p = preproc.transform(X_test)

# NOTE (2026 refresh): default-alpha Ridge() / ElasticNet() collapsed on the
# refreshed sales pull (Ridge R^2 -> -4.97) because the now-numeric unit-count
# columns are near-perfectly collinear (TOTAL UNITS ~ RESIDENTIAL + COMMERCIAL)
# and sklearn's default alpha=1.0 over-regularizes them. Both linear baselines
# are now fit with cross-validated regularization so neither is an artifact of a
# hand-picked alpha; the chosen alpha (and elasticnet's l1_ratio) are recorded in
# results/_metrics/registry for transparency. Ridge handles collinearity for
# PREDICTION once its penalty is sized correctly; ElasticNetCV additionally prunes
# the redundant column via its L1 term.
# Ridge's collinearity forces a very large optimal alpha; extend the grid top so
# the CV optimum is interior rather than capped at the grid edge.
ALPHAS = np.logspace(-4, 7, 40)
ELASTICNET_L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

for cls, name, fit_kwargs in [
    (RidgeCV, 'ridge', {'alphas': ALPHAS, 'cv': 5}),
    (ElasticNetCV, 'elasticnet', {'l1_ratio': ELASTICNET_L1_RATIOS,
                                  'alphas': np.logspace(-4, 1, 20),
                                  'cv': 5, 'max_iter': 100000, 'n_jobs': -1}),
]:
    model = cls(**fit_kwargs)
    model.fit(X_train_p, y_train)
    preds = model.predict(X_test_p)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    fullname = f"{name}_{args.variant}_{args.mode}"
    chosen_alpha = float(model.alpha_)
    chosen_l1 = float(model.l1_ratio_) if getattr(model, 'l1_ratio_', None) is not None else None
    hyper = {'alpha': chosen_alpha}
    if chosen_l1 is not None:
        hyper['l1_ratio'] = chosen_l1
    results = {
        'model': name.capitalize(),
        'name': fullname,
        'model_class': cls.__name__,
        'variant': args.variant,
        'mode': args.mode,
        'n_train': int(len(train)),
        'n_test': int(len(test)),
        'n_features': int(len(features)),
        'r2': float(r2),
        'mae': float(mae),
        'hyperparameters': hyper,
    }
    with open(RESULTS_DIR / f"{fullname}.json", 'w') as f:
        json.dump(results, f, indent=2)

    model_path = MODEL_DIR / f"{fullname}.joblib"
    dump(model, model_path)

    preproc_path = ARTIFACT_DIR / f"{fullname}_preproc.joblib"
    dump(preproc, preproc_path)

    with open(ARTIFACT_DIR / f"{fullname}_features.json", 'w') as f:
        json.dump(features, f, indent=2)
    with open(ARTIFACT_DIR / f"{fullname}_metrics.json", 'w') as f:
        json.dump({'r2': float(r2), 'mae': float(mae), 'hyperparameters': hyper}, f, indent=2)

    print(f"{name.capitalize()} training complete:", results)
    print('\n📦 Artifacts written:')
    print('Model:   ', model_path)
    print('Preproc: ', preproc_path)
    print('Metrics: ', ARTIFACT_DIR / f"{fullname}_metrics.json")

    # register
    try:
        git = subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
    except Exception:
        git = None

    exp = {
        'id': f"exp-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        'created_utc': datetime.utcnow().isoformat() + 'Z',
        'git_commit': git,
        'script': 'models/training/train_linear_baselines.py',
        'name': fullname,
        'mode': args.mode,
        'scope': '2017' if args.variant == 'year2017' else 'all',
        'hypothesis': f'{name.capitalize()} baseline',
        'reasoning': 'Baseline linear models for comparison',
        'metrics': {'r2': float(r2), 'mae': float(mae), **hyper},
        'results': results,
        # .as_posix(): str(Path) on Windows emits backslashes, which silently
        # breaks registry.json portability (this bug recurred twice — see
        # VERIFICATION_REPORT.md Rounds 1-2). as_posix() is the structural fix.
        'features_path': (ARTIFACT_DIR / f"{fullname}_features.json").as_posix(),
        'model_path': model_path.as_posix(),
    }

    if REGISTRY.exists():
        registry = json.loads(REGISTRY.read_text())
    else:
        registry = []
    # Replace, don't duplicate: a re-run of this exact script+config produces
    # the identical (script, name) identity, so drop the prior entry instead
    # of accumulating stale duplicate rows in the registry.
    registry = [e for e in registry
                if not (e.get('script') == exp['script'] and e.get('name') == exp['name'])]
    registry.append(exp)
    REGISTRY.write_text(json.dumps(registry, indent=2))
    print('Registered experiment:', exp['id'])

# Keep champion.json pointing at a live entry: replacing entries above could
# otherwise orphan the previous champion id until the next full pipeline run.
subprocess.run([sys.executable, str(Path('src/ops/experiment_runner.py'))], check=False)
