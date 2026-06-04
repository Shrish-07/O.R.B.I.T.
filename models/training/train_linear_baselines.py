import argparse
from pathlib import Path
import pandas as pd
import json
import yaml
import subprocess
from datetime import datetime
from sklearn.linear_model import Ridge, ElasticNet
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

for cls, name in [(Ridge, 'ridge'), (ElasticNet, 'elasticnet')]:
    model = cls()
    model.fit(X_train_p, y_train)
    preds = model.predict(X_test_p)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    fullname = f"{name}_{args.variant}_{args.mode}"
    results = {
        'model': name.capitalize(),
        'name': fullname,
        'variant': args.variant,
        'mode': args.mode,
        'n_train': int(len(train)),
        'n_test': int(len(test)),
        'n_features': int(len(features)),
        'r2': float(r2),
        'mae': float(mae),
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
        json.dump({'r2': r2, 'mae': mae}, f, indent=2)

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
        'metrics': {'r2': float(r2), 'mae': float(mae)},
        'results': results,
        'features_path': str(ARTIFACT_DIR / f"{fullname}_features.json"),
        'model_path': str(model_path),
    }

    if REGISTRY.exists():
        registry = json.loads(REGISTRY.read_text())
    else:
        registry = []
    registry.append(exp)
    REGISTRY.write_text(json.dumps(registry, indent=2))
    print('Registered experiment:', exp['id'])
