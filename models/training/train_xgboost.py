import argparse
from pathlib import Path
import pandas as pd
import json
import yaml
import subprocess
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error

try:
    import xgboost as xgb
    from joblib import dump
except Exception:
    raise

parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['all_years','year2017'], required=True)
parser.add_argument('--mode', choices=['base','political'], required=True)
parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--max_depth', type=int, default=6)
parser.add_argument('--n_estimators', type=int, default=1000)
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

X_train = train[features].fillna(0)
y_train = train[TARGET]
X_test = test[features].fillna(0)
y_test = test[TARGET]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params_xgb = {
    'objective': 'reg:squarederror',
    'learning_rate': args.learning_rate,
    'max_depth': args.max_depth,
    'verbosity': 0,
}

num_boost_round = args.n_estimators
bst = xgb.train(
    params_xgb,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, 'eval')],
    early_stopping_rounds=50,
    verbose_eval=100,
)

preds = bst.predict(dtest)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)

name = f"xgb_{args.variant}_{args.mode}_lr{args.learning_rate}_md{args.max_depth}"
results = {
    'model': 'XGBoost',
    'name': name,
    'variant': args.variant,
    'mode': args.mode,
    'n_train': int(len(train)),
    'n_test': int(len(test)),
    'n_features': int(len(features)),
    'r2': float(r2),
    'mae': float(mae),
}

with open(RESULTS_DIR / f"{name}.json", 'w') as f:
    json.dump(results, f, indent=2)

model_path = MODEL_DIR / f"{name}.model"
bst.save_model(str(model_path))

with open(ARTIFACT_DIR / f"{name}_features.json", 'w') as f:
    json.dump(features, f, indent=2)
with open(ARTIFACT_DIR / f"{name}_metrics.json", 'w') as f:
    json.dump({'r2': r2, 'mae': mae}, f, indent=2)

print('XGBoost training complete:', results)
print('\n📦 Artifacts written:')
print('Model:   ', model_path)
print('Metrics: ', ARTIFACT_DIR / f"{name}_metrics.json")
print('Features:', ARTIFACT_DIR / f"{name}_features.json")
print('Results: ', RESULTS_DIR / f"{name}.json")

# register in experiments/registry.json
try:
    git = subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
except Exception:
    git = None

exp = {
    'id': f"exp-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
    'created_utc': datetime.utcnow().isoformat() + 'Z',
    'git_commit': git,
    'script': 'models/training/train_xgboost.py',
    'name': name,
    'mode': args.mode,
    'scope': '2017' if args.variant == 'year2017' else 'all',
    'hypothesis': 'XGBoost baseline for diversity',
    'reasoning': 'Automated model family comparison',
    'metrics': {'r2': float(r2), 'mae': float(mae)},
    'results': results,
    'features_path': str(ARTIFACT_DIR / f"{name}_features.json"),
    'model_path': str(model_path),
}

if REGISTRY.exists():
    registry = json.loads(REGISTRY.read_text())
else:
    registry = []
registry.append(exp)
REGISTRY.write_text(json.dumps(registry, indent=2))

print('Registered experiment:', exp['id'])

try:
    # promote champion selection
    subprocess.check_call([Path('src') / 'experiment_runner.py'])
except Exception:
    pass
