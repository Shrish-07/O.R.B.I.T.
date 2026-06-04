import argparse
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import yaml
import json
from sklearn.metrics import r2_score, mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['all_years','year2017'], required=True)
parser.add_argument('--mode', choices=['base','political'], required=True)
parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--num_leaves', type=int, default=64)
args = parser.parse_args()

SPLIT_DIR = Path('data/splits')
MODEL_DIR = Path('models')
ARTIFACT_DIR = Path('models/artifacts')
RESULTS_DIR = Path('experiments/results')

MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'target_log_price'

TRAIN_PATH = SPLIT_DIR / f"{args.variant}_train.parquet"
TEST_PATH = SPLIT_DIR / f"{args.variant}_test.parquet"

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# load leakage blacklist and filter numeric features
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

params = {
    'objective': 'regression',
    'metric': 'l2',
    'learning_rate': args.learning_rate,
    'num_leaves': args.num_leaves,
    'verbosity': -1,
}

lgb_train = lgb.Dataset(X_train, y_train)
model = lgb.train(params, lgb_train, num_boost_round=800, valid_sets=[lgb_train], callbacks=[lgb.log_evaluation(period=200)])

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)

name = f"lgbm_tuned_{args.variant}_{args.mode}_lr{args.learning_rate}_nl{args.num_leaves}"
results = {
    'model': 'LightGBM',
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

model_path = MODEL_DIR / f"{name}.txt"
model.save_model(str(model_path))

with open(ARTIFACT_DIR / f"{name}_features.json", 'w') as f:
    json.dump(features, f, indent=2)
with open(ARTIFACT_DIR / f"{name}_metrics.json", 'w') as f:
    json.dump({'r2': r2, 'mae': mae}, f, indent=2)

print('Tuned training complete:', results)
