"""Lightweight smoke training used by CI to validate training pipeline quickly."""
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import json

SPLIT_DIR = Path('data/splits')
RESULTS_DIR = Path('experiments/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / 'all_years_train.parquet'
TEST_PATH = SPLIT_DIR / 'all_years_test.parquet'

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError('Required split files not found in data/splits')

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# select numeric features
TARGET = 'target_log_price'
features = [c for c in train.select_dtypes(include='number').columns if c != TARGET]

# sample small subset for quick run
train_small = train.sample(n=min(2000, len(train)), random_state=42)
test_small = test.sample(n=min(1000, len(test)), random_state=42)

X_train = train_small[features].fillna(0)
y_train = train_small[TARGET]
X_test = test_small[features].fillna(0)
y_test = test_small[TARGET]

params = {
    'objective': 'regression',
    'metric': 'l2',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbosity': -1,
}

train_set = lgb.Dataset(X_train, y_train)
model = lgb.train(params, train_set, num_boost_round=50)
preds = model.predict(X_test)

mae = float(mean_absolute_error(y_test, preds))
r2 = float(r2_score(y_test, preds))

out = {
    'smoke_test': True,
    'n_train': int(len(train_small)),
    'n_test': int(len(test_small)),
    'n_features': int(len(features)),
    'mae': mae,
    'r2': r2,
}

with open(RESULTS_DIR / 'smoke_train_result.json', 'w') as f:
    json.dump(out, f, indent=2)

print('Smoke training complete:', out)
