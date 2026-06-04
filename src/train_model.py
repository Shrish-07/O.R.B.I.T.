import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------------------
# CLI args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["base", "political"], required=True)
parser.add_argument("--scope", choices=["2017", "all"], required=True)
args = parser.parse_args()

MODE = args.mode
SCOPE = args.scope

print(f"\n🚀 ORBIT v2 — Training ({MODE.upper()} | {SCOPE.upper()})\n")

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/processed/modeling_dataset_fe_imputed.parquet")
RESULTS_DIR = Path("results") / MODE / SCOPE
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet(DATA_PATH)

# -----------------------------
# Scope filter
# -----------------------------
if SCOPE == "2017":
    df = df[df["SALE_YEAR"] == 2017].copy()

if df.empty:
    raise RuntimeError("❌ No rows left after scope filter")

# -----------------------------
# Target auto-detection
# -----------------------------
TARGET_CANDIDATES = [
    "log_sale_price",
    "LOG_SALE_PRICE",
    "sale_price_log",
    "SALE_PRICE_NUM",
    "sale_price",
    "SALE PRICE",
]

TARGET = None
for c in TARGET_CANDIDATES:
    if c in df.columns:
        TARGET = c
        break

if TARGET is None:
    raise RuntimeError(
        "❌ No target column found.\n"
        "Looked for: " + ", ".join(TARGET_CANDIDATES)
    )

print(f"🎯 Using target: {TARGET}")

# -----------------------------
# Create log_sale_price if needed
# -----------------------------
if TARGET != "log_sale_price":
    if (df[TARGET] <= 0).any():
        raise RuntimeError("❌ Non-positive sale prices found; cannot log-transform")

    df["log_sale_price"] = np.log(df[TARGET])
    TARGET = "log_sale_price"
    print("🔁 Created log_sale_price from raw target")

# -----------------------------
# Drop leakage features
# -----------------------------
LEAKAGE = [
    "SALE PRICE",
    "SALE_PRICE_NUM",
    "assessland",
    "assesstot",
    "sale_price",
    "saleprice",
]

leakage_cols = [c for c in LEAKAGE if c in df.columns]
if leakage_cols:
    print("🚫 Dropping leakage features:")
    for c in leakage_cols:
        print("  ", c)
    df = df.drop(columns=leakage_cols)

# -----------------------------
# ECON feature core
# -----------------------------
ECON_FEATURES = [
    "log_land_square_feet",
    "log_gross_square_feet",
    "log_total_units",
    "building_age",
    "far",
    "SALE_YEAR",
    "Latitude",
    "Longitude",
]

if MODE == "political":
    ECON_FEATURES.append("district_ideology")

ECON_FEATURES = [c for c in ECON_FEATURES if c in df.columns]

if len(ECON_FEATURES) < 6:
    raise RuntimeError(
        "❌ ECON feature core incomplete.\n"
        f"Available: {ECON_FEATURES}"
    )

print("🧠 Using ECON feature core:")
for f in ECON_FEATURES:
    print("  ", f)

features = ECON_FEATURES.copy()

# -----------------------------
# Subset + type cleanup
# -----------------------------
df_ = df[features + [TARGET]].copy()

for c in features:
    df_[c] = pd.to_numeric(df_[c], errors="coerce")

df_[TARGET] = pd.to_numeric(df_[TARGET], errors="coerce")

# -----------------------------
# Train/test split
# -----------------------------
df_train, df_test = train_test_split(
    df_,
    test_size=0.25,
    random_state=42,
)

# -----------------------------
# Impute + drop all-NaN features
# -----------------------------
bad_features = []

for c in features:
    med = df_train[c].median()
    if pd.isna(med):
        print(f"⚠️ Dropping all-NaN feature: {c}")
        bad_features.append(c)
    else:
        df_train[c] = df_train[c].fillna(med)
        df_test[c] = df_test[c].fillna(med)

if bad_features:
    df_train = df_train.drop(columns=bad_features)
    df_test = df_test.drop(columns=bad_features)
    features = [c for c in features if c not in bad_features]

# -----------------------------
# Final row drop (target NaNs)
# -----------------------------
df_train = df_train.dropna(subset=[TARGET])
df_test = df_test.dropna(subset=[TARGET])

print(
    f"⚠️  Dropped rows with NaNs — "
    f"train: {df_train.shape[0]}, test: {df_test.shape[0]}"
)

if df_train.empty or df_test.empty:
    raise RuntimeError("❌ Train or test set empty after cleaning")

# -----------------------------
# LightGBM datasets
# -----------------------------
X_train = df_train[features]
y_train = df_train[TARGET]

X_test = df_test[features]
y_test = df_test[TARGET]

if X_train.shape[1] == 0:
    raise RuntimeError("❌ No features left after filtering")

train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

# -----------------------------
# LightGBM params
# -----------------------------
params = {
    "objective": "regression",
    "metric": "l2",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=75),
    lgb.log_evaluation(period=50),
]

print("Training until validation scores don't improve for 75 rounds")

model = lgb.train(
    params,
    train_set,
    valid_sets=[train_set, valid_set],
    valid_names=["train", "valid"],
    callbacks=callbacks,
)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n✅ ORBIT v2 — LightGBM ({MODE.upper()} | {SCOPE.upper()})")
print(f"Features used: {len(features)}")
print(f"R²:  {r2:.4f}")
print(f"MAE: {mae:.4f}")

# -----------------------------
# Save artifacts
# -----------------------------
model.save_model(str(RESULTS_DIR / "model.txt"))

metrics = {
    "mode": MODE,
    "scope": SCOPE,
    "target": TARGET,
    "features": features,
    "r2": float(r2),
    "mae": float(mae),
}

with open(RESULTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n📦 Saved model + metrics to: {RESULTS_DIR}")
