# src/train_baseline.py

import pandas as pd
import yaml
import json
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# Paths
# -----------------------------
TRAIN_PATH = "data/splits/train.parquet"
TEST_PATH  = "data/splits/test.parquet"
BLACKLIST_PATH = "config/feature_blacklist.yaml"

ARTIFACT_DIR = Path("models/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_log_price"

# -----------------------------
# Load data
# -----------------------------
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

# -----------------------------
# Drop split-only columns
# -----------------------------
for df in (train, test):
    df.drop(columns=[c for c in ["SALE_YEAR"] if c in df.columns], inplace=True)

# -----------------------------
# Load blacklist
# -----------------------------
with open(BLACKLIST_PATH, "r") as f:
    blacklist = {c.lower() for c in yaml.safe_load(f)["leakage_features"]}

# -----------------------------
# Feature selection (numeric + leakage-safe)
# -----------------------------
features = [
    c for c in train.select_dtypes(include="number").columns
    if c.lower() not in blacklist and c != TARGET
]

# -----------------------------
# Assert no leakage
# -----------------------------
assert not any(c.lower() in blacklist for c in features), "Leakage features detected"
assert TARGET in train.columns
assert TARGET in test.columns

# -----------------------------
# Matrices
# -----------------------------
X_train = train[features]
y_train = train[TARGET]

X_test  = test[features]
y_test  = test[TARGET]

# -----------------------------
# Baseline pipeline (REQUIRED)
# -----------------------------
pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ]
)

# -----------------------------
# Train
# -----------------------------
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print("\n✅ ORBIT v2 — Ridge Baseline (Leakage-Safe + Imputed)")
print(f"Features used: {len(features)}")
print(f"R² (temporal): {r2:.4f}")
print(f"MAE:           {mae:.4f}")

# -----------------------------
# Save metrics
# -----------------------------
with open(ARTIFACT_DIR / "ridge_baseline_metrics.json", "w") as f:
    json.dump(
        {
            "model": "ridge + median imputation",
            "r2": r2,
            "mae": mae,
            "num_features": len(features),
        },
        f,
        indent=2,
    )

# -----------------------------
# Save features used (for audit parity with LGBM)
# -----------------------------
with open(ARTIFACT_DIR / "ridge_features.json", "w") as f:
    json.dump(features, f, indent=2)
