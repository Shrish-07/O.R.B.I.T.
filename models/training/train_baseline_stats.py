# models/training/train_baseline_stats.py

import pandas as pd
import yaml
import json
from pathlib import Path

TRAIN_PATH = "data/splits/train.parquet"
TEST_PATH  = "data/splits/test.parquet"
BLACKLIST_PATH = "config/feature_blacklist.yaml"

ARTIFACT_DIR = Path("models/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_log_price"

# -----------------------------
# Load
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
# Feature set
# -----------------------------
features = [
    c for c in train.select_dtypes(include="number").columns
    if c.lower() not in blacklist and c != TARGET
]

# -----------------------------
# Summary stats
# -----------------------------
stats = {
    "train_rows": len(train),
    "test_rows": len(test),
    "num_features": len(features),
    "feature_names": features,
}

# -----------------------------
# Save artifact
# -----------------------------
with open(ARTIFACT_DIR / "baseline_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("✅ Baseline stats saved")
print(json.dumps(stats, indent=2))
