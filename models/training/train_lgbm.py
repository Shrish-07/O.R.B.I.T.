# models/training/train_lgbm.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Allow emoji/unicode prints when this script is run directly on a cp1252
# Windows console. When launched via main.py this is already handled
# (main.py sets PYTHONUTF8/PYTHONIOENCODING for stages).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import pandas as pd
import lightgbm as lgb
import yaml
import json
import subprocess
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--variant",
    type=str,
    required=True,
    choices=["all_years", "year2017"],
    help="Which experimental split to use",
)
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["base", "political"],
    help="Whether to include ideology features",
)
args = parser.parse_args()

VARIANT = args.variant
MODE = args.mode

# -----------------------------
# Paths
# -----------------------------
SPLIT_DIR = Path("data/splits")
BLACKLIST_PATH = Path("config/feature_blacklist.yaml")

MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("models/artifacts")
RESULTS_DIR = Path("experiments/results")
REGISTRY = Path("experiments/registry.json")

MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_log_price"

TRAIN_PATH = SPLIT_DIR / f"{VARIANT}_train.parquet"
TEST_PATH = SPLIT_DIR / f"{VARIANT}_test.parquet"

MODEL_PATH = MODEL_DIR / f"lgbm_{VARIANT}_{MODE}.txt"
METRICS_PATH = ARTIFACT_DIR / f"lgbm_{VARIANT}_{MODE}_metrics.json"
FEATURES_PATH = ARTIFACT_DIR / f"lgbm_{VARIANT}_{MODE}_features.json"
RESULTS_PATH = RESULTS_DIR / f"lgbm_{VARIANT}_{MODE}.json"

# -----------------------------
# Load data
# -----------------------------
if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Missing split file: {TRAIN_PATH}")
if not TEST_PATH.exists():
    raise FileNotFoundError(f"Missing split file: {TEST_PATH}")

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# -----------------------------
# Drop split-only columns
# -----------------------------
for df in (train, test):
    if "SALE_YEAR" in df.columns:
        df.drop(columns=["SALE_YEAR"], inplace=True)

# -----------------------------
# Load leakage blacklist
# -----------------------------
with open(BLACKLIST_PATH, "r") as f:
    blacklist = {c.lower() for c in yaml.safe_load(f)["leakage_features"]}

# -----------------------------
# Select numeric features (leakage-safe)
# -----------------------------
features = [
    c
    for c in train.select_dtypes(include="number").columns
    if c.lower() not in blacklist and c != TARGET
]

# -----------------------------
# Ideology mode
# -----------------------------
IDEOLOGY_FEATURES = [
    "dem_share",
    "rep_share",
    "turnout",
]

# Exclude ideology features from base mode (they may be numeric columns)
if MODE == "base":
    features = [c for c in features if c not in IDEOLOGY_FEATURES]

# Add ideology features for political mode if present
if MODE == "political":
    features = features + [c for c in IDEOLOGY_FEATURES if c in train.columns]

# Deduplicate features in case of overlap
features = list(dict.fromkeys(features))

if len(features) == 0:
    raise RuntimeError("No features selected after blacklist filtering")

# -----------------------------
# Assert no leakage
# -----------------------------
leaks = [c for c in features if c.lower() in blacklist]
assert not leaks, f"Leakage detected: {leaks}"

# -----------------------------
# Matrices
# -----------------------------
X_train = train[features]
y_train = train[TARGET]

X_test = test[features]
y_test = test[TARGET]

# -----------------------------
# LightGBM params (governance-safe)
# -----------------------------
params = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "seed": 42,
    "verbosity": -1,
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# -----------------------------
# Train
# -----------------------------
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1500,
    valid_sets=[lgb_test],
    callbacks=[
        lgb.early_stopping(stopping_rounds=75),
        lgb.log_evaluation(period=100),
    ],
)

# -----------------------------
# Evaluate
# -----------------------------
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)

# -----------------------------
# Compare-models–compatible results
# -----------------------------
scope = "2017" if VARIANT == "year2017" else "all"

results = {
    "model": "LightGBM",
    "mode": MODE,
    "scope": scope,
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "features_used": int(len(features)),
    "r2": float(r2),
    "mae": float(mae),
}

print(f"\n✅ ORBIT v2 — LightGBM ({VARIANT} | {MODE})")
print(f"Train rows: {len(train):,}")
print(f"Test rows:  {len(test):,}")
print(f"Features used: {len(features)}")
print(f"R² (temporal): {r2:.4f}")
print(f"MAE:           {mae:.4f}")
print(f"Best iteration: {model.best_iteration}")

# -----------------------------
# Save artifacts
# -----------------------------
model.save_model(MODEL_PATH)

with open(METRICS_PATH, "w") as f:
    json.dump(
        {
            "variant": VARIANT,
            "mode": MODE,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "num_features": int(len(features)),
            "r2_temporal": float(round(r2, 6)),
            "mae_log_price": float(round(mae, 6)),
            "best_iteration": int(model.best_iteration),
        },
        f,
        indent=2,
    )

with open(FEATURES_PATH, "w") as f:
    json.dump(features, f, indent=2)

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

# -----------------------------
# Register experiment (registry.json)
# -----------------------------
# Same pattern as train_catboost.py / train_random_forest.py: every retrain
# (base and political) must leave a registry entry so champion selection
# (src/ops/experiment_runner.py, run by main.py as the refresh_champion stage)
# sees the current run with the current feature count.
try:
    git = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
except Exception:
    git = None

exp = {
    "id": f"exp-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
    "created_utc": datetime.utcnow().isoformat() + "Z",
    "git_commit": git,
    "script": "models/training/train_lgbm.py",
    "name": f"lgbm_{VARIANT}_{MODE}",
    "mode": MODE,
    "scope": scope,
    "hypothesis": "Champion LightGBM base spec" if MODE == "base" else
                  "LightGBM ideology-augmented spec (adds dem_share/turnout features)",
    "reasoning": "Production training run (the models the paper's Table 2/3 report)",
    "metrics": {"r2": float(r2), "mae": float(mae)},
    "results": results,
    "features_used": int(len(features)),
    # .as_posix(): str(Path) on Windows emits backslashes, which silently
    # breaks registry.json portability (this bug recurred twice — see
    # VERIFICATION_REPORT.md Rounds 1-2). as_posix() is the structural fix.
    "features_path": FEATURES_PATH.as_posix(),
    "model_path": MODEL_PATH.as_posix(),
}

if REGISTRY.exists():
    registry = json.loads(REGISTRY.read_text())
else:
    registry = []
# Replace, don't duplicate: a re-run of this exact script+config produces the
# identical (script, name) identity, so drop the prior entry instead of
# accumulating stale duplicate rows in the registry.
registry = [e for e in registry
            if not (e.get("script") == exp["script"] and e.get("name") == exp["name"])]
registry.append(exp)
REGISTRY.write_text(json.dumps(registry, indent=2))
print("Registered experiment:", exp["id"])

# Keep champion.json pointing at a live entry: replacing an entry above could
# otherwise orphan the previous champion id until the next full pipeline run.
subprocess.run([sys.executable, str(Path("src/ops/experiment_runner.py"))], check=False)

print("\n📦 Artifacts written:")
print("Model:   ", MODEL_PATH)
print("Metrics: ", METRICS_PATH)
print("Features:", FEATURES_PATH)
print("Results: ", RESULTS_PATH)
