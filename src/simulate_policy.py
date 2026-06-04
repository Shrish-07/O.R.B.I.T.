# simulate_policy.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# === Paths ===
INTERMEDIATE = Path("data/intermediate")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# === Load merged dataset ===
df_path = INTERMEDIATE / "sales_pluto_ideology.parquet"
print("Loading merged dataset...")
df = pd.read_parquet(df_path)
print("Rows loaded:", len(df))

# === Ensure required columns exist ===
required_numeric = ["GROSS SQUARE FEET", "LAND SQUARE FEET", "YEAR BUILT", "weighted_dem"]
required_categorical = ["BOROUGH", "BUILDING CLASS CATEGORY", "Council District", "landuse"]

# Fix numeric columns
for col in required_numeric:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Fix categorical columns
for col in required_categorical:
    if col not in df.columns:
        df[col] = "Missing"
    df[col] = df[col].astype("category")

# === Create policy flag ===
if 'SALE_DATE' not in df.columns:
    # Attempt to find alternative date column
    if 'SALE DATE' in df.columns:
        df['SALE_DATE'] = pd.to_datetime(df['SALE DATE'], errors='coerce')
    else:
        raise ValueError("No SALE_DATE or SALE DATE column found in dataset")

df['policy_flag'] = (df['SALE_DATE'] > pd.to_datetime('2023-01-01')).astype(int)

# === Prepare features for prediction ===
feature_cols = required_numeric + required_categorical
X = df[feature_cols]

# === Load trained model (joblib) ===
model_path = MODEL_DIR / "xgb_model.pkl"
print("Loading trained model...")
model = joblib.load(model_path)

# === Predict log-prices ===
print("Predicting log-prices...")
df['log_price_pred'] = model.predict(X)

# Convert back to sale price
df['predicted_sale_price'] = np.expm1(df['log_price_pred'])

# === Optional: simulate counterfactual policy ===
# Example: what if policy never happened?
X_cf = X.copy()
X_cf['policy_flag'] = 0  # simulate no policy
# Ensure column exists in pipeline (if used in training)
if 'policy_flag' in X_cf.columns and 'policy_flag' in feature_cols:
    df['log_price_cf'] = model.predict(X_cf)
    df['predicted_sale_price_cf'] = np.expm1(df['log_price_cf'])

# === Save predictions ===
output_file = OUTPUT_DIR / "sales_predictions.parquet"
df.to_parquet(output_file, index=False)
print(f"✅ Predictions saved at: {output_file}")
