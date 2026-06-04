# src/causal_analysis.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import geopandas as gpd
from causalml.inference.meta import BaseXLearner
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import warnings

# -----------------------------
# Global warning suppression (visual spam killer)
# -----------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# === Paths ===
DATA_PATH = Path("data/intermediate/processed_data.parquet")
MODEL_PATH = Path("models/xgb_model.pkl")
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNCIL_SHP = Path("data/raw/council_districts.shp")  # shapefile for council maps

# === Load processed data ===
df = pd.read_parquet(DATA_PATH)
print("Rows loaded:", len(df))

# -----------------------------
# Column existence guards
# -----------------------------
REQUIRED_COLS = {
    "GROSS SQUARE FEET", "LAND SQUARE FEET", "YEAR BUILT",
    "weighted_dem", "BOROUGH", "BUILDING CLASS CATEGORY",
    "Council District", "landuse", "ideology_score",
    "post_2023_reform", "log_sale_price", "BBL"
}

missing = REQUIRED_COLS - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# === Load baseline model ===
model = joblib.load(MODEL_PATH)
print("Baseline model loaded.")

# === Select feature columns for causal model ===
X_cols = [
    "GROSS SQUARE FEET", "LAND SQUARE FEET", "YEAR BUILT",
    "weighted_dem", "BOROUGH", "BUILDING CLASS CATEGORY",
    "Council District", "landuse", "ideology_score"
]

treatment_col = "post_2023_reform"
outcome_col = "log_sale_price"

# -----------------------------
# Coerce numeric safety (XGB hard-fail guard)
# -----------------------------
numeric_cols = [
    "GROSS SQUARE FEET",
    "LAND SQUARE FEET",
    "YEAR BUILT",
    "weighted_dem",
    "ideology_score",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Coerce categorical features → codes (XGB cannot take object)
# -----------------------------
categorical_cols = [
    "BOROUGH",
    "BUILDING CLASS CATEGORY",
    "Council District",
    "landuse",
]

for col in categorical_cols:
    df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

# -----------------------------
# Coerce treatment to {0,1}
# -----------------------------
df[treatment_col] = pd.to_numeric(df[treatment_col], errors="coerce")
df[treatment_col] = (df[treatment_col] > 0).astype(int)

# -----------------------------
# Drop rows with any required NA
# -----------------------------
df = df.dropna(subset=X_cols + [treatment_col, outcome_col])

print("Rows after cleaning:", len(df))

# -----------------------------
# Final feature matrix
# -----------------------------
X = df[X_cols].copy()
t = df[treatment_col].copy()
y = df[outcome_col].copy()

# -----------------------------
# Fit BaseXLearner
# -----------------------------
print("Fitting causal model...")

learner = BaseXLearner(
    learner=XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist"
    )
)

learner.fit(X=X, treatment=t, y=y)

print("Causal model trained.")

# -----------------------------
# Estimate counterfactuals
# -----------------------------
df["cf_log_price"] = learner.predict(X=X, treatment=1)
df["ate"] = df["cf_log_price"] - y

# === Average treatment effect (ATE) ===
ate = float(df["ate"].mean())
print(f"Average Treatment Effect (ATE): {ate:.4f}")

# -----------------------------
# Subgroup analysis by ideology quartiles
# -----------------------------
df["ideology_quartile"] = pd.qcut(
    df["ideology_score"],
    q=4,
    labels=False,
    duplicates="drop"
)

quartile_ates = (
    df.groupby("ideology_quartile", as_index=False)["ate"]
      .mean()
)

print("ATE by ideology quartile:")
print(quartile_ates.to_string(index=False))

# -----------------------------
# Save results
# -----------------------------
out_cols = [
    "BBL",
    outcome_col,
    "cf_log_price",
    "ate",
    "ideology_score",
    "ideology_quartile"
]

df[out_cols].to_parquet(
    OUTPUT_DIR / "causal_results.parquet",
    index=False
)

print("✅ Causal predictions saved.")

# -----------------------------
# Visualizations (quiet)
# -----------------------------
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.boxplot(x="ideology_quartile", y="ate", data=df)
plt.title("Policy Shock Effect by Ideology Quartile")
plt.xlabel("Ideology Quartile (0=Low, 3=High)")
plt.ylabel("Predicted Price Change (log scale)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "policy_effect_boxplot.png")
plt.close()

print("✅ Boxplot saved.")

# -----------------------------
# Map ATE by Council District (guarded + quiet)
# -----------------------------
if COUNCIL_SHP.exists():
    council_gdf = gpd.read_file(COUNCIL_SHP)

    # Aggregate mean ATE by district
    district_ates = (
        df.groupby("Council District", as_index=False)["ate"]
          .mean()
    )

    if "CounDist" not in council_gdf.columns:
        raise ValueError("Shapefile missing 'CounDist' column")

    council_gdf = council_gdf.merge(
        district_ates,
        left_on="CounDist",
        right_on="Council District",
        how="left"
    )

    # Simple Folium map
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

    folium.Choropleth(
        geo_data=council_gdf,
        name="ATE",
        data=council_gdf,
        columns=["CounDist", "ate"],
        key_on="feature.properties.CounDist",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Average Treatment Effect (log-price)"
    ).add_to(m)

    m.save(OUTPUT_DIR / "policy_effect_map.html")
    print("✅ Map saved as HTML.")

else:
    print("⚠️  Council shapefile not found — skipping map.")
