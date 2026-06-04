import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/modeling_dataset_fe.parquet")
OUTPUT_PATH = Path("data/processed/modeling_dataset_fe_imputed.parquet")

assert INPUT_PATH.exists(), "❌ Input dataset not found"

df = pd.read_parquet(INPUT_PATH)

assert "district_ideology" in df.columns, "❌ district_ideology column missing"

# -----------------------------
# Add missing indicator
# -----------------------------
df["district_ideology_missing"] = df["district_ideology"].isna().astype(int)

# -----------------------------
# Impute with borough-level median
# -----------------------------
if "BOROUGH" in df.columns:
    borough_medians = (
        df.groupby("BOROUGH")["district_ideology"]
        .median()
    )

    df["district_ideology"] = df.apply(
        lambda r: borough_medians.get(r["BOROUGH"], None)
        if pd.isna(r["district_ideology"])
        else r["district_ideology"],
        axis=1
    )

# -----------------------------
# Global fallback
# -----------------------------
global_median = df["district_ideology"].median()
df["district_ideology"] = df["district_ideology"].fillna(global_median)

assert df["district_ideology"].notna().all(), "❌ Ideology still has missing values"

df.to_parquet(OUTPUT_PATH, index=False)

print("✅ Ideology imputation complete")
print("Saved to:", OUTPUT_PATH)
