import pandas as pd
from pathlib import Path

RAW_DATA_PATH = "data/raw/merged_sales_pluto.csv"
OUTPUT_PATH = "data/intermediate/processed_data.parquet"

print("Loading raw merged dataset...")
df = pd.read_csv(RAW_DATA_PATH)

print(f"Rows loaded: {len(df)}")

# -------------------------
# BASIC CLEANING
# -------------------------

df["SALE_DATE"] = pd.to_datetime(df["SALE_DATE"])
df = df[(df["SALE_PRICE"] > 0) & (df["SALE_PRICE"] < 1e8)]

df["log_sale_price"] = np.log(df["SALE_PRICE"])

# Example treatment variable (adjust date if needed)
df["post_policy"] = (df["SALE_DATE"] >= "2023-01-01").astype(int)

# Drop rows missing required columns
required_cols = [
    "log_sale_price",
    "post_policy",
    "council",
    "GROSS_SQUARE_FEET",
    "LAND_SQUARE_FEET",
    "YEAR_BUILT"
]

df = df.dropna(subset=required_cols)

# -------------------------
# SAVE
# -------------------------

Path("data/intermediate").mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH)

print(f"✅ Processed data saved to {OUTPUT_PATH}")
