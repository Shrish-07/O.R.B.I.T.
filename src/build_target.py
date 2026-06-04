# src/build_target.py

import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/sales_pluto_ideology.parquet"
OUTPUT_PATH = "data/processed/modeling_dataset_with_target.parquet"

df = pd.read_parquet(INPUT_PATH)

price_col = "SALE PRICE"

df["SALE_PRICE_NUM"] = (
    df[price_col]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("$", "", regex=False)
)

df["SALE_PRICE_NUM"] = pd.to_numeric(df["SALE_PRICE_NUM"], errors="coerce")

df = df[df["SALE_PRICE_NUM"] > 0].copy()

df["target_log_price"] = np.log(df["SALE_PRICE_NUM"])

assert df["target_log_price"].isna().sum() == 0

df.to_parquet(OUTPUT_PATH, index=False)

print("✅ Target built")
print("Shape:", df.shape)
