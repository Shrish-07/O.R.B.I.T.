import pandas as pd
from pathlib import Path

Path("docs").mkdir(exist_ok=True)

df = pd.read_parquet("data/processed/modeling_dataset.parquet")

schema = (
    df.dtypes
    .astype(str)
    .reset_index()
    .rename(columns={"index": "column", 0: "dtype"})
)

schema.to_csv("docs/modeling_schema.csv", index=False)

print("✅ Schema frozen")
print("Shape:", df.shape)
