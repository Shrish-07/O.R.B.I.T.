# src/create_feature_manifest.py
import pandas as pd
import json

df = pd.read_parquet("data/processed/modeling_dataset.parquet")

manifest = {
    "n_rows": int(df.shape[0]),
    "n_columns": int(df.shape[1]),
    "columns": sorted(df.columns.tolist()),
    "target": "target_log_price"
}

with open("docs/feature_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("✅ Feature manifest saved")
