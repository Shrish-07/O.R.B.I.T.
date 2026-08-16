"""Quick dataset dimension stats."""
import pandas as pd
import json

df = pd.read_parquet("data/canonical/modeling_dataset_canonical_v2.parquet")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Shape: {df.shape}")

# Feature counts
for name in ["lgbm_all_years_base", "lgbm_all_years_political"]:
    fpath = f"models/artifacts/{name}_features.json"
    feats = json.loads(open(fpath).read())
    print(f"{name}: {len(feats)} features")
