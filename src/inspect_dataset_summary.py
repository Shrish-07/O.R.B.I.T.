import sys, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import pandas as pd
pd.set_option("display.max_rows", 0)
pd.set_option("display.max_columns", 0)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", 0)


import sys, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import pandas as pd

df = pd.read_parquet("data/processed/modeling_dataset_with_target.parquet")

rows = len(df)
cols = len(df.columns)

years = None
if "sale_year" in df.columns:
    years = (int(df["sale_year"].min()), int(df["sale_year"].max()))

target_candidates = [
    c for c in df.columns
    if "target" in c.lower() or "price" in c.lower()
]

if target_candidates:
    t = target_candidates[0]
    target_mean = float(df[t].mean())
    target_std = float(df[t].std())
else:
    t = None
    target_mean = None
    target_std = None

missing_pct = float(df.isna().mean().mean()) * 100

print("Rows:", rows)
print("Columns:", cols)
print("Years:", years)
print("Target column:", t)
print("Target mean:", round(target_mean, 2) if target_mean else None)
print("Target std:", round(target_std, 2) if target_std else None)
print("Overall missing %:", round(missing_pct, 4))
