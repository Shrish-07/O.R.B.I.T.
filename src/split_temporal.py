# src/split_temporal.py

"""
Creates explicit temporal splits for ORBIT v2 experiments.

Splits produced:
1. all_years_train / all_years_test
   - Train: all years < TEST_YEAR
   - Test:  TEST_YEAR only
   - Purpose: global predictive performance

2. year2017_train / year2017_test
   - Train: 2017 only
   - Test:  2018 only
   - Purpose: controlled-year + political attribution

NOTE:
- SALE_YEAR is split-only and must never persist into modeling features
- These files are authoritative for ALL training runs
"""

import pandas as pd
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
INPUT_PATH = "data/canonical/modeling_dataset_canonical_v2.parquet"
SPLIT_DIR = Path("data/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TEST_YEAR = 2018
CONTROL_YEAR = 2017
TARGET = "target_log_price"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet(INPUT_PATH)

# -----------------------------
# Resolve SALE DATE
# -----------------------------
if "SALE DATE" in df.columns:
    sale_date_col = "SALE DATE"
elif "SALE_DATE" in df.columns:
    sale_date_col = "SALE_DATE"
else:
    raise KeyError("No SALE DATE column found")

df[sale_date_col] = pd.to_datetime(df[sale_date_col], errors="coerce")

before = len(df)
df = df.dropna(subset=[sale_date_col])
after = len(df)

print(f"Dropped {before - after} rows with invalid SALE DATE")

# -----------------------------
# Authoritative year (SPLIT ONLY)
# -----------------------------
df["SALE_YEAR"] = df[sale_date_col].dt.year.astype(int)

# =============================
# SPLIT 1 — ALL YEARS MODEL
# =============================
all_train = df[df["SALE_YEAR"] < TEST_YEAR].copy()
all_test  = df[df["SALE_YEAR"] == TEST_YEAR].copy()

# -----------------------------
# Safety checks
# -----------------------------
assert TARGET in all_train.columns
assert TARGET in all_test.columns
assert all_test["SALE_YEAR"].nunique() == 1
assert len(all_train) > 0, "All-years train split is empty"
assert len(all_test) > 0, "All-years test split is empty"

# -----------------------------
# Save
# -----------------------------
all_train.to_parquet(SPLIT_DIR / "all_years_train.parquet", index=False)
all_test.to_parquet(SPLIT_DIR / "all_years_test.parquet", index=False)

print("\n✅ All-years split complete")
print("Train shape:", all_train.shape)
print("Test shape: ", all_test.shape)

# =============================
# SPLIT 2 — 2017 CONTROLLED MODEL
# =============================
yr_train = df[df["SALE_YEAR"] == CONTROL_YEAR].copy()
yr_test  = df[df["SALE_YEAR"] == TEST_YEAR].copy()

# -----------------------------
# Safety checks
# -----------------------------
assert TARGET in yr_train.columns
assert TARGET in yr_test.columns
assert yr_train["SALE_YEAR"].nunique() == 1
assert yr_test["SALE_YEAR"].nunique() == 1
assert len(yr_train) > 0, "2017 train split is empty"
assert len(yr_test) > 0, "2018 test split is empty"

# -----------------------------
# Save
# -----------------------------
yr_train.to_parquet(SPLIT_DIR / "year2017_train.parquet", index=False)
yr_test.to_parquet(SPLIT_DIR / "year2017_test.parquet", index=False)

print("\n✅ 2017-only controlled split complete")
print("Train shape:", yr_train.shape)
print("Test shape: ", yr_test.shape)

print("\n🎯 Splits written to data/splits/")
