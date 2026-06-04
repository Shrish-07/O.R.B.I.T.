import pandas as pd
from pathlib import Path

print("\n🔍 ORBIT V2 — PIPELINE SANITY CHECK\n")

DATASET_PATH = Path("data/processed/modeling_dataset_fe_imputed.parquet")

assert DATASET_PATH.exists(), "❌ modeling_dataset_fe_imputed.parquet not found"

df = pd.read_parquet(DATASET_PATH)

# -----------------------------
# Required columns
# -----------------------------
required_cols = [
    "target_log_price",
    "SALE_YEAR",
    "LAND SQUARE FEET",
    "GROSS SQUARE FEET",
    "district_ideology",
]

for c in required_cols:
    assert c in df.columns, f"❌ Missing required column: {c}"

# -----------------------------
# Coerce SALE_YEAR to numeric
# -----------------------------
df["SALE_YEAR"] = pd.to_numeric(df["SALE_YEAR"], errors="coerce")

# -----------------------------
# Target integrity
# -----------------------------
assert df["target_log_price"].notna().mean() > 0.999, "❌ Target has missing values"

# -----------------------------
# Sale year sanity (data-driven)
# -----------------------------
valid_years = df["SALE_YEAR"].dropna()

min_year = int(valid_years.min())
max_year = int(valid_years.max())
coverage = valid_years.between(2003, 2025).mean()

print(f"SALE_YEAR range: {min_year} → {max_year}")
print(f"SALE_YEAR in [2003, 2025]: {coverage:.4f}")

# Relaxed but meaningful guardrail
assert coverage > 0.95, "❌ Too many rows with implausible SALE_YEAR values"

# -----------------------------
# Ideology coverage
# -----------------------------
ideo_coverage = df["district_ideology"].notna().mean()

print(f"Ideology coverage: {ideo_coverage:.4f}")

assert ideo_coverage > 0.70, "❌ Ideology merge coverage too low to justify political model"

# -----------------------------
# Missingness audit
# -----------------------------
missing_rate = df.isna().mean().mean()

print(f"Overall missing %: {missing_rate:.4%}")

assert missing_rate < 0.25, "❌ Excessive missingness after feature engineering"

print("\n✅ Pipeline sanity check PASSED\n")
