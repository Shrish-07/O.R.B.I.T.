import pandas as pd
from pathlib import Path

print("\n🔍 ORBIT V2 — PIPELINE SANITY CHECK\n")

DATASET_PATH = Path("data/canonical/modeling_dataset_canonical_v2.parquet")

assert DATASET_PATH.exists(), "❌ modeling_dataset_canonical_v2.parquet not found"

df = pd.read_parquet(DATASET_PATH)

# -----------------------------
# Required columns
# NOTE: canonical v2 schema uses lowercase `sale_year` (was SALE_YEAR
# in the old processed pipeline) and `dem_share` as the ideology feature
# (was district_ideology). Map to canonical column names.
# -----------------------------
required_cols = [
    "target_log_price",
    "sale_year",
    "LAND SQUARE FEET",
    "GROSS SQUARE FEET",
    "dem_share",
]

for c in required_cols:
    assert c in df.columns, f"❌ Missing required column: {c}"

# -----------------------------
# Coerce sale_year to numeric
# -----------------------------
df["sale_year"] = pd.to_numeric(df["sale_year"], errors="coerce")

# -----------------------------
# Target integrity
# -----------------------------
assert df["target_log_price"].notna().mean() > 0.999, "❌ Target has missing values"

# -----------------------------
# Sale year sanity (data-driven)
# -----------------------------
valid_years = df["sale_year"].dropna()

min_year = int(valid_years.min())
max_year = int(valid_years.max())
coverage = valid_years.between(2003, 2025).mean()

print(f"sale_year range: {min_year} → {max_year}")
print(f"sale_year in [2003, 2025]: {coverage:.4f}")

# Relaxed but meaningful guardrail
assert coverage > 0.95, "❌ Too many rows with implausible sale_year values"

# -----------------------------
# Ideology coverage
# -----------------------------
ideo_coverage = df["dem_share"].notna().mean()

print(f"Ideology coverage (dem_share): {ideo_coverage:.4f}")

assert ideo_coverage > 0.70, "❌ Ideology merge coverage too low to justify political model"

# -----------------------------
# Missingness audit
# -----------------------------
missing_rate = df.isna().mean().mean()

print(f"Overall missing %: {missing_rate:.4%}")

assert missing_rate < 0.25, "❌ Excessive missingness after feature engineering"

print("\n✅ Pipeline sanity check PASSED\n")
