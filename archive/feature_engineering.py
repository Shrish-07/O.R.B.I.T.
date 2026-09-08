# src/feature_engineering.py

from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path("data/processed/modeling_dataset_with_target.parquet")
OUTPUT_PATH = Path("data/processed/modeling_dataset_fe.parquet")

NUM_COLS = [
    "LAND SQUARE FEET",
    "GROSS SQUARE FEET",
    "TOTAL UNITS",
    "YEAR BUILT"
]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    # -----------------------------
    # Load (quiet)
    # -----------------------------
    df = pd.read_parquet(INPUT_PATH)

    # -----------------------------
    # Coerce core numeric columns
    # -----------------------------
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------
    # Core numeric transforms (safe)
    # -----------------------------
    for col in ["LAND SQUARE FEET", "GROSS SQUARE FEET", "TOTAL UNITS"]:
        if col in df.columns:
            new_col = f"log_{col.lower().replace(' ', '_')}"
            df[new_col] = np.log1p(df[col].fillna(0))

    # -----------------------------
    # Building age (safe)
    # -----------------------------
    if "SALE DATE" in df.columns and "YEAR BUILT" in df.columns:
        sale_year = pd.to_datetime(df["SALE DATE"], errors="coerce").dt.year
        df["building_age"] = (sale_year - df["YEAR BUILT"]).clip(lower=0)

    # -----------------------------
    # FAR (safe)
    # -----------------------------
    if "GROSS SQUARE FEET" in df.columns and "LAND SQUARE FEET" in df.columns:
        df["far"] = df["GROSS SQUARE FEET"] / df["LAND SQUARE FEET"]
        df["far"] = df["far"].replace([np.inf, -np.inf], np.nan)

    # -----------------------------
    # Guard: never use target for features
    # -----------------------------
    if "target_log_price" not in df.columns:
        raise ValueError("Missing target_log_price column")

    if any("mean_price" in c.lower() for c in df.columns):
        raise ValueError("Leakage column detected: mean_price")

    # -----------------------------
    # Guard: ideology must exist + numeric
    # -----------------------------
    if "district_ideology" in df.columns:
        df["district_ideology"] = pd.to_numeric(
            df["district_ideology"], errors="coerce"
        )

    # -----------------------------
    # Save
    # -----------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    # Scalar-only prints (no repr risk)
    print("FEATURE_ENGINEERING_OK")
    print("ROWS:", len(df))
    print("COLS:", len(df.columns))
    print("OUTPUT:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
