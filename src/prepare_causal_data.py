import pandas as pd
from pathlib import Path

DATA_INTERMEDIATE = Path("data/intermediate")
DATA_PROCESSED = Path("data/processed")

def main():
    df = pd.read_parquet(DATA_INTERMEDIATE / "sales_pluto_ideology.parquet")

    df = df.dropna(subset=["district_ideology"])
    df = df[df["district_ideology"].between(0, 1)]

    keep_cols = [
        "sale_price",
        "gross_sqft",
        "land_sqft",
        "year",
        "CounDist",
        "district_ideology"
    ]

    df = df[keep_cols]

    df.to_parquet(DATA_PROCESSED / "causal_dataset.parquet", index=False)
    print("✅ causal_dataset.parquet written")

if __name__ == "__main__":
    main()
