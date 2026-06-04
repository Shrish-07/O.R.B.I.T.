# src/build_sales_pluto.py

import pandas as pd
from pathlib import Path

RAW_PLUTO = Path("data/raw/pluto/Primary_Land_Use_Tax_Lot_Output_(PLUTO)_20260105.csv")
RAW_SALES = Path("data/raw/sales/NYC_Citywide_Annualized_Calendar_Sales_Update_20260105.csv")
OUT_PATH = Path("data/intermediate/sales_pluto.parquet")


def main():
    pluto = pd.read_csv(RAW_PLUTO, low_memory=False)
    sales = pd.read_csv(RAW_SALES, low_memory=False)

    if "BBL" not in pluto.columns or "BBL" not in sales.columns:
        raise RuntimeError("BBL column missing from PLUTO or sales data")

    pluto["BBL"] = pd.to_numeric(pluto["BBL"], errors="coerce")
    sales["BBL"] = pd.to_numeric(sales["BBL"], errors="coerce")

    pluto = pluto.dropna(subset=["BBL"])
    sales = sales.dropna(subset=["BBL"])

    pluto = pluto.drop_duplicates(subset=["BBL"])

    df = sales.merge(
        pluto,
        on="BBL",
        how="left",
        validate="many_to_one"
    )

    df.to_parquet(OUT_PATH, index=False)

    print("✅ sales_pluto.parquet saved")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))


if __name__ == "__main__":
    main()
