# src/merge_sales_pluto_ideology.py

import pandas as pd
from pathlib import Path


def main():
    sales_path = Path("data/intermediate/sales_pluto.parquet")
    ideology_path = Path("data/processed/district_ideology.parquet")
    out_path = Path("data/processed/sales_pluto_ideology.parquet")

    sales = pd.read_parquet(sales_path)
    ideology = pd.read_parquet(ideology_path)

    sale_date_col = next(
        c for c in sales.columns if c.lower().replace("_", " ") == "sale date"
    )

    sales["SALE_YEAR"] = pd.to_datetime(
        sales[sale_date_col], errors="coerce"
    ).dt.year

    council_col = next(
        c for c in sales.columns
        if "council" in c.lower() and "district" in c.lower()
    )

    sales["COUNCIL_DISTRICT"] = pd.to_numeric(
        sales[council_col], errors="coerce"
    )

    sales = sales.dropna(subset=["SALE_YEAR", "COUNCIL_DISTRICT"])
    sales["SALE_YEAR"] = sales["SALE_YEAR"].astype(int)
    sales["COUNCIL_DISTRICT"] = sales["COUNCIL_DISTRICT"].astype(int)

    ideology = ideology.rename(columns={"year": "SALE_YEAR"})
    ideology["SALE_YEAR"] = ideology["SALE_YEAR"].astype(int)
    ideology["COUNCIL_DISTRICT"] = ideology["COUNCIL_DISTRICT"].astype(int)

    merged = sales.merge(
        ideology[["COUNCIL_DISTRICT", "SALE_YEAR", "district_ideology"]],
        on=["COUNCIL_DISTRICT", "SALE_YEAR"],
        how="left",
        validate="many_to_one"
    )

    ideology_years = set(ideology["SALE_YEAR"].unique())
    invalid_mask = ~merged["SALE_YEAR"].isin(ideology_years)
    merged.loc[invalid_mask, "district_ideology"] = pd.NA

    if "district_ideology" not in merged.columns:
        raise KeyError("district_ideology column missing after merge")

    merged.to_parquet(out_path, index=False)

    total = len(merged)
    covered = merged["district_ideology"].notna().sum()
    coverage = covered / total if total > 0 else 0

    print("✅ sales_pluto_ideology.parquet saved")
    print("Rows:", total)
    print("Ideology coverage:", round(coverage, 3))
    print("Ideology years:", sorted(ideology_years))


if __name__ == "__main__":
    main()
