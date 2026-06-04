# src/normalize_ed_results.py

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/election_results")
OUT = Path("data/processed/ed_results_clean.parquet")

EXPECTED = {
    "AD",
    "ED",
    "Office_Position_Title",
    "Unit_Name",
    "Tally",
}

dfs = []

for path in sorted(RAW_DIR.glob("*.csv")):
    year_tokens = [x for x in path.stem.split("_") if x.isdigit()]
    if not year_tokens:
        raise ValueError(f"Could not infer year from {path.name}")

    year = int(year_tokens[0])
    print(f"Loading {path.name} ({year})")

    df = pd.read_csv(path)

    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    if not EXPECTED.issubset(df.columns):
        raise ValueError(f"{path.name} missing required columns")

    df = df[df["AD"] != "AD"]

    df["AD"] = pd.to_numeric(df["AD"], errors="coerce")
    df["ED"] = pd.to_numeric(df["ED"], errors="coerce")
    df["Tally"] = pd.to_numeric(df["Tally"], errors="coerce")

    df = df.dropna(subset=["AD", "ED", "Tally"])

    df["ED_CODE"] = df["AD"] * 1000 + df["ED"]
    df["year"] = year

    dfs.append(df)

final = pd.concat(dfs, ignore_index=True)
final.to_parquet(OUT, index=False)

print("✅ ed_results_clean.parquet saved")
print("Rows:", len(final))
print("Years:", sorted(final["year"].unique()))
