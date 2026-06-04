import pandas as pd
from pathlib import Path

DATA_PROCESSED = Path("data/processed")

def main():
    ideo = pd.read_parquet(DATA_PROCESSED / "ed_ideology.parquet")
    cw = pd.read_parquet(DATA_PROCESSED / "ed_to_council_crosswalk.parquet")

    # Build ED join key
    ideo["ElectDist"] = (ideo["AD"] * 1000 + ideo["ED"]).astype(float)
    cw["ElectDist"] = cw["ElectDist"].astype(float)

    # MANY-TO-MANY merge is expected here
    merged = ideo.merge(cw, on="ElectDist", how="inner")

    # Area-weighted ideology
    merged["weighted_ideo"] = merged["ideology_score"] * merged["area"]

    district = (
        merged
        .groupby(["year", "CounDist"], as_index=False)
        .agg(
            district_ideology=("weighted_ideo", "sum"),
            total_area=("area", "sum")
        )
    )

    district["district_ideology"] = (
        district["district_ideology"] / district["total_area"]
    )

    district = district.rename(
        columns={"CounDist": "COUNCIL_DISTRICT"}
    )

    district = district[
        ["year", "COUNCIL_DISTRICT", "district_ideology"]
    ]

    district["year"] = district["year"].astype(int)
    district["COUNCIL_DISTRICT"] = district["COUNCIL_DISTRICT"].astype(int)
    district["district_ideology"] = pd.to_numeric(
        district["district_ideology"], errors="coerce"
    )

    district.to_parquet(
        DATA_PROCESSED / "district_ideology.parquet",
        index=False
    )

    print("✅ district_ideology.parquet saved")
    print("Rows:", len(district))
    print("Years:", sorted(district["year"].unique()))

if __name__ == "__main__":
    main()
