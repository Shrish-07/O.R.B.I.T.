# src/compute_ideology.py

import pandas as pd

df = pd.read_parquet("data/processed/ed_results_clean.parquet")

df = df[df["Office_Position_Title"] == "Mayor"].copy()

votes = (
    df.groupby(["year", "AD", "ED", "ED_CODE", "Unit_Name"])["Tally"]
    .sum()
    .unstack(fill_value=0)
)

votes["dem"] = votes.filter(like="Democratic").sum(axis=1)
votes["rep"] = votes.filter(like="Republican").sum(axis=1)

votes = votes.reset_index()

votes["ideology_score"] = votes["dem"] / (votes["dem"] + votes["rep"])
votes = votes.dropna(subset=["ideology_score"])

votes.to_parquet("data/processed/ed_ideology.parquet", index=False)

print("✅ ed_ideology.parquet saved")
print("Rows:", len(votes))
print("Years:", sorted(votes["year"].unique()))
