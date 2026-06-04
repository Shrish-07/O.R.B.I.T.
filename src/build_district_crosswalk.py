import pandas as pd

cw = pd.read_parquet("data/processed/ed_to_council_crosswalk.parquet")

cw["ElectDist"] = cw["ElectDist"].astype(int)

assert cw["ElectDist"].between(23001, 87055).all()
assert cw["CounDist"].between(1, 51).all()

print("✅ ED → Council crosswalk valid")
print(cw.shape)
