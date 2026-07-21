import pandas as pd

df = pd.read_parquet("experiments/predictions/lgbm_all_years_base_prediction_intervals.parquet")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Sample:")
print(df.head(3).to_string())
