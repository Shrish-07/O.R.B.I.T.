import pandas as pd
from pathlib import Path

RESULTS_CSV = Path("experiments/results/model_comparison_raw.csv")
OUT_CSV = Path("figures/model_performance_table.csv")

# Read raw comparison
df = pd.read_csv(RESULTS_CSV)

# Sort so it’s consistent
df = df.sort_values(["model", "scope", "mode"])

# Select desired columns
table_df = df[[
    "model", "scope", "mode", "r2", "mae", "features_used"
]]

# Round numeric columns for readability
table_df["r2"] = table_df["r2"].round(4)
table_df["mae"] = table_df["mae"].round(4)

# Save
OUT_CSV.parent.mkdir(exist_ok=True)
table_df.to_csv(OUT_CSV, index=False)

print(f"📊 Model performance table saved to {OUT_CSV}")
