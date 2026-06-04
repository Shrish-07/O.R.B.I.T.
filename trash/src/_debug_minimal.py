# moved to trash: debug helper

from pathlib import Path
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/processed/modeling_dataset_with_target.parquet")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

# -----------------------------
# Load (quiet)
# -----------------------------
df = pd.read_parquet(DATA_PATH)

# -----------------------------
# Output (scalar only)
# -----------------------------
print("START")
print("IMPORTED PANDAS")
print("READ PARQUET OK")
print("ROWS:", len(df))
print("DONE")
