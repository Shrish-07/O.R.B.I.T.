import pandas as pd
from pathlib import Path

Path("docs").mkdir(exist_ok=True)

# Freeze the canonical v2 modeling dataset (forensic canonical schema)
CANON_V2 = Path('data/canonical/modeling_dataset_canonical_v2.parquet')
CANON_FALLBACK = Path('data/canonical/modeling_dataset_canonical.parquet')

if CANON_V2.exists():
    df = pd.read_parquet(CANON_V2)
elif CANON_FALLBACK.exists():
    df = pd.read_parquet(CANON_FALLBACK)
else:
    # The old processed fallback (data/processed/modeling_dataset.parquet)
    # was removed; there is no valid fallback anymore. Fail loudly.
    raise SystemExit(
        f"❌ No canonical dataset found. Tried:\n  {CANON_V2}\n  {CANON_FALLBACK}\n"
        "Reproduce/copy the canonical dataset before freezing a schema."
    )

schema = (
    df.dtypes
    .astype(str)
    .reset_index()
    .rename(columns={"index": "column", 0: "dtype"})
)

schema.to_csv("docs/modeling_schema.csv", index=False)

print("✅ Schema frozen")
print("Shape:", df.shape)
