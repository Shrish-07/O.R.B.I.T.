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

# FIX B2/B5: also refresh the two derived schema records other code/docs
# treat as authoritative. canonical_schema.json (read by
# src/verification/verify_canonical_dataset_and_split.py) and
# docs/feature_manifest.json were hand-frozen 8/31-era artifacts
# (stale at 514,618 rows). From now on every build regenerates them.
import json as _json
schema_json = {"shape": [int(df.shape[0]), int(df.shape[1])], "columns": list(df.columns)}
Path("models/artifacts").mkdir(parents=True, exist_ok=True)
Path("models/artifacts/canonical_schema.json").write_text(
    _json.dumps(schema_json, indent=2) + "\n", encoding="utf-8")
manifest = {"n_rows": int(df.shape[0]), "n_columns": int(df.shape[1]),
            "columns": sorted(df.columns), "target": "target_log_price"}
Path("docs/feature_manifest.json").write_text(
    _json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("Wrote models/artifacts/canonical_schema.json and docs/feature_manifest.json")

print("✅ Schema frozen")
print("Shape:", df.shape)
