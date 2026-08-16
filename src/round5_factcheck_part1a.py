"""Round 5 fact-check Part 1a: load canonical + ideology files, shared loader."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_canonical():
    paths = [
        ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet',
        ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical.parquet',
    ]
    p = next(x for x in paths if x.exists())
    return pd.read_parquet(p), p.name

def parse_feature_names(txt_path):
    txt = Path(txt_path).read_text(errors='ignore')
    line = next((ln for ln in txt.splitlines() if ln.startswith('feature_names=')), None)
    names = line.split('=', 1)[1].split() if line else []
    mfi = next((ln for ln in txt.splitlines() if ln.startswith('max_feature_idx=')), None)
    mfi_val = int(mfi.split('=', 1)[1]) if mfi else None
    return names, mfi_val

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))

if __name__ == '__main__':
    df, name = load_canonical()
    print('canonical rows', len(df), 'cols', df.shape[1], 'file', name)
