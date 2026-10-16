"""Round 5 fact-check Part 1a: load canonical + ideology files, shared loader.

Round 11: adds the machine-countable status recorder (record_status) that
EVERY verifier now calls at the end, plus the check_val helper. The summary
stage (src/ops/verification_summary.py) aggregates the per-verifier status
files into results/verification_summary.json and exits non-zero if any
verifier reports a mismatch or fails to report.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATUS_DIR = ROOT / 'results' / 'verification'

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

# ---------------------------------------------------------------------------
# Round 11: machine-countable verification status
# ---------------------------------------------------------------------------

def check_val(name, expected, actual, tol=0.0):
    """One check record. match is exact (<= tol on |actual - expected|).
    Non-numeric values (bools/strings/None) compare by equality."""
    if expected is None or actual is None:
        match = expected is actual
    elif isinstance(expected, bool) or isinstance(actual, bool):
        match = bool(expected) == bool(actual)
    else:
        try:
            match = abs(float(actual) - float(expected)) <= tol
        except (TypeError, ValueError):
            match = expected == actual
    return {'check': name, 'expected': expected, 'actual': actual,
            'tolerance': tol, 'match': bool(match)}


def record_status(verifier, checks, note=None):
    """Write results/verification/<verifier>.json: {verifier, generated_utc,
    n_checks, n_mismatches, mismatches}. The mismatches list carries the full
    failing check records (check/expected/actual/tolerance). Returns payload."""
    checks = [dict(c) for c in checks]
    for c in checks:
        c['match'] = bool(c.get('match'))
    mismatches = [c for c in checks if not c['match']]
    payload = {'verifier': verifier,
               'generated_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
               'n_checks': len(checks),
               'n_mismatches': len(mismatches),
               'mismatches': mismatches}
    if note:
        payload['note'] = note
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    out = STATUS_DIR / f'{verifier}.json'
    out.write_text(json.dumps(payload, indent=2, default=str),
                   encoding='utf-8')
    print(f'[status] {verifier}: {len(checks)} checks, '
          f'{len(mismatches)} mismatch(es) -> {out.relative_to(ROOT)}'
          + ('' if not mismatches else '  *** MISMATCHES -- see file'))
    return payload

if __name__ == '__main__':
    df, name = load_canonical()
    print('canonical rows', len(df), 'cols', df.shape[1], 'file', name)
