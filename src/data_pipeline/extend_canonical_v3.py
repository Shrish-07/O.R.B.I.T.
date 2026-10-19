"""Extend the canonical dataset with the 9 new _aw/_dom columns (Round 13, R1).

Writes data/canonical/modeling_dataset_canonical_v3.parquet = the existing
modeling_dataset_canonical_v2.parquet (566,956 rows, 64 columns, every
legacy column byte-identical) plus the Round-13 ideology columns merged
with the same semantics as rebuild_canonical.py:

  * mayoral _aw/_dom columns  <- data/processed/ideology_by_council_v3.parquet
        on (CounDist, election_year)
  * presidential _aw/_dom cols <- data/processed/ideology_pres_by_council_v3.parquet
        on (CounDist, election_year_pres)

All 9 new columns are blacklisted from models (config/feature_blacklist.yaml);
no model is retrained. A small provenance sidecar is written to
results/canonical_v3_provenance.json.

Run: python src/data_pipeline/extend_canonical_v3.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CANON_V2 = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet'
IDEO_V3 = ROOT / 'data' / 'processed' / 'ideology_by_council_v3.parquet'
IDEO_PRES_V3 = ROOT / 'data' / 'processed' / 'ideology_pres_by_council_v3.parquet'
OUT = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v3.parquet'
PROV = ROOT / 'results' / 'canonical_v3_provenance.json'

NEW_MAYORAL = ['dem_share_aw', 'dem_share_mayor_allcand_aw',
               'dem_share_vintage_aw', 'dem_share_mayor_allcand_vintage_aw',
               'dem_share_vintage_dom', 'dem_share_mayor_allcand_vintage_dom']
NEW_PRES = ['dem_share_pres_aw', 'dem_share_pres_vintage_aw',
            'dem_share_pres_vintage_dom']


def main():
    for p in [CANON_V2, IDEO_V3, IDEO_PRES_V3]:
        if not p.exists():
            raise SystemExit(f'MISSING INPUT: {p}')

    v2 = pd.read_parquet(CANON_V2)
    ideo = pd.read_parquet(IDEO_V3)
    ipres = pd.read_parquet(IDEO_PRES_V3)

    merged = v2.merge(
        ideo[['CounDist', 'election_year'] + NEW_MAYORAL],
        on=['CounDist', 'election_year'], how='left')
    ipr = ipres.rename(columns={'pres_election_year': 'election_year_pres'})
    merged = merged.merge(
        ipr[['CounDist', 'election_year_pres'] + NEW_PRES],
        on=['CounDist', 'election_year_pres'], how='left')

    # --- integrity gates: nothing pre-existing may change -------------------
    assert len(merged) == len(v2), 'row count changed'
    for c in v2.columns:
        assert merged[c].equals(v2[c]), f'legacy column changed: {c}'
    for c in NEW_MAYORAL + NEW_PRES:
        if c not in merged.columns:
            raise SystemExit(f'INTERNAL ERROR: expected new column {c} missing')

    merged.to_parquet(OUT, index=False)
    cov = {c: round(float(merged[c].notna().mean()), 4)
           for c in NEW_MAYORAL + NEW_PRES}
    print(f'Wrote {OUT.relative_to(ROOT)} rows={len(merged)} '
          f'cols={merged.shape[1]} (v2 was {v2.shape[1]})')
    print('new column coverage:', cov)

    PROV.parent.mkdir(parents=True, exist_ok=True)
    PROV.write_text(json.dumps({
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'base': CANON_V2.relative_to(ROOT).as_posix(),
        'rows': int(len(merged)),
        'columns_v2': int(v2.shape[1]),
        'columns_v3': int(merged.shape[1]),
        'new_columns': NEW_MAYORAL + NEW_PRES,
        'merge_semantics': ('mayoral on (CounDist, election_year); presidential '
                            'on (CounDist, election_year_pres); identical to '
                            'rebuild_canonical.py'),
        'new_columns_blacklisted_from_models': True,
        'coverage': cov,
    }, indent=2), encoding='utf-8')
    print(f'Wrote {PROV.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
