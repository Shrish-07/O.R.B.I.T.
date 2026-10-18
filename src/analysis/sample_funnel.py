"""Sample funnel: raw sales -> canonical -> restricted analysis sample (Task D).

Re-executes the exact filter sequence of src/data_pipeline/rebuild_canonical.py
on the raw sales CSV, counting rows at every step, then continues the funnel
through the analysis-side filters down to the restricted analysis sample
used by the v2 regression suite:

    restricted analysis sample := canonical rows that additionally pass
        (1) non-missing outcome + ideology measure + core controls,
        (2) is_resid_tc12 (tax class 1 or 2 at time of sale),
        (3) price_ge_100k,
        (4) NOT multi_parcel_flag (same BOROUGH+BLOCK+SALE DATE+SALE PRICE
            appearing more than once).

Also reports the flag shares on the canonical rows and the
BUILDING CLASS CATEGORY whitespace normalization effect (distinct raw
categories vs distinct normalized categories).

Outputs: results/sample_funnel.json, results/sample_funnel.csv

Run: python src/analysis/sample_funnel.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SALES_GLOB = 'NYC_Citywide_Annualized_Calendar_Sales_Update_*.csv'
CANON = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet'
OUT_JSON = ROOT / 'results' / 'sample_funnel.json'
OUT_CSV = ROOT / 'results' / 'sample_funnel.csv'

CORE_CONTROLS = ['yearbuilt', 'numfloors', 'landuse', 'commfar']
MEASURES = ['dem_share', 'dem_share_mayor_allcand', 'dem_share_pres',
            'dem_share_vintage', 'dem_share_mayor_allcand_vintage',
            'dem_share_pres_vintage']


def clean_price(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = re.sub(r"[^0-9.\-]", "", str(s))
    try:
        return float(s) if s not in ('', 'nan') else np.nan
    except Exception:
        return np.nan


def resolve_single_file(directory, pattern):
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f'Expected exactly one {pattern} in {directory}, '
                         f'found {len(matches)}')
    return matches[0]


def main():
    steps = []

    def step(name, desc, rows, prev):
        steps.append({'step': name, 'description': desc, 'rows': int(rows),
                      'rows_lost_from_previous': int(prev - rows)})
        print(f'  {name:34s} {rows:>9,}  (-{prev - rows:,})')

    sales_file = resolve_single_file(ROOT / 'data' / 'raw' / 'sales', SALES_GLOB)
    sales = pd.read_csv(sales_file, low_memory=False)
    n = len(sales)
    step('raw_sales_rows', 'Raw citywide annualized sales CSV', n, n)
    sales['sale_price'] = sales['SALE PRICE'].map(clean_price)
    sales = sales[sales['sale_price'].notna()].copy()
    step('valid_sale_price', 'SALE PRICE parses to a number', len(sales), n)
    n = len(sales)
    sales = sales[sales['sale_price'] > 10000].copy()
    step('price_gt_10k', 'sale_price > $10,000 (rebuild_canonical filter)',
         len(sales), n)
    n = len(sales)
    step('after_pluto_merge', 'left-join PLUTO on BBL (no row loss by design)',
         n, n)
    step('after_ideology_merge', 'left-join ideology on (CounDist, '
         'election_year) (no row loss by design)', n, n)
    dedup_n = len(sales.drop_duplicates())
    step('canonical_rows', 'exact-duplicate rows dropped (rebuild_canonical '
         'final step; = canonical parquet rows)', dedup_n, n)

    df = pd.read_parquet(CANON)
    if len(df) != dedup_n:
        raise SystemExit(
            f'Funnel mismatch: recomputed canonical {dedup_n:,} vs parquet '
            f'{len(df):,} — rebuild_canonical changed?')
    n = len(df)

    flags = {c: int(df[c].sum()) for c in
             ['sample_pre2024', 'is_resid_tc12', 'price_ge_100k',
              'multi_parcel_flag']}
    raw_bcc_n = int(df['BUILDING CLASS CATEGORY'].astype(str).nunique())
    norm_bcc_n = int(df['building_class_category'].astype(str).nunique())

    base_ok = df['target_log_price'].notna() & df['CounDist'].notna() \
        & df['sale_year'].notna()
    for c in CORE_CONTROLS:
        base_ok &= df[c].notna()
    step('analysis_base_sample', 'non-missing outcome + core controls '
         '(yearbuilt, numfloors, landuse, commfar) + district/year',
         int(base_ok.sum()), n)
    n_base = int(base_ok.sum())

    restricted_mask = base_ok & df['is_resid_tc12'] & df['price_ge_100k'] \
        & (~df['multi_parcel_flag'])
    step('restricted_analysis_sample', 'base + is_resid_tc12 + price_ge_100k '
         '+ NOT multi_parcel_flag', int(restricted_mask.sum()), n_base)

    per_measure = {}
    for m in MEASURES:
        per_measure[m] = {
            'nonmissing_canonical': int(df[m].notna().sum()),
            'full_analysis_sample': int((base_ok & df[m].notna()).sum()),
            'restricted_analysis_sample':
                int((restricted_mask & df[m].notna()).sum()),
        }

    result = {
        'funnel': steps,
        'flags_on_canonical': flags,
        'flag_shares_on_canonical': {k: round(v / len(df), 6)
                                     for k, v in flags.items()},
        'building_class_category': {
            'distinct_raw': raw_bcc_n,
            'distinct_normalized': norm_bcc_n,
            'note': ('Whitespace normalization collapses e.g. '
                     '"01  TWO-FAMILY" style double-space variants; the raw '
                     'column is untouched, the normalized column is '
                     'building_class_category.')},
        'restricted_analysis_sample_definition': (
            'canonical rows with non-missing outcome + core controls, '
            'AND is_resid_tc12, AND price_ge_100k, AND NOT '
            'multi_parcel_flag'),
        'per_measure_sample_sizes': per_measure,
        'canonical_rows': int(len(df)),
    }
    import json
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(steps).to_csv(OUT_CSV, index=False)
    print(f'Wrote {OUT_JSON.relative_to(ROOT)}')
    print(f'Wrote {OUT_CSV.relative_to(ROOT)}')


if __name__ == '__main__':
    main()

