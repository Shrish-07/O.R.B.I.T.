"""Rebuild canonical modeling dataset v2 with ideology and selected PLUTO fields.

Saves: data/canonical/modeling_dataset_canonical_v2.parquet

Usage: python src/data_pipeline/rebuild_canonical.py [--output PATH]

Raw-data location is glob-based and self-enforcing. This script discovers the
single current sales CSV matching
``data/raw/sales/NYC_Citywide_Annualized_Calendar_Sales_Update_*.csv`` and the
single current PLUTO CSV matching
``data/raw/pluto/Primary_Land_Use_Tax_Lot_Output_(PLUTO)_*.csv``. It refuses to
run if either folder holds zero or more than one matching file, so a stale
duplicate can never be silently preferred over the current one. Every run
records exactly which raw files produced the build in data/DATA_MANIFEST.json.
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SALES_DIR = ROOT / 'data' / 'raw' / 'sales'
SALES_GLOB = 'NYC_Citywide_Annualized_Calendar_Sales_Update_*.csv'
PLUTO_DIR = ROOT / 'data' / 'raw' / 'pluto'
PLUTO_GLOB = 'Primary_Land_Use_Tax_Lot_Output_(PLUTO)_*.csv'
IDEO = ROOT / 'data' / 'processed' / 'ideology_by_council.parquet'
IDEO_V2 = ROOT / 'data' / 'processed' / 'ideology_by_council_v2.parquet'
IDEO_PRES = ROOT / 'data' / 'processed' / 'ideology_pres_by_council.parquet'
OUT_DIR = ROOT / 'data' / 'canonical'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUT = OUT_DIR / 'modeling_dataset_canonical_v2.parquet'
MANIFEST = ROOT / 'data' / 'DATA_MANIFEST.json'

# sale_year -> presidential election providing dem_share_pres (explicit,
# documented mapping from the task brief). Out-of-range years fall back to
# the nearest covered election and are recorded in the manifest note below.
PRES_MAP = {
    2016: 2016, 2017: 2016,
    2018: 2020, 2019: 2020, 2020: 2020, 2021: 2020,
    2022: 2024, 2023: 2024, 2024: 2024, 2025: 2024,
}
PRES_MAP_NOTE = (
    'dem_share_pres mapping (sale_year -> presidential election): '
    '{2016:2016, 2017:2016, 2018:2020, 2019:2020, 2020:2020, 2021:2020, '
    '2022:2024, 2023:2024, 2024:2024, 2025:2024}. sale_year < 2016 falls '
    'back to 2016 and sale_year > 2025 to 2024 (nearest covered election); '
    'the canonical build contains sale years 2016-2025 only.'
)

BAN_COLUMNS = {'assesstot','exempttot','assessland','appbbl','taxmap','plutomapid'}


def resolve_single_file(directory, pattern, label):
    """Return the single file matching ``pattern`` in ``directory``, or exit with
    a clear message if there are zero or multiple matches (never guess)."""
    directory = Path(directory)
    matches = sorted(directory.glob(pattern))
    if len(matches) == 0:
        raise SystemExit(
            f"[build] ERROR: no {label} found. Looked for pattern "
            f"'{pattern}' in '{directory}'. Place exactly one matching file "
            f"there and re-run."
        )
    if len(matches) > 1:
        listing = '\n    '.join(str(m.name) for m in matches)
        raise SystemExit(
            f"[build] ERROR: multiple {label} files found in '{directory}'. "
            f"Refusing to guess which is current. Archive or remove all but "
            f"one, then re-run.\n"
            f"  Matched files:\n    {listing}"
        )
    return matches[0]

def find_col(cols, keywords):
    keys = [k.lower() for k in keywords]
    for c in cols:
        cl = c.lower()
        if all(k in cl for k in keys):
            return c
    return None

def clean_price(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int,float)):
        return float(s)
    s = str(s)
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s) if s not in ('','nan') else np.nan
    except Exception:
        return np.nan

def map_election_year(y):
    if 2016 <= y <= 2018:
        return 2017
    if 2019 <= y <= 2023:
        return 2021
    if y == 2024:
        return 2025
    if y < 2016:
        return 2017
    return 2025

def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the canonical modeling dataset v2 from glob-discovered raw files."
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUT),
        help=f"Output parquet path (default: {DEFAULT_OUT}).",
    )
    args = parser.parse_args()
    OUT = Path(args.output)

    sales_file = resolve_single_file(SALES_DIR, SALES_GLOB, 'sales data file')
    pluto_file = resolve_single_file(PLUTO_DIR, PLUTO_GLOB, 'PLUTO file')

    # Prominent, plain-text log of exactly which raw file produced this build.
    print(f'[build] SALES FILE = {sales_file.name}')
    print(f'[build] PLUTO FILE = {pluto_file.name}')

    print('Loading sales:', sales_file)
    sales = pd.read_csv(sales_file, low_memory=False)
    raw_sales_rows = int(len(sales))
    # identify columns
    sale_price_col = find_col(sales.columns, ['sale','price']) or 'SALE PRICE'
    sale_date_col = find_col(sales.columns, ['sale','date']) or 'SALE DATE'
    council_col = find_col(sales.columns, ['council']) or 'Council District'
    bbl_col = find_col(sales.columns, ['bbl']) or 'BBL'

    # clean price
    sales['sale_price'] = sales[sale_price_col].map(clean_price)
    sales['sale_date'] = pd.to_datetime(sales[sale_date_col], errors='coerce')
    sales = sales[sales['sale_price'].notna()].copy()
    sales = sales[sales['sale_price'] > 10000].copy()
    sales['target_log_price'] = np.log(sales['sale_price'])
    # FIX A2 (sales side): LAND/GROSS SQUARE FEET also ship as strings; coerce
    # so they are numeric features instead of being silently excluded.
    for _sc in ['LAND SQUARE FEET', 'GROSS SQUARE FEET']:
        if _sc in sales.columns:
            sales[_sc] = pd.to_numeric(sales[_sc], errors='coerce')

    # normalize BBL for join
    sales['BBL_join'] = pd.to_numeric(sales[bbl_col], errors='coerce')

    # read PLUTO and select fields
    pluto_keep = ['BBL','zonedist1','yearbuilt','numfloors','bldgarea','lotarea','unitsres','unitstotal','residfar','commfar','facilfar','landuse','bldgclass']
    print('Loading PLUTO:', pluto_file)
    pluto = pd.read_csv(pluto_file, usecols=lambda c: c in pluto_keep, low_memory=False)
    # FIX A2 (dtype bug): PLUTO ships these structural numerics as STRINGS in
    # the CSV; without coercion they land in the canonical parquet as object
    # dtype and every trainer's select_dtypes(include='number') silently
    # excludes them from the feature set.
    for _pc in ['bldgarea','lotarea','unitsres','unitstotal']:
        if _pc in pluto.columns:
            pluto[_pc] = pd.to_numeric(pluto[_pc], errors='coerce')
    # ensure BBL numeric
    if 'BBL' in pluto.columns:
        pluto['BBL_join'] = pd.to_numeric(pluto['BBL'], errors='coerce')
    else:
        raise KeyError('PLUTO missing BBL column')

    # merge sales + pluto (left)
    df = sales.merge(pluto.drop(columns=[c for c in pluto.columns if c not in pluto_keep and c!='BBL_join']), how='left', left_on='BBL_join', right_on='BBL_join', suffixes=('','_pluto'))

    # drop blacklist columns if present
    for col in list(df.columns):
        if col.lower() in BAN_COLUMNS:
            df.drop(columns=[col], inplace=True, errors='ignore')

    # create zonedist1_cat
    if 'zonedist1' in df.columns:
        df['zonedist1_cat'] = df['zonedist1'].astype(str).str.strip()
    else:
        df['zonedist1_cat'] = ''

    # create zoning flags
    df['is_R3'] = df['zonedist1_cat'].str.upper().str.startswith('R3')
    df['is_R7'] = df['zonedist1_cat'].str.upper().str.startswith('R7')
    df['is_commercial'] = df['zonedist1_cat'].str.upper().str.startswith('C')

    # attach ideology
    print('Loading ideology:', IDEO)
    ideo = pd.read_parquet(IDEO)
    # map sale year -> election_year
    df['sale_year'] = df['sale_date'].dt.year.fillna(0).astype(int)
    df['election_year'] = df['sale_year'].map(map_election_year)
    # match Council District
    if council_col in df.columns:
        df['CounDist'] = pd.to_numeric(df[council_col], errors='coerce')
    else:
        df['CounDist'] = np.nan

    merged = df.merge(ideo, how='left', left_on=['CounDist','election_year'], right_on=['CounDist','election_year'])

    # Do NOT fill missing dem_share from 2017 here — keep missing to surface parsing issues
    # (previous behavior silently masked missing ideology for 2021/2025).

    # ---- v2 ideology measures (NEW columns; legacy dem_share untouched) --
    # The two v2 files are required inputs for the v2 analysis suite; a
    # missing file is a hard stop (never silently skipped).
    if not IDEO_V2.exists():
        raise SystemExit(
            f'MISSING INPUT: {IDEO_V2} — run '
            'src/data_pipeline/build_ideology_scores_v2.py first.')
    if not IDEO_PRES.exists():
        raise SystemExit(
            f'MISSING INPUT: {IDEO_PRES} — run '
            'src/data_pipeline/build_ideology_scores_v2.py first.')
    ideo_v2 = pd.read_parquet(IDEO_V2)
    ideo_v2 = ideo_v2[['CounDist', 'election_year',
                       'dem_share_mayor_allcand', 'dem_share_vintage',
                       'dem_share_mayor_allcand_vintage']]
    merged = merged.merge(ideo_v2, how='left',
                          left_on=['CounDist', 'election_year'],
                          right_on=['CounDist', 'election_year'])
    ideo_pres = pd.read_parquet(IDEO_PRES)
    ideo_pres = ideo_pres[['CounDist', 'pres_election_year',
                           'dem_share_pres', 'dem_share_pres_vintage']]
    ideo_pres = ideo_pres.rename(columns={'pres_election_year': 'election_year_pres'})
    merged['election_year_pres'] = merged['sale_year'].map(
        lambda y: PRES_MAP.get(int(y), 2016 if int(y) < 2016 else 2024))
    merged = merged.merge(ideo_pres, how='left',
                          left_on=['CounDist', 'election_year_pres'],
                          right_on=['CounDist', 'election_year_pres'])

    # ---- sample flags (Task D) are computed AFTER the dedup step below, so
    # multi_parcel_flag reflects genuine multi-parcel transactions rather
    # than exact duplicate rows that dedup removes. ----

    # drop banned columns again and remove PLUTO-only unneeded fields
    for bad in BAN_COLUMNS:
        if bad in merged.columns:
            merged.drop(columns=[bad], inplace=True)

    # final columns: keep sale info, target_log_price, zonedist1_cat, flags, and selected pluto fields and dem_share
    final_cols = [c for c in merged.columns if c not in ['BBL_join','BBL_join_pluto']]
    out = merged[final_cols]

    # FIX C13: drop exact duplicate rows. The canonical build carried ~1,041
    # duplicated transactions (529 unique, 2-3 copies each); they double-weight
    # those sales in every model and regression. Duplicates share a sale date,
    # so they can never straddle the 2018 temporal split boundary (verified).
    _n_before_dedup = len(out)
    out = out.drop_duplicates()
    print(f'Dedup: dropped {_n_before_dedup - len(out)} exact duplicate rows ({_n_before_dedup} -> {len(out)})')

    # ---- v2 sample flags (Task D; computed on the post-dedup canonical
    # rows). NEW columns; nothing pre-existing is modified. ----------------
    out['sample_pre2024'] = (out['sale_year'] <= 2023).astype(bool)
    _tc_col = 'TAX CLASS AT TIME OF SALE'
    if _tc_col in out.columns:
        _tc = pd.to_numeric(out[_tc_col], errors='coerce')
        out['is_resid_tc12'] = _tc.isin([1, 2]).astype(bool)
    else:
        out['is_resid_tc12'] = False
    out['price_ge_100k'] = (out['sale_price'] >= 100000).astype(bool)
    _mp_key = ['BOROUGH', 'BLOCK', 'sale_date', 'sale_price']
    _mp_size = out.groupby(_mp_key, dropna=False)[_mp_key[0]].transform('size')
    out['multi_parcel_flag'] = (_mp_size > 1).astype(bool)
    # BUILDING CLASS CATEGORY whitespace normalization ("01 ONE FAMILY" vs
    # "01  ONE FAMILY" are split today): NEW column; original untouched.
    if 'BUILDING CLASS CATEGORY' in out.columns:
        out['building_class_category'] = (
            out['BUILDING CLASS CATEGORY'].astype(str)
            .str.strip().str.replace(r'\s+', ' ', regex=True))
    else:
        out['building_class_category'] = ''
    for _c in ['dem_share_mayor_allcand', 'dem_share_pres',
               'dem_share_vintage', 'dem_share_mayor_allcand_vintage',
               'dem_share_pres_vintage']:
        if _c not in out.columns:
            raise SystemExit(f'INTERNAL ERROR: expected new column {_c} missing')
    print('v2 column coverage:',
          {c: round(float(out[c].notna().mean()), 4)
           for c in ['dem_share_mayor_allcand', 'dem_share_pres',
                     'dem_share_vintage', 'dem_share_mayor_allcand_vintage',
                     'dem_share_pres_vintage']})
    print('flag shares:',
          {c: round(float(out[c].mean()), 4)
           for c in ['sample_pre2024', 'is_resid_tc12', 'price_ge_100k',
                     'multi_parcel_flag']})

    out.to_parquet(OUT, index=False)

    # Reporting
    nrows, ncols = out.shape
    dem_cov = out['dem_share'].notna().mean() if 'dem_share' in out.columns else 0.0
    print(f'Wrote canonical v2: {OUT} rows={nrows} cols={ncols}')
    print(f'dem_share coverage: {dem_cov:.4f}')
    _numcheck = [c for c in ['bldgarea','lotarea','unitsres','unitstotal','LAND SQUARE FEET','GROSS SQUARE FEET'] if c in out.columns]
    _bad = {c: str(out[c].dtype) for c in _numcheck if not pd.api.types.is_numeric_dtype(out[c])}
    print('Numeric-eligibility dtypes:', {c: str(out[c].dtype) for c in _numcheck})
    if _bad:
        print('WARNING: still object dtype (trainers will silently skip these):', _bad)
    sample_cols = [c for c in ['dem_share','zonedist1_cat','target_log_price'] if c in out.columns]
    print('Sample rows:')
    if sample_cols:
        print(out[sample_cols].head(5).to_string(index=False))
    else:
        print('No sample columns present')

    # Write the auditable data manifest: exactly which raw files produced this
    # build, so every future raw-data swap self-documents.
    manifest = {
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'sales_file': sales_file.name,
        'sales_row_count': raw_sales_rows,
        'sales_modified_utc': datetime.fromtimestamp(
            sales_file.stat().st_mtime, timezone.utc).isoformat(),
        'pluto_file': pluto_file.name,
        'canonical_output_rows': int(nrows),
        # Round 11: repo-relative POSIX path (was an absolute C:\... path).
        'canonical_output_path': OUT.relative_to(ROOT).as_posix(),
        # v2 additions (new ideology measures + sample flags):
        'ideology_v2_file': IDEO_V2.relative_to(ROOT).as_posix(),
        'ideology_pres_file': IDEO_PRES.relative_to(ROOT).as_posix(),
        'pres_map_note': PRES_MAP_NOTE,
        'new_v2_columns': [
            'dem_share_mayor_allcand', 'dem_share_vintage',
            'dem_share_mayor_allcand_vintage',
            'dem_share_pres', 'dem_share_pres_vintage',
            'election_year_pres', 'sample_pre2024', 'is_resid_tc12',
            'price_ge_100k', 'multi_parcel_flag', 'building_class_category'],
        'new_columns_blacklisted_from_models': True,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote data manifest: {MANIFEST}')

if __name__ == '__main__':
    main()
