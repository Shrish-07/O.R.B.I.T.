"""Rebuild canonical modeling dataset v2 with ideology and selected PLUTO fields.

Saves: data/canonical/modeling_dataset_canonical_v2.parquet

Usage: .venv\Scripts\python.exe src/rebuild_canonical.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SALES = ROOT / 'data' / 'raw' / 'sales' / 'NYC_Citywide_Annualized_Calendar_Sales_Update_20260105.csv'
PLUTO = ROOT / 'data' / 'raw' / 'pluto' / 'Primary_Land_Use_Tax_Lot_Output_(PLUTO)_20260105.csv'
IDEO = ROOT / 'data' / 'processed' / 'ideology_by_council.parquet'
OUT_DIR = ROOT / 'data' / 'canonical'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / 'modeling_dataset_canonical_v2.parquet'

BAN_COLUMNS = {'assesstot','exempttot','assessland','appbbl','taxmap','plutomapid'}

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
    print('Loading sales:', SALES)
    sales = pd.read_csv(SALES, low_memory=False)
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

    # normalize BBL for join
    sales['BBL_join'] = pd.to_numeric(sales[bbl_col], errors='coerce')

    # read PLUTO and select fields
    pluto_keep = ['BBL','zonedist1','yearbuilt','numfloors','bldgarea','lotarea','unitsres','unitstotal','residfar','commfar','facilfar','landuse','bldgclass']
    print('Loading PLUTO:', PLUTO)
    pluto = pd.read_csv(PLUTO, usecols=lambda c: c in pluto_keep, low_memory=False)
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

    # drop banned columns again and remove PLUTO-only unneeded fields
    for bad in BAN_COLUMNS:
        if bad in merged.columns:
            merged.drop(columns=[bad], inplace=True)

    # final columns: keep sale info, target_log_price, zonedist1_cat, flags, and selected pluto fields and dem_share
    final_cols = [c for c in merged.columns if c not in ['BBL_join','BBL_join_pluto']]
    out = merged[final_cols]

    out.to_parquet(OUT, index=False)

    # Reporting
    nrows, ncols = out.shape
    dem_cov = out['dem_share'].notna().mean() if 'dem_share' in out.columns else 0.0
    print(f'Wrote canonical v2: {OUT} rows={nrows} cols={ncols}')
    print(f'dem_share coverage: {dem_cov:.4f}')
    sample_cols = [c for c in ['dem_share','zonedist1_cat','target_log_price'] if c in out.columns]
    print('Sample rows:')
    if sample_cols:
        print(out[sample_cols].head(5).to_string(index=False))
    else:
        print('No sample columns present')

if __name__ == '__main__':
    main()
