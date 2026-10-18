"""Build a vintage-matched ED->council-district crosswalk (full-mile extension).

The production crosswalk (data/processed/ed_to_council_crosswalk.parquet)
is built from the 2026 election-district file and therefore has no entry for
the pre-redistricting ED keys under which the 2016-2021 BOE returns were
cast: results/crosswalk_audit.json quantifies that 17.8-24.1% of ED-level
candidate votes for those years sit on keys missing from it.

This stage reconstructs those assignments from the historical DCP "Bytes
of the Big Apple" election-district boundary files (nyed_16a, nyed_18a,
nyed_20a — pre-2022 ED vintages covering the 2016-2021 elections) overlaid
on the CURRENT council-district boundaries (the same
NYC_City_Council_Districts.shp the production crosswalk uses), so the output
is keyed to the SAME current council-district numbering as PLUTO and the
sales file.

It also records the 2013-plan council districts (nycc_13a — the lines in
force for the 2013/2017/2021 council races) and their areal correspondence
to current districts, documenting the council-vintage mismatch flagged in
the crosswalk audit.

Input files (downloaded from NYC DCP Bytes of the Big Apple,
https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/):
  data/raw/election_districts_legacy/nyed_16a/nyed_16a/nyed.shp
  data/raw/election_districts_legacy/nyed_18a/nyed_18a/nyed.shp
  data/raw/election_districts_legacy/nyed_20a/nyed_20a/nyed.shp
  data/raw/election_districts_legacy/nycc_13a/nycc_13a/nycc.shp

Outputs:
  data/processed/ed_to_council_crosswalk_legacy.parquet
      columns: ElectDist (int, AD*1000+ED), CounDist (int, CURRENT
      district number), area (intersection area, EPSG:2263 sq ft),
      ed_vintage ('nyed_16a'|'nyed_18a'|'nyed_20a')
  data/processed/council_2013_to_current.parquet
      columns: CounDist_2013, CounDist (current), area,
      area_share_of_2013_district

Run: python src/data_pipeline/build_legacy_ed_crosswalk.py
"""
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = ROOT / 'data' / 'raw' / 'election_districts_legacy'
COUNCIL_SHP = ROOT / 'data' / 'raw' / 'election_districts' / 'NYC_City_Council_Districts.shp'
OUT_ED = ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk_legacy.parquet'
OUT_CC = ROOT / 'data' / 'processed' / 'council_2013_to_current.parquet'

ED_VINTAGES = ['nyed_16a', 'nyed_18a', 'nyed_20a']
VINTAGE_RANK = {'nyed_16a': 0, 'nyed_18a': 1, 'nyed_20a': 2}


def overlay_to_council(ed_gdf, council_gdf, vintage):
    """Intersect one ED vintage with current council districts."""
    inter = gpd.overlay(
        ed_gdf[['ElectDist', 'geometry']],
        council_gdf[['CounDist', 'geometry']],
        how='intersection', keep_geom_type=True)
    inter['area'] = inter.geometry.area
    inter = inter[inter['area'] > 0]
    out = inter[['ElectDist', 'CounDist', 'area']].copy()
    out['ElectDist'] = pd.to_numeric(out['ElectDist'], errors='coerce')
    out['CounDist'] = pd.to_numeric(out['CounDist'], errors='coerce')
    out = out.dropna()
    out['ElectDist'] = out['ElectDist'].astype(int)
    out['CounDist'] = out['CounDist'].astype(int)
    out['ed_vintage'] = vintage
    return out

def main():
    council = gpd.read_file(COUNCIL_SHP)
    council['CounDist'] = pd.to_numeric(council['CounDist'], errors='coerce')
    council = council.dropna(subset=['CounDist'])
    council['CounDist'] = council['CounDist'].astype(int)
    if council.crs is None:
        raise SystemExit('Council shapefile has no CRS')
    print(f'Council districts: {len(council)} (crs {council.crs})')

    tables = []
    for vintage in ED_VINTAGES:
        path = LEGACY_DIR / vintage / vintage / 'nyed.shp'
        if not path.exists():
            raise SystemExit(f'MISSING INPUT: {path}')
        ed = gpd.read_file(path)
        ed['ElectDist'] = pd.to_numeric(ed['ElectDist'], errors='coerce')
        ed = ed.dropna(subset=['ElectDist'])
        ed['ElectDist'] = ed['ElectDist'].astype(int)
        if ed.crs != council.crs:
            ed = ed.to_crs(council.crs)
        t = overlay_to_council(ed, council, vintage)
        print(f'{vintage}: {ed["ElectDist"].nunique()} EDs -> '
              f'{len(t)} (ED, CounDist) intersection rows')
        tables.append(t)

    combined = pd.concat(tables, ignore_index=True)
    # One geometry per ED key: prefer the vintage with the largest total
    # intersection area for that key (most complete coverage); ties resolve
    # to the newer vintage.
    key_area = (combined.groupby(['ElectDist', 'ed_vintage'])['area']
                .sum().reset_index())
    key_area['_rank'] = key_area['ed_vintage'].map(VINTAGE_RANK)
    best = (key_area.sort_values(['area', '_rank'], ascending=[False, True])
            .groupby('ElectDist').first().reset_index()[['ElectDist', 'ed_vintage']]
            .rename(columns={'ed_vintage': 'best_vintage'}))
    combined = combined.merge(best, on='ElectDist', how='left')
    keep = combined[combined['ed_vintage'] == combined['best_vintage']]
    keep = keep[['ElectDist', 'CounDist', 'area', 'ed_vintage']]
    keep = keep.sort_values(['ElectDist', 'area'], ascending=[True, False])
    print(f'Legacy crosswalk: {keep["ElectDist"].nunique()} distinct ED keys, '
          f'{len(keep)} (ED, CounDist) rows')
    OUT_ED.parent.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(OUT_ED, index=False)
    print(f'Wrote {OUT_ED.relative_to(ROOT)}')

    # --- 2013-plan council districts -> current council correspondence ----
    cc13_path = LEGACY_DIR / 'nycc_13a' / 'nycc_13a' / 'nycc.shp'
    if cc13_path.exists():
        cc13 = gpd.read_file(cc13_path)
        cc13['CounDist_2013'] = pd.to_numeric(cc13['CounDist'], errors='coerce')
        cc13 = cc13.dropna(subset=['CounDist_2013'])
        cc13['CounDist_2013'] = cc13['CounDist_2013'].astype(int)
        if cc13.crs != council.crs:
            cc13 = cc13.to_crs(council.crs)
        inter = gpd.overlay(
            cc13[['CounDist_2013', 'geometry']],
            council[['CounDist', 'geometry']],
            how='intersection', keep_geom_type=True)
        inter['area'] = inter.geometry.area
        inter = inter[inter['area'] > 0]
        tot = inter.groupby('CounDist_2013')['area'].transform('sum')
        inter['area_share_of_2013_district'] = inter['area'] / tot
        corr = inter[['CounDist_2013', 'CounDist', 'area',
                      'area_share_of_2013_district']].copy()
        corr['CounDist'] = corr['CounDist'].astype(int)
        corr = corr.sort_values(['CounDist_2013', 'area'], ascending=[True, False])
        corr.to_parquet(OUT_CC, index=False)
        print(f'Wrote {OUT_CC.relative_to(ROOT)} '
              f'({corr["CounDist_2013"].nunique()} old-plan -> '
              f'{corr["CounDist"].nunique()} current districts)')
    else:
        print(f'NOTE: {cc13_path} not found; skipping 2013-plan correspondence.')

    # coverage summary vs every raw file's candidate-vote ED keys
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from boe_ed_reader import read_boe_dataframe
    have = set(keep['ElectDist'])
    for yr, off in [(2016, 'president'), (2017, 'mayor'),
                    (2020, 'president'), (2021, 'mayor'),
                    (2024, 'president'), (2025, 'mayor')]:
        df = read_boe_dataframe(
            ROOT / 'data' / 'raw' / 'election_results' / f'ed_results_{yr}_{off}.csv', yr)
        per_ed = df[df['class'] != 'NonCandidate'].groupby('ElectDist')['Tally'].sum()
        tot = int(per_ed.sum())
        miss = per_ed[~per_ed.index.isin(have)]
        print(f'{yr} ({off}): candidate votes={tot:,}; not in legacy '
              f'crosswalk: {len(miss)} EDs / {int(miss.sum())} votes '
              f'({100.0 * miss.sum() / tot:.4f}%)')


if __name__ == '__main__':
    main()

