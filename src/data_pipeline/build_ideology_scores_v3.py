"""Build area-weighted (_aw) and dominant-district (_dom) ideology measures (Round 13, R1).

The legacy aggregation semantics (build_ideology_scores.py,
build_ideology_scores_v2.py) send each ED's FULL vote totals to every
council district the ED touches, so split EDs are double counted:
results/crosswalk_audit.json quantifies summed CD votes at 1.31x raw
votes in 2024/2025 and up to +62% per district. This stage adds
allocation-corrected versions of the six existing measures WITHOUT
touching any legacy output:

  * _aw (area-weighted): for each ED-CD pair, w = intersection_area /
    total ED area taken from the crosswalk 'area' column, normalized so
    each ED's weights sum to 1 (equal split across the ED's districts if
    area is missing or non-positive). Crosswalk choice is by ELECTION
    YEAR: the legacy-vintage crosswalk for 2016/2017/2020/2021
    (pre-redistricting ED keys), the production (2026) crosswalk for
    2024/2025.
  * _dom (dominant district): each ED is assigned wholly to the CD with
    the largest overlap area (ties resolve to the smaller CounDist),
    for the three vintage measures only.

New measures (9 canonical columns, all blacklisted from models):
  dem_share_aw, dem_share_mayor_allcand_aw, dem_share_pres_aw,
  dem_share_vintage_aw, dem_share_mayor_allcand_vintage_aw,
  dem_share_pres_vintage_aw, dem_share_vintage_dom,
  dem_share_mayor_allcand_vintage_dom, dem_share_pres_vintage_dom.

Shares are ratios of weighted vote totals, matching the v2 builders'
ratio-of-totals convention. Because the crosswalk choice above is
vintage-matched for the pre-2024 years by construction, each _aw measure
numerically equals its _vintage_aw counterpart; both are written as
specified and the identity is documented in results/crosswalk_audit_v3.json.

The mayoral ED-level Dem/Rep totals are reconciled against the production
builder's parse_year_file() output exactly as in the v2 stage.

Outputs (new files; nothing pre-existing is modified):
  data/processed/ideology_by_council_v3.parquet
  data/processed/ideology_pres_by_council_v3.parquet

Run: python src/data_pipeline/build_ideology_scores_v3.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boe_ed_reader import read_boe_dataframe  # noqa: E402
import build_ideology_scores_v2 as v2builder  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / 'data' / 'raw' / 'election_results'
CROSSWALK = ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk.parquet'
LEGACY_XW = ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk_legacy.parquet'
OUT_MAYOR = ROOT / 'data' / 'processed' / 'ideology_by_council_v3.parquet'
OUT_PRES = ROOT / 'data' / 'processed' / 'ideology_pres_by_council_v3.parquet'

MAYORAL_YEARS = [2017, 2021, 2025]
PRES_YEARS = [2016, 2020, 2024]
# election year -> crosswalk vintage (R1: production for 2024/2025,
# legacy-vintage for 2016/2017/2020/2021)
YEAR_USES_LEGACY_XW = {2016, 2017, 2020, 2021}


def load_crosswalk(path):
    if not path.exists():
        raise SystemExit(f'MISSING INPUT: {path}')
    cw = pd.read_parquet(path)
    cw['ElectDist_num'] = pd.to_numeric(cw['ElectDist'], errors='coerce')
    return cw[['ElectDist_num', 'CounDist', 'area']]


def weighted_key(key):
    """ED-CD pairs with normalized area weights (each ED sums to exactly 1)."""
    k = key.copy()
    tot = k.groupby('ElectDist_num')['area'].transform('sum')
    usable = k['area'].notna() & (k['area'] > 0) & tot.notna() & (tot > 0)
    w = (k['area'] / tot).where(usable)
    n = k.groupby('ElectDist_num')['CounDist'].transform('size')
    w = w.fillna(1.0 / n)
    k['w'] = w.astype(float)
    chk = k.groupby('ElectDist_num')['w'].sum()
    assert ((chk - 1.0).abs() < 1e-9).all(), 'weight normalization failed'
    return k[['ElectDist_num', 'CounDist', 'w']].reset_index(drop=True)


def dominant_key(key):
    """One (ED -> CD) row per ED: the CD with the largest overlap area."""
    k = key.copy()
    k['_area'] = k['area'].fillna(-1.0)
    k = k.sort_values(['ElectDist_num', '_area', 'CounDist'],
                      ascending=[True, False, True])
    dom = k.groupby('ElectDist_num', as_index=False).first()
    return dom[['ElectDist_num', 'CounDist']]


def agg_weighted(votes_ed, wkey, cols):
    """Weighted vote totals by council district (w applied to each ED's votes)."""
    m = votes_ed.merge(wkey, how='inner', left_on='ElectDist',
                       right_on='ElectDist_num')
    out = m[['CounDist']].copy()
    for c in cols:
        out[c] = (m[c].astype(float) * m['w']).values
    return out.groupby('CounDist')[cols].sum().reset_index()


def agg_full(votes_ed, key_one_row):
    """Full ED votes to the single assigned CD (dominant-district semantics)."""
    m = votes_ed.merge(key_one_row, how='inner', left_on='ElectDist',
                       right_on='ElectDist_num')
    cols = [c for c in votes_ed.columns if c != 'ElectDist']
    return m.groupby('CounDist')[cols].sum().reset_index()


def keys_for_year(year):
    path = LEGACY_XW if year in YEAR_USES_LEGACY_XW else CROSSWALK
    raw = load_crosswalk(path)
    return weighted_key(raw), dominant_key(raw), path.name


def main():
    for yr in MAYORAL_YEARS + PRES_YEARS:
        office = 'mayor' if yr in MAYORAL_YEARS else 'president'
        p = RAW_DIR / f'ed_results_{yr}_{office}.csv'
        if not p.exists():
            raise SystemExit(
                f'MISSING INPUT: {p} — required for the v3 ideology stage. '
                'Fetch it from vote.nyc (Statement and Return Report by '
                'Election District, Citywide '
                + ('Mayor' if office == 'mayor'
                   else 'President/Vice President')
                + ', ED level) before re-running.')

    # ---- mayoral: per-ED dem/rep/allcand (identical parse to the v2 stage)
    frames = []
    for year in MAYORAL_YEARS:
        df = read_boe_dataframe(RAW_DIR / f'ed_results_{year}_mayor.csv', year)
        per_ed = v2builder.ed_class_matrix(df)
        per_ed = per_ed.rename(columns={'Dem': 'dem', 'Rep': 'rep'})
        per_ed['allcand'] = per_ed[['dem', 'rep', 'Other']].sum(axis=1)
        per_ed['year'] = year
        per_ed = per_ed.merge(
            df.groupby('ElectDist')[['AD', 'ED']].first().reset_index(),
            on='ElectDist', how='left')
        frames.append(per_ed[['ElectDist', 'AD', 'ED', 'year',
                              'dem', 'rep', 'allcand']])
    mayor_ed = pd.concat(frames, ignore_index=True)
    v2builder.reconcile_with_legacy_builder(mayor_ed)
    build_mayoral(mayor_ed)
    build_presidential()


def build_mayoral(mayor_ed):
    cols = ['dem', 'rep', 'allcand']
    aw_parts, dom_parts = [], []
    for year in MAYORAL_YEARS:
        wkey, dkey, xw_name = keys_for_year(year)
        sub = mayor_ed[mayor_ed['year'] == year]
        aw = agg_weighted(sub, wkey, cols)
        aw['election_year'] = year
        aw_parts.append(aw)
        dm = agg_full(sub[['ElectDist'] + cols], dkey)
        dm['election_year'] = year
        dom_parts.append(dm)
        print(f'mayoral {year} via {xw_name}: {len(aw)} CDs, '
              f'{int(round(aw["allcand"].sum())):,} weighted all-candidate votes')

    aw = pd.concat(aw_parts, ignore_index=True)
    aw['dem_share_aw'] = aw['dem'] / (aw['dem'] + aw['rep'])
    aw['dem_share_mayor_allcand_aw'] = aw['dem'] / aw['allcand']
    # vintage _aw variants: same year->crosswalk mapping, therefore identical
    aw['dem_share_vintage_aw'] = aw['dem_share_aw']
    aw['dem_share_mayor_allcand_vintage_aw'] = aw['dem_share_mayor_allcand_aw']

    dm = pd.concat(dom_parts, ignore_index=True)
    dm = dm.rename(columns={'dem': 'dem_dom', 'rep': 'rep_dom',
                            'allcand': 'allcand_dom'})
    aw = aw.merge(dm, on=['CounDist', 'election_year'], how='left')
    aw['dem_share_vintage_dom'] = aw['dem_dom'] / (aw['dem_dom'] + aw['rep_dom'])
    aw['dem_share_mayor_allcand_vintage_dom'] = aw['dem_dom'] / aw['allcand_dom']

    aw['CounDist'] = aw['CounDist'].astype(int)
    aw['election_year'] = aw['election_year'].astype(int)
    aw = aw.sort_values(['CounDist', 'election_year']).reset_index(drop=True)
    OUT_MAYOR.parent.mkdir(parents=True, exist_ok=True)
    aw.to_parquet(OUT_MAYOR, index=False)
    print(f'Wrote {OUT_MAYOR.relative_to(ROOT)} ({len(aw)} CD-year rows)')
    for year in MAYORAL_YEARS:
        s = aw[aw['election_year'] == year]
        print(f'mayoral {year}: mean dem_share_aw={s["dem_share_aw"].mean():.4f} '
              f'mean dom={s["dem_share_vintage_dom"].mean():.4f}')


def build_presidential():
    pframes = []
    for year in PRES_YEARS:
        df = read_boe_dataframe(RAW_DIR / f'ed_results_{year}_president.csv', year)
        per_ed = v2builder.ed_class_matrix(df)
        per_ed['dem_pres'] = per_ed['Dem']
        per_ed['rep_pres'] = per_ed['Rep']
        per_ed['pres_election_year'] = year
        pframes.append(per_ed[['ElectDist', 'dem_pres', 'rep_pres',
                               'pres_election_year']])
    pres_ed = pd.concat(pframes, ignore_index=True)

    cols = ['dem_pres', 'rep_pres']
    paw_parts, pdom_parts = [], []
    for year in PRES_YEARS:
        wkey, dkey, xw_name = keys_for_year(year)
        sub = pres_ed[pres_ed['pres_election_year'] == year]
        paw = agg_weighted(sub, wkey, cols)
        paw['pres_election_year'] = year
        paw_parts.append(paw)
        pdm = agg_full(sub[['ElectDist'] + cols], dkey)
        pdm['pres_election_year'] = year
        pdom_parts.append(pdm)
        print(f'pres {year} via {xw_name}: {len(paw)} CDs, '
              f'{int(round((paw["dem_pres"] + paw["rep_pres"]).sum())):,} '
              'weighted Dem+Rep votes')

    paw = pd.concat(paw_parts, ignore_index=True)
    paw['dem_share_pres_aw'] = (paw['dem_pres']
                               / (paw['dem_pres'] + paw['rep_pres']))
    paw['dem_share_pres_vintage_aw'] = paw['dem_share_pres_aw']
    pdm = pd.concat(pdom_parts, ignore_index=True)
    pdm = pdm.rename(columns={'dem_pres': 'dem_pres_dom',
                              'rep_pres': 'rep_pres_dom'})
    paw = paw.merge(pdm, on=['CounDist', 'pres_election_year'], how='left')
    paw['dem_share_pres_vintage_dom'] = (
        paw['dem_pres_dom'] / (paw['dem_pres_dom'] + paw['rep_pres_dom']))

    paw['CounDist'] = paw['CounDist'].astype(int)
    paw['pres_election_year'] = paw['pres_election_year'].astype(int)
    paw = paw.sort_values(['CounDist', 'pres_election_year']).reset_index(drop=True)
    paw.to_parquet(OUT_PRES, index=False)
    print(f'Wrote {OUT_PRES.relative_to(ROOT)} ({len(paw)} CD-year rows)')
    for year in PRES_YEARS:
        s = paw[paw['pres_election_year'] == year]
        print(f'pres {year}: mean dem_share_pres_aw='
              f'{s["dem_share_pres_aw"].mean():.4f} '
              f'mean dom={s["dem_share_pres_vintage_dom"].mean():.4f}')


if __name__ == '__main__':
    main()

