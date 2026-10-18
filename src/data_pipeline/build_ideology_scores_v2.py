"""Build v2 ideology measures by council district (Task B).

Keeps the legacy dem_share (data/processed/ideology_by_council.parquet,
built by build_ideology_scores.py) COMPLETELY UNTOUCHED. Adds two new
ideology measures, aggregated to council district through the same
ED->council crosswalk with the same merge semantics as the legacy builder
(each ED's full vote counts go to every council district the ED touches;
split EDs are therefore counted in each district they overlap — documented
and quantified in results/crosswalk_audit.json):

1. dem_share_mayor_allcand = Dem / (ALL candidate-line votes)
   Dem = Democratic + Working Families lines, from the mayoral races
   2017/2021/2025 (raw: ed_results_{year}_mayor.csv). Unlike the legacy
   Dem/(Dem+Rep), this denominator counts every candidate ballot line, so
   the 2025 "Fight and Deliver" line (Cuomo) and every other minor line
   enter the denominator.

2. dem_share_pres = (Dem+WFP) / (Dem+WFP+Rep+Conservative) for President
   2016, 2020, 2024 (raw: ed_results_{year}_president.csv, fetched from
   vote.nyc "Statement and Return Report by Election District",
   Citywide President/Vice President, ED level).

Both raw layouts (clean 2016/2017 vs header-prepended 2020/2021/2024/2025)
are handled by the shared reader in src/data_pipeline/boe_ed_reader.py,
which re-uses the working line-oriented parse from
src/data_pipeline/build_ideology_scores.py.

Election-year mapping (explicit, documented; sale_year -> presidential
election used for dem_share_pres):
    {2016:2016, 2017:2016, 2018:2020, 2019:2020, 2020:2020, 2021:2020,
     2022:2024, 2023:2024, 2024:2024, 2025:2024}
The canonical merge attaches dem_share_pres via a separate
election_year_pres column so the mayoral election_year merge (and legacy
dem_share) is unchanged.

Reconciliation gate: the ED-level Dem/Rep totals re-derived here for the
mayoral years are compared against the production builder's own
parse_year_file() output; any mismatch aborts the stage (no silent drift
from the legacy numbers).

Outputs:
  data/processed/ideology_by_council_v2.parquet
      (CounDist, election_year in {2017,2021,2025}; columns:
       dem, rep, allcand, dem_share [= legacy definition, recomputed],
       dem_share_mayor_allcand)
  data/processed/ideology_pres_by_council.parquet
      (CounDist, pres_election_year in {2016,2020,2024}; columns:
       dem_pres, rep_pres, dem_share_pres)

Run: python src/data_pipeline/build_ideology_scores_v2.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boe_ed_reader import read_boe_dataframe  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / 'data' / 'raw' / 'election_results'
CROSSWALK = ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk.parquet'
LEGACY_XW = ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk_legacy.parquet'
OUT_MAYOR = ROOT / 'data' / 'processed' / 'ideology_by_council_v2.parquet'
OUT_PRES = ROOT / 'data' / 'processed' / 'ideology_pres_by_council.parquet'
LEGACY_BUILDER = ROOT / 'src' / 'data_pipeline' / 'build_ideology_scores.py'

MAYORAL_YEARS = [2017, 2021, 2025]
PRES_YEARS = [2016, 2020, 2024]

def load_crosswalk():
    if not CROSSWALK.exists():
        raise FileNotFoundError(f'Missing ED->Council crosswalk: {CROSSWALK}')
    cw = pd.read_parquet(CROSSWALK)
    cw['ElectDist_num'] = pd.to_numeric(cw['ElectDist'], errors='coerce')
    return cw[['ElectDist_num', 'CounDist']]


def load_legacy_crosswalk():
    """Vintage-matched crosswalk (old ED boundaries -> CURRENT council
    districts). Recovers the 17.8-24.1% of 2016-2021 candidate votes whose
    pre-redistricting ED keys are absent from the 2026 crosswalk."""
    if not LEGACY_XW.exists():
        raise SystemExit(
            f'MISSING INPUT: {LEGACY_XW} — run '
            'src/data_pipeline/build_legacy_ed_crosswalk.py first (it needs '
            'the historical ED boundary files under '
            'data/raw/election_districts_legacy/).')
    cw = pd.read_parquet(LEGACY_XW)
    cw['ElectDist_num'] = pd.to_numeric(cw['ElectDist'], errors='coerce')
    return cw[['ElectDist_num', 'CounDist']]


def aggregate_to_council(edm, key):
    """Merge ED rows to the crosswalk (legacy semantics: an ED that touches
    k council districts contributes its FULL vote totals to each of the k
    districts), then sum by council district."""
    return edm.merge(key, how='inner', left_on='ElectDist',
                     right_on='ElectDist_num')


def reconcile_with_legacy_builder(mayor_ed):
    """Cross-check this reader's Dem/Rep totals against the production
    builder's parse_year_file() per (AD,ED) — must match exactly."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('legacy_builder', LEGACY_BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for year in MAYORAL_YEARS:
        legacy = mod.parse_year_file(year, RAW_DIR / f'ed_results_{year}_mayor.csv')
        mine = mayor_ed[mayor_ed['year'] == year]
        mism = 0
        for _, r in mine.iterrows():
            k = (int(r['AD']), int(r['ED']))
            lv = legacy.get(k, {'dem': 0, 'rep': 0})
            if abs(r['dem'] - lv['dem']) > 0.5 or abs(r['rep'] - lv['rep']) > 0.5:
                mism += 1
        if mism:
            raise SystemExit(
                f'RECONCILIATION FAILED for {year}: {mism} ED rows differ '
                'from build_ideology_scores.parse_year_file. Aborting.')
        print(f'  reconcile {year}: OK '
              f'({len(mine)} ED rows match legacy builder Dem/Rep totals)')


def ed_class_matrix(df):
    """Candidate-line class totals per ED."""
    cand = df[df['class'] != 'NonCandidate']
    per_ed = cand.groupby(['ElectDist', 'class'])['Tally'].sum().unstack(fill_value=0)
    for c in ['Dem', 'Rep', 'Other']:
        if c not in per_ed.columns:
            per_ed[c] = 0
    return per_ed.reset_index()


def main():
    for yr in MAYORAL_YEARS + PRES_YEARS:
        office = 'mayor' if yr in MAYORAL_YEARS else 'president'
        p = RAW_DIR / f'ed_results_{yr}_{office}.csv'
        if not p.exists():
            raise SystemExit(
                f'MISSING INPUT: {p} — required for the v2 ideology stage. '
                'Fetch it from vote.nyc (Statement and Return Report by '
                'Election District, Citywide ' +
                ('Mayor' if office == 'mayor' else 'President/Vice President') +
                ', ED level) before re-running.')

    key = load_crosswalk()

    # ---- mayoral: dem, rep, allcand per ED per year -------------------
    frames = []
    for year in MAYORAL_YEARS:
        df = read_boe_dataframe(RAW_DIR / f'ed_results_{year}_mayor.csv', year)
        per_ed = ed_class_matrix(df)
        per_ed = per_ed.rename(columns={'Dem': 'dem', 'Rep': 'rep'})
        per_ed['allcand'] = per_ed[['dem', 'rep', 'Other']].sum(axis=1)
        per_ed['year'] = year
        per_ed = per_ed.merge(
            df.groupby('ElectDist')[['AD', 'ED']].first().reset_index(),
            on='ElectDist', how='left')
        frames.append(per_ed[['ElectDist', 'AD', 'ED', 'year', 'dem', 'rep', 'allcand']])
    mayor_ed = pd.concat(frames, ignore_index=True)
    reconcile_with_legacy_builder(mayor_ed)

    m = aggregate_to_council(mayor_ed, key)
    agg = (m.groupby(['CounDist', 'year'])[['dem', 'rep', 'allcand']]
           .sum().reset_index().rename(columns={'year': 'election_year'}))
    agg['dem_share'] = agg['dem'] / (agg['dem'] + agg['rep'])
    agg['dem_share_mayor_allcand'] = agg['dem'] / agg['allcand']
    agg['CounDist'] = agg['CounDist'].astype(int)
    agg['election_year'] = agg['election_year'].astype(int)

    # ---- vintage-corrected mayoral (2017/2021 via the legacy ED
    # crosswalk built from the pre-2022 boundary files; 2025 keys already
    # match the 2026 crosswalk, so its vintage column equals the base) ----
    lkey = load_legacy_crosswalk()
    mv = aggregate_to_council(mayor_ed[mayor_ed['year'].isin([2017, 2021])], lkey)
    aggv = (mv.groupby(['CounDist', 'year'])[['dem', 'rep', 'allcand']]
            .sum().reset_index().rename(columns={'year': 'election_year'}))
    aggv['dem_share_mayor_allcand_vintage'] = aggv['dem'] / aggv['allcand']
    aggv['dem_share_vintage'] = aggv['dem'] / (aggv['dem'] + aggv['rep'])
    aggv['CounDist'] = aggv['CounDist'].astype(int)
    aggv['election_year'] = aggv['election_year'].astype(int)
    aggv = aggv[['CounDist', 'election_year',
                 'dem_share_mayor_allcand_vintage', 'dem_share_vintage']]
    agg = agg.merge(aggv, on=['CounDist', 'election_year'], how='left')
    agg['dem_share_mayor_allcand_vintage'] = agg[
        'dem_share_mayor_allcand_vintage'].fillna(agg['dem_share_mayor_allcand'])
    agg['dem_share_vintage'] = agg['dem_share_vintage'].fillna(agg['dem_share'])

    agg = agg.sort_values(['CounDist', 'election_year'])
    OUT_MAYOR.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT_MAYOR, index=False)
    print(f'Wrote {OUT_MAYOR.relative_to(ROOT)} ({len(agg)} CD-year rows)')

    # ---- presidential: (Dem+WFP)/(Dem+WFP+Rep+Conservative) per ED ----
    frames = []
    for year in PRES_YEARS:
        df = read_boe_dataframe(RAW_DIR / f'ed_results_{year}_president.csv', year)
        per_ed = ed_class_matrix(df)
        # spec formula counts ONLY the four major-party lines in BOTH
        # numerator and denominator (Women's Equality / Green / Libertarian
        # / Independence etc. lines are excluded by definition).
        per_ed['dem_pres'] = per_ed['Dem']
        per_ed['rep_pres'] = per_ed['Rep']
        per_ed['pres_election_year'] = year
        frames.append(per_ed[['ElectDist', 'dem_pres', 'rep_pres',
                              'pres_election_year']])
    pres_ed = pd.concat(frames, ignore_index=True)
    p = aggregate_to_council(pres_ed, key)
    pagg = (p.groupby(['CounDist', 'pres_election_year'])
            [['dem_pres', 'rep_pres']].sum().reset_index())
    pagg['dem_share_pres'] = pagg['dem_pres'] / (pagg['dem_pres'] + pagg['rep_pres'])
    pagg['CounDist'] = pagg['CounDist'].astype(int)
    pagg['pres_election_year'] = pagg['pres_election_year'].astype(int)

    # ---- vintage-corrected presidential (2016/2020 via the legacy ED
    # crosswalk; 2024 keys match the 2026 crosswalk already) ------------
    pv = aggregate_to_council(
        pres_ed[pres_ed['pres_election_year'].isin([2016, 2020])], lkey)
    paggv = (pv.groupby(['CounDist', 'pres_election_year'])
             [['dem_pres', 'rep_pres']].sum().reset_index())
    paggv['dem_share_pres_vintage'] = (paggv['dem_pres']
                                       / (paggv['dem_pres'] + paggv['rep_pres']))
    paggv['CounDist'] = paggv['CounDist'].astype(int)
    paggv['pres_election_year'] = paggv['pres_election_year'].astype(int)
    paggv = paggv[['CounDist', 'pres_election_year', 'dem_share_pres_vintage']]
    pagg = pagg.merge(paggv, on=['CounDist', 'pres_election_year'], how='left')
    pagg['dem_share_pres_vintage'] = pagg[
        'dem_share_pres_vintage'].fillna(pagg['dem_share_pres'])

    pagg = pagg.sort_values(['CounDist', 'pres_election_year'])
    pagg.to_parquet(OUT_PRES, index=False)
    print(f'Wrote {OUT_PRES.relative_to(ROOT)} ({len(pagg)} CD-year rows)')

    for year in MAYORAL_YEARS:
        sub = agg[agg['election_year'] == year]
        print(f'mayoral {year}: mean dem_share={sub["dem_share"].mean():.4f} '
              f'mean allcand={sub["dem_share_mayor_allcand"].mean():.4f}')
    for year in PRES_YEARS:
        sub = pagg[pagg['pres_election_year'] == year]
        print(f'pres {year}: mean dem_share_pres={sub["dem_share_pres"].mean():.4f} '
              f'across {len(sub)} districts')


if __name__ == '__main__':
    main()

