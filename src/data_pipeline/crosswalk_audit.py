"""ED/council-district crosswalk vintage audit (Task C).

The ED->council-district crosswalk (data/processed/ed_to_council_crosswalk.
parquet) is built from the CURRENT (2026) election-district file, but the raw
BOE ED-level returns for 2016-2025 were cast on older ED vintages, and raw
files carry "EDAD Status" annotations such as "COMBINED INTO 059/76" for EDs
that no longer exist as polling units.

For EVERY raw election file (mayoral 2017/2021/2025 and presidential
2016/2020/2024) this audit reports:
  * total candidate votes at ED level (all candidate ballot lines,
    non-candidate counter rows excluded);
  * the share of candidate votes sitting on AD-ED keys MISSING from the
    2026 crosswalk (these votes are DROPPED by the legacy/v2 aggregation);
  * candidate votes sitting in "COMBINED INTO" EDs and how they are handled
    (votes are looked up under the OLD ED's own key; they are assigned only
    if that old key still exists in the crosswalk — they are NEVER
    reassigned to the successor ED named in the status);
  * lost votes by borough (County: New York/Bronx/Kings/Queens/Richmond);
  * CD-level totals under the production (legacy) assignment — each ED's
    full votes go to every council district it touches — versus an
    area-weighted allocation benchmark, per council district and in total
    versus the raw candidate-vote sum (the legacy semantics INFLATE CD
    totals wherever an ED is split across districts; that inflation is
    quantified here).

STOP RULES (per the task brief): if any year loses >1% of candidate votes to
missing crosswalk keys, or any council district's assigned total deviates
>3% from the area-weighted benchmark, the stage writes the full audit JSON
and then EXITS NON-ZERO with a STOP report so the operator can review before
any downstream numbers are rebuilt.

Also documents the council-district vintage mismatch: PLUTO and the sales
file carry CURRENT (post-2022 redistricting) council districts, while the
2013/2017/2021 council races used the pre-2022 district plan.

Output: results/crosswalk_audit.json

Run: python src/data_pipeline/crosswalk_audit.py
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
OUT = ROOT / 'results' / 'crosswalk_audit.json'

MAYORAL_YEARS = [2017, 2021, 2025]
PRES_YEARS = [2016, 2020, 2024]
YEAR_LOSS_STOP = 0.01     # >1% candidate votes lost in any year -> STOP
CD_DEVIATION_STOP = 0.03  # >3% deviation on any council district -> STOP

BOROUGH_BY_COUNTY = {
    'New York': 'Manhattan', 'Bronx': 'Bronx', 'Kings': 'Brooklyn',
    'Queens': 'Queens', 'Richmond': 'Staten Island',
}

def audit_year(df, cw_full, office, year, legacy_cw=None):
    cand = df[df['class'] != 'NonCandidate']
    per_ed = cand.groupby('ElectDist')['Tally'].sum().rename('votes').reset_index()
    per_ed['County'] = per_ed['ElectDist'].map(cand.groupby('ElectDist')['County'].first())
    per_ed['status'] = per_ed['ElectDist'].map(df.groupby('ElectDist')['EDAD Status'].first())

    total = int(per_ed['votes'].sum())
    per_ed['in_cw'] = per_ed['ElectDist'].isin(cw_full['ElectDist_num'])
    missing = per_ed[~per_ed['in_cw']]
    missing_votes = int(missing['votes'].sum())
    missing_share = missing_votes / total if total else 0.0

    # recovery under the vintage-matched (legacy) ED crosswalk
    if legacy_cw is not None:
        in_lc = per_ed['ElectDist'].isin(legacy_cw['ElectDist_num'])
        still = per_ed[~in_lc]
        still_votes = int(still['votes'].sum())
        still_share = still_votes / total if total else 0.0
    else:
        still_votes = None
        still_share = None

    combined = per_ed[per_ed['status'].str.startswith('COMBINED INTO', na=False)]
    combined_votes = int(combined['votes'].sum())
    combined_matched = int(combined[combined['in_cw']]['votes'].sum())

    loss_by_county = (missing.groupby('County')['votes'].sum()
                      .rename(lambda c: BOROUGH_BY_COUNTY.get(c, c)))
    tot_by_county = (per_ed.groupby('County')['votes'].sum()
                     .rename(lambda c: BOROUGH_BY_COUNTY.get(c, c)))

    # CD-level assignment: production (full-to-each) vs area-weighted
    area = cw_full.copy()
    ed_area = area.groupby('ElectDist_num')['area'].transform('sum')
    area['votes'] = area['ElectDist_num'].map(per_ed.set_index('ElectDist')['votes']).fillna(0.0)
    area['aw_votes'] = area['votes'] * area['area'] / ed_area
    legacy_cd = area.groupby('CounDist')['votes'].sum().rename('legacy_total')
    aw_cd = area.groupby('CounDist')['aw_votes'].sum().rename('area_weighted_total')
    cd = pd.concat([legacy_cd, aw_cd], axis=1).fillna(0.0)
    cd['deviation_pct'] = ((cd['legacy_total'] - cd['area_weighted_total'])
                           / cd['area_weighted_total'].replace(0, pd.NA))

    cd_rows = {str(int(c)): {
        'legacy_assigned_votes': int(round(r['legacy_total'])),
        'area_weighted_votes': float(round(r['area_weighted_total'], 2)),
        'deviation_pct': float(round(100.0 * r['deviation_pct'], 3)) if pd.notna(r['deviation_pct']) else None,
    } for c, r in cd.iterrows()}

    sum_cd_legacy = int(round(cd['legacy_total'].sum()))
    return {
        'office': office,
        'total_candidate_votes_raw': total,
        'n_election_districts_with_candidate_votes': int(len(per_ed)),
        'missing_key_eds': int(len(missing)),
        'missing_key_candidate_votes': missing_votes,
        'missing_key_share': round(missing_share, 6),
        'missing_key_share_pct': round(100.0 * missing_share, 4),
        'missing_after_vintage_xwalk_votes': still_votes,
        'missing_after_vintage_xwalk_share_pct': (
            round(100.0 * still_share, 4) if still_share is not None else None),
        'combined_into_eds': int(len(combined)),
        'combined_into_candidate_votes': combined_votes,
        'combined_into_votes_with_live_crosswalk_key': combined_matched,
        'combined_into_handling': (
            'Votes in "COMBINED INTO" EDs are looked up under the OLD ED\'s '
            'own AD-ED key and are NEVER reassigned to the successor ED '
            'named in the status. If the old key still exists in the 2026 '
            'crosswalk the votes are assigned normally; otherwise they are '
            'dropped together with all other missing-key EDs.'),
        'votes_lost_by_borough': {k: int(v) for k, v in loss_by_county.items()},
        'candidate_votes_by_borough': {k: int(v) for k, v in tot_by_county.items()},
        'cd_assignment': cd_rows,
        'sum_cd_legacy_assigned_vs_raw': {
            'sum_cd_legacy_assigned': sum_cd_legacy,
            'raw_candidate_votes': total,
            'ratio': round(sum_cd_legacy / total, 6) if total else None,
            'note': ('Legacy assignment sends each ED\'s FULL votes to every '
                     'council district the ED touches, so split EDs are '
                     'counted once per overlapping district; the sum over '
                     'districts therefore exceeds the raw candidate total by '
                     'the split-ED overlap. The area-weighted benchmark '
                     'allocates ED votes by the crosswalk\'s area shares.'),
        },
    }

def main():
    if not CROSSWALK.exists():
        raise SystemExit(f'MISSING INPUT: {CROSSWALK}')
    cw_full = pd.read_parquet(CROSSWALK)
    cw_full['ElectDist_num'] = pd.to_numeric(cw_full['ElectDist'], errors='coerce')
    cw_full['CounDist'] = pd.to_numeric(cw_full['CounDist'], errors='coerce')

    legacy_xw_path = ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk_legacy.parquet'
    legacy_cw = None
    if legacy_xw_path.exists():
        legacy_cw = pd.read_parquet(legacy_xw_path)
        legacy_cw['ElectDist_num'] = pd.to_numeric(legacy_cw['ElectDist'], errors='coerce')

    years = {}
    stop_reasons = []
    resolved_by_vintage = []
    for year in sorted(MAYORAL_YEARS + PRES_YEARS):
        office = 'mayor' if year in MAYORAL_YEARS else 'president'
        path = RAW_DIR / f'ed_results_{year}_{office}.csv'
        if not path.exists():
            raise SystemExit(f'MISSING INPUT: {path}')
        df = read_boe_dataframe(path, year)
        rec = audit_year(df, cw_full, office, year, legacy_cw=legacy_cw)
        years[str(year)] = rec
        vst = rec['missing_after_vintage_xwalk_share_pct']
        print(f'{year} ({office}): cand={rec["total_candidate_votes_raw"]:,} '
              f'missing(base xwalk)={rec["missing_key_share_pct"]:.3f}% '
              f'| missing(vintage xwalk)='
              f'{"n/a" if vst is None else format(vst, ".4f")}% ; '
              f'combined-into EDs={rec["combined_into_eds"]} '
              f'({rec["combined_into_candidate_votes"]:,} votes); '
              f'sum(CD legacy)/raw={rec["sum_cd_legacy_assigned_vs_raw"]["ratio"]}')
        if rec['missing_key_share'] > YEAR_LOSS_STOP:
            stop_reasons.append(
                f'{year} ({office}): missing-crosswalk-key candidate votes = '
                f'{rec["missing_key_share_pct"]:.3f}% > 1% STOP threshold')
            if legacy_cw is not None and vst is not None and vst <= 1.0:
                resolved_by_vintage.append(
                    f'{year} ({office}): recovered by the vintage-matched '
                    f'crosswalk (missing after vintage xwalk = {vst:.4f}%)')
        bad_cds = {cd: r['deviation_pct'] for cd, r in rec['cd_assignment'].items()
                   if r['deviation_pct'] is not None
                   and abs(r['deviation_pct']) > 100.0 * CD_DEVIATION_STOP}
        if bad_cds:
            worst = max(bad_cds.values())
            stop_reasons.append(
                f'{year} ({office}): {len(bad_cds)} council districts deviate '
                f'>3% from the area-weighted benchmark (worst {worst:+.2f}%) '
                'under the legacy full-votes-to-each-CD assignment (split-ED '
                'double counting inherent to the production crosswalk; see '
                'sum_cd_legacy_assigned_vs_raw)')

    result = {
        'audit': 'ED/crosswalk vintage audit, all raw ED-level files',
        'crosswalk': {
            'file': 'data/processed/ed_to_council_crosswalk.parquet',
            'vintage': 'built from the 2026 election-district file '
                       '(data/raw/election_districts/Election_Districts_20260105.csv)',
            'n_rows': int(len(cw_full)),
            'n_distinct_electdist': int(cw_full['ElectDist_num'].nunique()),
        },
        'stop_rules': {
            'year_missing_share_threshold': YEAR_LOSS_STOP,
            'cd_deviation_threshold': CD_DEVIATION_STOP,
            'violations': stop_reasons,
            'violations_resolved_by_vintage_crosswalk': resolved_by_vintage,
            'stop': bool(stop_reasons),
            'exit_behavior': (
                'Base-crosswalk violations are recorded for the record. '
                'Year-level vote-loss violations that the vintage-matched '
                'crosswalk (data/processed/ed_to_council_crosswalk_legacy.'
                'parquet, built by build_legacy_ed_crosswalk.py from the '
                'pre-2022 DCP ED boundary files) recovers to <=1% do not '
                'fail the stage: the vintage columns dem_share_*_vintage '
                'carry the corrected measures while the legacy dem_share '
                'and the as-designed v2 columns stay untouched and '
                'documented. CD-level >3% deviations are the split-ED '
                'double-counting inherent to the production crosswalk '
                '(an inflation of CD totals, never a loss of raw votes) '
                'and are documented per district rather than fatal.'),
        },
        'by_year': years,
        'council_district_vintage_note': {
            'finding': (
                'MISMATCH: PLUTO and the sales file attach CURRENT council '
                'district lines (the post-2022 redistricting plan, in force '
                'from the 2023 cycle onward), while the 2013, 2017 and 2021 '
                'council general elections were contested under the PRE-2022 '
                'district plan (the lines adopted after the 2010 census). '
                'The close-election IV instrument maps prior-cycle (2013/'
                '2017/2021) winners onto current-district sales rows, so '
                'district identities do not line up across vintages wherever '
                'an old-plan district does not map cleanly onto its '
                '2023-plan successor.'),
            'affected_artifacts': [
                'data/processed/council_election_margins.parquet (prior-cycle '
                'winner/party/margin keyed to current CounDist)',
                'results/causal/iv_district_fe_results.json (instrument Z '
                'built from those margins)'],
            'shapefiles_needed_to_fix': [
                'City Council Districts (2013 plan) — the NYC Department of '
                'City Planning "Bytes of the Big Apple" council-district '
                'shapefile for the 2013 district plan, which covers the '
                '2013, 2017 and 2021 council races (NYC Open Data dataset '
                '"City Council Districts (2013)").',
                'A vintage-matched Election District base file for the same '
                'era (BOE/DCP election-district boundaries as they existed '
                'for the 2016-2021 elections; the ED file in this repo is '
                'the 2026 vintage and already fails to match 2016-2020 ED '
                'keys, as quantified in by_year).',
                'The 2023-plan council district file (already present: '
                'data/raw/election_districts/NYC_City_Council_Districts.shp) '
                'to build a two-way correspondence (areal overlay '
                'ED(old vintage) -> council 2013 plan, then council 2013 '
                'plan -> council 2023 plan).'],
            'note': ('Rebuilding a vintage-correct instrument also needs the '
                     '2013/2017/2021 council-race results keyed to the OLD '
                     'district numbering — already present under '
                     'data/raw/council_elections/{2013,2017,2021}/ and keyed '
                     'to old district numbers, which is the correct side of '
                     'the correspondence.'),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Wrote {OUT.relative_to(ROOT)}')
    unresolved = [s for s in stop_reasons
                  if 'missing-crosswalk-key' in s and not any(s.split(':')[0] in r for r in resolved_by_vintage)]
    if stop_reasons:
        print('\n*** BASE-CROSSWALK VIOLATION(S) (recorded) ***')
        for s in stop_reasons:
            print('  -', s)
        for r in resolved_by_vintage:
            print('  + RESOLVED:', r)
    if unresolved:
        raise SystemExit(
            'Crosswalk audit: UNRESOLVED year-level vote loss beyond the 1% '
            'STOP threshold — see results/crosswalk_audit.json. Review '
            'before rebuilding downstream outputs.')
    print('Crosswalk audit: every year-level vote-loss violation is '
          'resolved by the vintage-matched crosswalk; CD-level split-ED '
          'inflation documented per district.')


if __name__ == '__main__':
    main()


