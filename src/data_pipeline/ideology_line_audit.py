"""Ideology ballot-line audit (v2, Task A).

For the mayoral races 2017/2021/2025, tally citywide votes by ballot line
(the parenthetical in "Unit Name"), classify each line
(Dem = Democratic + Working Families; Rep = Republican + Conservative;
Other = every other candidate line), and report Other's share of all
candidate-line votes. Non-candidate counter rows (Public Counter, Manually
Counted Emergency, Absentee / Military, Affidavit, Federal, Special
Presidential, Scattered/Scattering, Blank, Void, Total Ballots) are excluded
from every candidate-vote figure.

The legacy dem_share = Dem/(Dem+Rep) silently drops every Other-class line;
this audit quantifies exactly how large that dropped share is. 2025 is
expected to show a very large Other share (Andrew M. Cuomo ran on the "Fight
and Deliver" line, roughly 40% of candidate votes) — the script flags any
year whose Other share exceeds 30% and hard-flags 2025 in the output.

Reads via the shared robust reader in src/data_pipeline/boe_ed_reader.py
(same header-prepended-row handling as build_ideology_scores.py).

Outputs:
  results/ideology_line_audit.json
  results/ideology_line_audit.csv

Run: python src/data_pipeline/ideology_line_audit.py
"""
import json
from pathlib import Path

import pandas as pd

from boe_ed_reader import read_boe_dataframe

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / 'data' / 'raw' / 'election_results'
OUT_JSON = ROOT / 'results' / 'ideology_line_audit.json'
OUT_CSV = ROOT / 'results' / 'ideology_line_audit.csv'

MAYORAL_YEARS = [2017, 2021, 2025]
FLAG_OTHER_SHARE_ABOVE = 0.30


def main():
    rows = []
    year_summary = {}
    for year in MAYORAL_YEARS:
        path = RAW_DIR / f'ed_results_{year}_mayor.csv'
        if not path.exists():
            raise SystemExit(f'MISSING INPUT: {path} — cannot build the '
                             'ballot-line audit without it.')
        df = read_boe_dataframe(path, year)
        cand = df[df['class'] != 'NonCandidate']
        noncand = df[df['class'] == 'NonCandidate']
        cand_votes = int(cand.groupby('Unit Name')['Tally'].sum().sum())
        by_unit = cand.groupby(['Unit Name', 'line', 'class'])['Tally'].sum().reset_index()
        by_unit = by_unit.sort_values('Tally', ascending=False)
        dem = int(by_unit.loc[by_unit['class'] == 'Dem', 'Tally'].sum())
        rep = int(by_unit.loc[by_unit['class'] == 'Rep', 'Tally'].sum())
        other = int(by_unit.loc[by_unit['class'] == 'Other', 'Tally'].sum())
        noncand_votes = int(noncand['Tally'].sum())
        other_share = other / cand_votes if cand_votes else float('nan')
        legacy_share = dem / (dem + rep) if (dem + rep) else float('nan')
        allcand_share = dem / cand_votes if cand_votes else float('nan')
        flagged = bool(other_share > FLAG_OTHER_SHARE_ABOVE)
        for _, r in by_unit.iterrows():
            rows.append({
                'year': year,
                'unit_name': r['Unit Name'],
                'ballot_line': r['line'],
                'class': r['class'],
                'votes': int(r['Tally']),
                'share_of_candidate_votes': round(r['Tally'] / cand_votes, 6),
            })
        year_summary[str(year)] = {
            'candidate_line_votes': cand_votes,
            'non_candidate_counter_votes': noncand_votes,
            'class_votes': {'Dem': dem, 'Rep': rep, 'Other': other},
            'other_share_of_candidate_votes': round(other_share, 6),
            'legacy_dem_share_dem_over_dem_plus_rep': round(legacy_share, 6),
            'dem_share_mayor_allcand': round(allcand_share, 6),
            'n_candidate_lines': int(by_unit.shape[0]),
            'other_share_flag': flagged,
            'flag_note': (
                'Other-class ballot lines hold more than '
                f'{FLAG_OTHER_SHARE_ABOVE:.0%} of candidate votes; the '
                'legacy Dem/(Dem+Rep) ratio drops all of them.'
            ) if flagged else None,
        }
        print(f'{year}: candidate lines={by_unit.shape[0]} '
              f'cand_votes={cand_votes:,} Dem={dem:,} Rep={rep:,} '
              f'Other={other:,} ({other_share:.2%})'
              + ('  <-- FLAGGED' if flagged else ''))

    csv_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(OUT_CSV, index=False)

    flag_2025 = year_summary.get('2025', {}).get('other_share_flag', False)
    result = {
        'audit': 'mayoral ballot-line classification, 2017/2021/2025',
        'reader': 'src/data_pipeline/boe_ed_reader.py (shared with '
                  'build_ideology_scores_v2.py and normalize_ed_results.py)',
        'classification': {
            'Dem': 'Democratic + Working Families lines',
            'Rep': 'Republican + Conservative lines',
            'Other': 'all other candidate ballot lines',
            'excluded_non_candidate_rows': [
                'Public Counter', 'Manually Counted Emergency',
                'Absentee / Military', 'Affidavit', 'Federal',
                'Special Presidential', 'Scattered', 'Scattering',
                'Blank', 'Void', 'Total Ballots'],
        },
        'by_year': year_summary,
        'by_line': rows,
        'expected_2025_flag': {
            'expected': True,
            'observed': bool(flag_2025),
            'match': bool(flag_2025),
            'note': ('2025 shows the expected very large Other share: '
                     'Andrew M. Cuomo ran citywide on the "Fight and '
                     'Deliver" line (~40% of candidate votes), which the '
                     'legacy dem_share = Dem/(Dem+Rep) silently drops.'),
        },
        'headline_warning': (
            '2025: Other-class lines hold '
            f'{year_summary["2025"]["other_share_of_candidate_votes"]:.1%} '
            'of mayoral candidate votes. The legacy dem_share '
            '(Dem/(Dem+Rep)) excludes these votes by construction; use '
            'dem_share_mayor_allcand (Dem / all candidate-line votes) for a '
            'denominator that counts every candidate line.'
        ) if flag_2025 else None,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Wrote {OUT_JSON.relative_to(ROOT)}')
    print(f'Wrote {OUT_CSV.relative_to(ROOT)}')


if __name__ == '__main__':
    main()

