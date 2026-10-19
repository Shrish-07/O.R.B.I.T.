"""Crosswalk audit v3: split-ED double-counting reconciliation (Round 13, R1).

For EVERY raw ED-level election file (mayoral 2017/2021/2025, presidential
2016/2020/2024) this audit reconciles the new area-weighted (_aw) and
dominant-district (_dom) allocations against the raw ED-level candidate
votes, using the same year->crosswalk mapping as build_ideology_scores_v3
(legacy-vintage crosswalk for 2016/2017/2020/2021, production 2026
crosswalk for 2024/2025):

  * overcount_pct  = 100 * (sum over CDs of weighted votes - raw votes on
    crosswalk-matched EDs) / matched votes   -> 0.00% by construction
    (each matched ED's weights sum to exactly 1);
  * loss_pct       = 100 * (matched votes - sum over CDs of weighted
    votes) / matched votes                    -> 0.00% by construction;
  * coverage_pct   = matched votes / all raw ED candidate votes (the
    production crosswalk has no key for a small share of 2024 ED votes;
    reported explicitly so nothing is hidden);
  * the legacy full-to-each assignment is reported alongside for
    contrast (summed CD votes / matched votes, the 1.31x style inflation
    quantified in results/crosswalk_audit.json).

Also reports, for every new _aw/_dom measure, the per-CD Pearson
correlation with its full-to-each counterpart measure across the CD's
election waves (and, supplementarily, per-wave across-district and
overall correlations).

Output: results/crosswalk_audit_v3.json

Run: python src/data_pipeline/crosswalk_audit_v3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boe_ed_reader import read_boe_dataframe  # noqa: E402
from build_ideology_scores_v3 import (  # noqa: E402
    MAYORAL_YEARS, PRES_YEARS, YEAR_USES_LEGACY_XW, keys_for_year,
    weighted_key, dominant_key, load_crosswalk)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / 'data' / 'raw' / 'election_results'
IDEO_V2 = ROOT / 'data' / 'processed' / 'ideology_by_council_v2.parquet'
IDEO_PRES_V2 = ROOT / 'data' / 'processed' / 'ideology_pres_by_council.parquet'
IDEO_V3 = ROOT / 'data' / 'processed' / 'ideology_by_council_v3.parquet'
IDEO_PRES_V3 = ROOT / 'data' / 'processed' / 'ideology_pres_by_council_v3.parquet'
OUT = ROOT / 'results' / 'crosswalk_audit_v3.json'


def per_ed_candidate_votes(year, office):
    df = read_boe_dataframe(RAW_DIR / f'ed_results_{year}_{office}.csv', year)
    cand = df[df['class'] != 'NonCandidate']
    return cand.groupby('ElectDist')['Tally'].sum().rename('votes').reset_index()


def audit_year(year, office):
    per_ed = per_ed_candidate_votes(year, office)
    wkey, dkey, xw_name = keys_for_year(year)
    raw_all = float(per_ed['votes'].sum())
    matched = per_ed[per_ed['ElectDist'].isin(wkey['ElectDist_num'])]
    raw_matched = float(matched['votes'].sum())

    m = matched.merge(wkey, how='inner', left_on='ElectDist',
                      right_on='ElectDist_num')
    m['aw_votes'] = m['votes'] * m['w']
    aw_cd = m.groupby('CounDist')['aw_votes'].sum()
    sum_cd_aw = float(aw_cd.sum())

    md = matched.merge(dkey, how='inner', left_on='ElectDist',
                       right_on='ElectDist_num')
    dom_cd = md.groupby('CounDist')['votes'].sum()
    sum_cd_dom = float(dom_cd.sum())

    full_cd = m.groupby('CounDist')['votes'].sum()
    sum_cd_full = float(full_cd.sum())

    overcount_pct = 100.0 * (sum_cd_aw - raw_matched) / raw_matched
    loss_pct = 100.0 * (raw_matched - sum_cd_aw) / raw_matched
    overcount_dom_pct = 100.0 * (sum_cd_dom - raw_matched) / raw_matched
    loss_dom_pct = 100.0 * (raw_matched - sum_cd_dom) / raw_matched

    cd_rows = {}
    for cd in sorted(set(aw_cd.index) | set(dom_cd.index) | set(full_cd.index)):
        cd_rows[str(int(cd))] = {
            'aw_votes': round(float(aw_cd.get(cd, 0.0)), 2),
            'dom_votes': round(float(dom_cd.get(cd, 0.0)), 2),
            'full_to_each_votes': round(float(full_cd.get(cd, 0.0)), 2),
        }

    return {
        'office': office,
        'crosswalk_used': xw_name,
        'raw_candidate_votes_all_eds': int(round(raw_all)),
        'raw_candidate_votes_matched_eds': int(round(raw_matched)),
        'unmatched_ed_votes': int(round(raw_all - raw_matched)),
        'coverage_pct_of_raw': round(100.0 * raw_matched / raw_all, 4),
        'sum_cd_aw_votes': round(sum_cd_aw, 2),
        'sum_cd_dom_votes': round(sum_cd_dom, 2),
        'sum_cd_full_to_each_votes': round(sum_cd_full, 2),
        'overcount_pct_aw_vs_matched': round(overcount_pct, 4),
        'loss_pct_aw_vs_matched': round(loss_pct, 4),
        'overcount_pct_dom_vs_matched': round(overcount_dom_pct, 4),
        'loss_pct_dom_vs_matched': round(loss_dom_pct, 4),
        'full_to_each_overcount_ratio_vs_matched': (
            round(sum_cd_full / raw_matched, 6) if raw_matched else None),
        'per_cd_votes': cd_rows,
    }


CORR_PAIRS_MAYORAL = [
    ('dem_share_aw', 'dem_share'),
    ('dem_share_mayor_allcand_aw', 'dem_share_mayor_allcand'),
    ('dem_share_vintage_aw', 'dem_share_vintage'),
    ('dem_share_mayor_allcand_vintage_aw', 'dem_share_mayor_allcand_vintage'),
    ('dem_share_vintage_dom', 'dem_share_vintage'),
    ('dem_share_mayor_allcand_vintage_dom', 'dem_share_mayor_allcand_vintage'),
]
CORR_PAIRS_PRES = [
    ('dem_share_pres_aw', 'dem_share_pres'),
    ('dem_share_pres_vintage_aw', 'dem_share_pres_vintage'),
    ('dem_share_pres_vintage_dom', 'dem_share_pres_vintage'),
]


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def correlation_report(new, old, wave_col, pairs):
    """Per-CD (across waves), per-wave (across districts), overall r."""
    rep = {}
    j = new.merge(old, on=['CounDist', wave_col], how='inner')
    for new_m, old_m in pairs:
        sub = j.dropna(subset=[new_m, old_m])
        per_cd, per_wave = {}, {}
        for cd, g in sub.groupby('CounDist'):
            r = safe_corr(g[new_m], g[old_m])
            if r is not None:
                per_cd[str(int(cd))] = round(r, 4)
        for wv, g in sub.groupby(wave_col):
            r = safe_corr(g[new_m], g[old_m])
            if r is not None:
                per_wave[str(int(wv))] = round(r, 4)
        rep[new_m] = {
            'counterpart_full_to_each': old_m,
            'n_cd_wave_pairs': int(len(sub)),
            'per_cd_correlation_across_waves': per_cd,
            'per_wave_correlation_across_districts': per_wave,
            'overall_correlation': round(safe_corr(sub[new_m], sub[old_m]), 6),
        }
    return rep


def main():
    for p in [IDEO_V2, IDEO_PRES_V2]:
        if not p.exists():
            raise SystemExit(f'MISSING INPUT: {p} — run build_ideology_scores_v2 first.')
    if not (IDEO_V3.exists() and IDEO_PRES_V3.exists()):
        raise SystemExit('MISSING INPUT: data/processed/ideology_by_council_v3.parquet '
                          '— run build_ideology_scores_v3 first.')

    years = {}
    for year in MAYORAL_YEARS:
        years[str(year)] = audit_year(year, 'mayor')
        r = years[str(year)]
        print(f'{year} (mayor, {r["crosswalk_used"]}): coverage='
              f'{r["coverage_pct_of_raw"]:.4f}% | overcount_aw='
              f'{r["overcount_pct_aw_vs_matched"]:+.4f}% loss_aw='
              f'{r["loss_pct_aw_vs_matched"]:+.4f}% | full-to-each ratio='
              f'{r["full_to_each_overcount_ratio_vs_matched"]}')
    for year in PRES_YEARS:
        years[str(year)] = audit_year(year, 'president')
        r = years[str(year)]
        print(f'{year} (pres, {r["crosswalk_used"]}): coverage='
              f'{r["coverage_pct_of_raw"]:.4f}% | overcount_aw='
              f'{r["overcount_pct_aw_vs_matched"]:+.4f}% loss_aw='
              f'{r["loss_pct_aw_vs_matched"]:+.4f}% | full-to-each ratio='
              f'{r["full_to_each_overcount_ratio_vs_matched"]}')

    new_m = pd.read_parquet(IDEO_V3)
    old_m = pd.read_parquet(IDEO_V2)
    new_p = pd.read_parquet(IDEO_PRES_V3)
    old_p = pd.read_parquet(IDEO_PRES_V2)

    result = {
        'audit': ('Split-ED double-counting reconciliation for the '
                  'area-weighted (_aw) and dominant-district (_dom) '
                  'measures (Round 13, R1)'),
        'weights_methodology': {
            'definition': ('w(ED,CD) = intersection_area(ED,CD) / sum over '
                           'CDs of intersection_area(ED,CD), from the '
                           "crosswalk 'area' column; each ED's weights sum "
                           'to exactly 1; equal split across the ED\'s '
                           'districts if the area is missing or non-positive'),
            'crosswalk_by_election_year': {
                '2016/2017/2020/2021': 'ed_to_council_crosswalk_legacy.parquet',
                '2024/2025': 'ed_to_council_crosswalk.parquet'},
            'dominant_district_rule': ('ED assigned wholly to the CD with the '
                                       'largest overlap area; ties resolve '
                                       'to the smaller CounDist'),
            'share_convention': 'ratios of weighted vote totals',
            'aw_equals_vintage_aw_identity': (
                'Because the _aw crosswalk choice is vintage-matched for the '
                'pre-2024 years by construction, each _aw measure is '
                'numerically identical to its _vintage_aw counterpart; both '
                'columns are written as the task specifies and both are '
                'verified.'),
        },
        'reconciliation_convention': (
            'overcount/loss are computed against raw candidate votes on '
            'crosswalk-MATCHED EDs (weights sum to 1 per matched ED, so '
            'both are exactly 0). The share of raw ED votes on keys absent '
            'from the chosen crosswalk is reported separately as '
            'coverage_pct_of_raw / unmatched_ed_votes and is a pre-existing '
            'crosswalk-key property, not a property of the weighting.'),
        'by_year': years,
        'per_cd_correlations': {
            'mayoral_measures': correlation_report(
                new_m, old_m, 'election_year', CORR_PAIRS_MAYORAL),
            'presidential_measures': correlation_report(
                new_p, old_p, 'pres_election_year', CORR_PAIRS_PRES),
        },
    }

    bad = [f'{y}: overcount {r["overcount_pct_aw_vs_matched"]}%, '
           f'loss {r["loss_pct_aw_vs_matched"]}%'
           for y, r in years.items()
           if abs(r['overcount_pct_aw_vs_matched']) > 1e-6
           or abs(r['loss_pct_aw_vs_matched']) > 1e-6]
    if bad:
        raise SystemExit('RECONCILIATION FAILED: ' + '; '.join(bad))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Wrote {OUT.relative_to(ROOT)}')
    print('Reconciliation: overcount = 0.00% and loss = 0.00% for every '
          'year (vs matched ED votes); per-CD correlations recorded.')


if __name__ == '__main__':
    main()
