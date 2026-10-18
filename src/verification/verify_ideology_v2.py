"""Independent verification of the v2 ideology measures (Round 12, Task B/C).

Re-derives every new ideology number WITHOUT importing
src/data_pipeline/build_ideology_scores_v2.py or boe_ed_reader.py: the
re-derivation reads data/processed/ed_results_clean.parquet (written by the
fixed normalize_ed_results stage, whose own reconciliation gate checks it
against the production builder) and recomputes class totals, the
ED->council aggregations (both the 2026 crosswalk and the vintage-matched
legacy crosswalk) and the council-level shares with plain pandas groupbys.

Checks (all machine-countable via _shared.record_status):
  1. ideology_by_council_v2.parquet dem_share == legacy
     ideology_by_council.parquet dem_share for every (CounDist, year) —
     the legacy column is untouched by construction.
  2. dem_share_mayor_allcand / dem_share_pres (and the *_vintage variants)
     re-derived from ed_results_clean.parquet match the v2 parquets to 1e-9.
  3. Canonical columns match the v2 parquets under the documented
     sale-year -> election-year (mayoral) and -> presidential-year mappings,
     including spot re-derivation of election_year_pres for every sale year.
  4. results/ideology_line_audit.json: 2025 Other-class share matches the
     re-derived value and the expected large-Other-share flag fired.
  5. results/crosswalk_audit.json: 2016-2021 base-crosswalk losses recorded
     (17-24%, the documented pre-existing finding), vintage crosswalk
     recovers every year to <=1%, and the audit reports the vintage
     resolution.
  6. Sample flags (sample_pre2024, is_resid_tc12, price_ge_100k,
     multi_parcel_flag) re-derived from the canonical fields match the
     stored columns exactly; building_class_category contains no
     double-space variants.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ROOT, record_status, check_val  # noqa: E402

CANON = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet'
CLEAN = ROOT / 'data' / 'processed' / 'ed_results_clean.parquet'
LEGACY_IDEO = ROOT / 'data' / 'processed' / 'ideology_by_council.parquet'
IDEO_V2 = ROOT / 'data' / 'processed' / 'ideology_by_council_v2.parquet'
IDEO_PRES = ROOT / 'data' / 'processed' / 'ideology_pres_by_council.parquet'
XWALK = ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk.parquet'
XWALK_LEGACY = ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk_legacy.parquet'
LINE_AUDIT = ROOT / 'results' / 'ideology_line_audit.json'
CWAUDIT = ROOT / 'results' / 'crosswalk_audit.json'

PRES_MAP = {2016: 2016, 2017: 2016, 2018: 2020, 2019: 2020, 2020: 2020,
            2021: 2020, 2022: 2024, 2023: 2024, 2024: 2024, 2025: 2024}
MAYOR_YEARS = [2017, 2021, 2025]
PRES_YEARS = [2016, 2020, 2024]
TOL = 1e-9

def aggregate_measures(per_ed, key):
    """per_ed: index (ElectDist, year) -> Dem/Rep/Other pivot; returns
    council-level class sums with derived shares."""
    m = per_ed.reset_index().merge(key, how='inner',
                                   left_on='ElectDist', right_on='ElectDist_num')
    g = m.groupby(['CounDist', 'year'])[['Dem', 'Rep', 'Other']].sum().reset_index()
    g['allcand'] = g[['Dem', 'Rep', 'Other']].sum(axis=1)
    g['dem_share'] = g['Dem'] / (g['Dem'] + g['Rep'])
    g['dem_share_mayor_allcand'] = g['Dem'] / g['allcand']
    return g


def cmp_canonical_col(df, src, keys, col):
    """Compare a canonical ideology column against its source parquet,
    keyed by (CounDist, year-ish). Returns mismatch count."""
    src2 = src.rename(columns={'pres_election_year': 'election_year_pres'}) \
        if 'pres_election_year' in src.columns else src
    uniq = df[keys + [col]].drop_duplicates(subset=keys)
    mm = uniq.merge(src2[keys + [col]].rename(columns={col: col + '_src'}),
                    on=keys, how='left')
    na_mismatch = int((mm[col].isna() != mm[col + '_src'].isna()).sum())
    both = mm[col].notna() & mm[col + '_src'].notna()
    val_mismatch = int(((mm.loc[both, col] - mm.loc[both, col + '_src'])
                        .abs() > 1e-9).sum())
    return na_mismatch + val_mismatch

def main():
    checks = []
    for p in [CANON, CLEAN, IDEO_V2, IDEO_PRES, XWALK, XWALK_LEGACY,
              LINE_AUDIT, CWAUDIT, LEGACY_IDEO]:
        if not p.exists():
            raise SystemExit(f'Verify-ideology-v2: MISSING INPUT {p}')

    clean = pd.read_parquet(CLEAN)
    cand = clean[clean['class'] != 'NonCandidate']

    key26 = pd.read_parquet(XWALK)
    key26['ElectDist_num'] = pd.to_numeric(key26['ElectDist'], errors='coerce')
    key26 = key26[['ElectDist_num', 'CounDist']].dropna()
    keyL = pd.read_parquet(XWALK_LEGACY)
    keyL['ElectDist_num'] = pd.to_numeric(keyL['ElectDist'], errors='coerce')
    keyL = keyL[['ElectDist_num', 'CounDist']].dropna()

    def per_ed_office(office):
        t = (cand[cand['office'] == office]
             .pivot_table(index=['ElectDist', 'year'], columns='class',
                          values='Tally', aggfunc='sum', fill_value=0))
        for c in ['Dem', 'Rep', 'Other']:
            if c not in t.columns:
                t[c] = 0
        return t

    pem = per_ed_office('mayor')
    pep = per_ed_office('president')
    v2 = pd.read_parquet(IDEO_V2)
    pres = pd.read_parquet(IDEO_PRES)

    m26 = aggregate_measures(pem.reset_index(level='year'), key26) \
        .rename(columns={'year': 'election_year'})
    mL = aggregate_measures(
        pem[pem.index.get_level_values('year').isin([2017, 2021])]
        .reset_index(level='year'), keyL).rename(columns={'year': 'election_year'})
    merged = v2.merge(m26, on=['CounDist', 'election_year'], how='outer',
                      suffixes=('', '_re'))
    checks.append(check_val(
        'v2_dem_share_matches_rederived', 0,
        int((merged['dem_share'] - merged['dem_share_re']).abs().gt(1e-9).sum()),
        tol=0))
    checks.append(check_val(
        'v2_allcand_matches_rederived', 0,
        int((merged['dem_share_mayor_allcand'] - merged['dem_share_mayor_allcand_re'])
            .abs().gt(1e-9).sum()), tol=0))
    checks.append(check_val('v2_cd_year_rows', 153, int(len(v2)), tol=0))
    mergedL = v2.merge(mL, on=['CounDist', 'election_year'], how='inner',
                      suffixes=('', '_re'))
    checks.append(check_val(
        'v2_allcand_vintage_matches_rederived', 0,
        int((mergedL['dem_share_mayor_allcand_vintage']
             - mergedL['dem_share_mayor_allcand_re']).abs().gt(1e-9).sum()), tol=0))

    legacy = pd.read_parquet(LEGACY_IDEO)
    lg = legacy.merge(v2, on=['CounDist', 'election_year'], how='outer',
                      suffixes=('_legacy', '_v2'))
    checks.append(check_val(
        'legacy_dem_share_identical_in_v2', 0,
        int((lg['dem_share_legacy'] - lg['dem_share_v2']).abs().gt(1e-12).sum()),
        tol=0))

    p26 = aggregate_measures(pep.reset_index(level='year'), key26) \
        .rename(columns={'year': 'pres_election_year'})
    pL = aggregate_measures(
        pep[pep.index.get_level_values('year').isin([2016, 2020])]
        .reset_index(level='year'), keyL).rename(columns={'year': 'pres_election_year'})
    pres_base = (p26.assign(dem_share_pres=lambda d: d['Dem'] / (d['Dem'] + d['Rep']))
                 [['CounDist', 'pres_election_year', 'dem_share_pres']])
    pm = pres.merge(pres_base, on=['CounDist', 'pres_election_year'],
                    how='outer', suffixes=('', '_re'))
    checks.append(check_val(
        'pres_dem_share_matches_rederived', 0,
        int((pm['dem_share_pres'] - pm['dem_share_pres_re']).abs().gt(1e-9).sum()),
        tol=0))
    checks.append(check_val('pres_cd_year_rows', 153, int(len(pres)), tol=0))
    pmL = pres.merge(
        pL.assign(dem_share_pres_vintage=lambda d: d['Dem'] / (d['Dem'] + d['Rep']))
        [['CounDist', 'pres_election_year', 'dem_share_pres_vintage']],
        on=['CounDist', 'pres_election_year'], how='inner', suffixes=('', '_re'))
    checks.append(check_val(
        'pres_vintage_matches_rederived', 0,
        int((pmL['dem_share_pres_vintage'] - pmL['dem_share_pres_vintage_re'])
            .abs().gt(1e-9).sum()), tol=0))

    # ---- 3: canonical columns + mapping ----------------------------------
    df = pd.read_parquet(CANON)
    bad_map = int(sum(
        1 for y, e in zip(df['sale_year'], df['election_year_pres'])
        if PRES_MAP.get(int(y), None) != int(e)))
    checks.append(check_val('canonical_election_year_pres_map', 0, bad_map, tol=0))
    checks.append(check_val(
        'canonical_allcand_matches_source', 0,
        cmp_canonical_col(df, v2, ['CounDist', 'election_year'],
                          'dem_share_mayor_allcand'), tol=0))
    checks.append(check_val(
        'canonical_allcand_vintage_matches_source', 0,
        cmp_canonical_col(df, v2, ['CounDist', 'election_year'],
                          'dem_share_mayor_allcand_vintage'), tol=0))
    checks.append(check_val(
        'canonical_pres_matches_source', 0,
        cmp_canonical_col(df, pres, ['CounDist', 'election_year_pres'],
                          'dem_share_pres'), tol=0))
    checks.append(check_val(
        'canonical_pres_vintage_matches_source', 0,
        cmp_canonical_col(df, pres, ['CounDist', 'election_year_pres'],
                          'dem_share_pres_vintage'), tol=0))
    checks.append(check_val(
        'canonical_dem_share_vintage_matches_source', 0,
        cmp_canonical_col(df, v2, ['CounDist', 'election_year'],
                          'dem_share_vintage'), tol=0))

    # ---- 4: line audit ---------------------------------------------------
    la = json.loads(LINE_AUDIT.read_text())
    y25 = cand[(cand['year'] == 2025) & (cand['office'] == 'mayor')]
    tot = int(y25['Tally'].sum())
    other25 = float(y25.loc[y25['class'] == 'Other', 'Tally'].sum())
    checks.append(check_val(
        'line_audit_2025_other_share', round(other25 / tot, 6),
        la['by_year']['2025']['other_share_of_candidate_votes'], tol=1e-6))
    checks.append(check_val(
        'line_audit_2025_expected_flag', True,
        bool(la['by_year']['2025']['other_share_flag'])))

    # ---- 5: crosswalk audit resolution ----------------------------------
    cw = json.loads(CWAUDIT.read_text())
    unresolved_years = [
        y for y, r in cw['by_year'].items()
        if r['missing_key_share'] > 0.01
        and (r['missing_after_vintage_xwalk_share_pct'] is None
             or r['missing_after_vintage_xwalk_share_pct'] > 1.0)]
    checks.append(check_val('crosswalk_audit_unresolved_years', 0,
                            len(unresolved_years), tol=0))
    checks.append(check_val(
        'crosswalk_audit_2017_loss_documented', True,
        bool(cw['by_year']['2017']['missing_key_share'] > 0.17)))
    checks.append(check_val(
        'crosswalk_audit_vintage_2017_recovery', True,
        bool(cw['by_year']['2017']['missing_after_vintage_xwalk_share_pct'] is not None
             and cw['by_year']['2017']['missing_after_vintage_xwalk_share_pct'] <= 1.0)))

    # ---- 6: sample flags --------------------------------------------------
    tc = pd.to_numeric(df['TAX CLASS AT TIME OF SALE'], errors='coerce')
    checks.append(check_val(
        'flag_is_resid_tc12', 0,
        int((df['is_resid_tc12'] != tc.isin([1, 2])).sum()), tol=0))
    checks.append(check_val(
        'flag_price_ge_100k', 0,
        int((df['price_ge_100k'] != (df['sale_price'] >= 100000)).sum()), tol=0))
    checks.append(check_val(
        'flag_sample_pre2024', 0,
        int((df['sample_pre2024'] != (df['sale_year'] <= 2023)).sum()), tol=0))
    _size = df.groupby(['BOROUGH', 'BLOCK', 'sale_date', 'sale_price'],
                       dropna=False)['BOROUGH'].transform('size')
    checks.append(check_val(
        'flag_multi_parcel', 0,
        int((df['multi_parcel_flag'] != (_size > 1)).sum()), tol=0))
    bcc = df['building_class_category'].astype(str)
    checks.append(check_val(
        'building_class_category_no_double_space', 0,
        int(bcc.str.contains('  ').sum()), tol=0))
    norm_ref = (df['BUILDING CLASS CATEGORY'].astype(str)
                .str.strip().str.replace(r'\s+', ' ', regex=True))
    checks.append(check_val(
        'building_class_category_matches_normalizer', 0,
        int((bcc != norm_ref).sum()), tol=0))

    record_status('verify_ideology_v2', checks,
                  note='Independent re-derivation of the v2 ideology measures '
                       '(allcand, pres, vintage variants), the canonical merge '
                       'mappings, the ballot-line audit, the crosswalk-audit '
                       'vintage resolution, and the Task-D sample flags.')


if __name__ == '__main__':
    main()



