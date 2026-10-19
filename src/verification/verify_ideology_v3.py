"""Independent verification of the Round-13 _aw/_dom ideology outputs (R1).

Re-derives every new number WITHOUT importing build_ideology_scores_v3.py
or boe_ed_reader.py: the re-derivation reads
data/processed/ed_results_clean.parquet (whose own reconciliation gate in
the v2 stage checks it against the production builder) and recomputes:

  1. the area-weighted (_aw) and dominant-district (_dom) council-level
     measures with plain pandas groupbys, using normalized
     intersection-area weights (legacy-vintage crosswalk for
     2016/2017/2020/2021, production crosswalk for 2024/2025) — compared
     to ideology_by_council_v3.parquet / ideology_pres_by_council_v3.parquet;
  2. the weight-normalization identity (each ED's weights sum to exactly 1);
  3. the per-year reconciliation in results/crosswalk_audit_v3.json:
     sum over CDs of weighted votes == raw votes on matched EDs
     (overcount = 0.00%, loss = 0.00%), re-derived from the clean ED file;
  4. the _aw == _vintage_aw documented identity;
  5. canonical v3: 73 columns, all 9 new columns match the v3 parquets
     under the documented merge keys, a spot-set of legacy columns is
     byte-identical to canonical v2, and all 9 columns are blacklisted
     in config/feature_blacklist.yaml;
  6. the overall per-CD correlation recorded in the audit for one
     representative measure pair, re-derived.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ROOT, record_status, check_val  # noqa: E402

CANON_V3 = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v3.parquet'
CANON_V2 = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet'
CLEAN = ROOT / 'data' / 'processed' / 'ed_results_clean.parquet'
IDEO_V2 = ROOT / 'data' / 'processed' / 'ideology_by_council_v2.parquet'
IDEO_PRES_V2 = ROOT / 'data' / 'processed' / 'ideology_pres_by_council.parquet'
IDEO_V3 = ROOT / 'data' / 'processed' / 'ideology_by_council_v3.parquet'
IDEO_PRES_V3 = ROOT / 'data' / 'processed' / 'ideology_pres_by_council_v3.parquet'
XWALK = ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk.parquet'
XWALK_LEGACY = ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk_legacy.parquet'
CWAUDIT3 = ROOT / 'results' / 'crosswalk_audit_v3.json'
BLACKLIST = ROOT / 'config' / 'feature_blacklist.yaml'
PROV = ROOT / 'results' / 'canonical_v3_provenance.json'

MAYOR_YEARS = [2017, 2021, 2025]
PRES_YEARS = [2016, 2020, 2024]
LEGACY_XW_YEARS = {2016, 2017, 2020, 2021}
TOL = 1e-9


def load_xw(path):
    k = pd.read_parquet(path)
    k['ElectDist_num'] = pd.to_numeric(k['ElectDist'], errors='coerce')
    return k[['ElectDist_num', 'CounDist', 'area']].dropna()


def xw_for_year(year):
    return load_xw(XWALK_LEGACY if year in LEGACY_XW_YEARS else XWALK)


def weighted_agg(per_ed_year, xw):
    """per_ed_year: DataFrame indexed by ElectDist with vote columns."""
    k = xw.copy()
    tot = k.groupby('ElectDist_num')['area'].transform('sum')
    k['w'] = (k['area'] / tot)
    bad = ~k['w'].notna() | (k['w'] <= 0) | ~tot.notna() | (tot <= 0)
    if bad.any():
        n = k.groupby('ElectDist_num')['CounDist'].transform('size')
        k.loc[bad, 'w'] = (1.0 / n[bad])
    k['w'] = k['w'].astype(float)
    chk = k.groupby('ElectDist_num')['w'].sum()
    norm_ok = bool(((chk - 1.0).abs() < 1e-9).all())
    m = per_ed_year.merge(k[['ElectDist_num', 'CounDist', 'w']],
                          how='inner', left_on='ElectDist',
                          right_on='ElectDist_num')
    return m, norm_ok


def dom_agg(per_ed_year, xw):
    k = xw.copy()
    k['_area'] = k['area'].fillna(-1.0)
    k = k.sort_values(['ElectDist_num', '_area', 'CounDist'],
                      ascending=[True, False, True])
    dom = k.groupby('ElectDist_num', as_index=False).first()[['ElectDist_num', 'CounDist']]
    m = per_ed_year.merge(dom, how='inner', left_on='ElectDist',
                          right_on='ElectDist_num')
    return m



def main():
    checks = []
    for p in [CANON_V3, CANON_V2, CLEAN, IDEO_V2, IDEO_PRES_V2, IDEO_V3,
              IDEO_PRES_V3, XWALK, XWALK_LEGACY, CWAUDIT3, BLACKLIST, PROV]:
        if not p.exists():
            raise SystemExit(f'Verify-ideology-v3: MISSING INPUT {p}')

    clean = pd.read_parquet(CLEAN)
    cand = clean[clean['class'] != 'NonCandidate']

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
    v3 = pd.read_parquet(IDEO_V3)
    pv3 = pd.read_parquet(IDEO_PRES_V3)
    cw3 = json.loads(CWAUDIT3.read_text(encoding='utf-8'))

    # ---- 1: re-derive the _aw measures (mayoral) --------------------------
    aw_frames, dom_frames = [], []
    norm_ok_all = True
    recon = {}
    for year in MAYOR_YEARS:
        t = pem.xs(year, level='year').reset_index()
        t['allcand'] = t[['Dem', 'Rep', 'Other']].sum(axis=1)
        m, norm_ok = weighted_agg(t, xw_for_year(year))
        norm_ok_all &= norm_ok
        w = m[['CounDist']].copy()
        for c in ['Dem', 'Rep', 'allcand']:
            w[c] = (m[c] * m['w']).values
        aw = w.groupby('CounDist')[['Dem', 'Rep', 'allcand']].sum().reset_index()
        aw['election_year'] = year
        aw_frames.append(aw)
        dm = dom_agg(t, xw_for_year(year))
        dd = dm.groupby('CounDist')[['Dem', 'Rep', 'allcand']].sum().reset_index()
        dd['election_year'] = year
        dom_frames.append(dd)
        raw_matched = float(t.loc[t['ElectDist'].isin(
            xw_for_year(year)['ElectDist_num']), 'allcand'].sum())
        recon[str(year)] = float(aw['allcand'].sum()) - raw_matched

    aw = pd.concat(aw_frames, ignore_index=True)
    aw['dem_share_aw'] = aw['Dem'] / (aw['Dem'] + aw['Rep'])
    aw['dem_share_mayor_allcand_aw'] = aw['Dem'] / aw['allcand']
    dm = pd.concat(dom_frames, ignore_index=True)
    dm = dm.rename(columns={'Dem': 'Dem_d', 'Rep': 'Rep_d', 'allcand': 'allcand_d'})
    aw = aw.merge(dm, on=['CounDist', 'election_year'], how='inner')
    aw['dem_share_vintage_dom'] = aw['Dem_d'] / (aw['Dem_d'] + aw['Rep_d'])
    aw['dem_share_mayor_allcand_vintage_dom'] = aw['Dem_d'] / aw['allcand_d']

    mg = v3.merge(aw, on=['CounDist', 'election_year'], how='inner',
                  suffixes=('', '_re'))
    for col in ['dem_share_aw', 'dem_share_mayor_allcand_aw',
                'dem_share_vintage_dom', 'dem_share_mayor_allcand_vintage_dom']:
        checks.append(check_val(f'v3_{col}_matches_rederived', 0,
                                int((mg[col] - mg[col + '_re'])
                                    .abs().gt(TOL).sum()), tol=0))
    checks.append(check_val('v3_mayoral_cd_year_rows', 153, int(len(v3)), tol=0))
    checks.append(check_val('v3_weights_sum_to_one', True, norm_ok_all))

    # ---- 1b: re-derive the _aw measures (presidential) ---------------------
    paw_frames, pdom_frames = [], []
    for year in PRES_YEARS:
        t = pep.xs(year, level='year').reset_index()
        m, norm_ok = weighted_agg(t, xw_for_year(year))
        norm_ok_all &= norm_ok
        w = m[['CounDist']].copy()
        w['dem_pres'] = m['Dem'] * m['w']
        w['rep_pres'] = m['Rep'] * m['w']
        paw = w.groupby('CounDist')[['dem_pres', 'rep_pres']].sum().reset_index()
        paw['pres_election_year'] = year
        paw_frames.append(paw)
        dmo = dom_agg(t, xw_for_year(year))
        pdm = dmo.groupby('CounDist')[['Dem', 'Rep']].sum().reset_index()
        pdm['pres_election_year'] = year
        pdom_frames.append(pdm)
        raw_matched = float(t.loc[t['ElectDist'].isin(
            xw_for_year(year)['ElectDist_num']), ['Dem', 'Rep']].sum().sum())
        recon[str(year)] = float(paw['dem_pres'].sum()
                                 + paw['rep_pres'].sum()) - raw_matched

    paw = pd.concat(paw_frames, ignore_index=True)
    paw['dem_share_pres_aw'] = paw['dem_pres'] / (paw['dem_pres'] + paw['rep_pres'])
    pdm = pd.concat(pdom_frames, ignore_index=True)
    pdm = pdm.rename(columns={'Dem': 'Dem_d', 'Rep': 'Rep_d'})
    paw = paw.merge(pdm, on=['CounDist', 'pres_election_year'], how='inner')
    paw['dem_share_pres_vintage_dom'] = paw['Dem_d'] / (paw['Dem_d'] + paw['Rep_d'])

    mpg = pv3.merge(paw, on=['CounDist', 'pres_election_year'], how='inner',
                    suffixes=('', '_re'))
    for col in ['dem_share_pres_aw', 'dem_share_pres_vintage_dom']:
        checks.append(check_val(f'v3_{col}_matches_rederived', 0,
                                int((mpg[col] - mpg[col + '_re'])
                                    .abs().gt(TOL).sum()), tol=0))
    checks.append(check_val('v3_pres_cd_year_rows', 153, int(len(pv3)), tol=0))

    # ---- 2: reconciliation fields in the audit -----------------------------
    for y, rec in cw3['by_year'].items():
        checks.append(check_val(f'cw3_{y}_overcount_zero', 0.0,
                                rec['overcount_pct_aw_vs_matched'], tol=1e-6))
        checks.append(check_val(f'cw3_{y}_loss_zero', 0.0,
                                rec['loss_pct_aw_vs_matched'], tol=1e-6))
        checks.append(check_val(f'cw3_{y}_overcount_dom_zero', 0.0,
                                rec['overcount_pct_dom_vs_matched'], tol=1e-6))
        checks.append(check_val(f'cw3_{y}_recon_rederived', 0.0,
                                round(recon[y], 2), tol=0.01))
    checks.append(check_val('cw3_crosswalk_2017_is_legacy', True,
                            'ed_to_council_crosswalk_legacy.parquet'
                            == cw3['by_year']['2017']['crosswalk_used']))
    checks.append(check_val('cw3_crosswalk_2024_is_production', True,
                            'ed_to_council_crosswalk.parquet'
                            == cw3['by_year']['2024']['crosswalk_used']))

    # ---- 3: _aw == _vintage_aw documented identity ------------------------
    checks.append(check_val('v3_aw_equals_vintage_aw_mayoral', 0,
                            int((v3['dem_share_aw'] != v3['dem_share_vintage_aw']).sum()
                                + (v3['dem_share_mayor_allcand_aw']
                                   != v3['dem_share_mayor_allcand_vintage_aw']).sum()),
                            tol=0))
    checks.append(check_val('v3_aw_equals_vintage_aw_pres', 0,
                            int((pv3['dem_share_pres_aw']
                                 != pv3['dem_share_pres_vintage_aw']).sum()), tol=0))
    checks.append(check_val('cw3_identity_documented', True,
                            bool(cw3['weights_methodology']
                                 .get('aw_equals_vintage_aw_identity'))))

    # ---- 4: correlation spot-check ----------------------------------------
    old_m = pd.read_parquet(IDEO_V2)
    j = v3.merge(old_m, on=['CounDist', 'election_year'], how='inner')
    r = float(np.corrcoef(j['dem_share_aw'], j['dem_share'])[0, 1])
    checks.append(check_val(
        'cw3_overall_corr_dem_share_aw', round(r, 6),
        cw3['per_cd_correlations']['mayoral_measures']['dem_share_aw']
        ['overall_correlation'], tol=1e-6))

    # ---- 5: canonical v3 ----------------------------------------------------
    df3 = pd.read_parquet(CANON_V3)
    df2 = pd.read_parquet(CANON_V2)
    checks.append(check_val('canon_v3_rows', int(len(df2)), int(len(df3)), tol=0))
    checks.append(check_val('canon_v3_cols', 73, int(df3.shape[1]), tol=0))

    def cmp_col(df, src, keys, col):
        uniq = df[keys + [col]].drop_duplicates(subset=keys)
        mm = uniq.merge(src[keys + [col]].rename(columns={col: col + '_src'}),
                        on=keys, how='left')
        na_m = int((mm[col].isna() != mm[col + '_src'].isna()).sum())
        both = mm[col].notna() & mm[col + '_src'].notna()
        val_m = int(((mm.loc[both, col] - mm.loc[both, col + '_src'])
                     .abs() > TOL).sum())
        return na_m + val_m

    for col in ['dem_share_aw', 'dem_share_mayor_allcand_aw',
                'dem_share_vintage_aw', 'dem_share_mayor_allcand_vintage_aw',
                'dem_share_vintage_dom', 'dem_share_mayor_allcand_vintage_dom']:
        checks.append(check_val(f'canon_v3_{col}', 0,
                                cmp_col(df3, v3, ['CounDist', 'election_year'],
                                        col), tol=0))
    for col in ['dem_share_pres_aw', 'dem_share_pres_vintage_aw',
                'dem_share_pres_vintage_dom']:
        pv3_r = pv3.rename(columns={'pres_election_year': 'election_year_pres'})
        checks.append(check_val(f'canon_v3_{col}', 0,
                                cmp_col(df3, pv3_r,
                                        ['CounDist', 'election_year_pres'],
                                        col), tol=0))

    # legacy columns byte-identical to v2 (spot set incl. all legacy ideology)
    spot = ['dem_share', 'dem_share_pres', 'dem_share_vintage',
            'dem_share_mayor_allcand_vintage', 'dem_share_pres_vintage',
            'target_log_price', 'CounDist', 'sale_year', 'election_year',
            'election_year_pres', 'sample_pre2024']
    n_changed = sum(0 if df3[c].equals(df2[c]) else 1 for c in spot)
    checks.append(check_val('canon_v3_legacy_columns_unchanged', 0,
                            n_changed, tol=0))

    # blacklist registration
    bl = BLACKLIST.read_text(encoding='utf-8')
    missing_bl = [c for c in ['dem_share_aw', 'dem_share_mayor_allcand_aw',
                              'dem_share_pres_aw', 'dem_share_vintage_aw',
                              'dem_share_mayor_allcand_vintage_aw',
                              'dem_share_pres_vintage_aw',
                              'dem_share_vintage_dom',
                              'dem_share_mayor_allcand_vintage_dom',
                              'dem_share_pres_vintage_dom']
                  if f'- {c}' not in bl]
    checks.append(check_val('blacklist_all_nine_new_measures', 0,
                            len(missing_bl), tol=0))

    prov = json.loads(PROV.read_text(encoding='utf-8'))
    checks.append(check_val('prov_new_columns_count', 9,
                            len(prov['new_columns']), tol=0))
    checks.append(check_val('prov_blacklisted_flag', True,
                            bool(prov['new_columns_blacklisted_from_models'])))

    record_status('verify_ideology_v3', checks,
                  note='Independent re-derivation of the Round-13 _aw/_dom '
                       'measures, the split-ED reconciliation (overcount = '
                       '0.00%, loss = 0.00% per year), the documented '
                       '_aw == _vintage_aw identity, canonical v3 column '
                       'integrity, and blacklist registration.')


if __name__ == '__main__':
    main()

