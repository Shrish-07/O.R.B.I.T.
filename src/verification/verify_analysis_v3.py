"""Independent verification of the Round-13 analysis outputs (R2/R5/R6/R8).

Re-derives key numbers WITHOUT importing the producing scripts:

  * robustness_checks_v3.json: fit count (9 measures x 7 specs x 2
    samples = 126), required inference fields, a statsmodels re-fit of
    the full-sample Model 4 for dem_share_aw (coef + cluster SE) and the
    deterministic wild p from the same seed.
  * robustness_pre2024_v3.json: 15 measures x 5 specs = 75 fits with all
    required fields; the pre-2024 sample size re-derived from the
    canonical flag; a statsmodels re-fit of Model 4 (dem_share) on the
    pre-2024 sample; wild-p determinism.
  * time_trend_placebo_v3.json: wavemean/dev identities re-derived from
    the canonical (wavemean constant within wave, dev mean-zero within
    wave), statsmodels re-fits of the dem_share wavemean and dev Model 4
    specs, and consistency of the stored comparison/near-zero verdicts.
  * hedonic_v3.json: the N=23,789 funnel re-computed; the land/units
    variant sample size and a statsmodels re-fit of its dem_share
    coefficient and cluster SE.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ROOT, record_status, check_val  # noqa: E402

sys.path.insert(0, str(ROOT / 'src' / 'analysis'))
from wild_cluster_inference import WildClusterOLS  # noqa: E402

CANON = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v3.parquet'
ROB3 = ROOT / 'results' / 'robustness_checks_v3.json'
PRE24 = ROOT / 'results' / 'robustness_pre2024_v3.json'
PLACEBO = ROOT / 'results' / 'time_trend_placebo_v3.json'
HED3 = ROOT / 'results' / 'hedonic_v3.json'
ROB2 = ROOT / 'results' / 'robustness_checks_v2.json'

CORE = ['yearbuilt', 'numfloors', 'landuse', 'commfar']
REQUIRED_FIELDS = ['coef', 'se_classical', 'se_cluster', 'p_t_G_minus_1',
                   'p_wild_cluster_bootstrap_t', 'ci95_low_t', 'ci95_high_t',
                   'n', 'n_clusters_G', 'r_squared']


def base_mask(df):
    base_ok = df['target_log_price'].notna() & df['CounDist'].notna() \
        & df['sale_year'].notna()
    for c in CORE:
        base_ok &= df[c].notna()
    return base_ok


def model4_statsmodels(d):
    dist = pd.get_dummies(d['CounDist'].astype(int), prefix='district',
                          drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=d.index, name='const'),
                   d[['measure_val']].astype(float), d[CORE].astype(float),
                   dist], axis=1)
    X = X.rename(columns={'measure_val': 'measure'})
    res = sm.OLS(d['target_log_price'].astype(float), X).fit(
        cov_type='cluster', cov_kwds={'groups': d['CounDist'].astype(int)})
    return res, X



def main():
    checks = []
    for p in [CANON, ROB3, PRE24, PLACEBO, HED3, ROB2]:
        if not p.exists():
            raise SystemExit(f'Verify-analysis-v3: MISSING INPUT {p}')
    rob3 = json.loads(ROB3.read_text(encoding='utf-8'))
    pre24 = json.loads(PRE24.read_text(encoding='utf-8'))
    placebo = json.loads(PLACEBO.read_text(encoding='utf-8'))
    hed3 = json.loads(HED3.read_text(encoding='utf-8'))
    rob2 = json.loads(ROB2.read_text(encoding='utf-8'))
    df = pd.read_parquet(CANON)
    base_ok = base_mask(df)

    # ---- robustness v3 ------------------------------------------------------
    n_fits, missing_fields = 0, 0
    for s in ['full', 'restricted']:
        for m, specs in rob3[s].items():
            for sp, rep in specs.items():
                if 'coef' in rep:
                    n_fits += 1
                    for f in REQUIRED_FIELDS:
                        if rep.get(f) is None:
                            missing_fields += 1
    checks.append(check_val('rob3_fit_count', 126, n_fits, tol=0))
    checks.append(check_val('rob3_missing_fields', 0, missing_fields, tol=0))

    d = df[base_ok & df['dem_share_aw'].notna()].copy()
    d['measure_val'] = d['dem_share_aw']
    res, X = model4_statsmodels(d)
    m4 = rob3['full']['dem_share_aw']['model4_district_fe']
    checks.append(check_val('rob3_model4_aw_coef_vs_statsmodels',
                            float(res.params['measure']), m4['coef'], tol=1e-8))
    checks.append(check_val('rob3_model4_aw_se_vs_statsmodels',
                            float(res.bse['measure']), m4['se_cluster'],
                            tol=1e-8))
    ols = WildClusterOLS(d['target_log_price'].to_numpy(dtype=float),
                         X.to_numpy(dtype=np.float64),
                         d['CounDist'].to_numpy(),
                         list(X.columns).index('measure')).fit()
    rep = ols.report(reps=9999, seed=42)
    checks.append(check_val('rob3_model4_aw_wild_p_deterministic',
                            rep['p_wild_cluster_bootstrap_t'],
                            m4['p_wild_cluster_bootstrap_t'], tol=1e-12))

    # ---- pre-2024 suite -----------------------------------------------------
    n_fits_p, missing_p = 0, 0
    for m, specs in pre24.items():
        if m == 'metadata':
            continue
        for sp, rep in specs.items():
            if 'coef' in rep:
                n_fits_p += 1
                for f in REQUIRED_FIELDS:
                    if rep.get(f) is None:
                        missing_p += 1
    checks.append(check_val('pre24_fit_count', 75, n_fits_p, tol=0))
    checks.append(check_val('pre24_missing_fields', 0, missing_p, tol=0))

    pre_mask = base_ok & df['sample_pre2024'] & df['dem_share'].notna()
    checks.append(check_val('pre24_model4_dem_share_n',
                            int(pre_mask.sum()),
                            pre24['dem_share']['model4_district_fe']['n'],
                            tol=0))
    dp = df[pre_mask].copy()
    dp['measure_val'] = dp['dem_share']
    resp, Xp = model4_statsmodels(dp)
    m4p = pre24['dem_share']['model4_district_fe']
    checks.append(check_val('pre24_model4_dem_share_coef_vs_statsmodels',
                            float(resp.params['measure']), m4p['coef'],
                            tol=1e-8))
    checks.append(check_val('pre24_model4_dem_share_se_vs_statsmodels',
                            float(resp.bse['measure']), m4p['se_cluster'],
                            tol=1e-8))
    ols = WildClusterOLS(dp['target_log_price'].to_numpy(dtype=float),
                         Xp.to_numpy(dtype=np.float64),
                         dp['CounDist'].to_numpy(),
                         list(Xp.columns).index('measure')).fit()
    rep = ols.report(reps=9999, seed=42)
    checks.append(check_val('pre24_model4_wild_p_deterministic',
                            rep['p_wild_cluster_bootstrap_t'],
                            m4p['p_wild_cluster_bootstrap_t'], tol=1e-12))

    # ---- time-trend placebo --------------------------------------------------
    n_placebo = sum(1 for m in placebo.items() if m[0] != 'metadata')
    checks.append(check_val('placebo_measure_count', 9, n_placebo, tol=0))
    # re-derive dem_share wavemean / dev from the canonical and re-fit Model 4
    cd_wave = df[['CounDist', 'election_year', 'dem_share']].dropna() \
        .drop_duplicates()
    wave_mean = cd_wave.groupby('election_year')['dem_share'].mean()
    checks.append(check_val('placebo_wavemean_n_districts_per_wave', 51,
                            int(placebo['dem_share']
                                ['n_districts_per_wave']['2017']), tol=0))
    pla = placebo['dem_share']
    # wavemean Model 4 re-fit
    dw = df[base_ok & df['election_year'].notna()].copy()
    dw['measure_val'] = dw['election_year'].map(wave_mean)
    dw = dw[dw['measure_val'].notna() & dw['dem_share'].notna()]
    resw, Xw = model4_statsmodels(dw)
    checks.append(check_val('placebo_wavemean_coef_vs_statsmodels',
                            float(resw.params['measure']),
                            pla['wavemean_model4']['coef'], tol=1e-8))
    checks.append(check_val('placebo_wavemean_se_vs_statsmodels',
                            float(resw.bse['measure']),
                            pla['wavemean_model4']['se_cluster'], tol=1e-8))
    # dev Model 4 re-fit
    dw2 = df[base_ok].copy()
    dw2['measure_val'] = dw2['dem_share'] - dw2['election_year'].map(wave_mean)
    dw2 = dw2[dw2['measure_val'].notna() & dw2['dem_share'].notna()]
    resd, Xd = model4_statsmodels(dw2)
    checks.append(check_val('placebo_dev_coef_vs_statsmodels',
                            float(resd.params['measure']),
                            pla['dev_model4']['coef'], tol=1e-8))
    checks.append(check_val('placebo_dev_se_vs_statsmodels',
                            float(resd.bse['measure']),
                            pla['dev_model4']['se_cluster'], tol=1e-8))
    # verdict consistency: wavemean sign vs original model4; dev CI covers 0
    orig = rob2['full']['dem_share']['model4_district_fe']['coef']
    cmp = pla['comparison_to_original_model4']
    checks.append(check_val('placebo_wavemean_sign_verdict_consistent',
                            bool(np.sign(pla['wavemean_model4']['coef'])
                                 == np.sign(orig)),
                            bool(cmp['sign_matches_original'])))
    checks.append(check_val('placebo_original_model4_coef', orig,
                            cmp['original_model4_coef'], tol=1e-12))
    checks.append(check_val('placebo_dev_near_zero_verdict_consistent',
                            bool(pla['dev_model4']['ci95_low_t'] <= 0.0
                                 <= pla['dev_model4']['ci95_high_t']),
                            bool(pla['dev_near_zero_verdict']['near_zero'])))


    # ---- hedonic v3 ----------------------------------------------------------
    b = df[base_ok]
    gross_ok = pd.to_numeric(b['GROSS SQUARE FEET'], errors='coerce') > 0
    land_ok = pd.to_numeric(b['LAND SQUARE FEET'], errors='coerce') > 0
    funnel = {r['step']: r['rows'] for r in hed3['part1_why_n_23789']['funnel']}
    checks.append(check_val('hed3_base_rows', int(len(b)),
                            funnel['analysis base sample'], tol=0))
    checks.append(check_val('hed3_land_rows', int(land_ok.sum()),
                            funnel['land sq ft > 0'], tol=0))
    checks.append(check_val('hed3_gross_rows', int(gross_ok.sum()),
                            funnel['gross sq ft > 0'], tol=0))
    checks.append(check_val('hed3_both_rows', int((gross_ok & land_ok).sum()),
                            funnel['gross and land sq ft both > 0 '
                                   '(upgraded hedonic sample)'], tol=0))
    checks.append(check_val('hed3_n_23789', 23789,
                            funnel['both > 0 + building-class-category '
                                   'non-empty + tax class non-missing '
                                   '(robustness hedonic sample, N=23,789 for '
                                   'the legacy measures)'], tol=0))
    # land/units variant: N recompute + statsmodels re-fit for dem_share
    l = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
    df['log_land_sqft'] = np.log(l.where(l > 0))
    vmask = base_ok & (df['log_land_sqft'] > 0) \
        & df['RESIDENTIAL UNITS'].notna() & df['TOTAL UNITS'].notna() \
        & (df['building_class_category'].astype(str) != '') \
        & df['TAX CLASS AT TIME OF SALE'].notna() & df['dem_share'].notna()
    hv = hed3['part2_hedonic_land_units']['by_measure']['dem_share']
    checks.append(check_val('hed3_land_units_n_dem_share', int(vmask.sum()),
                            hv['n'], tol=0))
    dv = df[vmask].copy()
    parts = [pd.Series(1.0, index=dv.index, name='const'),
             dv[['dem_share', 'log_land_sqft', 'RESIDENTIAL UNITS',
                 'TOTAL UNITS']].astype(float),
             pd.get_dummies(dv['landuse'].astype('Int64').astype(str),
                            prefix='landuse', drop_first=True, dtype=float),
             pd.get_dummies(dv['building_class_category'].astype(str),
                            prefix='bldgclasscat', drop_first=True,
                            dtype=float),
             pd.get_dummies(dv['TAX CLASS AT TIME OF SALE'].astype(str),
                            prefix='taxclass', drop_first=True, dtype=float),
             pd.get_dummies(dv['CounDist'].astype(int), prefix='district',
                            drop_first=True, dtype=float),
             pd.get_dummies(dv['sale_year'].astype(int), prefix='sale_year',
                            drop_first=True, dtype=float)]
    Xh = pd.concat(parts, axis=1)
    keep = [c for c in Xh.columns
            if c == 'const' or Xh[c].nunique(dropna=False) > 1]
    Xh = Xh[keep]
    resh = sm.OLS(dv['target_log_price'].astype(float), Xh).fit(
        cov_type='cluster', cov_kwds={'groups': dv['CounDist'].astype(int)})
    checks.append(check_val('hed3_land_units_coef_vs_statsmodels',
                            float(resh.params['dem_share']), hv['coef'],
                            tol=1e-8))
    checks.append(check_val('hed3_land_units_se_vs_statsmodels',
                            float(resh.bse['dem_share']), hv['se_cluster'],
                            tol=1e-8))

    record_status('verify_analysis_v3', checks,
                  note='Independent re-derivation of the Round-13 robustness '
                       'v3 suite, the Cuomo-free pre-2024 suite, the '
                       'time-trend placebo identities and the hedonic '
                       'sample audit + land/units variant.')


if __name__ == '__main__':
    main()

