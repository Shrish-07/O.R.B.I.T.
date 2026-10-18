"""Independent verification of the Round-12 analysis outputs (Tasks D/E/I).

Re-derives key numbers WITHOUT importing the producing scripts:
  * sample_funnel.json: canonical row count, analysis-base and restricted
    sample sizes re-computed from the canonical parquet.
  * robustness_checks_v2.json: 84 spec records present with every required
    inference field; the full-sample Model 4 (dem_share) coefficient and
    cluster SE re-derived with statsmodels, compared to the file AND to
    the paper's recorded Model 4 values; wild-bootstrap p-value reproduced
    deterministically from the same seed; hedonic R^2 spot-check.
  * ml_dem_share_collinearity.json: dem_share single-valued per district
    in train+test, R^2 of dem_share on district dummies == 1, six
    duplicated pairs reported.
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

CANON = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet'
FUNNEL = ROOT / 'results' / 'sample_funnel.json'
ROB2 = ROOT / 'results' / 'robustness_checks_v2.json'
MLC = ROOT / 'results' / 'ml_dem_share_collinearity.json'
LEGACY_ROB = ROOT / 'results' / 'robustness_checks.json'
CORE = ['yearbuilt', 'numfloors', 'landuse', 'commfar']


def main():
    checks = []
    for p in [CANON, FUNNEL, ROB2, MLC, LEGACY_ROB]:
        if not p.exists():
            raise SystemExit(f'MISSING INPUT: {p}')
    funnel = json.loads(FUNNEL.read_text())
    rob = json.loads(ROB2.read_text())
    mlc = json.loads(MLC.read_text())
    legacy_rob = json.loads(LEGACY_ROB.read_text())

    # ---- funnel ----------------------------------------------------------
    df = pd.read_parquet(CANON)
    checks.append(check_val('funnel_canonical_rows', 566956,
                             funnel['canonical_rows'], tol=0))
    base_ok = df['target_log_price'].notna() & df['CounDist'].notna() \
        & df['sale_year'].notna()
    for c in CORE:
        base_ok &= df[c].notna()
    restricted = base_ok & df['is_resid_tc12'] & df['price_ge_100k'] \
        & (~df['multi_parcel_flag'])
    base_n = {r['step']: r['rows'] for r in funnel['funnel']}
    checks.append(check_val('funnel_base_sample_matches_recompute',
                            int(base_ok.sum()),
                            base_n['analysis_base_sample'], tol=0))
    checks.append(check_val('funnel_restricted_matches_recompute',
                            int(restricted.sum()),
                            base_n['restricted_analysis_sample'], tol=0))
    checks.append(check_val('funnel_raw_rows', 845607,
                            base_n['raw_sales_rows'], tol=0))

    # ---- robustness v2 ----------------------------------------------------
    n_fits = 0
    missing_fields = 0
    for s in ['full', 'restricted']:
        for m, specs in rob[s].items():
            for sp, rep in specs.items():
                if 'coef' in rep:
                    n_fits += 1
                    for f in ['coef', 'se_classical', 'se_cluster',
                              'p_t_G_minus_1', 'p_wild_cluster_bootstrap_t',
                              'ci95_low_t', 'ci95_high_t', 'n',
                              'n_clusters_G', 'r_squared']:
                        if rep.get(f) is None:
                            missing_fields += 1
    checks.append(check_val('rob2_fit_count', 84, n_fits, tol=0))
    checks.append(check_val('rob2_missing_fields', 0, missing_fields, tol=0))

    # Model 4 full dem_share vs statsmodels + paper values
    d = df[base_ok & df['dem_share'].notna()].copy()
    dist = pd.get_dummies(d['CounDist'].astype(int), prefix='district',
                          drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=d.index, name='const'),
                   d[['dem_share']].astype(float), d[CORE].astype(float),
                   dist], axis=1)
    res = sm.OLS(d['target_log_price'].astype(float), X).fit(
        cov_type='cluster', cov_kwds={'groups': d['CounDist'].astype(int)})
    m4 = rob['full']['dem_share']['model4_district_fe']
    checks.append(check_val('rob2_model4_coef_vs_statsmodels',
                            float(res.params['dem_share']), m4['coef'],
                            tol=1e-8))
    checks.append(check_val('rob2_model4_se_vs_statsmodels',
                            float(res.bse['dem_share']), m4['se_cluster'],
                            tol=1e-8))
    legacy_m4 = legacy_rob['check2_model4_council_district_fe_classical_se']
    checks.append(check_val('rob2_model4_coef_vs_paper',
                            legacy_m4['dem_share_coef'], m4['coef'], tol=1e-8))
    checks.append(check_val('rob2_model4_se_vs_paper',
                            legacy_m4['dem_share_se'], m4['se_classical'],
                            tol=1e-8))

    # wild p determinism: re-run the same design through the engine
    ols = WildClusterOLS(d['target_log_price'].to_numpy(dtype=float),
                         X.to_numpy(dtype=np.float64),
                         d['CounDist'].to_numpy(),
                         list(X.columns).index('dem_share')).fit()
    rep = ols.report(reps=9999, seed=42)
    checks.append(check_val('rob2_model4_wild_p_deterministic',
                            rep['p_wild_cluster_bootstrap_t'],
                            m4['p_wild_cluster_bootstrap_t'], tol=1e-12))
    checks.append(check_val('rob2_model4_p_t_rederived',
                            rep['p_t_G_minus_1'], m4['p_t_G_minus_1'],
                            tol=1e-12))

    # hedonic R^2 / N spot check vs statsmodels (full sample, legacy measure)
    g = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
    l = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
    with np.errstate(invalid='ignore'):
        df['log_gross_sqft'] = np.log(g.where(g > 0))
        df['log_land_sqft'] = np.log(l.where(l > 0))
    hd_mask = base_ok & df['dem_share'].notna() \
        & df['log_gross_sqft'].notna() & df['log_land_sqft'].notna() \
        & df['RESIDENTIAL UNITS'].notna() & df['TOTAL UNITS'].notna() \
        & (df['building_class_category'].astype(str) != '') \
        & df['TAX CLASS AT TIME OF SALE'].notna()
    hd = df[hd_mask].copy()
    parts = [pd.Series(1.0, index=hd.index, name='const'),
             hd[['dem_share', 'log_gross_sqft', 'log_land_sqft',
                 'RESIDENTIAL UNITS', 'TOTAL UNITS']].astype(float),
             pd.get_dummies(hd['landuse'].astype('Int64').astype(str),
                            prefix='landuse', drop_first=True, dtype=float),
             pd.get_dummies(hd['building_class_category'].astype(str),
                            prefix='bldgclasscat', drop_first=True,
                            dtype=float),
             pd.get_dummies(hd['TAX CLASS AT TIME OF SALE'].astype(str),
                            prefix='taxclass', drop_first=True, dtype=float),
             pd.get_dummies(hd['CounDist'].astype(int), prefix='district',
                            drop_first=True, dtype=float),
             pd.get_dummies(hd['sale_year'].astype(int), prefix='sale_year',
                            drop_first=True, dtype=float)]
    Xh = pd.concat(parts, axis=1)
    Xh = Xh[[c for c in Xh.columns
             if c == 'const' or Xh[c].nunique(dropna=False) > 1]]
    r2h = sm.OLS(hd['target_log_price'].astype(float), Xh).fit().rsquared
    checks.append(check_val('rob2_hedonic_r2_vs_statsmodels', float(r2h),
                            rob['full']['dem_share']['hedonic_upgraded']['r_squared'],
                            tol=1e-8))
    checks.append(check_val('rob2_hedonic_n_vs_statsmodels', int(len(hd)),
                            rob['full']['dem_share']['hedonic_upgraded']['n'],
                            tol=0))

    # ---- ML collinearity ---------------------------------------------------
    train = pd.read_parquet(ROOT / 'data' / 'splits' / 'all_years_train.parquet')
    test = pd.read_parquet(ROOT / 'data' / 'splits' / 'all_years_test.parquet')
    both = pd.concat([train, test], ignore_index=True)
    nu = both.groupby('CounDist')['dem_share'].nunique()
    checks.append(check_val('mlc_dem_share_single_valued_per_district', 51,
                            int((nu == 1).sum()), tol=0))
    checks.append(check_val('mlc_r2_reported', 1.0,
                            mlc['dem_share_deterministic_by_district']
                            ['r2_dem_share_on_district_dummies'], tol=1e-9))
    checks.append(check_val('mlc_six_dup_pairs', 6,
                            len(mlc['duplicated_feature_pairs']), tol=0))

    record_status('verify_analysis_v2', checks,
                  note='Independent re-derivation of the sample funnel, the '
                       'v2 robustness suite (statsmodels cross-check + '
                       'deterministic wild bootstrap), and the ML '
                       'collinearity diagnostics.')


if __name__ == '__main__':
    main()


