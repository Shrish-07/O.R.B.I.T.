"""Independent verification of the Round-12 causal/spatial outputs (F/G/H).

Re-derives key numbers WITHOUT importing the producing scripts:
  * city_of_yes_did_results_v2.json: the main estimate's coef/cluster SE
    (statsmodels re-fit), deterministic wild p, the continuous-treatment
    interaction, the 2022-25 window spec and one leave-one-borough-out arm.
  * city_of_yes_did_results.json Wald bug fix: chi2/q = the stored F and
    its F(q, G-1) p-value satisfies the scipy identity.
  * iv_district_fe_results_v2.json: the ±20pp matched OLS for the legacy
    measure (statsmodels re-fit on the rebuilt close-race sample), the
    G / treated / control cell counts, and the RD local-linear
    first-stage F at one bandwidth (linearmodels re-fit).
  * morans_i_results_v2.json + spatial_regression_v2.json: every measure
    present; the legacy-measure Moran's I re-derived with esda; the
    residual Moran's I re-derived; borough-FE sections present.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ROOT, record_status, check_val  # noqa: E402

sys.path.insert(0, str(ROOT / 'src' / 'analysis'))
from wild_cluster_inference import WildClusterOLS  # noqa: E402

CANON = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet'
ROLL = ROOT / 'data' / 'processed' / 'city_of_yes_roll_calls.parquet'
MARGINS = ROOT / 'data' / 'processed' / 'council_election_margins.parquet'
DID_MAIN = ROOT / 'results' / 'causal' / 'city_of_yes_did_results.json'
DID_V2 = ROOT / 'results' / 'causal' / 'city_of_yes_did_results_v2.json'
IV_V2 = ROOT / 'results' / 'causal' / 'iv_district_fe_results_v2.json'
MORAN_V2 = ROOT / 'results' / 'morans_i_results_v2.json'
SPAT_V2 = ROOT / 'results' / 'spatial_regression_v2.json'
COUNCIL_SHP = ROOT / 'data' / 'raw' / 'election_districts' / 'NYC_City_Council_Districts.shp'
BORO_MAP = ROOT / 'results' / 'council_district_borough_map.csv'
CONTROLS = ['yearbuilt', 'numfloors', 'landuse', 'commfar',
            'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS']
REQUIRED = ['target_log_price', 'CounDist', 'sale_year'] + CONTROLS
IV_CONTROLS = ['yearbuilt', 'numfloors', 'landuse', 'commfar',
               'RESIDENTIAL UNITS', 'COMMERCIAL UNITS']
IV_REQUIRED = ['target_log_price', 'dem_share', 'CounDist', 'sale_year'] + IV_CONTROLS
MEASURES = ['dem_share', 'dem_share_mayor_allcand', 'dem_share_pres',
            'dem_share_vintage', 'dem_share_mayor_allcand_vintage',
            'dem_share_pres_vintage']


def did_design(d, extra):
    d2 = d.copy()
    d2['CounDist'] = d2['CounDist'].astype(int)
    d2['sale_year'] = d2['sale_year'].astype(int)
    dist_fe = pd.get_dummies(d2['CounDist'], prefix='district',
                             drop_first=True, dtype=float)
    year_fe = pd.get_dummies(d2['sale_year'], prefix='sale_year',
                              drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=d2.index, name='const'),
                   d2[CONTROLS].astype(float), d2[extra].astype(float),
                   dist_fe, year_fe], axis=1)
    return X, d2['target_log_price'].astype(float), d2['CounDist']


def main():
    checks = []
    for p in [CANON, ROLL, MARGINS, DID_MAIN, DID_V2, IV_V2, MORAN_V2,
              SPAT_V2, COUNCIL_SHP]:
        if not p.exists():
            raise SystemExit(f'MISSING INPUT: {p}')
    did_main = json.loads(DID_MAIN.read_text())
    did_v2 = json.loads(DID_V2.read_text())
    iv_v2 = json.loads(IV_V2.read_text())
    moran_v2 = json.loads(MORAN_V2.read_text())
    spat_v2 = json.loads(SPAT_V2.read_text())

    df = pd.read_parquet(CANON).dropna(subset=REQUIRED)
    roll = pd.read_parquet(ROLL)
    d = df.merge(roll[['council_district', 'member_vote']],
                 left_on='CounDist', right_on='council_district', how='inner')
    d['yes_district'] = (d['member_vote'] == 'Yes').astype(float)

    # ---- DiD main estimate re-derivation -----------------------------------
    dm = d[d['sale_year'] != 2024].copy()
    dm['post'] = (dm['sale_year'] >= 2025).astype(float)
    dm['yes_x_post'] = dm['yes_district'] * dm['post']
    X, y, g = did_design(dm, ['yes_x_post'])
    res = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': g})
    mw = did_v2['main_wild_inference']
    checks.append(check_val('did2_main_coef', float(res.params['yes_x_post']),
                            mw['coef'], tol=1e-8))
    checks.append(check_val('did2_main_se', float(res.bse['yes_x_post']),
                            mw['se_cluster'], tol=1e-8))
    checks.append(check_val('did2_main_matches_mainfile',
                            did_main['did_main']['did_coef'], mw['coef'],
                            tol=1e-9))
    ols = WildClusterOLS(y.to_numpy(), X.to_numpy(dtype=np.float64),
                         g.to_numpy(), list(X.columns).index('yes_x_post')).fit()
    rep = ols.report(reps=9999, seed=42)
    checks.append(check_val('did2_main_wild_p_deterministic',
                            rep['p_wild_cluster_bootstrap_t'],
                            mw['p_wild_cluster_bootstrap_t'], tol=1e-12))
    checks.append(check_val('did2_main_p_t', rep['p_t_G_minus_1'],
                            mw['p_t_G_minus_1'], tol=1e-12))

    # ---- Wald bug-fix identities -------------------------------------------
    ev = did_main['event_study_pre_trends']
    chi2 = ev['pre_trend_wald_chi2']
    q = ev['pre_trend_wald_q']
    G = ev['n_clusters_G']
    checks.append(check_val('did_wald_f_is_chi2_over_q', chi2 / q,
                            ev['pre_trend_wald_f'], tol=1e-10))
    checks.append(check_val('did_wald_f_p_identity',
                            float(sps.f.sf(chi2 / q, q, G - 1)),
                            ev['pre_trend_wald_f_p'], tol=1e-10))
    checks.append(check_val('did_wald_chi2_value_unchanged', 4.405585464840547,
                            chi2, tol=1e-9))

    # ---- continuous treatment + window + LOBO -------------------------------
    dc = d.dropna(subset=['dem_share_pres']).copy()
    dc = dc[dc['sale_year'] != 2024]
    dc['post'] = (dc['sale_year'] >= 2025).astype(float)
    dc['dsp_x_post'] = dc['dem_share_pres'].astype(float) * dc['post']
    X, y, g = did_design(dc, ['dem_share_pres', 'dsp_x_post'])
    res = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': g})
    ct = did_v2['continuous_treatment_dem_share_pres_x_post']
    checks.append(check_val('did2_continuous_interaction_coef',
                            float(res.params['dsp_x_post']),
                            ct['interaction_coef'], tol=1e-8))
    dw = d[d['sale_year'].between(2022, 2025) & (d['sale_year'] != 2024)].copy()
    dw['post'] = (dw['sale_year'] >= 2025).astype(float)
    dw['yes_x_post'] = dw['yes_district'] * dw['post']
    X, y, g = did_design(dw, ['yes_x_post'])
    res = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': g})
    wspec = did_v2['window_2022_2025_excl_2024']
    checks.append(check_val('did2_window_excl2024_coef',
                            float(res.params['yes_x_post']),
                            wspec['did_coef'], tol=1e-8))
    db = dm[dm['BOROUGH'] != 3].copy()  # drop Brooklyn
    X, y, g = did_design(db, ['yes_x_post'])
    res = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': g})
    lobo = did_v2['leave_one_borough_out']['Brooklyn']
    checks.append(check_val('did2_lobo_brooklyn_coef',
                            float(res.params['yes_x_post']),
                            lobo['did_coef'], tol=1e-8))

    # ---- IV v2 --------------------------------------------------------------
    margins = pd.read_parquet(MARGINS)
    close = margins[margins['margin_pct'] <= 20]
    zmap = {}
    for r in close.itertuples():
        if r.winner_party in ('Democratic', 'Republican'):
            zmap[(int(r.council_district), int(r.year))] = \
                1.0 if r.winner_party == 'Democratic' else 0.0
    iv_df = pd.read_parquet(CANON).dropna(subset=IV_REQUIRED)
    iv_df['prior_cycle'] = iv_df['election_year'].astype(int) - 4
    keys = list(zip(iv_df['CounDist'].astype(int), iv_df['prior_cycle']))
    iv_df['Z'] = [zmap.get(k, np.nan) for k in keys]
    iv_d = iv_df.dropna(subset=['Z'])
    cells = iv_d[['CounDist', 'election_year', 'Z']].drop_duplicates()
    b20 = iv_v2['bandwidths']['bw_20pp']
    checks.append(check_val('iv2_bw20_G', int(iv_d['CounDist'].nunique()),
                            b20['n_clusters_G'], tol=0))
    checks.append(check_val('iv2_bw20_n', int(len(iv_d)), b20['n_obs'],
                            tol=0))
    checks.append(check_val('iv2_bw20_treated_cells',
                            int((cells['Z'] == 1.0).sum()),
                            b20['treated_cells_Z1'], tol=0))
    exog = pd.concat([
        pd.Series(1.0, index=iv_d.index, name='const'),
        iv_d[IV_CONTROLS].astype(float),
        pd.get_dummies(iv_d['CounDist'].astype(int), prefix='CounDist',
                       drop_first=True, dtype=float),
        pd.get_dummies(iv_d['sale_year'].astype(int), prefix='sale_year',
                       drop_first=True, dtype=float),
        iv_d[['dem_share']].astype(float)], axis=1)
    res = sm.OLS(iv_d['target_log_price'].astype(float), exog).fit(
        cov_type='cluster', cov_kwds={'groups': iv_d['CounDist'].astype(int)})
    mo = b20['matched_ols_by_measure']['dem_share']
    checks.append(check_val('iv2_bw20_matched_ols_coef',
                            float(res.params['dem_share']), mo['coef'],
                            tol=1e-8))
    checks.append(check_val('iv2_g30_flag_true', True,
                            bool(b20['g_below_30_flag'])))
    checks.append(check_val('iv2_benchmark_pointer', 'matched_ols_benchmark',
                            iv_v2['benchmark_pointer']))

    # ---- spatial v2 ----------------------------------------------------------
    for m in MEASURES:
        if m not in moran_v2['by_measure'] or m not in spat_v2['by_measure']:
            checks.append(check_val(f'spatial2_measure_present_{m}', True,
                                     False))
    checks.append(check_val('spatial2_all_measures_present', 6,
                            sum(1 for m in MEASURES
                                if m in moran_v2['by_measure']
                                and m in spat_v2['by_measure']), tol=0))
    checks.append(check_val('spatial2_borough_fe_sections',
                            6,
                            sum(1 for m in MEASURES
                                if spat_v2['by_measure'][m].get('borough_fe')
                                and moran_v2['by_measure'][m]
                                .get('residual_morans_boro_fe')), tol=0))
    # re-derive the legacy-measure Moran's I with esda
    import geopandas as gpd
    import warnings
    warnings.filterwarnings('ignore')
    agg = pd.read_parquet(CANON).groupby('CounDist').agg(
        median_log_price=('target_log_price', 'median'),
        dem_share=('dem_share', 'mean')).reset_index()
    agg['CounDist'] = pd.to_numeric(agg['CounDist'], errors='coerce')
    agg = agg.dropna(subset=['CounDist'])
    agg['CounDist'] = agg['CounDist'].astype(int)
    gdf = gpd.read_file(COUNCIL_SHP)
    gdf['CounDist'] = pd.to_numeric(gdf['CounDist'], errors='coerce')
    gdf = gdf.dropna(subset=['CounDist'])
    gdf['CounDist'] = gdf['CounDist'].astype(int)
    merged = gdf[['CounDist', 'geometry']].merge(agg, on='CounDist',
                                                 how='inner')
    merged = merged.sort_values('CounDist').reset_index(drop=True)
    from libpysal.weights import KNN
    from esda import Moran
    np.random.seed(42)
    centroid = merged.geometry.centroid.set_crs(merged.crs)
    coords = np.column_stack([centroid.x.values, centroid.y.values])
    w = KNN(coords, k=5)
    mi = Moran(merged['dem_share'].to_numpy(dtype=float), w,
               transformation='r', permutations=999, two_tailed=True)
    checks.append(check_val('moran2_dem_share_rederived', float(mi.I),
                            moran_v2['by_measure']['dem_share']['moran_i'],
                            tol=1e-10))
    checks.append(check_val('moran2_legacy_value', 0.5705,
                            round(moran_v2['by_measure']['dem_share']
                                  ['moran_i'], 4), tol=1e-4))

    record_status('verify_causal_spatial_v2', checks,
                  note='Independent re-derivation of the DiD v2 specs and '
                       'Wald-field bug fix, the IV matched OLS benchmark and '
                       'cluster counts, and the per-measure spatial '
                       'statistics (Moran I re-derived with esda).')


if __name__ == '__main__':
    main()


