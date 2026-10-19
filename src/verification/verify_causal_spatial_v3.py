"""Independent verification of the Round-13 causal/spatial outputs (R3/R4/R7 + v3 spatial).

Re-derives key numbers WITHOUT importing the producing scripts:

  * morans_i_results_v2.json (R3 fix): the legacy dem_share pooled residual
    Moran's I re-derived with esda WITH the measure among the regressors
    (must equal 0.3950), the recorded spreg z cross-check (6.14), and the
    residual I values no longer uniform across measures.
  * morans_i_results_v3.json + spatial_regression_v3.json (R2): all nine
    _aw/_dom measures present with residual + borough-FE sections; the
    dem_share_aw raw and residual Moran's I re-derived with esda; the
    spatial OLS-anchor residual I cross-checked against the Moran file.
  * iv_district_fe_results_v2.json (R4 fix): the RD local-linear spec
    re-fit at the 40pp bandwidth with cov_type='clustered' (F, coef, SE,
    G); the 10-20pp bandwidths must be marked 'not estimable' with no
    unadjusted fallback.
  * city_of_yes_did_results_v3.json (R7): the window / district-trends
    coefficients re-fit with statsmodels, wild-p determinism re-run, and
    the continuous-treatment dem_share_pres_vintage_aw x post arm
    re-fit in both 2024 variants.
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
MORAN_V2 = ROOT / 'results' / 'morans_i_results_v2.json'
MORAN_V3 = ROOT / 'results' / 'morans_i_results_v3.json'
SPAT_V3 = ROOT / 'results' / 'spatial_regression_v3.json'
IV_V2 = ROOT / 'results' / 'causal' / 'iv_district_fe_results_v2.json'
DID_V3 = ROOT / 'results' / 'causal' / 'city_of_yes_did_results_v3.json'
MARGINS = ROOT / 'data' / 'processed' / 'council_election_margins.parquet'
ROLL = ROOT / 'data' / 'processed' / 'city_of_yes_roll_calls.parquet'
COUNCIL_SHP = ROOT / 'data' / 'raw' / 'election_districts' / 'NYC_City_Council_Districts.shp'

CONTROLS = ['yearbuilt', 'numfloors', 'landuse', 'commfar',
            'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS']
IV_CONTROLS = ['yearbuilt', 'numfloors', 'landuse', 'commfar',
               'RESIDENTIAL UNITS', 'COMMERCIAL UNITS']
IV_REQUIRED = ['target_log_price', 'dem_share', 'CounDist', 'sale_year'] + IV_CONTROLS
V2_MEASURES = ['dem_share', 'dem_share_mayor_allcand', 'dem_share_pres',
               'dem_share_vintage', 'dem_share_mayor_allcand_vintage',
               'dem_share_pres_vintage']
V3_MEASURES = ['dem_share_aw', 'dem_share_mayor_allcand_aw', 'dem_share_pres_aw',
               'dem_share_vintage_aw', 'dem_share_mayor_allcand_vintage_aw',
               'dem_share_pres_vintage_aw', 'dem_share_vintage_dom',
               'dem_share_mayor_allcand_vintage_dom', 'dem_share_pres_vintage_dom']


def district_agg(df, measures):
    agg = df.groupby('CounDist').agg(
        median_log_price=('target_log_price', 'median'),
        mean_numfloors=('numfloors', 'mean'),
        mean_yearbuilt=('yearbuilt', 'mean'),
        mean_commfar=('commfar', 'mean'),
        **{m: (m, 'mean') for m in measures},
    ).reset_index()
    agg['CounDist'] = pd.to_numeric(agg['CounDist'], errors='coerce')
    agg = agg.dropna(subset=['CounDist'])
    agg['CounDist'] = agg['CounDist'].astype(int)
    return agg


def merged_geo(agg):
    import geopandas as gpd
    import warnings
    warnings.filterwarnings('ignore')
    gdf = gpd.read_file(COUNCIL_SHP)
    gdf['CounDist'] = pd.to_numeric(gdf['CounDist'], errors='coerce')
    gdf = gdf.dropna(subset=['CounDist'])
    gdf['CounDist'] = gdf['CounDist'].astype(int)
    merged = gdf[['CounDist', 'geometry']].merge(agg, on='CounDist', how='inner')
    merged = merged.sort_values('CounDist').reset_index(drop=True)
    centroid = merged.geometry.centroid.set_crs(merged.crs)
    coords = np.column_stack([centroid.x.values, centroid.y.values]).astype(float)
    return merged, coords


def esda_moran(y, coords, k=5, n_perm=999, seed=42):
    from libpysal.weights import KNN
    from esda import Moran
    np.random.seed(seed)
    w = KNN(coords, k=k)
    mi = Moran(np.asarray(y, dtype=float), w, transformation='r',
               permutations=n_perm, two_tailed=True)
    return float(mi.I), float(mi.z_sim), float(mi.p_sim)


def resid_I(sub, m, coords):
    X = sub[[m, 'mean_numfloors', 'mean_yearbuilt', 'mean_commfar']]
    res = sm.OLS(sub['median_log_price'].to_numpy(dtype=float),
                 sm.add_constant(X.to_numpy(dtype=float))).fit()
    I, z, p = esda_moran(res.resid, coords)
    return I



def main():
    checks = []
    for p in [CANON, MORAN_V2, MORAN_V3, SPAT_V3, IV_V2, DID_V3, MARGINS,
              ROLL, COUNCIL_SHP]:
        if not p.exists():
            raise SystemExit(f'Verify-causal-spatial-v3: MISSING INPUT {p}')
    moran_v2 = json.loads(MORAN_V2.read_text(encoding='utf-8'))
    moran_v3 = json.loads(MORAN_V3.read_text(encoding='utf-8'))
    spat_v3 = json.loads(SPAT_V3.read_text(encoding='utf-8'))
    iv_v2 = json.loads(IV_V2.read_text(encoding='utf-8'))
    did_v3 = json.loads(DID_V3.read_text(encoding='utf-8'))

    df = pd.read_parquet(CANON)

    # ---- R3 fix: residual Moran's I re-derivation ---------------------------
    agg2 = district_agg(df, V2_MEASURES)
    merged2, coords2 = merged_geo(agg2)
    sub2 = merged2.dropna(subset=['dem_share'])
    I_fix = resid_I(sub2, 'dem_share', coords2[sub2.index])
    checks.append(check_val('moran2_dem_share_residual_I_fixed', I_fix,
                            moran_v2['by_measure']['dem_share']
                            ['residual_morans']['I'], tol=1e-10))
    checks.append(check_val('moran2_dem_share_residual_I_value', 0.3950,
                            round(moran_v2['by_measure']['dem_share']
                                  ['residual_morans']['I'], 4), tol=5e-5))
    zc = moran_v2['by_measure']['dem_share'].get(
        'residual_morans_z_spreg_crosscheck', {})
    checks.append(check_val('moran2_spreg_z_crosscheck', 6.14,
                            round(float(zc.get('z')), 4), tol=0.005))
    distinct_I = len({round(moran_v2['by_measure'][m]['residual_morans']['I'], 6)
                      for m in V2_MEASURES})
    checks.append(check_val('moran2_residual_I_now_measure_specific', True,
                            distinct_I >= 5))
    for m in V2_MEASURES:
        if not moran_v2['by_measure'][m].get('residual_morans_boro_fe'):
            checks.append(check_val(f'moran2_boro_fe_present_{m}', True, False))

    # ---- R2 v3 spatial files ------------------------------------------------
    for m in V3_MEASURES:
        if m not in moran_v3['by_measure'] or m not in spat_v3['by_measure']:
            checks.append(check_val(f'spatial3_measure_present_{m}', True, False))
        elif not (moran_v3['by_measure'][m].get('residual_morans_boro_fe')
                  and spat_v3['by_measure'][m].get('borough_fe')):
            checks.append(check_val(f'spatial3_boro_fe_{m}', True, False))
    agg3 = district_agg(df, V3_MEASURES)
    merged3, coords3 = merged_geo(agg3)
    sub3 = merged3.dropna(subset=['dem_share_aw'])
    I_raw, _, _ = esda_moran(sub3['dem_share_aw'].to_numpy(dtype=float),
                             coords3[sub3.index])
    checks.append(check_val('moran3_dem_share_aw_raw_I', I_raw,
                            moran_v3['by_measure']['dem_share_aw']['moran_i'],
                            tol=1e-10))
    I_res3 = resid_I(sub3, 'dem_share_aw', coords3[sub3.index])
    checks.append(check_val('moran3_dem_share_aw_residual_I', I_res3,
                            moran_v3['by_measure']['dem_share_aw']
                            ['residual_morans']['I'], tol=1e-10))
    # cross-file invariant: spreg OLS anchor residual I == esda residual I
    checks.append(check_val('spatial3_anchor_residual_I_matches_moran3',
                            moran_v3['by_measure']['dem_share_aw']
                            ['residual_morans']['I'],
                            spat_v3['by_measure']['dem_share_aw']['pooled']
                            ['ols_residual_morans_i']['I'], tol=1e-9))


    # ---- R4: IV RD clustered re-fit at bw 40pp ------------------------------
    from linearmodels.iv import IV2SLS
    margins = pd.read_parquet(MARGINS)
    close = margins[margins['margin_pct'] <= 40]
    zmap, mmap = {}, {}
    for r in close.itertuples():
        if r.winner_party in ('Democratic', 'Republican'):
            zmap[(int(r.council_district), int(r.year))] = \
                1.0 if r.winner_party == 'Democratic' else 0.0
            mmap[(int(r.council_district), int(r.year))] = float(r.margin_pct)
    iv_df = df.dropna(subset=IV_REQUIRED).copy()
    iv_df['prior_cycle'] = iv_df['election_year'].astype(int) - 4
    keys = list(zip(iv_df['CounDist'].astype(int), iv_df['prior_cycle']))
    iv_df['Z'] = [zmap.get(k, np.nan) for k in keys]
    iv_df['margin'] = [mmap.get(k, np.nan) for k in keys]
    dd = iv_df.dropna(subset=['Z', 'margin']).copy()
    dd['Zxmargin'] = dd['Z'] * dd['margin']
    exog = pd.concat([
        pd.Series(1.0, index=dd.index, name='const'),
        dd[IV_CONTROLS].astype(float),
        pd.get_dummies(dd['CounDist'].astype(int), prefix='CounDist',
                       drop_first=True, dtype=float),
        pd.get_dummies(dd['sale_year'].astype(int), prefix='sale_year',
                       drop_first=True, dtype=float),
        dd[['margin']].astype(float)], axis=1)
    res40 = IV2SLS(dd['target_log_price'].astype(float), exog,
                   dd['dem_share'].astype(float),
                   dd[['Z', 'Zxmargin']].astype(float)).fit(
                       cov_type='clustered', clusters=dd['CounDist'].astype(int))
    fs40 = res40.first_stage.diagnostics.loc['dem_share']
    rec40 = iv_v2['rd_local_linear']['bw_40pp']
    checks.append(check_val('iv2_rd40_clustered_first_stage_F',
                            float(fs40['f.stat']),
                            rec40['first_stage_joint_f_excluded_instruments'],
                            tol=1e-6))
    checks.append(check_val('iv2_rd40_clustered_coef',
                            float(res40.params['dem_share']),
                            rec40['dem_share_coef'], tol=1e-8))
    checks.append(check_val('iv2_rd40_clustered_se',
                            float(res40.std_errors['dem_share']),
                            rec40['dem_share_se'], tol=1e-8))
    checks.append(check_val('iv2_rd40_G', int(dd['CounDist'].nunique()),
                            rec40['n_clusters_G'], tol=0))
    for bw in ['bw_10pp', 'bw_15pp', 'bw_20pp']:
        rec = iv_v2['rd_local_linear'][bw]
        checks.append(check_val(f'iv2_{bw}_not_estimable', 'not estimable',
                                rec.get('status')))
        checks.append(check_val(f'iv2_{bw}_no_unadjusted_fallback', False,
                                bool(rec.get('identified'))))
    for bw, rec in iv_v2['rd_local_linear'].items():
        if rec.get('identified'):
            checks.append(check_val(f'iv2_{bw}_covariance_clustered', True,
                                    'clustered by council district'
                                    in rec.get('covariance', '')))
    checks.append(check_val('iv2_vintage_mismatch_flag', True,
                            bool(iv_v2.get('council_district_vintage_mismatch',
                                           {}).get('flag'))))

    # ---- R7: DiD v3 re-derivations -------------------------------------------
    roll = pd.read_parquet(ROLL)
    d = df.merge(roll[['council_district', 'member_vote']],
                 left_on='CounDist', right_on='council_district', how='inner')
    d = d.dropna(subset=['target_log_price', 'CounDist', 'sale_year']
                 + CONTROLS)
    d['yes_district'] = (d['member_vote'] == 'Yes').astype(float)

    def did_design(dd, extra_cols):
        dd = dd.copy()
        dd['CounDist'] = dd['CounDist'].astype(int)
        dd['sale_year'] = dd['sale_year'].astype(int)
        dist_fe = pd.get_dummies(dd['CounDist'], prefix='district',
                                 drop_first=True, dtype=float)
        year_fe = pd.get_dummies(dd['sale_year'], prefix='sale_year',
                                 drop_first=True, dtype=float)
        X = pd.concat([pd.Series(1.0, index=dd.index, name='const'),
                       dd[CONTROLS].astype(float), dd[extra_cols].astype(float),
                       dist_fe, year_fe], axis=1)
        keep = [c for c in X.columns
                if c == 'const' or X[c].nunique(dropna=False) > 1]
        return X[keep], dd['target_log_price'].astype(float), dd['CounDist']

    dw = d[d['sale_year'].between(2022, 2025) & (d['sale_year'] != 2024)].copy()
    dw['post'] = (dw['sale_year'] >= 2025).astype(float)
    dw['yes_x_post'] = dw['yes_district'] * dw['post']
    Xw, yw, gw = did_design(dw, ['yes_x_post'])
    resw = sm.OLS(yw, Xw).fit(cov_type='cluster', cov_kwds={'groups': gw})
    wspec = did_v3['window_2022_2025_excl_2024']
    checks.append(check_val('did3_window_excl_coef', float(resw.params['yes_x_post']),
                            wspec['coef'], tol=1e-8))
    checks.append(check_val('did3_window_excl_se', float(resw.bse['yes_x_post']),
                            wspec['se_cluster'], tol=1e-6))
    ols = WildClusterOLS(yw.to_numpy(), Xw.to_numpy(dtype=np.float64),
                         gw.to_numpy(), list(Xw.columns).index('yes_x_post')).fit()
    rep = ols.report(reps=9999, seed=42)
    checks.append(check_val('did3_window_excl_wild_p_deterministic',
                            rep['p_wild_cluster_bootstrap_t'],
                            wspec['p_wild_cluster_bootstrap_t'], tol=1e-12))
    # incl-2024 variant present with wild inference
    winc = did_v3['window_2022_2025_incl_2024_as_pre']
    checks.append(check_val('did3_window_incl_wild_p_present', True,
                            wspec['p_wild_cluster_bootstrap_t'] is not None
                            and winc['p_wild_cluster_bootstrap_t'] is not None))

    # district-trends spec re-fit
    dm = d[d['sale_year'] != 2024].copy()
    dm['post'] = (dm['sale_year'] >= 2025).astype(float)
    dm['yes_x_post'] = dm['yes_district'] * dm['post']
    dist_fe = pd.get_dummies(dm['CounDist'], prefix='district',
                             drop_first=True, dtype=float)
    t = (dm['sale_year'].astype(float) - 2016.0)
    dm = pd.concat([dm, dist_fe.mul(t.values, axis=0).add_suffix('_xtrend')], axis=1)
    trend_cols = [c for c in dm.columns if c.endswith('_xtrend')]
    Xt, yt, gt = did_design(dm, ['yes_x_post'] + trend_cols)
    rest = sm.OLS(yt, Xt).fit(cov_type='cluster', cov_kwds={'groups': gt})
    tspec = did_v3['district_specific_linear_trends']
    checks.append(check_val('did3_trends_coef', float(rest.params['yes_x_post']),
                            tspec['coef'], tol=1e-8))
    checks.append(check_val('did3_trends_se', float(rest.bse['yes_x_post']),
                            tspec['se_cluster'], tol=1e-6))

    # continuous treatment: dem_share_pres_vintage_aw x post (2022-25 window)
    aw = 'dem_share_pres_vintage_aw'
    dc_all = d.dropna(subset=[aw])
    dc_all = dc_all[dc_all['sale_year'].between(2022, 2025)]
    arms = did_v3['continuous_treatment_dem_share_pres_vintage_aw_x_post']
    for tag, keep_2024 in [('excl_2024', False), ('incl_2024_as_pre', True)]:
        dc = dc_all.copy() if keep_2024 else \
            dc_all[dc_all['sale_year'] != 2024].copy()
        dc['post'] = (dc['sale_year'] >= 2025).astype(float)
        dc['dsp_x_post'] = dc[aw].astype(float) * dc['post']
        Xc, yc, gc = did_design(dc, [aw, 'dsp_x_post'])
        rescs = sm.OLS(yc, Xc).fit(cov_type='cluster', cov_kwds={'groups': gc})
        rec = arms[tag]['interaction_wild_inference']
        checks.append(check_val(f'did3_continuous_{tag}_coef',
                                float(rescs.params['dsp_x_post']),
                                rec['coef'], tol=1e-8))
        checks.append(check_val(f'did3_continuous_{tag}_se',
                                float(rescs.bse['dsp_x_post']),
                                rec['se_cluster'], tol=1e-6))

    record_status('verify_causal_spatial_v3', checks,
                  note='Independent re-derivation of the R3 Moran residual '
                       'fix (0.3950 / spreg z 6.14), the v3 spatial files, '
                       'the R4 clustered IV RD refit (with not-estimable '
                       'bandwidths), and the R7 DiD v3 wild-bootstrap specs.')


if __name__ == '__main__':
    main()

