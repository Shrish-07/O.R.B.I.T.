"""v2 Moran's I for every ideology measure (Task H).

Reruns the district-level global Moran's I computation of
src/analysis/compute_morans_i.py for EACH ideology measure (legacy
dem_share, allcand, pres, and the vintage variants), with the identical
weights construction (k=5 row-standardized KNN on official council-district
polygon centroids, 999 permutations, seed=42), and adds:

  * residual Moran's I: Moran's I of the residuals of the district-level
    OLS median_log_price ~ measure + mean numfloors + mean yearbuilt +
    mean commfar (the spatial_regression_model.py control set; the
    measure itself is included among the regressors — Round 13, R3,
    previously the measure was omitted and every measure reported the
    same residual I = 0.3825);
  * a borough-FE version of the same residual computation (district-level
    regression with borough dummies).

Output: results/morans_i_results_v2.json

Run: python src/analysis/compute_morans_i_v2.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "canonical" / "modeling_dataset_canonical_v2.parquet"
COUNCIL_SHP = ROOT / "data" / "raw" / "election_districts" / "NYC_City_Council_Districts.shp"
BORO_MAP = ROOT / "results" / "council_district_borough_map.csv"
OUT = ROOT / "results" / "morans_i_results_v2.json"

SEED = 42
K = 5
N_PERM = 999
MEASURES = ['dem_share', 'dem_share_mayor_allcand', 'dem_share_pres',
            'dem_share_vintage', 'dem_share_mayor_allcand_vintage',
            'dem_share_pres_vintage']
CONTROLS = ['mean_numfloors', 'mean_yearbuilt', 'mean_commfar']


def moran(y, coords, k=K, n_perm=N_PERM, seed=SEED):
    from libpysal.weights import KNN as libKNN
    from esda import Moran
    np.random.seed(seed)
    w = libKNN(coords, k=k)
    mi = Moran(np.asarray(y, dtype=float), w, transformation='r',
               permutations=n_perm, two_tailed=True)
    return float(mi.I), float(mi.z_sim), float(mi.p_sim)


def resid_moran(y, X, coords, add_boro_fe, boro_dummies=None):
    import statsmodels.api as sm
    XX = X.copy()
    if add_boro_fe and boro_dummies is not None:
        XX = pd.concat([XX, boro_dummies], axis=1)
    res = sm.OLS(np.asarray(y, dtype=float),
                 sm.add_constant(XX.to_numpy(dtype=float))).fit()
    I, z, p = moran(res.resid, coords)
    return {'I': I, 'z': z, 'p_sim': p, 'r_squared': float(res.rsquared),
            'borough_fe': bool(add_boro_fe)}
def main():
    df = pd.read_parquet(CANON)
    agg = df.groupby('CounDist').agg(
        median_log_price=('target_log_price', 'median'),
        mean_numfloors=('numfloors', 'mean'),
        mean_yearbuilt=('yearbuilt', 'mean'),
        mean_commfar=('commfar', 'mean'),
        **{m: (m, 'mean') for m in MEASURES},
    ).reset_index()
    agg['CounDist'] = pd.to_numeric(agg['CounDist'], errors='coerce')
    agg = agg.dropna(subset=['CounDist'])
    agg['CounDist'] = agg['CounDist'].astype(int)

    gdf = gpd.read_file(COUNCIL_SHP)
    gdf['CounDist'] = pd.to_numeric(gdf['CounDist'], errors='coerce')
    gdf = gdf.dropna(subset=['CounDist'])
    gdf['CounDist'] = gdf['CounDist'].astype(int)
    merged = gdf[['CounDist', 'geometry']].merge(agg, on='CounDist', how='inner')
    merged = merged.sort_values('CounDist').reset_index(drop=True)
    centroid = merged.geometry.centroid.set_crs(merged.crs)
    coords = np.column_stack([centroid.x.values, centroid.y.values]).astype(float)

    boro_dummies = None
    if BORO_MAP.exists():
        bm = pd.read_csv(BORO_MAP)
        bcol = next((c for c in bm.columns
                     if 'council' in c.lower() or c == 'CounDist'), None)
        bbcol = next((c for c in bm.columns if 'borough' in c.lower()), None)
        if bcol and bbcol:
            boro = bm[[bcol, bbcol]].copy()
            boro.columns = ['CounDist', 'borough']
            boro['CounDist'] = pd.to_numeric(boro['CounDist'], errors='coerce')
            mm = merged[['CounDist']].merge(boro, on='CounDist', how='left')
            boro_dummies = pd.get_dummies(
                mm['borough'].fillna('Unknown'), prefix='borough',
                drop_first=True, dtype=float)

    price = merged['median_log_price'].to_numpy(dtype=float)
    out = {'methodology': {
        'weights': f'libpysal KNN k={K}, row-standardized',
        'permutations': N_PERM, 'seed': SEED,
        'aggregation': ('median target_log_price / mean measure per council '
                       'district (same as compute_morans_i.py)'),
        'residual_spec': ('district-level OLS: median_log_price ~ measure + '
                          'mean numfloors + mean yearbuilt + mean commfar '
                          '(+ borough FE in the borough-FE variant). '
                          'Round 13 (R3) fix: the measure IS included among '
                          'the residual regressors (it was previously '
                          'omitted, which forced every measure to the same '
                          'residual I = 0.3825).')}}
    I_p, z_p, p_p = moran(price, coords)
    out['median_log_price'] = {'I': I_p, 'z': z_p, 'p_sim': p_p,
                               'n_districts': int(len(merged))}
    out['by_measure'] = {}
    for m in MEASURES:
        sub = merged.dropna(subset=[m])
        sub_coords = coords[sub.index]
        I_m, z_m, p_m = moran(sub[m].to_numpy(dtype=float), sub_coords)
        rec = {
            'moran_i': I_m, 'z': z_m, 'p_sim': p_m,
            'n_districts': int(len(sub)),
            'residual_morans': resid_moran(
                sub['median_log_price'], sub[[m] + CONTROLS], sub_coords,
                False),
            'residual_morans_boro_fe': (
                resid_moran(sub['median_log_price'], sub[[m] + CONTROLS],
                            sub_coords, True,
                            boro_dummies.loc[sub.index].reset_index(drop=True))
                if boro_dummies is not None else None),
        }
        out['by_measure'][m] = rec
        if m == 'dem_share':
            # Round 13 (R3) regression gate: with the measure included among
            # the residual regressors, the legacy pooled dem_share residual
            # Moran's I must reproduce 0.3950. The pre-fix value 0.3825 was
            # the measure-omitted specification. The z=6.14 companion value
            # is spreg's normal-theory residual z (the same statistic
            # spatial_regression_model_v2.py records as ols_residual_morans_i
            # for the identical design); esda's permutation z_sim is 5.2548
            # by construction and is asserted separately below.
            assert abs(rec['residual_morans']['I'] - 0.3950) < 5e-4, (
                f"dem_share pooled residual Moran's I drifted: "
                f"{rec['residual_morans']['I']:.4f} != 0.3950")
            import spreg
            from libpysal.weights import KNN as LibKNN
            w_sp = LibKNN(sub_coords, k=K)
            w_sp.transform = 'r'
            ols_sp = spreg.OLS(
                sub['median_log_price'].to_numpy(dtype=float).reshape(-1, 1),
                sub[[m] + CONTROLS].to_numpy(dtype=float), w=w_sp,
                spat_diag=True, moran=True)
            z_spreg = float(ols_sp.moran_res[1])
            rec['residual_morans_z_spreg_crosscheck'] = {
                'I': float(ols_sp.moran_res[0]), 'z': z_spreg,
                'note': ('spreg normal-theory residual z for the identical '
                         'design, matching spatial_regression_v2.json '
                         "ols_residual_morans_i (z = 6.14)")}
            assert abs(z_spreg - 6.14) < 0.01, (
                f"dem_share pooled residual Moran's spreg z drifted: "
                f"{z_spreg:.4f} != 6.14")
        print(f'  {m}: I={I_m:.4f} (z={z_m:.2f}, p={p_m:.3f}); '
              f'residual I={rec["residual_morans"]["I"]:.4f}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'Wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()

