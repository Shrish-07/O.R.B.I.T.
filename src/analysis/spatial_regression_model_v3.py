"""v3 district-level spatial regressions for the NEW measures (Round 13, R2).

Runs the identical SLM/SEM specifications of spatial_regression_model_v2.py
(OLS anchor + ML spatial lag + ML spatial error, pooled and borough-FE,
k=5 row-standardized KNN weights on council-district centroids, seed 42)
for the nine Round-13 (_aw/_dom) measures on
modeling_dataset_canonical_v3.parquet.

Output: results/spatial_regression_v3.json

Run: python src/analysis/spatial_regression_model_v3.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import KNN
import spreg

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "canonical" / "modeling_dataset_canonical_v3.parquet"
COUNCIL_SHP = ROOT / "data" / "raw" / "election_districts" / "NYC_City_Council_Districts.shp"
BORO_MAP = ROOT / "results" / "council_district_borough_map.csv"
OUT = ROOT / "results" / "spatial_regression_v3.json"

SEED = 42
K = 5
MEASURES = ['dem_share_aw', 'dem_share_mayor_allcand_aw', 'dem_share_pres_aw',
            'dem_share_vintage_aw', 'dem_share_mayor_allcand_vintage_aw',
            'dem_share_pres_vintage_aw', 'dem_share_vintage_dom',
            'dem_share_mayor_allcand_vintage_dom', 'dem_share_pres_vintage_dom']
CONTROLS = ['mean_numfloors', 'mean_yearbuilt', 'mean_commfar']


def extract(model):
    betas = model.betas.flatten()
    se = list(np.sqrt(np.diag(model.vm))) if getattr(model, 'vm', None) is not None else None
    coefs = {}
    for i, nm in enumerate(model.name_x):
        if i < len(betas):
            c = {'coef': float(betas[i])}
            if se is not None and i < len(se):
                c['se'] = float(se[i])
            if hasattr(model, 'z_stat') and i < len(model.z_stat):
                c['z'] = float(model.z_stat[i][0])
                c['p'] = float(model.z_stat[i][1])
            coefs[nm] = c
    for extra in ('W_median_log_price', 'lambda'):
        if hasattr(model, 'name_z') and extra in getattr(model, 'name_z', []):
            idx = list(model.name_z).index(extra)
            b = model.betas.flatten()
            c = {'coef': float(b[idx])}
            if se is not None and idx < len(se):
                c['se'] = float(se[idx])
            if hasattr(model, 'z_stat') and idx < len(model.z_stat):
                c['z'] = float(model.z_stat[idx][0])
                c['p'] = float(model.z_stat[idx][1])
            coefs[extra] = c
    return coefs


def run_variant(y, X, w, x_names, label):
    ols = spreg.OLS(y, X, w=w, name_y='median_log_price',
                    name_x=x_names, name_w='knn5', spat_diag=True, moran=True)
    rec = {'label': label,
           'ols_anchor': extract(ols),
           'ols_residual_morans_i': {
               'I': float(ols.moran_res[0]), 'z': float(ols.moran_res[1]),
               'p': float(ols.moran_res[2])}}
    try:
        slm = spreg.ML_Lag(y, X, w=w, name_y='median_log_price',
                           name_x=x_names, name_w='knn5')
        rec['spatial_lag_model'] = extract(slm)
    except Exception as e:
        rec['spatial_lag_model_error'] = str(e)
    try:
        sem = spreg.ML_Error(y, X, w=w, name_y='median_log_price',
                             name_x=x_names, name_w='knn5')
        rec['spatial_error_model'] = extract(sem)
    except Exception as e:
        rec['spatial_error_model_error'] = str(e)
    return rec


def main():
    if not CANON.exists():
        raise SystemExit(f'MISSING INPUT: {CANON} — run '
                         'src/data_pipeline/extend_canonical_v3.py first.')
    df = pd.read_parquet(CANON)
    agg = df.groupby('CounDist').agg(
        median_log_price=('target_log_price', 'median'),
        mean_numfloors=('numfloors', 'mean'),
        mean_yearbuilt=('yearbuilt', 'mean'),
        mean_commfar=('commfar', 'mean'),
        **{m: (m, 'mean') for m in MEASURES},
    ).reset_index()
    agg['CounDist'] = pd.to_numeric(agg['CounDist'], errors='coerce')
    agg = agg.dropna(subset=['CounDist']).reset_index(drop=True)
    agg['CounDist'] = agg['CounDist'].astype(int)

    gdf = gpd.read_file(COUNCIL_SHP)
    gdf['CounDist'] = pd.to_numeric(gdf['CounDist'], errors='coerce')
    gdf = gdf.dropna(subset=['CounDist'])
    gdf['CounDist'] = gdf['CounDist'].astype(int)
    merged = gdf[['CounDist', 'geometry']].merge(agg, on='CounDist', how='inner')
    merged = merged.sort_values('CounDist').reset_index(drop=True)

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

    np.random.seed(SEED)
    centroid = merged.geometry.centroid.set_crs(merged.crs)
    coords = np.column_stack([centroid.x.values, centroid.y.values]).astype(float)
    w = KNN(coords, k=K)
    w.transform = 'r'

    out = {'n_districts': int(len(merged)), 'weights': f'knn{K}',
           'seed': SEED, 'by_measure': {}}
    for m in MEASURES:
        sub = merged.dropna(subset=[m]).reset_index(drop=True)
        y = sub[['median_log_price']].values
        X = sub[[m] + CONTROLS].values.astype(float)
        names = [m] + CONTROLS
        rec = {'pooled': run_variant(y, X, w, names, 'pooled (no borough FE)')}
        if boro_dummies is not None:
            bd = boro_dummies.loc[sub.index].reset_index(drop=True)
            Xb = np.column_stack([X, bd.to_numpy(dtype=float)])
            rec['borough_fe'] = run_variant(
                y, Xb, w, names + list(bd.columns),
                'borough FE (district-level borough dummies)')
        out['by_measure'][m] = rec
        rho = rec['pooled'].get('spatial_lag_model', {}) \
            .get('W_median_log_price', {}).get('coef')
        print(f'  {m}: pooled SLM rho={rho if rho is None else round(rho, 4)}; '
              f'OLS residual I='
              f'{rec["pooled"]["ols_residual_morans_i"]["I"]:.4f}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'Wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()

