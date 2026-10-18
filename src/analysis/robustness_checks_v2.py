"""v2 robustness suite (Task E): every spec x ideology measure x sample.

Recomputes the paper's Models 3, 4 and 5 with t(G-1) and wild-cluster-
bootstrap inference, adds Models 6, 7 and 8 and an upgraded hedonic
specification, and runs EVERY spec for EVERY ideology measure (legacy
dem_share, all-candidate mayoral share, presidential share, and the
vintage-corrected variants) on BOTH the full analysis sample and the
restricted analysis sample.

Specs (clustering by council district everywhere; each report carries
coef, classical SE, cluster SE, p from t(G-1), wild-cluster-bootstrap-t
p (Webb weights, 9,999 reps, seed 42), 95% CIs, N, G, R^2):

  model3  : measure + [yearbuilt, numfloors, landuse, commfar] + borough FE
            (the paper's Model 3, controls identical)
  model4  : measure + core controls + 51 council-district FE (paper Model 4)
  model5  : measure + core controls + borough FE + sale-year FE (paper M5)
  model6  : measure + core controls + RESIDENTIAL/COMMERCIAL UNITS +
            district FE + sale-year FE
  model7  : model6 with borough x sale-year FE replacing plain sale-year FE
  model8  : measure + core + units + district FE + district-specific linear
            trends (district x (sale_year-2016)) + sale-year FE
  hedonic : measure + log(GROSS SQUARE FEET) + log(LAND SQUARE FEET) +
            RESIDENTIAL UNITS + TOTAL UNITS + landuse as CATEGORICAL +
            building-class FE (normalized building_class_category) +
            tax-class-at-sale FE + district FE + sale-year FE (R^2 reported)

Samples:
  full       : canonical rows with non-missing outcome + core controls
  restricted : full + is_resid_tc12 + price_ge_100k + NOT multi_parcel_flag
               (definition recorded in results/sample_funnel.json)

Output: results/robustness_checks_v2.json

Run: python src/analysis/robustness_checks_v2.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wild_cluster_inference import WildClusterOLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet'
OUT = ROOT / 'results' / 'robustness_checks_v2.json'

CORE = ['yearbuilt', 'numfloors', 'landuse', 'commfar']
UNITS = ['RESIDENTIAL UNITS', 'COMMERCIAL UNITS']
MEASURES = ['dem_share', 'dem_share_mayor_allcand', 'dem_share_pres',
            'dem_share_vintage', 'dem_share_mayor_allcand_vintage',
            'dem_share_pres_vintage']
SPECS = ['model3_boro_fe', 'model4_district_fe', 'model5_boro_year_fe',
         'model6_district_year_fe_units', 'model7_district_boro_x_year_fe',
         'model8_district_linear_trends', 'hedonic_upgraded']
REPS = 9999
SEED = 42

def samples(df):
    """Return dict sample_name -> boolean mask (mirrors sample_funnel.py)."""
    base_ok = df['target_log_price'].notna() & df['CounDist'].notna() \
        & df['sale_year'].notna()
    for c in CORE:
        base_ok &= df[c].notna()
    restricted = base_ok & df['is_resid_tc12'] & df['price_ge_100k'] \
        & (~df['multi_parcel_flag'])
    return {'full': base_ok, 'restricted': restricted}


def build_design(d, spec, measure):
    """Assemble (X DataFrame, y, clusters) for one spec/measure on rows d."""
    parts = [pd.Series(1.0, index=d.index, name='const'),
             d[[measure]].astype(float)]
    if spec.startswith('hedonic'):
        cols = [measure, 'log_gross_sqft', 'log_land_sqft',
                'RESIDENTIAL UNITS', 'TOTAL UNITS']
        parts = [pd.Series(1.0, index=d.index, name='const'),
                 d[cols].astype(float)]
        parts.append(pd.get_dummies(d['landuse'].astype('Int64')
                                    .astype(str), prefix='landuse',
                                    drop_first=True, dtype=float))
        parts.append(pd.get_dummies(d['building_class_category'].astype(str),
                                     prefix='bldgclasscat', drop_first=True,
                                     dtype=float))
        parts.append(pd.get_dummies(d['TAX CLASS AT TIME OF SALE'].astype(str),
                                    prefix='taxclass', drop_first=True,
                                    dtype=float))
    else:
        parts.append(d[CORE].astype(float))
        if spec in ('model6_district_year_fe_units',
                    'model7_district_boro_x_year_fe',
                    'model8_district_linear_trends'):
            parts.append(d[UNITS].astype(float))
    dist = pd.get_dummies(d['CounDist'].astype(int), prefix='district',
                          drop_first=True, dtype=float)
    yr = pd.get_dummies(d['sale_year'].astype(int), prefix='sale_year',
                        drop_first=True, dtype=float)
    if spec == 'model3_boro_fe':
        parts.append(pd.get_dummies(d['BOROUGH'].astype(int), prefix='borough',
                                    drop_first=True, dtype=float))
    elif spec == 'model4_district_fe':
        parts.append(dist)
    elif spec == 'model5_boro_year_fe':
        parts.append(pd.get_dummies(d['BOROUGH'].astype(int), prefix='borough',
                                    drop_first=True, dtype=float))
        parts.append(yr)
    elif spec in ('model6_district_year_fe_units',
                  'model8_district_linear_trends'):
        parts.append(dist)
        parts.append(yr)
        if spec == 'model8_district_linear_trends':
            t = (d['sale_year'].astype(float) - 2016.0)
            parts.append(dist.mul(t.values, axis=0).add_suffix('_xtrend'))
    elif spec == 'model7_district_boro_x_year_fe':
        by = (d['BOROUGH'].astype(int).astype(str) + 'x'
              + d['sale_year'].astype(int).astype(str))
        parts.append(dist)
        parts.append(pd.get_dummies(by, prefix='boroXyear', drop_first=True,
                                    dtype=float))
    elif spec.startswith('hedonic'):
        parts.append(dist)
        parts.append(yr)
    X = pd.concat(parts, axis=1)
    # drop constant columns EXCEPT the intercept (needed for a well-posed
    # R^2 and the wild bootstrap's restricted model)
    keep = [c for c in X.columns
            if c == 'const' or X[c].nunique(dropna=False) > 1]
    X = X[keep]
    return X, d['target_log_price'].astype(float).to_numpy(), \
        d['CounDist'].astype(int).to_numpy()

HEDONIC_EXTRA_REQ = ['log_gross_sqft', 'log_land_sqft',
                     'RESIDENTIAL UNITS', 'TOTAL UNITS']
SPEC_EXTRA_REQ = {
    'model3_boro_fe': ['BOROUGH'],
    'model5_boro_year_fe': ['BOROUGH'],
    'model7_district_boro_x_year_fe': ['BOROUGH'] + UNITS,
    'model6_district_year_fe_units': UNITS,
    'model8_district_linear_trends': UNITS,
    'hedonic_upgraded': HEDONIC_EXTRA_REQ,
}


def main():
    t0 = time.time()
    df = pd.read_parquet(CANON)
    g = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
    l = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
    with np.errstate(invalid='ignore'):
        df['log_gross_sqft'] = np.log(g.where(g > 0))
        df['log_land_sqft'] = np.log(l.where(l > 0))
    masks = samples(df)

    results = {
        'metadata': {
            'canonical_rows': int(len(df)),
            'specs': SPECS,
            'measures': MEASURES,
            'samples': {
                'full': 'non-missing outcome + core controls + district/year',
                'restricted': 'full + is_resid_tc12 + price_ge_100k + NOT '
                              'multi_parcel_flag (see results/sample_funnel.json)'},
            'clustering': 'council district (CounDist)',
            'inference': 'classical SE, CR1 cluster SE (statsmodels/Stata '
                         'convention), p from t(G-1), wild-cluster-bootstrap-t '
                         'p with Webb 6-point weights, 9999 reps, seed 42',
            'controls': {
                'core': CORE,
                'units': UNITS,
                'hedonic': ['log_gross_sqft', 'log_land_sqft',
                            'RESIDENTIAL UNITS', 'TOTAL UNITS',
                            'landuse (categorical dummies)',
                            'building_class_category FE',
                            'TAX CLASS AT TIME OF SALE FE',
                            'district FE', 'sale-year FE']},
        },
    }

    for sample_name, mask in masks.items():
        results[sample_name] = {}
        for measure in MEASURES:
            results[sample_name][measure] = {}
            base = df[mask & df[measure].notna()].copy()
            for spec in SPECS:
                extra = SPEC_EXTRA_REQ.get(spec, [])
                d = base.dropna(subset=[c for c in extra if c in base.columns])
                if spec == 'hedonic_upgraded':
                    d = d[(d['log_gross_sqft'] > 0) & (d['log_land_sqft'] > 0)
                          & (d['building_class_category'].astype(str) != '')
                          & (d['TAX CLASS AT TIME OF SALE'].notna())]
                if len(d) < 1000:
                    results[sample_name][measure][spec] = {
                        'error': f'insufficient rows ({len(d)})'}
                    continue
                X, y, cl = build_design(d, spec, measure)
                focus_idx = list(X.columns).index(measure)
                t1 = time.time()
                ols = WildClusterOLS(y, X.to_numpy(dtype=np.float64), cl,
                                     focus_idx).fit()
                rep = ols.report(reps=REPS, seed=SEED)
                rep['k_params'] = int(X.shape[1])
                rep['fit_seconds'] = round(time.time() - t1, 2)
                results[sample_name][measure][spec] = rep
                del X, ols
                print(f'  [{sample_name}/{measure}/{spec}] '
                      f'coef={rep["coef"]:+.5f} p_t={rep["p_t_G_minus_1"]:.4f} '
                      f'p_wild={rep["p_wild_cluster_bootstrap_t"]:.4f} '
                      f'N={rep["n"]:,} G={rep["n_clusters_G"]} '
                      f'({rep["fit_seconds"]}s)', flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Wrote {OUT.relative_to(ROOT)} '
          f'(total {round((time.time() - t0) / 60, 1)} min)')


if __name__ == '__main__':
    main()

