"""Hedonic sample audit + land/units variant (Round 13, R8).

Part 1 — why N = 23,789: the upgraded hedonic specification
(robustness_checks_v2/v3) requires positive GROSS and LAND square feet,
and those fields are missing (or zero) for the large majority of the
520,365-row analysis base sample. This stage quantifies the
non-missingness shares by building-class category and tax class (at time
of sale) on the base sample, plus the marginal effect of each hedonic
requirement, so the funnel from 520,365 to the 23,789-row hedonic sample
is fully explained.

Part 2 — hedonic_land_units variant: the hedonic specification with the
size controls reduced to log land square feet + RESIDENTIAL UNITS +
TOTAL UNITS (log GROSS square feet dropped), on the LARGER sample that
only requires positive land square feet, for every measure (15). FE set
unchanged: landuse categorical + building-class-category FE + tax-class
FE + district FE + sale-year FE. N is reported per measure.

Output: results/hedonic_v3.json

Run: python src/analysis/hedonic_v3.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robustness_checks_v2 as v2  # noqa: E402
from wild_cluster_inference import WildClusterOLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v3.parquet'
OUT = ROOT / 'results' / 'hedonic_v3.json'

LEGACY_MEASURES = ['dem_share', 'dem_share_mayor_allcand', 'dem_share_pres',
                   'dem_share_vintage', 'dem_share_mayor_allcand_vintage',
                   'dem_share_pres_vintage']
NEW_MEASURES = ['dem_share_aw', 'dem_share_mayor_allcand_aw', 'dem_share_pres_aw',
                'dem_share_vintage_aw', 'dem_share_mayor_allcand_vintage_aw',
                'dem_share_pres_vintage_aw', 'dem_share_vintage_dom',
                'dem_share_mayor_allcand_vintage_dom', 'dem_share_pres_vintage_dom']
MEASURES = LEGACY_MEASURES + NEW_MEASURES
REPS = v2.REPS
SEED = v2.SEED


def build_land_units_design(d, measure):
    parts = [pd.Series(1.0, index=d.index, name='const'),
             d[[measure, 'log_land_sqft', 'RESIDENTIAL UNITS',
                'TOTAL UNITS']].astype(float)]
    parts.append(pd.get_dummies(d['landuse'].astype('Int64').astype(str),
                                prefix='landuse', drop_first=True,
                                dtype=float))
    parts.append(pd.get_dummies(d['building_class_category'].astype(str),
                                prefix='bldgclasscat', drop_first=True,
                                dtype=float))
    parts.append(pd.get_dummies(d['TAX CLASS AT TIME OF SALE'].astype(str),
                                prefix='taxclass', drop_first=True,
                                dtype=float))
    parts.append(pd.get_dummies(d['CounDist'].astype(int), prefix='district',
                                drop_first=True, dtype=float))
    parts.append(pd.get_dummies(d['sale_year'].astype(int), prefix='sale_year',
                                drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    keep = [c for c in X.columns
            if c == 'const' or X[c].nunique(dropna=False) > 1]
    X = X[keep]
    return X, d['target_log_price'].astype(float).to_numpy(), \
        d['CounDist'].astype(int).to_numpy()



def main():
    t0 = time.time()
    if not CANON.exists():
        raise SystemExit(f'MISSING INPUT: {CANON} — run '
                         'src/data_pipeline/extend_canonical_v3.py first.')
    df = pd.read_parquet(CANON)
    g = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
    l = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
    with np.errstate(invalid='ignore'):
        df['log_gross_sqft'] = np.log(g.where(g > 0))
        df['log_land_sqft'] = np.log(l.where(l > 0))
    base = v2.samples(df)['full']
    b = df[base].copy()

    gross_ok = pd.to_numeric(b['GROSS SQUARE FEET'], errors='coerce') > 0
    land_ok = pd.to_numeric(b['LAND SQUARE FEET'], errors='coerce') > 0

    def by_group(col):
        both = gross_ok & land_ok
        grp = b.groupby(col).agg(rows=('target_log_price', 'size'))
        grp['gross_sqft_nonmissing'] = gross_ok.groupby(b[col]).sum().astype(int)
        grp['land_sqft_nonmissing'] = land_ok.groupby(b[col]).sum().astype(int)
        grp['both_nonmissing'] = both.groupby(b[col]).sum().astype(int)
        for c in ['gross_sqft_nonmissing', 'land_sqft_nonmissing',
                  'both_nonmissing']:
            grp[c + '_pct'] = (100.0 * grp[c] / grp['rows']).round(2)
        return {str(k): {kk: (int(vv) if not isinstance(vv, float) else float(vv))
                         for kk, vv in r.items()}
                for k, r in grp.iterrows()}

    funnel = [
        {'step': 'analysis base sample', 'rows': int(len(b))},
        {'step': 'land sq ft > 0', 'rows': int(land_ok.sum())},
        {'step': 'gross sq ft > 0', 'rows': int(gross_ok.sum())},
        {'step': 'gross and land sq ft both > 0 (upgraded hedonic sample)',
         'rows': int((gross_ok & land_ok).sum())},
        {'step': ('both > 0 + building-class-category non-empty + tax class '
                  'non-missing (robustness hedonic sample, N=23,789 for the '
                  'legacy measures)'),
         'rows': int(((gross_ok & land_ok)
                      & (b['building_class_category'].astype(str) != '')
                      & b['TAX CLASS AT TIME OF SALE'].notna()).sum())},
    ]

    results = {
        'part1_why_n_23789': {
            'headline': ('The upgraded hedonic specification requires '
                         'positive GROSS and LAND square feet; only '
                         f'{int((gross_ok & land_ok).sum()):,} of the '
                         f'{len(b):,}-row analysis base sample satisfy both, '
                         'and after the building-class/tax-class FE '
                         'requirements 23,789 rows remain for the legacy '
                         'measures. Missing gross/land square footage is '
                         'concentrated in the smaller building-class and '
                         'tax-class categories (see shares below).'),
            'funnel': funnel,
            'share_nonmissing_by_building_class_category':
                by_group('building_class_category'),
            'share_nonmissing_by_tax_class_at_time_of_sale':
                by_group('TAX CLASS AT TIME OF SALE'),
        },
        'part2_hedonic_land_units': {
            'spec': ('measure + log(LAND SQUARE FEET) + RESIDENTIAL UNITS + '
                     'TOTAL UNITS + landuse categorical + building-class-'
                     'category FE + tax-class-at-sale FE + district FE + '
                     'sale-year FE; sample = analysis base + land sq ft > 0 '
                     '+ units non-missing + building-class/tax-class '
                     'non-missing (log GROSS square feet NOT required); '
                     'clustered by council district, wild p (Webb 6, 9999 '
                     'reps, seed 42)'),
            'by_measure': {},
        },
    }

    sample_mask = base & (df['log_land_sqft'] > 0) \
        & df['RESIDENTIAL UNITS'].notna() & df['TOTAL UNITS'].notna() \
        & (df['building_class_category'].astype(str) != '') \
        & df['TAX CLASS AT TIME OF SALE'].notna()
    print(f'land/units hedonic sample: {int(sample_mask.sum()):,} rows')

    for measure in MEASURES:
        d = df[sample_mask & df[measure].notna()].copy()
        if len(d) < 1000:
            results['part2_hedonic_land_units']['by_measure'][measure] = {
                'error': f'insufficient rows ({len(d)})'}
            continue
        X, y, cl = build_land_units_design(d, measure)
        focus_idx = list(X.columns).index(measure)
        t1 = time.time()
        ols = WildClusterOLS(y, X.to_numpy(dtype=np.float64), cl,
                             focus_idx).fit()
        rep = ols.report(reps=REPS, seed=SEED)
        rep['k_params'] = int(X.shape[1])
        rep['fit_seconds'] = round(time.time() - t1, 2)
        results['part2_hedonic_land_units']['by_measure'][measure] = rep
        del X, ols
        print(f'  [land_units/{measure}] coef={rep["coef"]:+.5f} '
              f'p_t={rep["p_t_G_minus_1"]:.4f} '
              f'p_wild={rep["p_wild_cluster_bootstrap_t"]:.4f} '
              f'N={rep["n"]:,} G={rep["n_clusters_G"]} '
              f'({rep["fit_seconds"]}s)', flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Wrote {OUT.relative_to(ROOT)} '
          f'(total {round((time.time() - t0) / 60, 1)} min)')


if __name__ == '__main__':
    main()
