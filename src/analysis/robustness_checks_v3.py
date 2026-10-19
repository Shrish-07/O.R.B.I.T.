"""v3 robustness suite (Round 13, R2): every spec x NEW ideology measure x sample.

Re-runs the exact robustness_checks_v2 specification set (Models 3, 4, 5,
6, 7, 8 + the upgraded hedonic; full and restricted samples; classical
SE, CR1 cluster SE, p from t(G-1), wild-cluster-bootstrap-t p with Webb
6-point weights, 9,999 reps, seed 42) for the NINE new Round-13 ideology
measures — the six area-weighted (_aw) variants and the three
dominant-district (_dom) variants built by build_ideology_scores_v3.py —
on data/canonical/modeling_dataset_canonical_v3.parquet.

The legacy robustness_checks_v2.json and every legacy column are
untouched; this file re-uses the v2 spec builders so the two suites are
comparable by construction.

Output: results/robustness_checks_v3.json

Run: python src/analysis/robustness_checks_v3.py
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
OUT = ROOT / 'results' / 'robustness_checks_v3.json'

MEASURES = ['dem_share_aw', 'dem_share_mayor_allcand_aw', 'dem_share_pres_aw',
            'dem_share_vintage_aw', 'dem_share_mayor_allcand_vintage_aw',
            'dem_share_pres_vintage_aw', 'dem_share_vintage_dom',
            'dem_share_mayor_allcand_vintage_dom', 'dem_share_pres_vintage_dom']
SPECS = v2.SPECS
REPS = v2.REPS
SEED = v2.SEED



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
    masks = v2.samples(df)

    results = {
        'metadata': {
            'canonical_rows': int(len(df)),
            'canonical_file': CANON.relative_to(ROOT).as_posix(),
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
            'measures_note': ('Round 13 split-ED fix: _aw = area-weighted '
                              '(each ED split across its council districts '
                              'by normalized intersection-area weights); '
                              '_dom = ED wholly to the largest-overlap CD. '
                              'See results/crosswalk_audit_v3.json.'),
        },
    }

    for sample_name, mask in masks.items():
        results[sample_name] = {}
        for measure in MEASURES:
            results[sample_name][measure] = {}
            base = df[mask & df[measure].notna()].copy()
            for spec in SPECS:
                extra = v2.SPEC_EXTRA_REQ.get(spec, [])
                d = base.dropna(subset=[c for c in extra if c in base.columns])
                if spec == 'hedonic_upgraded':
                    d = d[(d['log_gross_sqft'] > 0) & (d['log_land_sqft'] > 0)
                          & (d['building_class_category'].astype(str) != '')
                          & (d['TAX CLASS AT TIME OF SALE'].notna())]
                if len(d) < 1000:
                    results[sample_name][measure][spec] = {
                        'error': f'insufficient rows ({len(d)})'}
                    continue
                X, y, cl = v2.build_design(d, spec, measure)
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
