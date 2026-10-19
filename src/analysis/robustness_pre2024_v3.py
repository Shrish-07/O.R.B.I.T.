"""Cuomo-free robustness (Round 13, R5): pre-2024 sample for every measure.

Runs Models 3, 4, 6, 7 and 8 on the sample_pre2024 flag (sale_year <=
2023 — the canonical column is defined exactly that way) for EVERY
ideology measure: the six legacy/vintage measures, the six _aw
area-weighted measures and the three _dom dominant-district measures
(15 measures x 5 specs = 75 fits), each with coef, cluster SE, p from
t(G-1), wild-cluster-bootstrap-t p (Webb 6-point weights, 9,999 reps,
seed 42), 95% CI, N and G, clustered by council district.

Motivation: Model 7's positive mayoral-vote estimates may come from the
2025 election alone — the 2025 mayoral race put the Cuomo "Fight and
Deliver" line on the ballot and the 2025 returns post-date the 2024
sale-year rows entirely. Restricting to sale_year <= 2023 removes the
2024/2025 election waves (2024 presidential + 2025 mayoral contamination)
from the estimating sample; the mayoral measures then carry only the
2017/2021 waves and the presidential measures only 2016/2020.

Output: results/robustness_pre2024_v3.json

Run: python src/analysis/robustness_pre2024_v3.py
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
OUT = ROOT / 'results' / 'robustness_pre2024_v3.json'

LEGACY_MEASURES = ['dem_share', 'dem_share_mayor_allcand', 'dem_share_pres',
                   'dem_share_vintage', 'dem_share_mayor_allcand_vintage',
                   'dem_share_pres_vintage']
NEW_MEASURES = ['dem_share_aw', 'dem_share_mayor_allcand_aw', 'dem_share_pres_aw',
                'dem_share_vintage_aw', 'dem_share_mayor_allcand_vintage_aw',
                'dem_share_pres_vintage_aw', 'dem_share_vintage_dom',
                'dem_share_mayor_allcand_vintage_dom', 'dem_share_pres_vintage_dom']
MEASURES = LEGACY_MEASURES + NEW_MEASURES
SPECS = ['model3_boro_fe', 'model4_district_fe',
         'model6_district_year_fe_units', 'model7_district_boro_x_year_fe',
         'model8_district_linear_trends']
REPS = v2.REPS
SEED = v2.SEED


def main():
    t0 = time.time()
    if not CANON.exists():
        raise SystemExit(f'MISSING INPUT: {CANON} — run '
                         'src/data_pipeline/extend_canonical_v3.py first.')
    df = pd.read_parquet(CANON)
    masks = v2.samples(df)
    mask = masks['full'] & df['sample_pre2024']

    results = {
        'metadata': {
            'canonical_rows': int(len(df)),
            'canonical_file': CANON.relative_to(ROOT).as_posix(),
            'sample': ('full analysis sample restricted to sample_pre2024 '
                       '(sale_year <= 2023); pre-2024 rows = '
                       f'{int(mask.sum()):,}'),
            'motivation': ('Tests whether Model 7\'s positive mayoral '
                           'estimates come from the 2025 Cuomo-line '
                           'contamination: sale years 2024-2025 (and the '
                           '2024 presidential / 2025 mayoral election '
                           'waves) are excluded.'),
            'specs': SPECS,
            'measures': MEASURES,
            'clustering': 'council district (CounDist)',
            'inference': 'classical SE, CR1 cluster SE, p from t(G-1), '
                         'wild-cluster-bootstrap-t p (Webb 6, 9999 reps, '
                         'seed 42), 95% CI, N, G',
        },
    }

    for measure in MEASURES:
        results[measure] = {}
        base = df[mask & df[measure].notna()].copy()
        for spec in SPECS:
            extra = v2.SPEC_EXTRA_REQ.get(spec, [])
            d = base.dropna(subset=[c for c in extra if c in base.columns])
            if len(d) < 1000:
                results[measure][spec] = {'error': f'insufficient rows ({len(d)})'}
                continue
            X, y, cl = v2.build_design(d, spec, measure)
            focus_idx = list(X.columns).index(measure)
            t1 = time.time()
            ols = WildClusterOLS(y, X.to_numpy(dtype=np.float64), cl,
                                 focus_idx).fit()
            rep = ols.report(reps=REPS, seed=SEED)
            rep['k_params'] = int(X.shape[1])
            rep['fit_seconds'] = round(time.time() - t1, 2)
            results[measure][spec] = rep
            del X, ols
            print(f'  [pre2024/{measure}/{spec}] '
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

