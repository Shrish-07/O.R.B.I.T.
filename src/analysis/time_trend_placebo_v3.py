"""Time-trend placebo (Round 13, R6): wave-mean vs demeaned measures.

For each of the six existing measures plus the _aw versions of the three
vintage ones (9 measures) this stage builds:

  (a) measure_wavemean — the citywide (across the districts) mean of the
      measure in that election wave, identical for every district within
      the wave (mayoral measures wave on election_year; presidential
      measures wave on election_year_pres);
  (b) measure_dev — measure minus measure_wavemean (mean-zero within
      wave by construction).

It then runs Model 4 (district FE, no year FE, cluster by district, wild
p; Webb 6-point weights, 9,999 reps, seed 42) with each. Expected: (a)
reproduces the sign/magnitude of the original Model 4 coefficient — the
wave-mean carries only the across-wave (citywide time-trend) variation —
and (b) is near zero, because the demeaned measure carries only the
within-wave cross-district variation which the district FE absorb. Both
are reported side by side with the original Model 4 estimate.

Output: results/time_trend_placebo_v3.json

Run: python src/analysis/time_trend_placebo_v3.py
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
ROB2 = ROOT / 'results' / 'robustness_checks_v2.json'
ROB3 = ROOT / 'results' / 'robustness_checks_v3.json'
OUT = ROOT / 'results' / 'time_trend_placebo_v3.json'

MAYORAL_MEASURES = ['dem_share', 'dem_share_mayor_allcand', 'dem_share_vintage',
                    'dem_share_mayor_allcand_vintage', 'dem_share_vintage_aw',
                    'dem_share_mayor_allcand_vintage_aw']
PRES_MEASURES = ['dem_share_pres', 'dem_share_pres_vintage',
                 'dem_share_pres_vintage_aw']
MEASURES = MAYORAL_MEASURES + PRES_MEASURES
REPS = v2.REPS
SEED = v2.SEED


def wave_col_for(measure):
    return 'election_year' if measure in MAYORAL_MEASURES else 'election_year_pres'


def original_model4(measure, rob2, rob3):
    if measure in rob2['full']:
        return rob2['full'][measure]['model4_district_fe']
    if measure in rob3.get('full', {}):
        return rob3['full'][measure]['model4_district_fe']
    return None


def fit_model4(df, regressor):
    """Model 4 (district FE, no year FE, cluster by district, wild p)."""
    base_ok = df['target_log_price'].notna() & df['CounDist'].notna() \
        & df['sale_year'].notna() & df[regressor].notna()
    for c in v2.CORE:
        base_ok &= df[c].notna()
    d = df[base_ok].copy()
    X, y, cl = v2.build_design(d, 'model4_district_fe', regressor)
    focus_idx = list(X.columns).index(regressor)
    ols = WildClusterOLS(y, X.to_numpy(dtype=np.float64), cl, focus_idx).fit()
    rep = ols.report(reps=REPS, seed=SEED)
    del X, ols
    return rep


def main():
    t0 = time.time()
    if not CANON.exists():
        raise SystemExit(f'MISSING INPUT: {CANON} — run '
                         'src/data_pipeline/extend_canonical_v3.py first.')
    for p in [ROB2, ROB3]:
        if not p.exists():
            raise SystemExit(f'MISSING INPUT: {p} — run robustness_checks_v2 '
                             'and robustness_checks_v3 first.')
    rob2 = json.loads(ROB2.read_text(encoding='utf-8'))
    rob3 = json.loads(ROB3.read_text(encoding='utf-8'))
    df = pd.read_parquet(CANON)

    results = {
        'metadata': {
            'spec': ('Model 4: measure + core controls + 51 council-district '
                     'FE, NO sale-year FE, clustered by district, wild-'
                     'cluster-bootstrap-t p (Webb 6, 9999 reps, seed 42)'),
            'measures': MEASURES,
            'wavemean_definition': ('citywide mean of the CD-level measure '
                                    'across the districts within the election '
                                    'wave (mayoral: election_year; '
                                    'presidential: election_year_pres), '
                                    'identical for every district in the wave'),
            'dev_definition': 'measure - measure_wavemean (mean-zero in wave)',
            'expected': ('(a) wavemean reproduces the sign/magnitude of the '
                         'original Model 4; (b) dev is near zero'),
        },
    }

    for m in MEASURES:
        wc = wave_col_for(m)
        cd_wave = df[['CounDist', wc, m]].dropna().drop_duplicates()
        n_districts_per_wave = cd_wave.groupby(wc)['CounDist'].nunique()
        wave_mean = cd_wave.groupby(wc)[m].mean().rename(m + '_wavemean')
        # wavemean is a wave-level constant: map by wave only
        df = df.merge(wave_mean.reset_index(), on=wc, how='left')
        df[m + '_dev'] = df[m] - df[m + '_wavemean']

        # identity gates
        chk = df[['CounDist', wc, m + '_wavemean']].dropna().drop_duplicates()
        per_wave_nunique = chk.groupby(wc)[m + '_wavemean'].nunique()
        assert (per_wave_nunique == 1).all(), f'{m}: wavemean not constant in wave'
        dev_chk = df[[wc, m + '_dev']].dropna().drop_duplicates()
        dev_mean = dev_chk.groupby(wc)[m + '_dev'].mean()
        assert (dev_mean.abs() < 1e-9).all(), f'{m}: dev not mean-zero in wave'

        rep_a = fit_model4(df, m + '_wavemean')
        rep_b = fit_model4(df, m + '_dev')
        orig = original_model4(m, rob2, rob3)

        verdict_a = None
        verdict_b = None
        if orig is not None:
            verdict_a = {
                'original_model4_coef': orig['coef'],
                'sign_matches_original': bool(np.sign(rep_a['coef'])
                                             == np.sign(orig['coef'])),
                'ratio_wavemean_over_original': (
                    float(rep_a['coef'] / orig['coef'])
                    if orig['coef'] != 0 else None),
            }
        verdict_b = {
            'near_zero': bool(rep_b['ci95_low_t'] <= 0.0 <= rep_b['ci95_high_t']),
            'abs_coef': abs(rep_b['coef']),
        }

        results[m] = {
            'wave_column': wc,
            'n_districts_per_wave': {str(k): int(v)
                                     for k, v in n_districts_per_wave.items()},
            'wavemean_model4': rep_a,
            'dev_model4': rep_b,
            'comparison_to_original_model4': verdict_a,
            'dev_near_zero_verdict': verdict_b,
        }
        print(f'  {m}: wavemean coef={rep_a["coef"]:+.5f} '
              f'(orig {orig["coef"]:+.5f}) | dev coef={rep_b["coef"]:+.6f} '
              f'(CI [{rep_b["ci95_low_t"]:.5f}, {rep_b["ci95_high_t"]:.5f}])',
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Wrote {OUT.relative_to(ROOT)} '
          f'(total {round((time.time() - t0) / 60, 1)} min)')


if __name__ == '__main__':
    main()

