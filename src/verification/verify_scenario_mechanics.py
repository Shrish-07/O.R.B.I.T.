"""Round 5 fact-check (kept maintained through Round 11): Section K
(counterfactual mechanics + clip rates).

Round 9 (2026-09-27) sync: the producer (src/analysis/political_scenarios.py)
now runs all three scenarios for EVERY eligible ranked model. This verifier
mirrors that ranking exactly -- identical model list from
src/ops/champion_ranking.py (rank 1 = champion.json selection; ranks 2..N
ascending test MAE among untainted, non-stale, post-2026-05-18 runs), with
all predictions via src/ops/predictors.predict_entry_with_features so the
verifier uses the SAME family-optimized prediction convention as the
producer and agreement between them is by construction.

NaN-handling fix folded in (pre-existing latent bug, NOT introduced by Round
9): the pre-Round-9 verifier re-predicted with booster.predict(...
.fillna(0)) for every model, which disagreed with the producer's
native-NaN/FIX C16 behavior for the LightGBM scenario models. This only
never surfaced because the RF-era champion was sklearn-trained on fillna(0)
and the hard-coded political LightGBM's fillna(0)-vs-native difference is
small -- both coincidences masked the bug. Predictions now flow through the
same predictors.predict_entry_with_features factory as the producer.

Output: results/round5_part_K_mechanics_rank{N}.json per rank, plus the
legacy results/round5_part_K_mechanics.json carrying the rank-1 section
(cited by archive/_truth_assemble.py and the Round 5 verification record).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _shared import ROOT, write_json, record_status, check_val

sys.path.insert(0, str(ROOT))
from src.ops.champion_ranking import get_all_models
from src.ops.predictors import load_entry_model, features_of, predict_entry_with_features

MECHANICS_LIBERAL = ('R3->R7 substring substitution on zonedist1 + residfar (*first far col*) x1.5 '
                     '(Tier 1: read from src/analysis/political_scenarios.py -- unchanged Round 9)')
MECHANICS_CONSERVATIVE = ('R7->R3 substring substitution on zonedist1 + residfar (*first far col*) x0.7 '
                          '(Tier 1: read from src/analysis/political_scenarios.py -- unchanged Round 9)')
MECHANICS_MIXED = ('dem_share + 0.15 clipped to [0,1], re-predicted through the SAME rank model as '
                   'liberal/conservative for this rank (Round 9: mixed_governance is no longer '
                   'hard-wired to lgbm_all_years_political) (Tier 1: read from '
                   'src/analysis/political_scenarios.py)')


def build_frames(df):
    z_cols = [c for c in df.columns if 'zonedist' in c.lower() or 'zoning' in c.lower()]
    far_cols = [c for c in df.columns if 'far' in c.lower() or 'gross_floor' in c.lower()]
    zc = z_cols[0] if z_cols else None
    fc_col = far_cols[0] if far_cols else None
    frames = {}
    df_lib = df.copy(deep=True)
    if zc:
        df_lib[zc] = df_lib[zc].astype(str).apply(
            lambda z: z.replace('R3', 'R7') if isinstance(z, str) else z)
    if fc_col:
        df_lib[fc_col] = df_lib[fc_col].astype(float).fillna(0) * 1.5
    frames['liberal'] = df_lib
    df_cons = df.copy(deep=True)
    if zc:
        df_cons[zc] = df_cons[zc].astype(str).apply(
            lambda z: z.replace('R7', 'R3') if isinstance(z, str) else z)
    if fc_col:
        df_cons[fc_col] = df_cons[fc_col].astype(float).fillna(0) * 0.7
    frames['conservative'] = df_cons
    df_mix = df.copy(deep=True)
    if 'dem_share' in df_mix.columns:
        df_mix['dem_share'] = (df_mix['dem_share'] + 0.15).clip(0.0, 1.0)
    frames['mixed'] = df_mix
    return frames


def scenario_pct(predict_fn, scen_df, base_df):
    bp = predict_fn(base_df)
    sp = predict_fn(scen_df)
    pc = (np.exp(sp) - np.exp(bp)) / np.exp(bp) * 100.0
    mask = bp >= 9.0
    pc = pc[mask]
    pc_clipped = np.clip(pc, -100.0, 100.0)
    return pc, pc_clipped, mask


def main():
    df_path = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet'
    if not df_path.exists():
        df_path = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical.parquet'
    df = pd.read_parquet(df_path)
    frames = build_frames(df)

    _train_sp = pd.read_parquet(ROOT / 'data' / 'splits' / 'all_years_train.parquet',
                                columns=['dem_share'])
    _tr_max = float(_train_sp['dem_share'].max())
    _base_ds = df['dem_share']
    _n_ds = int(_base_ds.notna().sum())
    _shocked = _base_ds + 0.15
    _n_clip = int((_shocked > 1.0).sum())
    _n_beyond = int((_shocked > _tr_max).sum())

    diag_dir = ROOT / 'results' / 'political_scenarios'
    _status_checks = []

    for ent in get_all_models(write_audit=False):
        rank = int(ent['rank'])
        model, preproc = load_entry_model(ent)
        feats = features_of(ent)

        def predict_fn(X, _m=model, _p=preproc, _e=ent, _f=feats):
            return predict_entry_with_features(_e, _f, X, model=_m, preproc=_p)

        K = {}
        K['mechanics_liberal'] = MECHANICS_LIBERAL
        K['mechanics_conservative'] = MECHANICS_CONSERVATIVE
        K['mechanics_mixed'] = MECHANICS_MIXED
        K['filter_base_pred_lt_9_removed'] = ('base_pred >= 9.0 keeps rows; removes rows with base '
                                              'predicted log-price < 9.0')
        K['clip_to_neg100_100'] = 'pct_change clipped to [-100, 100]'
        K['rank'] = rank
        K['model_name'] = ent['name']
        K['model_family'] = ent['family']
        K['model_mode'] = ent['mode']
        K['test_mae'] = ent['test_mae']

        base_preds = predict_fn(df)
        K['base_pred_min'] = float(np.min(base_preds))
        K['base_pred_max'] = float(np.max(base_preds))
        K['filter_9.0_removed_rows'] = int(np.sum(base_preds < 9.0))
        K['filter_9.0_removes_zero_rows'] = bool(K['filter_9.0_removed_rows'] == 0)

        lib_pc, lib_pc_c, _m1 = scenario_pct(predict_fn, frames['liberal'], df)
        cons_pc, cons_pc_c, _m2 = scenario_pct(predict_fn, frames['conservative'], df)
        mix_pc, mix_pc_c, _m3 = scenario_pct(predict_fn, frames['mixed'], df)
        K['clip_counts'] = {
            'liberal_upper': int(np.sum(lib_pc > 100.0)),
            'liberal_lower': int(np.sum(lib_pc < -100.0)),
            'conservative_upper': int(np.sum(cons_pc > 100.0)),
            'conservative_lower': int(np.sum(cons_pc < -100.0)),
            'mixed_upper': int(np.sum(mix_pc > 100.0)),
            'mixed_lower': int(np.sum(mix_pc < -100.0)),
        }

        K['mixed_support_diag'] = {
            'n_rows_with_dem_share': _n_ds,
            'n_shock_clipped_at_1p0': _n_clip,
            'pct_shock_clipped_at_1p0': round(100.0 * _n_clip / _n_ds, 2),
            'train_dem_share_max_all_years_split': _tr_max,
            'n_shocked_beyond_train_support': _n_beyond,
            'pct_beyond_train_support': round(100.0 * _n_beyond / _n_ds, 2),
        }
        # producer comparison: per-rank diag file written by
        # political_scenarios.py on this run
        _diag_path = diag_dir / f'mixed_governance_support_diag_rank{rank}.json'
        if _diag_path.exists():
            _prod = json.loads(_diag_path.read_text())
            K['mixed_support_diag_producer_model'] = _prod.get('model_name')
            K['mixed_support_diag_matches_producer'] = bool(all(
                K['mixed_support_diag'].get(k) == _prod.get(k)
                for k in ['n_rows_with_dem_share', 'n_shock_clipped_at_1p0',
                          'n_shocked_beyond_train_support',
                          'train_dem_share_max_all_years_split']))
        else:
            K['mixed_support_diag_matches_producer'] = None
        _status_checks.append(check_val(
            f'rank{rank}.mixed_support_diag_matches_producer', True,
            K['mixed_support_diag_matches_producer']))

        out = ROOT / 'results' / f'round5_part_K_mechanics_rank{rank}.json'
        write_json(out, {'section_K_mechanics': K})
        print(f'WROTE {out.name}', '| rank', rank, '| model', ent['name'],
              '| base_pred_min', K['base_pred_min'], '| filter removes',
              K['filter_9.0_removed_rows'], 'rows | clips', K['clip_counts'])

        if rank == 1:
            # legacy filename keeps working (rank-1 / champion content), cited
            # by archive/_truth_assemble.py and the Round 5 report tables.
            write_json(ROOT / 'results' / 'round5_part_K_mechanics.json',
                       {'section_K_mechanics': K})
            print('WROTE results/round5_part_K_mechanics.json (legacy = rank 1)')
    # Round 11: machine-countable status (per-rank producer-diagnostic
    # concordance; clip counts printed above are recomputed fresh each run --
    # no paper-era constants are referenced).
    record_status('verify_scenario_mechanics', _status_checks,
                  note='Per-rank mixed-governance support diagnostics '
                       "re-derived and compared to the producer's rank diag "
                       'files; all scenario counts recomputed live.')


if __name__ == '__main__':
    main()
