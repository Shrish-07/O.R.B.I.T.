"""Political counterfactual scenarios -- Section 4.7 / Appendix A.1 (Round 9).

BEFORE Round 9: liberal/conservative followed the experiments/champion.json
pointer family-agnostically (FIX C16: native NaN for boosters, no fillna(0)
on LightGBM), but mixed_governance was HARD-WIRED to lgbm_all_years_political
regardless of which model was champion. That asymmetry is the bug this
round removes.

NOW: all three scenario types (liberal_policy, conservative_policy,
mixed_governance) are scored for EVERY eligible ranked model resolved by
src/ops/champion_ranking.py (rank 1 = exactly champion.json's selection;
ranks 2..N = remaining eligible entries in ascending test MAE; untainted,
non-stale, post-2026-05-18). No scenario type is exempt, and none is pinned
to a specific model family. Each scenario frame is built ONCE (it is
model-independent apart from the dem_share mutation) and re-scored per rank.

Per-family input conventions come from src/ops/predictors.py -- each model is
scored exactly the way its own trainer scored it (LightGBM native NaN;
XGBoost/CatBoost/RandomForest fillna(0); ridge/elasticnet through their saved
median-impute+scale preproc). The old fillna(0)-for-everything bug (FIX C16)
is gone for all families.

Ranked output naming (results/political_scenarios/):
  rank 1 keeps the LEGACY filenames (*_by_council.csv, *_all_properties.parquet,
  mixed_governance_support_diag.json) so every existing consumer keeps working
  unchanged (generate_figures.py, generate_fig5_scenario_impact.py,
  generate_scenario_map.py, research_summary, verify_appendix_a1_scenarios.py,
  app/app.py, archive/*);
  ranks 2..N write *_rank{N}_by_council.csv / *_rank{N}_all_properties.parquet
  and mixed_governance_support_diag_rank{N}.json.

scenario_run_metadata.json (same directory) maps each rank to the actual
model file/features/MAE used on this run -- the traceability record when the
rank assignment changes between runs. It also carries the base-mode note: a
model whose feature list lacks dem_share produces identically-zero
mixed_governance deltas BY CONSTRUCTION (a spec property, not a null finding).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ops.champion_ranking import get_all_models
from src.ops.predictors import load_entry_model, features_of, predict_entry_with_features

CANON = Path('data/canonical/modeling_dataset_canonical_v2.parquet')
CANON_FALLBACK = Path('data/canonical/modeling_dataset_canonical.parquet')
OUT_DIR = Path('results/political_scenarios')
OUT_DIR.mkdir(parents=True, exist_ok=True)

if CANON.exists():
    df = pd.read_parquet(CANON)
elif CANON_FALLBACK.exists():
    df = pd.read_parquet(CANON_FALLBACK)
else:
    # The old processed fallback (data/processed/modeling_dataset_fe_imputed
    # .parquet) was removed; there is no valid fallback anymore. Fail loudly
    # instead of silently loading a missing/legacy dataset.
    raise SystemExit(
        'ERROR: No canonical dataset found. Tried:\n  ' + str(CANON) + '\n  '
        + str(CANON_FALLBACK) + '\nReproduce/copy the canonical dataset '
        'before running political scenarios.')

# detect council district column
council_candidates = [c for c in df.columns if 'council' in c.lower()]
council_col = council_candidates[0] if council_candidates else None

# dem_share training-support ceiling for the mixed_governance diagnostic
# (FIX A4 mirror; identical to the pre-Round-9 computation).
_split = Path('data/splits/all_years_train.parquet')
if _split.exists():
    _tr_max = float(pd.read_parquet(_split, columns=['dem_share'])['dem_share'].max())
else:
    _tr_max = float(df['dem_share'].max())


def build_scenario_frames(base_df):
    """The three scenario frames, built ONCE (they are model-independent
    apart from the dem_share mutation); each rank re-scores them below.''
    - liberal_policy: R3->R7 substring on the first zonedist column +
      first 'far'-matching column scaled x1.5
    - conservative_policy: R7->R3 on zonedist + first 'far' column x0.7
    - mixed_governance: dem_share + 0.15 clipped to [0,1]
    """
    z_cols = [c for c in base_df.columns if 'zonedist' in c.lower()
              or 'zoning' in c.lower() or 'zonedist1' in c.lower()]
    far_cols = [c for c in base_df.columns if 'far' in c.lower()
                or 'gross_floor' in c.lower()]
    frames = {}

    df_lib = base_df.copy(deep=True)
    if z_cols:
        zc = z_cols[0]
        df_lib[zc] = df_lib[zc].astype(str).apply(
            lambda z: z.replace('R3', 'R7') if isinstance(z, str) else z)
    if far_cols:
        df_lib[far_cols[0]] = df_lib[far_cols[0]].astype(float).fillna(0) * 1.5
    frames['liberal_policy'] = df_lib

    df_cons = base_df.copy(deep=True)
    if z_cols:
        zc = z_cols[0]
        df_cons[zc] = df_cons[zc].astype(str).apply(
            lambda z: z.replace('R7', 'R3') if isinstance(z, str) else z)
    if far_cols:
        df_cons[far_cols[0]] = df_cons[far_cols[0]].astype(float).fillna(0) * 0.7
    frames['conservative_policy'] = df_cons

    df_mix = base_df.copy(deep=True)
    if 'dem_share' in df_mix.columns:
        df_mix['dem_share'] = (df_mix['dem_share'] + 0.15).clip(0.0, 1.0)
    frames['mixed_governance'] = df_mix
    return frames


# ---------------------------------------------------------------------------
# Per-rank scoring loop (Round 9): fully symmetric over all three scenarios
# and every eligible ranked model. Legacy filenames are the rank-1 outputs.
# ---------------------------------------------------------------------------
frames = build_scenario_frames(df)
rank_meta = []

for ent in get_all_models(write_audit=False):
    model, preproc = load_entry_model(ent)
    feats = features_of(ent)
    rank = int(ent['rank'])
    suffix = '' if rank == 1 else f'_rank{rank}'
    print(f'Rank {rank}: {ent["name"]} ({ent["family"]}, MAE={ent["test_mae"]:.6f})')

    for scen_name, scen_df in frames.items():
        # family-correct NaN handling lives in predict_entry_with_features;
        # no fillna(0) is applied here (FIX C16 remains in force).
        base_preds = predict_entry_with_features(ent, feats, df, model=model,
                                                 preproc=preproc)
        scen_preds = predict_entry_with_features(ent, feats, scen_df,
                                                 model=model, preproc=preproc)
        delta = scen_preds - base_preds
        pct_change = (np.exp(scen_preds) - np.exp(base_preds)) / np.exp(base_preds) * 100.0
        out = scen_df[[council_col]] if council_col else pd.DataFrame({'idx': scen_df.index})
        out = out.copy()
        out['base_pred'] = base_preds
        out['scen_pred'] = scen_preds
        out['delta'] = delta
        out['pct_change'] = pct_change

        valid_mask = out['base_pred'] >= 9.0
        out = out[valid_mask].copy()
        out['pct_change'] = np.clip(out['pct_change'], -100.0, 100.0)

        if council_col:
            agg = out.groupby(council_col).agg(
                mean_delta=('delta', 'mean'),
                mean_pct_change=('pct_change', 'mean'),
                base_mean=('base_pred', 'mean'),
                scen_mean=('scen_pred', 'mean'),
                count=('pct_change', 'count')
            )
            agg.to_csv(OUT_DIR / f'{scen_name}{suffix}_by_council.csv')
        out.to_parquet(OUT_DIR / f'{scen_name}{suffix}_all_properties.parquet')
        # Round 10: name-keyed copies. Downstream producers/verifiers
        # (generate_paper_assets.py, verify_appendix_a1_scenarios.py, fig5)
        # resolve models by experiment NAME -- rank numbers shift whenever
        # the eligible set changes; names never do. Rank-keyed and legacy
        # rank-1 filenames above are unchanged.
        if council_col:
            _bm_dir = OUT_DIR / 'by_model' / ent['name']
            _bm_dir.mkdir(parents=True, exist_ok=True)
            agg.to_csv(_bm_dir / f'{scen_name}_by_council.csv')
        print(f'Saved scenario {scen_name}{suffix} (rank {rank}, model {ent["name"]},'
              f' rows kept {len(out)})')

        if scen_name == 'mixed_governance':
            # FIX A4 (disclosure only -- the shock mechanics are unchanged):
            # quantify how much of the mixed-governance +0.15 dem_share shock
            # runs outside the model's training support. Boosters cannot
            # extrapolate trends past their largest training value; deltas for
            # those rows are artifacts, not identified responses. Written
            # per-manual rank and re-derived independently by
            # verify_scenario_mechanics.py.
            _base_ds = df['dem_share']
            _n_ds = int(_base_ds.notna().sum())
            _shocked = _base_ds + 0.15
            _n_clip = int((_shocked > 1.0).sum())
            _n_beyond = int((_shocked > _tr_max).sum())
            _diag = {
                'note': ('The mixed_governance shock (dem_share + 0.15 clipped to [0,1]) is a '
                         'model-level counterfactual. Rows counted below receive shocked dem_share '
                         'values at the [0,1] ceiling or beyond the political training '
                         'support; their deltas are extrapolation artifacts, not identified effects.'),
                'rank': rank,
                'model_name': ent['name'],
                'model_family': ent['family'],
                'n_rows_with_dem_share': _n_ds,
                'n_shock_clipped_at_1p0': _n_clip,
                'pct_shock_clipped_at_1p0': round(100.0 * _n_clip / _n_ds, 2),
                'train_dem_share_max_all_years_split': _tr_max,
                'n_shocked_beyond_train_support': _n_beyond,
                'pct_beyond_train_support': round(100.0 * _n_beyond / _n_ds, 2),
            }
            (OUT_DIR / f'mixed_governance_support_diag_rank{rank}.json').write_text(
                json.dumps(_diag, indent=2))
            if rank == 1:
                # legacy single-model filename preserved for existing consumers
                (OUT_DIR / 'mixed_governance_support_diag.json').write_text(
                    json.dumps(_diag, indent=2))
            print('Mixed-governance support diagnostic (rank ' + str(rank) + '):',
                  json.dumps(_diag, indent=2))
            if _diag['pct_beyond_train_support'] > 0:
                print(('WARNING: {:.1f}% of rows receive a dem_share shock '
                       'beyond the model training support; treat '
                       'mixed_governance magnitudes as extrapolation.').format(
                          _diag['pct_beyond_train_support']))

    # per-rank metadata entry (traceability: which model produced which rank)
    _meta = {
        'rank': rank,
        'experiment_id': ent['experiment_id'],
        'name': ent['name'],
        'family': ent['family'],
        'mode': ent['mode'],
        'test_mae': ent['test_mae'],
        'test_r2': ent['test_r2'],
        'model_path': ent['model_path'],
        'features_path': ent['features_path'],
        'outputs': {
            s: {'by_council': f'{s}{suffix}_by_council.csv',
                'all_properties': f'{s}{suffix}_all_properties.parquet',
                'by_model_by_council': f"by_model/{ent['name']}/{s}_by_council.csv"}
            for s in frames
        },
        'mixed_governance_diag': (f'mixed_governance_support_diag_rank{rank}.json'
                                  + (' (+ legacy mixed_governance_support_diag.json)'
                                     if rank == 1 else '')),
    }
    if 'dem_share' not in feats:
        _meta['base_mode_dem_share_note'] = (
            'BASE-MODE model: dem_share is NOT in this models feature list, so '
            'the mixed_governance dem_share shift cannot affect its '
            'predictions -- identical-zero deltas are expected BY CONSTRUCTION '
            '(model cannot see ideology), NOT a null-effect finding.')
    rank_meta.append(_meta)

_run_meta = {
    'generated_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    'producer': 'src/analysis/political_scenarios.py (Round 9 full-coverage)',
    'rank_selection_audit': 'results/rank1_rank2_selection.json',
    'note': ('Rank 1 writes the legacy unsuffixed filenames; ranks 2..N write '
             '*_rank{N}_* files. All scenario mechanics identical across ranks; '
             'each rank scored with its own model under that families trained '
             'NaN/preproc convention (src/ops/predictors.py). Round 10 also writes '
              'by_model/<model_name>/<scenario>_by_council.csv copies so '
              'consumers can resolve scenario outputs by experiment NAME, '
              'never by rank number (rank assignments shift; names do not).'),
    'ranks': rank_meta,
}
(OUT_DIR / 'scenario_run_metadata.json').write_text(json.dumps(_run_meta, indent=2))
print('Wrote scenario_run_metadata.json with', len(rank_meta), 'rank entries')
print('Political scenarios completed and saved to', OUT_DIR)
