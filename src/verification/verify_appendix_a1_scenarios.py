"""Appendix A.1 verifier (Round 11 rewrite): NO embedded manuscript table.

The pre-Round-11 version embedded a 51x3 copy of the OLD manuscript's Table
A.1, which no longer matched anything the pipeline produces (152 spurious
exceptions -- a comparison against stale constants that could only drift,
never verify). This rewrite is self-sufficient:

  1. Resolve the paper scenario model (lgbm_all_years_political) BY NAME via
     results/political_scenarios/scenario_run_metadata.json (rank->name map;
     rank assignments shift, names do not).
  2. Independently re-derive the 51x3 district means from the property-level
     parquets it wrote this run
     (results/political_scenarios/{scenario}[_rankN]_all_properties.parquet),
     applying the producer's identical filters: keep rows with
     base_pred >= 9.0, clip pct_change to [-100, 100], then
     group by council district and average pct_change.
  3. Compare the re-derived matrix, per cell, against:
       (i)  the name-keyed by_model CSVs
            (results/political_scenarios/by_model/<model>/<scenario>_by_council.csv), and
       (ii) the values printed in docs/paper_assets/appendix_a1_scenarios.tex
            (parsed from the two-block longtable).
     Tolerance: 0.005 percentage points (the .tex prints 2 decimals).
     NOT loosened.

Outputs: results/round5_part_K_appendixA1.json (legacy path, kept name),
         results/round5_part_K_appendixA1_all_models.json (informational
         per-rank summary, unchanged role), and the Round 11 machine-countable
         status results/verification/verify_appendix_a1_scenarios.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from _shared import ROOT, write_json, record_status, check_val

sys.path.insert(0, str(ROOT))

K = {}
CHECKS = []
TOL = 0.005  # percentage points -- the .tex prints 2 decimals; NOT loosened.
SCEN_DIR = ROOT / 'results' / 'political_scenarios'
SCEN_LIST = ['liberal_policy', 'conservative_policy', 'mixed_governance']
PAPER_SCEN_MODEL = 'lgbm_all_years_political'
A1_TEX = ROOT / 'docs' / 'paper_assets' / 'appendix_a1_scenarios.tex'


# ---- 1. resolve the model's parquet suffix by NAME -------------------------
def _rank_for_name(name):
    meta_path = SCEN_DIR / 'scenario_run_metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    for r in meta.get('ranks', []):
        if r.get('name') == name:
            return int(r['rank'])
    raise SystemExit(
        f'{name} not found in scenario_run_metadata.json ranks. '
        'Run the political_scenarios stage first.')


_rank = _rank_for_name(PAPER_SCEN_MODEL)
_suffix = '' if _rank == 1 else f'_rank{_rank}'
K['paper_model'] = PAPER_SCEN_MODEL
K['paper_model_rank_this_run'] = _rank
CHECKS.append(check_val('paper_model_resolved_by_name', True, _rank >= 1))


# ---- 2. independent re-derivation from the property-level parquets --------
def rederive(scen):
    p = SCEN_DIR / f'{scen}{_suffix}_all_properties.parquet'
    d = pd.read_parquet(p)
    # Producer filters (src/analysis/political_scenarios.py), replicated:
    d = d[d['base_pred'] >= 9.0]
    d = d.assign(pct_change=d['pct_change'].clip(-100.0, 100.0))
    return d.groupby('Council District')['pct_change'].mean()


rederived = pd.concat([rederive(s).rename(s) for s in SCEN_LIST], axis=1)
rederived.index = rederived.index.astype(int)
K['rederived_n_districts'] = int(len(rederived))
CHECKS.append(check_val('rederived_has_51_districts', 51, int(len(rederived))))


# ---- (i) compare vs the name-keyed by_model CSVs ----------------------------
def load_by_model(scen):
    p = SCEN_DIR / 'by_model' / PAPER_SCEN_MODEL / f'{scen}_by_council.csv'
    d = pd.read_csv(p)
    ccol = next(c for c in d.columns if 'coun' in c.lower())
    return d.set_index(d[ccol].astype(int))['mean_pct_change']


by_model_diffs = {}
for scen in SCEN_LIST:
    _csvp = SCEN_DIR / 'by_model' / PAPER_SCEN_MODEL / f'{scen}_by_council.csv'
    if _csvp.exists():
        live = load_by_model(scen)
        joined = pd.concat([rederived[scen].rename('rederived'),
                            live.rename('by_model')], axis=1)
        for cd, row in joined.iterrows():
            CHECKS.append(check_val(
                f'by_model.{scen}.cd{int(cd)}',
                round(float(row['by_model']), 4),
                round(float(row['rederived']), 4), tol=TOL))
        by_model_diffs[scen] = {
            'max_abs_diff': float((joined['by_model']
                                   - joined['rederived']).abs().max()),
            'n_cells': int(len(joined))}
    else:
        CHECKS.append(check_val(f'by_model.{scen}.csv_exists', True, False))
        by_model_diffs[scen] = {'status': 'MISSING by_model copy'}
K['rederived_vs_by_model_csv'] = by_model_diffs



# ---- (ii) compare vs the generated appendix_a1_scenarios.tex ---------------
def parse_tex_a1(path):
    """Parse the two-block longtable: 26 data rows x 2 blocks of
    (District & Borough & Liberal & Conservative & Mixed)."""
    rows = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not (line and line[0].isdigit() and line.endswith('\\\\')):
            continue
        cells = [c.strip() for c in line[:-2].split('&')]
        for block in (cells[0:5], cells[5:10]):
            if not block or not block[0]:
                continue
            rows.append({'CounDist': int(float(block[0])),
                         'liberal_policy': float(block[2]),
                         'conservative_policy': float(block[3]),
                         'mixed_governance': float(block[4])})
    return pd.DataFrame(rows).set_index('CounDist')


tex_diffs = {}
if A1_TEX.exists():
    tex_vals = parse_tex_a1(A1_TEX)
    K['tex_parsed_n_rows'] = int(len(tex_vals))
    CHECKS.append(check_val('tex_parsed_has_51_districts', 51,
                            int(len(tex_vals))))
    for scen in SCEN_LIST:
        joined = pd.concat([rederived[scen].rename('rederived'),
                            tex_vals[scen].rename('tex')], axis=1)
        for cd, row in joined.iterrows():
            CHECKS.append(check_val(
                f'tex.{scen}.cd{int(cd)}', round(float(row['tex']), 2),
                round(float(row['rederived']), 2), tol=TOL))
        tex_diffs[scen] = {
            'max_abs_diff': float((joined['tex']
                                   - joined['rederived']).abs().max()),
            'n_cells': int(len(joined))}
else:
    CHECKS.append(check_val('tex_exists', True, False))
    tex_diffs['status'] = 'docs/paper_assets/appendix_a1_scenarios.tex MISSING'
K['rederived_vs_tex'] = tex_diffs

K['tolerance_pct_points'] = TOL
write_json(ROOT / 'results' / 'round5_part_K_appendixA1.json',
           {'section_K_appendixA1_rederived': K})
print('WROTE results/round5_part_K_appendixA1.json')
print('by_model max_abs_diff:', {s: v.get('max_abs_diff')
                               for s, v in by_model_diffs.items()
                               if isinstance(v, dict)})
print('tex max_abs_diff:', {s: v.get('max_abs_diff')
                           for s, v in tex_diffs.items() if isinstance(v, dict)})

# ---------------------------------------------------------------------------
# Informational addendum (kept from Round 9/10): per-rank scenario summaries
# for every eligible ranked model. NOT a paper comparison.
# ---------------------------------------------------------------------------
from src.ops.champion_ranking import get_all_models

_all_models_summary = {}
for _e9 in get_all_models(write_audit=False):
    _r9 = int(_e9['rank'])
    _suffix9 = '' if _r9 == 1 else f'_rank{_r9}'
    row = {'model_name': _e9['name'], 'family': _e9.get('family'),
           'mode': _e9.get('mode'), 'test_mae': _e9.get('test_mae')}
    for _scen9 in SCEN_LIST:
        _csv9 = SCEN_DIR / f'{_scen9}{_suffix9}_by_council.csv'
        if _csv9.exists():
            _d9 = pd.read_csv(_csv9)
            row[_scen9 + '_mean_pct_change'] = round(
                float(_d9['mean_pct_change'].mean()), 4)
            row[_scen9 + '_n_districts'] = int(len(_d9))
        else:
            row[_scen9 + '_status'] = 'missing'
    _diag9 = SCEN_DIR / f'mixed_governance_support_diag_rank{_r9}.json'
    if _diag9.exists():
        row['mixed_gov_pct_beyond_train_support'] = json.loads(
            _diag9.read_text()).get('pct_beyond_train_support')
    _all_models_summary[f'rank{_r9}'] = row

write_json(ROOT / 'results' / 'round5_part_K_appendixA1_all_models.json',
           {'appendixA1_all_models_informational': {
               'round9_note': ('INFORMATIONAL ONLY. Rank 1 = current champion '
                               'per experiments/champion.json; ranks 2..N = '
                               'next-lowest-MAE eligible models. Models other '
                               'than the paper model are not expected to '
                               'reproduce the Appendix A.1 values.'),
               'per_rank_summaries': _all_models_summary}})
print('WROTE results/round5_part_K_appendixA1_all_models.json')

record_status('verify_appendix_a1_scenarios', CHECKS,
              note=('Independent re-derivation from the property-level '
                    'parquets vs (i) by_model CSVs and (ii) the generated '
                    'appendix_a1_scenarios.tex; tolerance 0.005 pct points.'))
