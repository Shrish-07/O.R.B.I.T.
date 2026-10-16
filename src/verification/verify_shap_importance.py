"""Round 5 fact-check: Section J (SHAP Table 3). Tier 1: parse
lgbm_all_years_political_shap_summary.json directly + base model Community
Board comparison. Round 10: refreshed to the 25-feature era -- the
manuscript's headline constants are Community Board 0.32088 and dem_share
0.04685 (rank 11 of 25)."""
import json
from pathlib import Path
from _shared import ROOT, write_json, record_status, check_val

shap = json.loads((ROOT / 'models' / 'artifacts' / 'lgbm_all_years_political_shap_summary.json').read_text())
J = {}
J['n_entries_in_json'] = len(shap['summary'])

# Manuscript-quoted headline cells (Round 10). The full 25-row table is
# reproduced by the shap_tiers pipeline each run; the audit checks the two
# published cells against the live artifact at 5 dp, verifies the table size
# and the dem_share rank, and lists the full table for review.
PAPER_CELLS = [
    ('Community Board', 0.32088),
    ('dem_share', 0.04685),
]
J['paper_shap_table_size'] = 25
shap_map = {e['feature']: float(e['mean_abs_shap']) for e in shap['summary']}
matches = {}
for feat, paper_val in PAPER_CELLS:
    json_val = shap_map.get(feat)
    matches[feat] = {'paper_value': paper_val, 'json_value': json_val,
                     'json_value_rounded5': round(json_val, 5) if json_val is not None else None,
                     'matches_at_5dp': (json_val is not None
                                        and round(json_val, 5) == paper_val)}
J['headline_cells'] = matches

# dem_share rank: by descending mean_abs_shap (paper says 11th of 25)
nonzero = sorted([e for e in shap['summary'] if e['mean_abs_shap'] > 0], key=lambda x: -x['mean_abs_shap'])
zero = [e for e in shap['summary'] if e['mean_abs_shap'] == 0]
J['nonzero_count'] = len(nonzero)
J['zero_count'] = len(zero)
J['zero_features'] = [e['feature'] for e in zero]
dem_rank = next((i + 1 for i, e in enumerate(nonzero) if e['feature'] == 'dem_share'), None)
J['dem_share_rank_by_descending_nonzero'] = dem_rank
J['paper_dem_share_rank'] = '11th of 25'

# All entries sorted by descending value (including zeros) for completeness
all_sorted = sorted(shap['summary'], key=lambda x: -x['mean_abs_shap'])
J['all_features_sorted_descending'] = [(e['feature'], e['mean_abs_shap']) for e in all_sorted]
dem_rank_all = next((i + 1 for i, e in enumerate(all_sorted) if e['feature'] == 'dem_share'), None)
J['dem_share_rank_including_zeros'] = dem_rank_all

# Base model Community Board SHAP comparison. Round 10: no Round-10 manuscript
# constant was supplied for the base model's Community Board cell -- this is
# informational (report the recomputed value; do not invent a paper target).
base_shap = json.loads((ROOT / 'models' / 'artifacts' / 'lgbm_all_years_base_shap_summary.json').read_text())
cb_base = next((e['mean_abs_shap'] for e in base_shap['summary'] if e['feature'] == 'Community Board'), None)
J['base_model_community_board_shap'] = {'paper': None, 'recomputed': cb_base,
                                        'note': 'informational only; base-model CB cell was not quoted in the current manuscript draft'}
write_json(ROOT / 'results' / 'round5_part_J_shap.json', {'section_J_shap': J})
print('WROTE results/round5_part_J_shap.json')
print('nonzero', len(nonzero), 'zero', [e['feature'] for e in zero])
print('dem_share rank (nonzero):', dem_rank, 'paper: 11th of 25')
print('base Community Board:', cb_base, '(informational; no current paper target)')

record_status('verify_shap_importance', [
    check_val('table_size_25_features', 25, J['n_entries_in_json']),
    *[{'check': f'headline_{f}', 'expected': pv, 'actual': matches[f]['json_value_rounded5'],
       'tolerance': 5e-6, 'match': matches[f]['matches_at_5dp']}
      for f, pv in PAPER_CELLS],
    check_val('dem_share_rank_nonzero', 11, dem_rank),
], note='SHAP headline cells at 5dp vs manuscript Table 3; paper target: 11th of 25.')
