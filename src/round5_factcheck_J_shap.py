"""Round 5 fact-check: Section J (full 17-row SHAP Table 3). Tier 1: parse lgbm_all_years_political_shap_summary.json directly + base model Community Board comparison."""
import json
from pathlib import Path
from round5_factcheck_part1a import ROOT, write_json

shap = json.loads((ROOT / 'models' / 'artifacts' / 'lgbm_all_years_political_shap_summary.json').read_text())
J = {}
J['n_entries_in_json'] = len(shap['summary'])

paper_shap_rows = [
    ('Community Board', 0.27472),
    ('landuse', 0.11688),
    ('yearbuilt', 0.10600),
    ('numfloors', 0.09722),
    ('TAX CLASS AT TIME OF SALE', 0.08728),
    ('BBL_pluto', 0.07654),
    ('YEAR BUILT', 0.06552),
    ('ZIP CODE', 0.05823),
    ('Council District', 0.05419),
    ('facilfar', 0.04773),
    ('Census Tract 2020', 0.04500),
    ('dem_share', 0.03473),
    ('residfar', 0.02662),
    ('commfar', 0.01983),
    ('BOROUGH', 0.01099),
    ('CounDist', 0.00446),
    ('Election Year', 0.00000),
]
J['paper_shap_table_size'] = 17
# Map each paper entry to the JSON entry
shap_map = {e['feature']: float(e['mean_abs_shap']) for e in shap['summary']}
matches = {}
for i, (feat, paper_val) in enumerate(paper_shap_rows):
    json_val = shap_map.get(feat)
    matches[feat] = {'paper_value': paper_val, 'json_value': json_val,
                     'json_value_rounded5': round(json_val, 5) if json_val is not None else None,
                     'rank_in_paper_table': i + 1}
J['row_by_row'] = matches

# dem_share rank: by descending mean_abs_shap (paper says 12th of 17)
nonzero = sorted([e for e in shap['summary'] if e['mean_abs_shap'] > 0], key=lambda x: -x['mean_abs_shap'])
zero = [e for e in shap['summary'] if e['mean_abs_shap'] == 0]
J['nonzero_count'] = len(nonzero)
J['zero_count'] = len(zero)
J['zero_features'] = [e['feature'] for e in zero]
dem_rank = next((i + 1 for i, e in enumerate(nonzero) if e['feature'] == 'dem_share'), None)
J['dem_share_rank_by_descending_nonzero'] = dem_rank
J['paper_dem_share_rank'] = '12th of 17'

# All entries sorted by descending value (including zeros) for completeness
all_sorted = sorted(shap['summary'], key=lambda x: -x['mean_abs_shap'])
J['all_features_sorted_descending'] = [(e['feature'], e['mean_abs_shap']) for e in all_sorted]
dem_rank_all = next((i + 1 for i, e in enumerate(all_sorted) if e['feature'] == 'dem_share'), None)
J['dem_share_rank_including_zeros'] = dem_rank_all

# Base model Community Board SHAP comparison
base_shap = json.loads((ROOT / 'models' / 'artifacts' / 'lgbm_all_years_base_shap_summary.json').read_text())
cb_base = next((e['mean_abs_shap'] for e in base_shap['summary'] if e['feature'] == 'Community Board'), None)
J['base_model_community_board_shap'] = {'paper': 0.30101, 'recomputed': cb_base}
write_json(ROOT / 'results' / 'round5_part_J_shap.json', {'section_J_shap': J})
print('WROTE results/round5_part_J_shap.json')
print('nonzero', len(nonzero), 'zero', [e['feature'] for e in zero])
print('dem_share rank (nonzero):', dem_rank, 'paper: 12th of 17')
print('base Community Board:', cb_base, 'paper 0.30101')
