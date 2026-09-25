"""Round 5 fact-check: Section L (borough mapping) + Section M (blacklist/PNG/tex-docx cross-cutting)."""
import json
import glob
from pathlib import Path
import pandas as pd
from _shared import ROOT, write_json, load_canonical, parse_feature_names

L = {}
boromap = pd.read_csv(ROOT / 'results' / 'council_district_borough_map.csv')
boromap['CounDist'] = boromap['CounDist'].astype(int)
expected = {
    'Manhattan': list(range(1, 11)),
    'Bronx': list(range(11, 19)),
    'Queens': list(range(19, 33)),
    'Brooklyn': list(range(33, 49)),
    'Staten Island': list(range(49, 52)),
}
check = {}
for boro, dists in expected.items():
    actual = boromap[boromap['borough_name'] == boro]['CounDist'].tolist()
    check[boro] = {'paper_districts': dists, 'actual_districts': actual, 'match': actual == dists}
L['borough_mapping_from_csv'] = check
L['all_borough_ranges_match'] = all(v['match'] for v in check.values())

# Also confirm via canonical dataset BOROUGH field per CounDist
df, _ = load_canonical()
canon_map = df.groupby('CounDist')['BOROUGH'].agg(lambda s: s.mode().iloc[0]).to_dict()
canon_map = {int(k): int(v) for k, v in canon_map.items()}
# BOROUGH codes: 1=Manhattan,2=Bronx,3=Brooklyn,4=Queens,5=Staten Island
boro_code_name = {1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn', 4: 'Queens', 5: 'Staten Island'}
canon_check = {boro_code_name[canon_map[d]]: d for d in sorted(canon_map)}
# group by borough name
by_name = {}
for d, c in canon_map.items():
    name = boro_code_name[c]
    by_name.setdefault(name, []).append(d)
canon_expected = {b: sorted(by_name[b]) for b in by_name}
L['borough_mapping_from_canonical_BOROUGH'] = {b: canon_expected.get(b) for b in expected}
write_json(ROOT / 'results' / 'round5_part_L.json', {'section_L_borough_mapping': L})
print('WROTE results/round5_part_L.json')
print('all borough ranges match (csv):', L['all_borough_ranges_match'])
for b, v in check.items():
    print(' ', b, v['match'])
print('canonical BOROUGH mapping:', L['borough_mapping_from_canonical_BOROUGH'])

# ===== M cross-cutting =====
M = {}
# (1) FIX B6: blacklist status is COMPUTED from the actual trained model
# files on every run. This block previously hardcoded blacklist_clean=True plus
# a summary string describing a superseded 16/17-feature spec - a false
# verification record. (The fail-loud enforcement gate lives in
# editorial_audit/verify_blacklist_fully.py; this is a read-only cross-check.)
import yaml as _yaml
with open(ROOT / 'config' / 'feature_blacklist.yaml') as _bf:
    _bl = {str(i).strip().lower() for items in _yaml.safe_load(_bf).values() for i in items}
base_names, base_mfi = parse_feature_names(ROOT / 'models' / 'lgbm_all_years_base.txt')
pol_names, pol_mfi = parse_feature_names(ROOT / 'models' / 'lgbm_all_years_political.txt')
def _violations(names):
    return [f for f in names if f.lower() in _bl or f.replace('_', ' ').lower() in _bl]
_base_bad = _violations(base_names)
_pol_bad = _violations(pol_names)
M['champion_base_feature_count'] = {'paper': 16, 'from_feature_names_line': len(base_names),
                                    'max_feature_idx_plus_1': (base_mfi + 1) if base_mfi is not None else None,
                                    'blacklist_violations': _base_bad,
                                    'blacklist_clean': len(_base_bad) == 0}
M['champion_political_feature_count'] = {'paper': 17, 'from_feature_names_line': len(pol_names),
                                          'max_feature_idx_plus_1': (pol_mfi + 1) if pol_mfi is not None else None,
                                          'blacklist_violations': _pol_bad,
                                          'blacklist_clean': len(_pol_bad) == 0}
M['blacklist_summary'] = (f'Computed live this run: lgbm_all_years_base.txt has {len(base_names)} '
                          f'features, blacklist violations: {_base_bad or "none"}; '
                          f'lgbm_all_years_political.txt has {len(pol_names)} features, '
                          f'violations: {_pol_bad or "none"}.')

# (2) PIL opens shap_top10.png cleanly
try:
    from PIL import Image
    im = Image.open(ROOT / 'docs' / 'figures' / 'shap_top10.png')
    im.verify()
    im2 = Image.open(ROOT / 'docs' / 'figures' / 'shap_top10.png')
    M['shap_top10_png_opens'] = True
    M['shap_top10_png_size'] = im2.size
except Exception as e:
    M['shap_top10_png_opens'] = False
    M['shap_top10_png_error'] = str(e)

# (3) no .tex/.docx in repo
tex = glob.glob(str(ROOT / '**' / '*.tex'), recursive=True)
docx = glob.glob(str(ROOT / '**' / '*.docx'), recursive=True)
# exclude junk inside venv/.venv
tex = [t for t in tex if '.venv' not in t and 'venv' not in t]
docx = [d for d in docx if '.venv' not in d and 'venv' not in d]
M['tex_files'] = tex
M['docx_files'] = docx
M['tex_files_are_repo_tables_only'] = all(str(ROOT / 'tables') in t for t in tex)
M['no_manuscript_docx_in_repo'] = (len(docx) == 0)
M['note'] = ('tables/*.tex are pipeline-generated drop-in tables, not the manuscript '
             '(the manuscript lives outside this repo), so their presence here is expected.')
write_json(ROOT / 'results' / 'round5_part_M.json', {'section_M_cross_cutting': M})
print('WROTE results/round5_part_M.json')
print('shap_top10 opens', M['shap_top10_png_opens'], 'size', M.get('shap_top10_png_size'))
print('tex', tex, 'docx', docx)
