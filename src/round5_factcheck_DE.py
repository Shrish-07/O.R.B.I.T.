"""Round 5 fact-check: Section D (Canonical structure) + Section E (split)."""
import json
from pathlib import Path
import pandas as pd
import round5_factcheck_part1a as p1a
from round5_factcheck_part1a import ROOT, write_json, parse_feature_names

df, cname = p1a.load_canonical()
OUT = {'raw': {'canonical_file': cname}, 'section_D': {}, 'section_E': {}}

# ===== D =====
D = {}
schema = json.loads((ROOT / 'models' / 'artifacts' / 'canonical_schema.json').read_text())
D['dataset_rows'] = {'paper': 514618, 'recomputed': int(schema['shape'][0]), 'actual_loaded': int(len(df))}
D['dataset_cols'] = {'paper': 53, 'recomputed': int(schema['shape'][1]), 'actual_loaded': int(df.shape[1])}
base_names, base_mfi = parse_feature_names(ROOT / 'models' / 'lgbm_all_years_base.txt')
pol_names, pol_mfi = parse_feature_names(ROOT / 'models' / 'lgbm_all_years_political.txt')
D['base_feature_count'] = {'paper': 16, 'from_feature_names_line': len(base_names),
                           'from_max_feature_idx_plus1': (base_mfi + 1) if base_mfi is not None else None,
                           'feature_names': base_names}
D['political_feature_count'] = {'paper': 17, 'from_feature_names_line': len(pol_names),
                               'from_max_feature_idx_plus1': (pol_mfi + 1) if pol_mfi is not None else None,
                               'feature_names': pol_names}
# YEAR BUILT vs yearbuilt
yb, yb2 = 'YEAR BUILT', 'yearbuilt'
both = df[[yb, yb2]].notna().all(axis=1)
n_both = int(both.sum())
n_agree = int((df.loc[both, yb].astype(float) == df.loc[both, yb2].astype(float)).sum())
n_disagree = int(n_both - n_agree)
max_dis = float((df.loc[both, yb].astype(float) - df.loc[both, yb2].astype(float)).abs().max())
corr = float(df.loc[both, [yb, yb2]].astype(float).corr().iloc[0, 1])
D['yearbuilt_vs_YEARBUILT'] = {
    'paper_both_populated': 464204, 'recomputed_both_populated': n_both,
    'paper_agree': 410164, 'recomputed_agree': n_agree,
    'paper_disagree': 54040, 'recomputed_disagree': n_disagree,
    'paper_max_disagreement_years': 2025, 'recomputed_max_disagreement_years': max_dis,
    'paper_correlation': 0.146, 'recomputed_correlation': round(corr, 6),
    'paper_agree_pct': 88.4, 'recomputed_agree_pct': round(100.0 * n_agree / n_both, 2),
    'paper_disagree_pct': 11.6, 'recomputed_disagree_pct': round(100.0 * n_disagree / n_both, 2),
}
OUT['section_D'] = D

# ===== E =====
E = {}
train = pd.read_parquet(ROOT / 'data' / 'splits' / 'all_years_train.parquet')
test = pd.read_parquet(ROOT / 'data' / 'splits' / 'all_years_test.parquet')
E['train_rows'] = {'paper': 121622, 'recomputed': int(len(train))}
E['test_rows'] = {'paper': 56817, 'recomputed': int(len(test))}
ey_dist = df['election_year'].value_counts().sort_index()
E['election_year_distribution'] = {int(yr): {'rows': int(cnt), 'pct': round(100.0 * cnt / len(df), 2)}
                                   for yr, cnt in ey_dist.items()}
E['election_year_distribution_paper'] = {2017: {'rows': 178439, 'pct': 34.7},
                                         2021: {'rows': 285617, 'pct': 55.5},
                                         2025: {'rows': 50562, 'pct': 9.8}}
n_2017 = int((df['election_year'] == 2017).sum())
E['train_plus_test'] = {'value': int(len(train) + len(test)), 'paper': 178439}
E['election_year_2017_subset'] = {'value': n_2017, 'paper': 178439}
E['identity_holds'] = (int(len(train) + len(test)) == n_2017)
E['train_sale_year_range'] = [int(train['sale_year'].min()), int(train['sale_year'].max())]
E['test_sale_year_range'] = [int(test['sale_year'].min()), int(test['sale_year'].max())]
OUT['section_E'] = E

write_json(ROOT / 'results' / 'round5_part1_DE.json', OUT)
print('WROTE results/round5_part1_DE.json')
print('D base features', D['base_feature_count'], 'political', D['political_feature_count'])
print('E train', E['train_rows'], 'test', E['test_rows'], 'identity', E['identity_holds'])
