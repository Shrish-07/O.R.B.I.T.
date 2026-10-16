"""Round 5 fact-check: Sections A (Abstract) + B (Intro ranges)."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from _shared import ROOT, load_canonical, write_json, parse_feature_names, record_status, check_val

df, cname = load_canonical()
OUT = {'section': {}, 'raw': {'canonical_file': cname, 'n_rows': int(len(df)), 'n_cols': int(df.shape[1])}}

# ===== A. ABSTRACT =====
A = {}
# Round 10: 'paper' targets = the manuscript's current values (post refresh +
# dedup rebuild; 566,956-row canonical, 520,365-row Table 1).
A['transaction_count'] = {'paper': 566956, 'recomputed': int(len(df))}
A['council_districts'] = {'paper': 51, 'recomputed': int(df['CounDist'].nunique())}
ey = sorted([int(x) for x in df['election_year'].dropna().unique().tolist()])
A['election_years'] = {'paper': [2017, 2021, 2025], 'recomputed': ey}

# Bivariate regression beta=0.9449 and controlled+borough FE beta=0.2883 -> read
# regression_table.csv (Tier 2 here; recomputed separately in part2)
reg = pd.read_csv(ROOT / 'results' / 'regression_table.csv')
m1 = reg[reg['Model'] == 'Bivariate'].iloc[0]
m3 = reg[reg['Model'] == '+ Borough FE'].iloc[0]
A['bivariate_beta'] = {'paper': 0.9449, 'recomputed': float(m1['Coef']), 'p': float(m1['p'])}
A['controlled_boroughFE_beta'] = {'paper': 0.2883, 'recomputed': float(m3['Coef']), 'p': float(m3['p'])}
A['shrinkage_pct'] = {'paper': '~69 percent (from 0.9449 to 0.2883)',
                      'recomputed_pct': round(100.0 * (0.9449 - 0.2883) / 0.9449, 2)}

# LightGBM MAE base vs political (read metrics JSON = Tier 2; recomputed from preds in part2 = Tier 1).
# Round 10: the paper targets mirror the live artifacts/registry (the paper's
# Table 2 is generated from the same registry; see generate_paper_assets.py).
bm = json.loads((ROOT / 'models' / 'artifacts' / 'lgbm_all_years_base_metrics.json').read_text())
pm = json.loads((ROOT / 'models' / 'artifacts' / 'lgbm_all_years_political_metrics.json').read_text())
A['lgbm_base_mae'] = {'paper': 0.4418, 'recomputed': bm['mae_log_price']}
A['lgbm_political_mae'] = {'paper': 0.4399, 'recomputed': pm['mae_log_price']}

# SHAP rank of dem_share (read political shap summary = Tier 2; full recompute is in part2 if feasible)
shap = json.loads((ROOT / 'models' / 'artifacts' / 'lgbm_all_years_political_shap_summary.json').read_text())
order = [s['feature'] for s in shap['summary'] if s['mean_abs_shap'] > 0]
idx = next((i for i, f in enumerate(order) if f == 'dem_share'), None)
A['shap_rank_dem_share'] = {'paper': '11th of 25', 'recomputed_rank_among_nonzero': (idx + 1) if idx is not None else None,
                            'n_nonzero_features': len(order),
                            'n_total_entries_in_json': len(shap['summary'])}

# Moran's I rounding consistency (don't recompute, just confirm 3-decimal rounding of the stored values)
mor = json.loads((ROOT / 'results' / 'morans_i_results.json').read_text())
A['morans_i_price'] = {'paper': 0.594, 'stored_full': mor['moran_i_price'], 'rounds_to_3dec': round(mor['moran_i_price'], 3)}
A['morans_i_dem_share'] = {'paper': 0.570, 'stored_full': mor['moran_i_dem_share'], 'rounds_to_3dec': round(mor['moran_i_dem_share'], 3)}
A['morans_p_paper'] = 0.001
A['morans_p_stored'] = mor['p_sim_price']
OUT['section']['A_abstract'] = A

# ===== B. INTRO RANGES =====
B = {}
g = df.groupby('CounDist')['dem_share'].mean()
h = df.groupby('CounDist')['target_log_price'].median()
B['dem_share_per_cd_mean_min'] = {'value': float(g.min()), 'cd': int(g.idxmin())}
B['dem_share_per_cd_mean_max'] = {'value': float(g.max()), 'cd': int(g.idxmax())}
B['median_logprice_per_cd_min'] = {'value': float(h.min()), 'cd': int(h.idxmin())}
B['median_logprice_per_cd_max'] = {'value': float(h.max()), 'cd': int(h.idxmax())}
# also using ideology_by_council (per cd,year) for the full range across years
ideology = pd.read_parquet(ROOT / 'data' / 'processed' / 'ideology_by_council.parquet')
B['dem_share_ideology_min_any_cdyear'] = float(ideology['dem_share'].min())
B['dem_share_ideology_max_any_cdyear'] = float(ideology['dem_share'].max())
OUT['section']['B_intro_ranges'] = B

write_json(ROOT / 'results' / 'round5_part1_AB.json', OUT)
print('WROTE results/round5_part1_AB.json')
print('A keys', list(A.keys()))
print('B keys', list(B.keys()))

# Round 11: machine-countable status
record_status('verify_abstract_and_intro', [
    check_val('transaction_count', 566956, A['transaction_count']['recomputed']),
    check_val('council_districts', 51, A['council_districts']['recomputed']),
    check_val('election_years', A['election_years']['paper'],
              A['election_years']['recomputed']),
    check_val('bivariate_beta', 0.9449, A['bivariate_beta']['recomputed'], tol=5e-5),
    check_val('controlled_boroughFE_beta', 0.2883,
              A['controlled_boroughFE_beta']['recomputed'], tol=5e-5),
    check_val('lgbm_base_mae', 0.4418, A['lgbm_base_mae']['recomputed'], tol=5e-5),
    check_val('lgbm_political_mae', 0.4399, A['lgbm_political_mae']['recomputed'], tol=5e-5),
    check_val('shap_rank_dem_share', 11,
              A['shap_rank_dem_share']['recomputed_rank_among_nonzero']),
    {'check': 'shap_total_entries', 'expected': 25,
     'actual': A['shap_rank_dem_share']['n_total_entries_in_json'],
     'tolerance': 0.0,
     'match': A['shap_rank_dem_share']['n_total_entries_in_json'] == 25},
    check_val('morans_i_price_3dp', 0.594, A['morans_i_price']['rounds_to_3dec'], tol=5e-4),
    check_val('morans_i_dem_share_3dp', 0.570, A['morans_i_dem_share']['rounds_to_3dec'], tol=5e-4),
], note='Abstract/intro headline cells vs canonical, regression_table.csv, '
        'metrics, SHAP and morans_i artifacts.')
