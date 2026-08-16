"""Round 5 fact-check: Section H (full OLS fresh re-derivation, every cell, Tier 1) + missingness reconciliation (Tier 1)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import round5_factcheck_part1a as p1a
from round5_factcheck_part1a import ROOT, write_json

df, cname = p1a.load_canonical()
H = {}

# === Fresh OLS re-fit (independent of generate_regression_table.py) ===
sub = df.dropna(subset=['target_log_price', 'dem_share', 'yearbuilt', 'numfloors', 'landuse', 'BOROUGH', 'commfar']).copy()
# NOTE: generate_regression_table.py drops na on the same subset; residfar/facilfar are NOT in the drop list,
# but commfar is -> commfar-null rows drop, which also removes farf group (yearbuilt/commfar/residfar/facilfar all null together are subsumed).
N = int(len(sub))
X1 = sm.add_constant(sub[['dem_share']])
m1 = sm.OLS(sub['target_log_price'], X1).fit()
controls = ['yearbuilt', 'numfloors', 'landuse', 'commfar']
X2 = sm.add_constant(sub[['dem_share'] + controls])
m2 = sm.OLS(sub['target_log_price'], X2).fit()
_df_fe = pd.get_dummies(sub, columns=['BOROUGH'], drop_first=True, dtype=float)
boro_cols = sorted([c for c in _df_fe.columns if c.startswith('BOROUGH_')])
X3 = sm.add_constant(_df_fe[['dem_share'] + controls + boro_cols])
m3 = sm.OLS(sub['target_log_price'], X3).fit()

def cell(model, name):
    if name in model.params.index:
        return {'coef': float(model.params[name]), 'se': float(model.bse[name]),
                't': float(model.tvalues[name]), 'p': float(model.pvalues[name])}
    return None

H['N'] = {'paper': 470535, 'recomputed': N}
H['model1'] = {
    'dem_share': {'paper_coef': 0.8890, 'paper_se': 0.0070, **cell(m1, 'dem_share')},
    'R2': {'paper': 0.0336, 'recomputed': float(m1.rsquared)},
}
H['model2'] = {
    'dem_share': {'paper_coef': 0.5741, 'paper_se': 0.0071, **cell(m2, 'dem_share')},
    'yearbuilt': {'paper_coef': -0.0001, 'paper_se': None, **cell(m2, 'yearbuilt')},
    'numfloors': {'paper_coef': 0.0043, 'paper_se': 0.0002, **cell(m2, 'numfloors')},
    'landuse': {'paper_coef': 0.0778, 'paper_se': 0.0013, **cell(m2, 'landuse')},
    'commfar': {'paper_coef': 0.0585, 'paper_se': 0.0006, **cell(m2, 'commfar')},
    'R2': {'paper': 0.0932, 'recomputed': float(m2.rsquared)},
}
H['model3'] = {
    'dem_share': {'paper_coef': 0.1843, 'paper_se': 0.0093, 'paper_t': 19.859, **cell(m3, 'dem_share')},
    'yearbuilt': {'paper_coef': 0.0000, 'paper_se': None, 'paper_t': 1.559, 'paper_p': 0.119, **cell(m3, 'yearbuilt')},
    'numfloors': {'paper_coef': -0.0035, 'paper_se': 0.0002, 'paper_t': -19.215, **cell(m3, 'numfloors')},
    'landuse': {'paper_coef': 0.0307, 'paper_se': 0.0013, 'paper_t': 23.812, **cell(m3, 'landuse')},
    'commfar': {'paper_coef': 0.0494, 'paper_se': 0.0006, 'paper_t': 79.687, **cell(m3, 'commfar')},
    'const': {'paper_coef': 13.7219, 'paper_t': 557.921, **cell(m3, 'const')},
    'BOROUGH_2': {'paper_coef': -0.8127, 'paper_t': -138.285, **cell(m3, 'BOROUGH_2')},
    'BOROUGH_3': {'paper_coef': -0.2850, 'paper_t': -64.342, **cell(m3, 'BOROUGH_3')},
    'BOROUGH_4': {'paper_coef': -0.6609, 'paper_t': -142.580, **cell(m3, 'BOROUGH_4')},
    'BOROUGH_5': {'paper_coef': -0.6583, 'paper_t': -86.587, **cell(m3, 'BOROUGH_5')},
    'R2': {'paper': 0.1478, 'recomputed': float(m3.rsquared)},
    'adj_R2': {'paper': 0.148, 'recomputed': float(m3.rsquared_adj)},
    'F_stat': {'paper': 9067, 'recomputed': float(m3.fvalue)},
    'F_p': {'paper': 0.0, 'recomputed': float(m3.f_pvalue)},
    'borough_FE': 'Yes (drop_first=True => BOROUGH_1=Manhattan reference)',
}
H['borough_borough_codes'] = {2: 'Bronx', 3: 'Brooklyn', 4: 'Queens', 5: 'Staten Island'}
write_json(ROOT / 'results' / 'round5_part_H_ols.json', {'section_H_ols': H})
print('WROTE results/round5_part_H_ols.json')
print('N', N)
print('m3 dem_share coef', cell(m3, 'dem_share'))
print('m3 BOROUGH_2', cell(m3, 'BOROUGH_2'))
print('m3 BOROUGH_4', cell(m3, 'BOROUGH_4'))
