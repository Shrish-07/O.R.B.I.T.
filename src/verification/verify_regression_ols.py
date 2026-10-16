"""Round 5 fact-check: Section H (full OLS fresh re-derivation, every cell, Tier 1) + missingness reconciliation (Tier 1)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import _shared as p1a
from _shared import ROOT, write_json, record_status, check_val

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

# === Fresh OLS re-fit (independent of generate_regression_table.py) ===
# Round 10: 'paper' targets are the MANUSCRIPT's current values (post data
# refresh + 24/25-feature rebuild). Where the manuscript value is generated
# from a pipeline artifact it equals results/regression_table.csv /
# results/regression_model3_full_summary.txt; target cells are updated to
# match, tolerances unchanged.
H['N'] = {'paper': 520365, 'recomputed': N}
H['model1'] = {
    'dem_share': {'paper_coef': 0.9449, 'paper_se': 0.0066, **cell(m1, 'dem_share')},
    'R2': {'paper': 0.0374, 'recomputed': float(m1.rsquared)},
}
H['model2'] = {
    'dem_share': {'paper_coef': 0.6146, 'paper_se': 0.0067, **cell(m2, 'dem_share')},
    'yearbuilt': {'paper_coef': -0.0001, 'paper_se': None, **cell(m2, 'yearbuilt')},
    'numfloors': {'paper_coef': 0.0034, 'paper_se': 0.0002, **cell(m2, 'numfloors')},
    'landuse': {'paper_coef': 0.0782, 'paper_se': 0.0012, **cell(m2, 'landuse')},
    'commfar': {'paper_coef': 0.0676, 'paper_se': 0.0006, **cell(m2, 'commfar')},
    'R2': {'paper': 0.1042, 'recomputed': float(m2.rsquared)},
}
H['model3'] = {
    'dem_share': {'paper_coef': 0.2883, 'paper_se': 0.0087, 'paper_t': 33.243, **cell(m3, 'dem_share')},
    'yearbuilt': {'paper_coef': 0.0000, 'paper_se': None, 'paper_t': 2.407, 'paper_p': 0.016, **cell(m3, 'yearbuilt')},
    'numfloors': {'paper_coef': -0.0039, 'paper_se': 0.0002, 'paper_t': -22.492, **cell(m3, 'numfloors')},
    'landuse': {'paper_coef': 0.0353, 'paper_se': 0.0012, 'paper_t': 28.852, **cell(m3, 'landuse')},
    'commfar': {'paper_coef': 0.0586, 'paper_se': 0.0006, 'paper_t': 99.170, **cell(m3, 'commfar')},
    'const': {'paper_coef': 13.6074, 'paper_t': 578.350, **cell(m3, 'const')},
    'BOROUGH_2': {'paper_coef': -0.7856, 'paper_t': -139.579, **cell(m3, 'BOROUGH_2')},
    'BOROUGH_3': {'paper_coef': -0.2566, 'paper_t': -60.879, **cell(m3, 'BOROUGH_3')},
    'BOROUGH_4': {'paper_coef': -0.6272, 'paper_t': -143.022, **cell(m3, 'BOROUGH_4')},
    'BOROUGH_5': {'paper_coef': -0.5815, 'paper_t': -81.194, **cell(m3, 'BOROUGH_5')},
    'R2': {'paper': 0.1547, 'recomputed': float(m3.rsquared)},
    'adj_R2': {'paper': 0.155, 'recomputed': float(m3.rsquared_adj)},
    'F_stat': {'paper': 10580, 'recomputed': float(m3.fvalue)},
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

# Round 11: machine-countable status. Exact N; coefs at the manuscript's
# 4dp (tol 5e-5), SEs 4dp, t-stats 3dp (tol 5e-4); R2 4dp; F 10580
# printed rounded (tol 1); F_p 0.0 at tol 1e-3.
# Round 11b: paper_se=None cells are SKIPPED here (not recorded as
# mismatches); the manuscript prints those SEs as "(0.0000)", so the explicit
# value checks below compare against 0.0000 at half a unit of the 4th
# decimal (5e-5).
CHECKS = [check_val('N', 520365, N)]


def _wire(model_key, cells_dict):
    for var, d in cells_dict.items():
        if not isinstance(d, dict):
            continue
        # Round 11b: skip (don't record) cells with no printed manuscript
        # target (paper_* = None) instead of logging a vacuous mismatch.
        for pk, ck, tol in (('paper_coef', 'coef', 5e-5), ('paper_se', 'se', 5e-5),
                            ('paper_t', 't', 5e-4), ('paper_p', 'p', 5e-4)):
            if pk in d and d[pk] is not None and d.get(ck) is not None:
                CHECKS.append(check_val(f'{model_key}.{var}.{ck}',
                                        d[pk], d[ck], tol=tol))
        if 'paper' in d and 'recomputed' in d:
            # Round 11b: adj_R2 and F_stat have NO manuscript citation (grep of
            # README.md, PAPER_TO_CODE_MAP.md, VERIFICATION_REPORT.md, docs/
            # found no "10580"/"10,580"/"0.155" claims) -- they verify claims
            # the manuscript does not make, so they are dropped from the
            # machine-countable status here and in the explicit list below.
            if var in ('adj_R2', 'F_stat'):
                continue
            CHECKS.append(check_val(f'{model_key}.{var}.value', d['paper'],
                                    d['recomputed'], tol=5e-5))


for _mk in ('model1', 'model2', 'model3'):
    _wire(_mk, H[_mk])
# Round 11b: manuscript prints "(0.0000)" for yearbuilt SEs -- target 0.0,
# tol = half a unit of the last printed digit (4dp -> 5e-5).
CHECKS.append(check_val('model2.yearbuilt.se_4dp', 0.0000,
                        H['model2']['yearbuilt']['se'], tol=5e-5))
CHECKS.append(check_val('model3.yearbuilt.se_4dp', 0.0000,
                        H['model3']['yearbuilt']['se'], tol=5e-5))
CHECKS.append(check_val('model3.F_p', 0.0, H['model3']['F_p']['recomputed'], tol=1e-3))
record_status('verify_regression_ols', CHECKS,
              note='Fresh statsmodels OLS re-fit vs manuscript Table 1 cells. '
                   'Round 11b: yearbuilt SEs compared at the printed '
                   '(0.0000); adj_R2/F_stat dropped (not manuscript claims).')
