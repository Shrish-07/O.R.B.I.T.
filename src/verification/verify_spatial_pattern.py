"""Round 5 fact-check: Section F (4.1 descriptive spatial claims)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import _shared as p1a
from _shared import ROOT, write_json, record_status, check_val

df, cname = p1a.load_canonical()
F = {}

# r = 0.175 between mean dem_share and median log price across 51 districts
# (manuscript's current value; Round 10)
agg = df.groupby('CounDist').agg(mean_dem=('dem_share', 'mean'), median_log=('target_log_price', 'median'))
r = float(np.corrcoef(agg['mean_dem'], agg['median_log'])[0, 1])
F['bivariate_r_51districts'] = {'paper': 0.175, 'recomputed': round(r, 6), 'n_districts': int(len(agg))}

# District 1 (Round 11b targets, post the 2016-2025 refresh): the manuscript
# says District 1 dem_share runs 0.72 (2017) to 0.93 (2025), and median log
# price runs 14.37 (2021) to 14.55 (2025). Checked per-year below using
# ideology_by_council + canonical target_log_price.
cd1 = agg.loc[1] if 1 in agg.index else None
# But the paper's CD1 ranges refer to per-year ideology ranges. Use ideology_by_council for dem_share per year.
ideology = pd.read_parquet(ROOT / 'data' / 'processed' / 'ideology_by_council.parquet')
cd1_ideology = ideology[ideology['CounDist'] == 1]
# CD1 dem_share range across years
cd1_ds_min = float(cd1_ideology['dem_share'].min())
cd1_ds_max = float(cd1_ideology['dem_share'].max())
# CD1 median log price range across years -> compute per-year median of target_log_price for CD1
df_cd1 = df[df['CounDist'] == 1]
cd1_med_by_year = df_cd1.groupby('election_year')['target_log_price'].median()
F['district_1'] = {
    'paper_dem_share_2017': 0.72, 'paper_dem_share_2025': 0.93,
    'dem_share_min_across_years': round(cd1_ds_min, 4), 'dem_share_max_across_years': round(cd1_ds_max, 4),
    'dem_share_per_year': {int(y): float(v) for y, v in cd1_ideology.set_index('election_year')['dem_share'].items()},
    'paper_logprice_2021': 14.37, 'paper_logprice_2025': 14.55,
    'median_logprice_per_year': {int(y): float(v) for y, v in cd1_med_by_year.items()},
    'median_logprice_min_across_years': float(cd1_med_by_year.min()),
    'median_logprice_max_across_years': float(cd1_med_by_year.max()),
}

# Districts 11,16,18,27: dem_share above 0.80, log price 12.6-13.2
named = [11, 16, 18, 27]
named_out = {}
for d in named:
    di = ideology[ideology['CounDist'] == d]
    med = df[df['CounDist'] == d].groupby('election_year')['target_log_price'].median()
    named_out[d] = {
        'dem_share_min_across_years': float(di['dem_share'].min()),
        'dem_share_max_across_years': float(di['dem_share'].max()),
        'dem_share_any_above_080': bool((di['dem_share'] > 0.80).any()),
        'median_logprice_min_across_years': float(med.min()),
        'median_logprice_max_across_years': float(med.max()),
        'dem_share_per_year': {int(y): float(v) for y, v in di.set_index('election_year')['dem_share'].items()},
        'median_logprice_per_year': {int(y): float(v) for y, v in med.items()},
    }
F['districts_11_16_18_27'] = named_out

# 49 of 51 districts shifted toward higher dem_share (2025>2017), only 16 and 17 opposite
shifts = {}
neg_shift = []
for d in sorted(ideology['CounDist'].unique()):
    y17 = ideology[(ideology['CounDist'] == d) & (ideology['election_year'] == 2017)]['dem_share']
    y25 = ideology[(ideology['CounDist'] == d) & (ideology['election_year'] == 2025)]['dem_share']
    if len(y17) and len(y25):
        delta = float(y25.iloc[0]) - float(y17.iloc[0])
        shifts[int(d)] = round(delta, 6)
        if delta < 0:
            neg_shift.append(int(d))
n_positive = sum(1 for v in shifts.values() if v > 0)
n_negative = sum(1 for v in shifts.values() if v < 0)
F['shift_2017_to_2025'] = {
    'paper_positive': '49 of 51', 'recomputed_positive': n_positive,
    'paper_negative_districts': [16, 17], 'recomputed_negative_districts': sorted(neg_shift),
    'n_negative': n_negative,
    'all_shifts': shifts,
}
write_json(ROOT / 'results' / 'round5_part1_F.json', {'section_F': F})
print('WROTE results/round5_part1_F.json')
print('r', F['bivariate_r_51districts'])
print('shift positive', n_positive, 'negative', n_negative, 'neg districts', sorted(neg_shift))

# Round 11: machine-countable status
record_status('verify_spatial_pattern', [
    check_val('bivariate_r_3dp', 0.175, F['bivariate_r_51districts']['recomputed'], tol=5e-4),
    check_val('n_districts', 51, F['bivariate_r_51districts']['n_districts']),
    check_val('shift_2017_2025_positive_count', 49, n_positive),
    check_val('shift_2017_2025_negative_districts', [16, 17], sorted(neg_shift)),
    # Round 11b: District-1 checks now match the CURRENT manuscript text
    # (dem_share 0.72 in 2017 -> 0.93 in 2025; median log price 14.37 in 2021
    # -> 14.55 in 2025). Values are read from the in-memory recompute above;
    # the paper targets come from the manuscript, so this is not circular.
    check_val('cd1_dem_share_2017', 0.72,
              F['district_1']['dem_share_per_year'][2017], tol=0.005),
    check_val('cd1_dem_share_2025', 0.93,
              F['district_1']['dem_share_per_year'][2025], tol=0.005),
    check_val('cd1_logprice_2021', 14.37,
              F['district_1']['median_logprice_per_year'][2021], tol=0.005),
    check_val('cd1_logprice_2025', 14.55,
              F['district_1']['median_logprice_per_year'][2025], tol=0.005),
    *[{'check': f'd{d}_dem_share_any_above_0.80', 'expected': True,
       'actual': F['districts_11_16_18_27'][str(d)]['dem_share_any_above_080']
       if isinstance(list(F['districts_11_16_18_27'].keys())[0], str)
       else F['districts_11_16_18_27'][d]['dem_share_any_above_080'],
       'tolerance': None,
       'match': (F['districts_11_16_18_27'][d]['dem_share_any_above_080']
                 if not isinstance(list(F['districts_11_16_18_27'].keys())[0], str)
                 else F['districts_11_16_18_27'][str(d)]['dem_share_any_above_080'])}
      for d in [11, 16, 18, 27]],
], note='Section 4.1 descriptive spatial claims: r, 51-district coverage, '
        '2017->2025 ideology shifts, named-district ranges.')
