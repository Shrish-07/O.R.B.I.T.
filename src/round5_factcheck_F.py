"""Round 5 fact-check: Section F (4.1 descriptive spatial claims)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import round5_factcheck_part1a as p1a
from round5_factcheck_part1a import ROOT, write_json

df, cname = p1a.load_canonical()
F = {}

# r = 0.163 between mean dem_share and median log price across 51 districts
agg = df.groupby('CounDist').agg(mean_dem=('dem_share', 'mean'), median_log=('target_log_price', 'median'))
r = float(np.corrcoef(agg['mean_dem'], agg['median_log'])[0, 1])
F['bivariate_r_51districts'] = {'paper': 0.163, 'recomputed': round(r, 6), 'n_districts': int(len(agg))}

# District 1: dem_share 0.75-0.85, log price 14.4-14.6 -- check using CD1 across the canonical df mean/median
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
    'paper_dem_share_range': '0.75-0.85',
    'dem_share_min_across_years': round(cd1_ds_min, 4), 'dem_share_max_across_years': round(cd1_ds_max, 4),
    'dem_share_per_year': {int(y): float(v) for y, v in cd1_ideology.set_index('election_year')['dem_share'].items()},
    'paper_logprice_range': '14.4-14.6',
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
