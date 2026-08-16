"""Round 5 fact-check: Section C (Ideology construction) — FIXED with correct column names."""
import json
from pathlib import Path
import pandas as pd
import round5_factcheck_part1a as p1a
from round5_factcheck_part1a import ROOT, write_json

df, cname = p1a.load_canonical()
ideology = pd.read_parquet(ROOT / 'data' / 'processed' / 'ideology_by_council.parquet')
dist_ideology = pd.read_parquet(ROOT / 'data' / 'processed' / 'district_ideology.parquet')
crosswalk = pd.read_parquet(ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk.parquet')

C = {}
cd1 = ideology[(ideology['CounDist'] == 1) & (ideology['election_year'] == 2017)]
C['cd1_2017_dem_share'] = {'paper': 0.715908, 'recomputed': float(cd1['dem_share'].iloc[0]),
                           'diff': float(cd1['dem_share'].iloc[0]) - 0.715908}
C['district_ideology_columns'] = list(dist_ideology.columns)
C['district_ideology_years_present'] = sorted([int(x) for x in dist_ideology['year'].unique()])
cd1_di = dist_ideology[dist_ideology['COUNCIL_DISTRICT'] == 1]
C['cd1_district_ideology'] = {'paper': 0.711309, 'recomputed': float(cd1_di['district_ideology'].iloc[0]),
                             'value_column_used': 'district_ideology'}
n_missing = int(df['dem_share'].isna().sum())
n_total = int(len(df))
C['dem_share_missing'] = {'paper': 12558, 'recomputed': n_missing}
C['dem_share_coverage_pct'] = {'paper': 97.56, 'recomputed': round(100.0 * (1 - n_missing / n_total), 4)}
C['dem_share_missing_pct'] = {'paper': 2.44, 'recomputed': round(100.0 * n_missing / n_total, 4)}
C['canonical_total'] = n_total
C['crosswalk_columns'] = list(crosswalk.columns)
C['crosswalk_n_rows'] = {'paper': 5783, 'recomputed': int(len(crosswalk))}
ed_col = 'ElectDist'
distinct = int(crosswalk[ed_col].nunique())
ec = crosswalk[ed_col].value_counts()
multi = int((ec > 1).sum())
C['crosswalk_n_distinct_ed'] = {'paper': 4264, 'recomputed': distinct}
C['crosswalk_ed_in_gt1_row'] = {'paper': 1383, 'recomputed': multi}
C['crosswalk_ed_multi_pct'] = {'paper': 32.4, 'recomputed': round(100.0 * multi / distinct, 2)}
ed_total_area = crosswalk.groupby(ed_col)['area'].transform('sum')
crosswalk['_share'] = crosswalk['area'] / ed_total_area
split_eds = crosswalk[crosswalk[ed_col].map(ec) > 1]
max_share_per_ed = split_eds.groupby(ed_col)['_share'].max()
minority_share_per_ed = 1.0 - max_share_per_ed
max_minority_share = float(minority_share_per_ed.max()) if len(minority_share_per_ed) else 0.0
C['crosswalk_max_minority_share'] = {'paper_le': 0.11,
                                     'recomputed_fraction': max_minority_share,
                                     'recomputed_pct': round(100.0 * max_minority_share, 4),
                                     'n_split_eds': int(split_eds[ed_col].nunique())}
write_json(ROOT / 'results' / 'round5_part1_C.json', {'section_C_ideology': C})
print('WROTE results/round5_part1_C.json')
for k, v in C.items():
    print(k, v)
