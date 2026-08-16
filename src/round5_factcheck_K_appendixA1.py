"""Round 5 fact-check: Section K — full 153-value Appendix A.1 diff + summary stats + named extremes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from round5_factcheck_part1a import ROOT, write_json

K = {}
# Paper table (153 values: 51 districts x 3 scenarios). Borough column ignored for numeric diff.
PAPER = """District,Borough,Liberal,Conservative,MixedGov
1,Manhattan,6.85,-2.94,5.69
2,Manhattan,4.97,0.77,-2.87
3,Manhattan,7.13,-0.56,-13.03
4,Manhattan,2.54,-1.31,12.73
5,Manhattan,1.90,-2.23,13.32
6,Manhattan,2.83,-3.18,-5.55
7,Manhattan,4.11,-0.82,-11.15
8,Manhattan,8.15,-0.38,-12.38
9,Manhattan,6.57,-0.45,-11.25
10,Manhattan,2.33,-7.29,-16.46
11,Bronx,6.27,-0.56,1.49
12,Bronx,0.53,0.11,-5.02
13,Bronx,4.73,-0.16,-2.00
14,Bronx,11.28,-0.42,-4.43
15,Bronx,6.42,0.70,-7.14
16,Bronx,5.13,1.05,-7.66
17,Bronx,7.49,0.17,-13.72
18,Bronx,4.49,0.12,-4.25
19,Queens,11.33,0.09,-0.64
20,Queens,6.91,-0.80,1.20
21,Queens,7.29,-0.90,-0.85
22,Queens,4.04,0.08,3.50
23,Queens,8.53,0.02,-0.88
24,Queens,6.99,-0.33,0.27
25,Queens,3.22,-4.98,-4.62
26,Queens,0.35,-0.45,0.42
27,Queens,9.81,-0.17,-3.09
28,Queens,10.16,-0.16,-5.63
29,Queens,5.27,-2.53,-6.17
30,Queens,2.39,-0.51,-7.59
31,Queens,30.77,0.33,-0.40
32,Queens,18.33,0.20,0.02
33,Brooklyn,1.60,-0.77,-16.73
34,Brooklyn,0.41,-0.44,-3.55
35,Brooklyn,1.16,-1.41,-11.29
36,Brooklyn,0.76,-0.85,-8.83
37,Brooklyn,0.54,-0.44,-2.88
38,Brooklyn,2.59,-0.09,9.04
39,Brooklyn,1.27,-1.16,-16.04
40,Brooklyn,9.89,-2.18,-9.56
41,Brooklyn,-1.06,-1.38,-2.10
42,Brooklyn,-1.21,-0.95,-3.98
43,Brooklyn,6.21,-1.06,1.23
44,Brooklyn,5.97,-1.79,-1.12
45,Brooklyn,9.37,0.42,6.88
46,Brooklyn,16.35,0.93,-7.52
47,Brooklyn,6.85,-0.59,2.16
48,Brooklyn,8.68,-0.68,-0.70
49,StatenIsland,27.65,0.08,-0.08
50,StatenIsland,30.02,0.08,0.25
51,StatenIsland,32.44,0.14,-0.01"""

# Load actual scenario CSV
actual = pd.read_csv(ROOT / 'results' / 'scenario_comparison_clean.csv')
# Load paper
import io
paper_df = pd.read_csv(io.StringIO(PAPER))
# Match by district id (actual CounDist is 1.0..51.0)
actual = actual.copy()
actual['District'] = actual['Council District'].astype(int)
merged = paper_df.merge(actual, left_on='District', right_on='District', how='left')

TOL = 0.02
exceptions = []
for _, r in merged.iterrows():
    d = int(r['District'])
    for paper_col, actual_col, label in [('Liberal', 'liberal_pct', 'liberal'),
                                         ('Conservative', 'conservative_pct', 'conservative'),
                                         ('MixedGov', 'mixed_gov_pct', 'mixed')]:
        pv = float(r[paper_col])
        av = float(r[actual_col])
        diff = abs(pv - av)
        if diff > TOL:
            exceptions.append({'district': d, 'scenario': label, 'paper': pv, 'actual': round(av, 4), 'abs_diff': round(diff, 4)})
K['paper_table_rows'] = int(len(paper_df))
K['actual_csv_rows'] = int(len(actual))
K['merged_rows'] = int(len(merged))
K['n_values_compared'] = 153
K['tolerance_pct_points'] = TOL
K['n_exceptions'] = len(exceptions)
K['exceptions'] = exceptions

# Borough-mapping side check (column B in paper table) using the actual scenario CSV's implied borough map
boromap = pd.read_csv(ROOT / 'results' / 'council_district_borough_map.csv')
paper_boroughs = dict(zip(paper_df['District'], paper_df['Borough']))
boromap['District'] = boromap['CounDist'].astype(int)
boromap['paper'] = boromap['District'].map(paper_boroughs)
boromap['match'] = boromap.apply(lambda r: str(r['paper']).lower().replace(' ', '') == str(r['borough_name']).lower().replace(' ', ''), axis=1)
K['borough_mapping_mismatches'] = boromap[~boromap['match']][['District', 'borough_name', 'paper']].to_dict('records')
K['borough_mapping_all_match'] = bool(boromap['match'].all())

# Summary stats derived from actual CSV (using actual scenario_comparison_clean.csv)
K['summary_stats_from_actual_csv'] = {
    'liberal_mean_pct': round(float(actual['liberal_pct'].mean()), 4),
    'liberal_n_positive': int((actual['liberal_pct'] > 0).sum()),
    'conservative_mean_pct': round(float(actual['conservative_pct'].mean()), 4),
    'conservative_n_negative': int((actual['conservative_pct'] < 0).sum()),
    'mixed_mean_pct': round(float(actual['mixed_gov_pct'].mean()), 4),
    'mixed_n_positive': int((actual['mixed_gov_pct'] > 0).sum()),
}
K['paper_summary_stats'] = {'liberal_mean': 7.42, 'liberal_n_positive': '49 of 51',
                            'conservative_mean': -0.78, 'conservative_n_negative': '35 of 51',
                            'mixed_mean': -3.39, 'mixed_n_positive': '14 of 51'}

# Named extremes from the prose
d = actual.copy()
d['District'] = d['Council District'].astype(int)
def topk(col, k, largest=True):
    s = d.nlargest(k, col) if largest else d.nsmallest(k, col)
    return [int(x) for x in s['District']]
K['named_extremes'] = {
    'liberal_top5_positive_paper': [51, 31, 50, 49, 32],
    'liberal_top5_positive_actual': topk('liberal_pct', 5, True),
    'liberal_only_two_negatives_paper': [41, 42],
    'liberal_negatives_actual': sorted([int(x) for x in d[d['liberal_pct'] < 0]['District']]),
    'conservative_two_named_paper': [10, 25],
    'conservative_two_most_negative_actual': topk('conservative_pct', 2, False),
    'mixed_top5_positive_paper': [5, 4, 38, 45, 1],
    'mixed_top5_positive_actual': topk('mixed_gov_pct', 5, True),
    'mixed_top5_negative_paper': [33, 10, 39, 17, 3],
    'mixed_top5_negative_actual': topk('mixed_gov_pct', 5, False),
}
write_json(ROOT / 'results' / 'round5_part_K_appendixA1.json', {'section_K_appendixA1_diff': K})
print('WROTE results/round5_part_K_appendixA1.json')
print('n_exceptions', len(exceptions))
for e in exceptions[:30]:
    print(e)
print('summary_stats', K['summary_stats_from_actual_csv'])
print('named_extremes', K['named_extremes'])
print('borough mapping all match', K['borough_mapping_all_match'])
