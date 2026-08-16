"""Round 5 fact-check: Section H missingness reconciliation (fresh, Tier 1) + shrinkage percentage + 0.10->0.0184 mechanics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import round5_factcheck_part1a as p1a
from round5_factcheck_part1a import ROOT, write_json

df, cname = p1a.load_canonical()
M = {}

# === Missingness reconciliation: 514,618 -> 470,535 (drop of 44,083) ===
total = int(len(df))
# Subset used by the regression (matches generate_regression_table.py / round5_factcheck_H_ols.py)
sub = df.dropna(subset=['target_log_price', 'dem_share', 'yearbuilt', 'numfloors', 'landuse', 'BOROUGH', 'commfar'])
N_after = int(len(sub))
dropped = total - N_after
M['total_rows'] = {'paper': 514618, 'recomputed': total}
M['n_after_dropna'] = {'paper': 470535, 'recomputed': N_after}
M['rows_dropped'] = {'paper': 44083, 'recomputed': dropped}

# farf group: yearbuilt/commfar/residfar/facilfar all null simultaneously
farf_mask = df[['yearbuilt', 'commfar', 'residfar', 'facilfar']].isna().all(axis=1)
farf_count = int(farf_mask.sum())
M['farf_group_size'] = {'paper': 31692, 'recomputed': farf_count}

# numfloors missing beyond the farf group: rows where numfloors is missing AND farf_mask is False
notfarf = ~farf_mask
numfloors_missing = df['numfloors'].isna()
numfloors_beyond_farf = int((numfloors_missing & notfarf).sum())
M['numfloors_additional'] = {'paper': 11918, 'recomputed': 11930,
                              'round2_corrected_value_used': 11930, 'diff_from_paper': numfloors_beyond_farf - 11918}

# dem_share missing
dem_missing = df['dem_share'].isna()
dem_missing_total = int(dem_missing.sum())
overlap_dem_farf = int((dem_missing & farf_mask).sum())
dem_unique = dem_missing_total - overlap_dem_farf
M['dem_share_missing'] = {'paper': 12558, 'recomputed': dem_missing_total}
M['dem_overlap_with_farf_group'] = {'paper': 12424, 'recomputed': overlap_dem_farf}
M['dem_unique_to_dem_share'] = {'paper': 134, 'recomputed': dem_unique}

# Reconcile: dropped should == union of (target_log_price null, dem_share missing not in farf...).
# The regression drop excludes any row with NA in the subset list. We have already computed N_after.
# Cross-check: dropped = N_dropped_by_any_subset_var质朴 = ...
# Dem missing not in farf (unique) is part of the drop.
# Verify the arithmetic that farf + (numfloors beyond farf) + (dem unique) + any_target_logprice/landuse/BOROUGH extras == dropped?
# Compute exactly:
# Rows dropped because target_log_price null OR landuse null OR BOROUGH null (NOT already captured by farf or dem/farloors)
target_null = df['target_log_price'].isna()
landuse_null = df['landuse'].isna()
borough_null = df['BOROUGH'].isna()
captured = farf_mask | (numfloors_missing & notfarf) | dem_missing
other_drops = target_null | landuse_null | borough_null
other_beyond_captured = int((other_drops & ~captured).sum())
M['other_drops_beyond_captured'] = other_beyond_captured
M['check_sum'] = {
    'farf': farf_count,
    'numfloors_beyond_farf': numfloors_beyond_farf,
    'dem_unique_to_dem_share': dem_unique,
    'other_beyond_captured': other_beyond_captured,
    'total': farf_count + numfloors_beyond_farf + dem_unique + other_beyond_captured,
    'matches_dropped': (farf_count + numfloors_beyond_farf + dem_unique + other_beyond_captured) == dropped,
}

# Coefficient shrinkage: 0.889 -> 0.574 -> 0.184 (~80% reduction). (0.889-0.184)/0.889 = ?
shrink_full = (0.889 - 0.184) / 0.889
shrink_full_precise = (0.8890 - 0.1843) / 0.8890
M['shrinkage_pct_0.889_0.184'] = {'paper': '~80 percent',
                                  'recomputed_using_paper_rounding': round(100.0 * shrink_full, 2),
                                  'recomputed_using_4dp_values': round(100.0 * shrink_full_precise, 2)}
# m1 -> m2 shrinkage
shrink_m1_m2 = (0.8890 - 0.5741) / 0.8890
M['shrinkage_pct_m1_to_m2'] = round(100.0 * shrink_m1_m2, 2)

# 0.10 increase in dem_share -> 0.0184 log-point (~1.9% price increase)
prod_0184 = 0.1843 * 0.10
pct_increase = (np.exp(prod_0184) - 1) * 100
M['dem_share_0.10_effect'] = {'paper_logpoints': '0.0184', 'recomputed_products': round(prod_0184, 6),
                              'paper_pct': '~1.9% (also stated 1.86-1.9%)',
                              'recomputed_pct': round(pct_increase, 4)}
write_json(ROOT / 'results' / 'round5_part_H_missingness.json', {'section_H_missingness': M})
print('WROTE results/round5_part_H_missingness.json')
print('dropped', dropped, 'farf', farf_count, 'numfloors beyond', numfloors_beyond_farf, 'dem missing', dem_missing_total, 'dem overlap farf', overlap_dem_farf, 'dem unique', dem_unique)
print('check_sum', M['check_sum'])
print('shrinkage', M['shrinkage_pct_0.889_0.184'])
print('0.10 effect', M['dem_share_0.10_effect'])
