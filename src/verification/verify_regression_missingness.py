"""Round 5 fact-check: Section H missingness reconciliation (fresh, Tier 1) + shrinkage percentage + 0.10->0.0184 mechanics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import _shared as p1a
from _shared import ROOT, write_json, record_status, check_val

df, cname = p1a.load_canonical()
M = {}

# === Missingness reconciliation: 566,956 -> 520,365 (drop of 46,591) ===
# Round 10: 'paper' targets are the manuscript's CURRENT values. Published
# decomposition: 46,591 = 32,823 (farf group) + 13,277 (numfloors beyond
# farf) + 137 (dem_share-unique) + 407 (other) - 53 (overlap).
total = int(len(df))
# Subset used by the regression (matches generate_regression_table.py / verify_regression_ols.py)
sub = df.dropna(subset=['target_log_price', 'dem_share', 'yearbuilt', 'numfloors', 'landuse', 'BOROUGH', 'commfar'])
N_after = int(len(sub))
dropped = total - N_after
M['total_rows'] = {'paper': 566956, 'recomputed': total}
M['n_after_dropna'] = {'paper': 520365, 'recomputed': N_after}
M['rows_dropped'] = {'paper': 46591, 'recomputed': dropped}

# farf group: yearbuilt/commfar/residfar/facilfar all null simultaneously
farf_mask = df[['yearbuilt', 'commfar', 'residfar', 'facilfar']].isna().all(axis=1)
farf_count = int(farf_mask.sum())
M['farf_group_size'] = {'paper': 32823, 'recomputed': farf_count}

# numfloors missing beyond the farf group: rows where numfloors is missing AND farf_mask is False
notfarf = ~farf_mask
numfloors_missing = df['numfloors'].isna()
numfloors_beyond_farf = int((numfloors_missing & notfarf).sum())
M['numfloors_additional'] = {'paper': 13277, 'recomputed': numfloors_beyond_farf}

# dem_share missing
dem_missing = df['dem_share'].isna()
dem_missing_total = int(dem_missing.sum())
overlap_dem_farf = int((dem_missing & farf_mask).sum())
dem_unique = dem_missing_total - overlap_dem_farf
M['dem_share_missing'] = {'paper': 13371, 'recomputed': dem_missing_total}
M['dem_overlap_with_farf_group'] = {'paper': 13234, 'recomputed': overlap_dem_farf}
M['dem_unique_to_dem_share'] = {'paper': 137, 'recomputed': dem_unique}

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
M['other_drops_beyond_captured'] = {'paper': 407, 'recomputed': other_beyond_captured}
M['check_sum'] = {
    'farf': farf_count,
    'numfloors_beyond_farf': numfloors_beyond_farf,
    'dem_unique_to_dem_share': dem_unique,
    'other_beyond_captured': other_beyond_captured,
    'total': farf_count + numfloors_beyond_farf + dem_unique + other_beyond_captured,
    # Round 10: the manuscript's four components sum to 46,644; 53 rows are
    # shared between components (the additive sum double-counts them), so the
    # published figure is 46,644 - 53 = 46,591. The overlap structure is
    # computed, not assumed.
    'paper_components': [32823, 13277, 137, 407],
    'paper_overlap_adjustment': -53,
    'paper_total': 46591,
    'additive_overlap': (farf_count + numfloors_beyond_farf + dem_unique
                         + other_beyond_captured) - dropped,
    'additive_overlap_equals_paper_53': (farf_count + numfloors_beyond_farf + dem_unique + other_beyond_captured) - dropped == 53,
}

# Coefficient shrinkage (Round 10 targets): 0.9449 -> 0.6146 -> 0.2883.
shrink_full = (0.9449 - 0.2883) / 0.9449
M['shrinkage_pct_0.9449_0.2883'] = {'paper': '~69 percent',
                                    'recomputed_using_4dp_values': round(100.0 * shrink_full, 2)}
# m1 -> m2 shrinkage
shrink_m1_m2 = (0.9449 - 0.6146) / 0.9449
M['shrinkage_pct_m1_to_m2'] = round(100.0 * shrink_m1_m2, 2)

# 0.10 increase in dem_share -> 0.0288 log-point (~2.9% price increase)
prod_0288 = 0.2883 * 0.10
pct_increase = (np.exp(prod_0288) - 1) * 100
M['dem_share_0.10_effect'] = {'paper_logpoints': '0.0288', 'recomputed_products': round(prod_0288, 6),
                              'paper_pct': '~2.9%',
                              'recomputed_pct': round(pct_increase, 4)}
write_json(ROOT / 'results' / 'round5_part_H_missingness.json', {'section_H_missingness': M})
print('WROTE results/round5_part_H_missingness.json')
print('dropped', dropped, 'farf', farf_count, 'numfloors beyond', numfloors_beyond_farf, 'dem missing', dem_missing_total, 'dem overlap farf', overlap_dem_farf, 'dem unique', dem_unique)
print('check_sum', M['check_sum'])
print('shrinkage', M['shrinkage_pct_0.9449_0.2883'])
print('0.10 effect', M['dem_share_0.10_effect'])

# Round 11: machine-countable status
record_status('verify_regression_missingness', [
    check_val('total_rows', 566956, M['total_rows']['recomputed']),
    check_val('n_after_dropna', 520365, M['n_after_dropna']['recomputed']),
    check_val('rows_dropped', 46591, M['rows_dropped']['recomputed']),
    check_val('farf_group', 32823, farf_count),
    check_val('numfloors_beyond_farf', 13277, numfloors_beyond_farf),
    check_val('dem_share_missing', 13371, dem_missing_total),
    check_val('dem_overlap_with_farf', 13234, overlap_dem_farf),
    check_val('dem_unique', 137, dem_unique),
    check_val('other_drops_beyond_captured', 407, other_beyond_captured),
    check_val('additive_overlap_equals_53', True, M['check_sum']['additive_overlap_equals_paper_53']),
    check_val('check_sum_total', 46644, M['check_sum']['total']),
], note='Missingness decomposition (566,956 -> 520,365, drop 46,591) vs '
        'manuscript targets, plus the 53-overlap reconciliation.')
