"""Round 5 fact-check: Section D (Canonical structure) + Section E (split)."""
import json
from pathlib import Path
import pandas as pd
import _shared as p1a
from _shared import ROOT, write_json, parse_feature_names, record_status, check_val

df, cname = p1a.load_canonical()
OUT = {'raw': {'canonical_file': cname}, 'section_D': {}, 'section_E': {}}

# ===== D =====
D = {}
schema = json.loads((ROOT / 'models' / 'artifacts' / 'canonical_schema.json').read_text())
# Round 10: 'paper' targets = the manuscript's current values (post refresh
# + dedup rebuild; 24-feature base / 25-feature political specification).
# Round 12: the canonical dataset gains 11 analysis-side v2 columns (new
# ideology measures + sample flags, all blacklisted in
# config/feature_blacklist.yaml so no model trains on them). The legacy
# 53-column contract is verified as "all 53 legacy columns still present";
# the schema shape itself now legitimately exceeds 53.
LEGACY_53_COLUMNS = [
    'BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY',
    'TAX CLASS AS OF FINAL ROLL', 'BLOCK', 'LOT', 'EASE-MENT',
    'BUILDING CLASS AS OF FINAL ROLL', 'ADDRESS', 'APARTMENT NUMBER',
    'ZIP CODE', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS',
    'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT',
    'TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE',
    'SALE PRICE', 'SALE DATE', 'Latitude', 'Longitude', 'Community Board',
    'Council District', 'BIN', 'BBL', 'Census Tract 2020',
    'Neighborhood Tabulation Area (NTA) (2020)', 'sale_price', 'sale_date',
    'target_log_price', 'zonedist1', 'bldgclass', 'landuse', 'lotarea',
    'bldgarea', 'numfloors', 'unitsres', 'unitstotal', 'yearbuilt',
    'residfar', 'commfar', 'facilfar', 'BBL_pluto', 'zonedist1_cat',
    'is_R3', 'is_R7', 'is_commercial', 'sale_year', 'election_year',
    'CounDist', 'dem_share']
V2_NEW_COLUMNS = [
    'dem_share_mayor_allcand', 'dem_share_pres', 'dem_share_vintage',
    'dem_share_mayor_allcand_vintage', 'dem_share_pres_vintage',
    'election_year_pres', 'sample_pre2024', 'is_resid_tc12',
    'price_ge_100k', 'multi_parcel_flag', 'building_class_category']
D['dataset_rows'] = {'paper': 566956, 'recomputed': int(schema['shape'][0]), 'actual_loaded': int(len(df))}
D['dataset_cols'] = {'paper': 53, 'recomputed': int(schema['shape'][1]), 'actual_loaded': int(df.shape[1])}
D['dataset_cols_legacy_present'] = {
    'paper': 53,
    'recomputed': int(sum(c in df.columns for c in LEGACY_53_COLUMNS))}
D['dataset_cols_v2_new'] = {
    'expected': len(V2_NEW_COLUMNS),
    'recomputed': int(sum(c in df.columns for c in V2_NEW_COLUMNS))}
D['dataset_cols_v2_blacklisted_from_models'] = {
    'expected': True,
    'recomputed': None}  # filled below
import yaml as _yaml  # noqa: E402
try:
    _bl = _yaml.safe_load((ROOT / 'config' / 'feature_blacklist.yaml').read_text())
    _blset = {str(x).lower() for x in (_bl.get('leakage_features') or [])}
    D['dataset_cols_v2_blacklisted_from_models']['recomputed'] = bool(
        all(c.lower() in _blset for c in V2_NEW_COLUMNS))
except Exception:
    D['dataset_cols_v2_blacklisted_from_models']['recomputed'] = False
base_names, base_mfi = parse_feature_names(ROOT / 'models' / 'lgbm_all_years_base.txt')
pol_names, pol_mfi = parse_feature_names(ROOT / 'models' / 'lgbm_all_years_political.txt')
D['base_feature_count'] = {'paper': 24, 'from_feature_names_line': len(base_names),
                           'from_max_feature_idx_plus1': (base_mfi + 1) if base_mfi is not None else None,
                           'feature_names': base_names}
D['political_feature_count'] = {'paper': 25, 'from_feature_names_line': len(pol_names),
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
    # Round 10: refreshed to the current build (the manuscript was updated to
    # the post-dedup rebuild; these targets mirror the same computation).
    'paper_both_populated': 512349, 'recomputed_both_populated': n_both,
    'paper_agree': 453992, 'recomputed_agree': n_agree,
    'paper_disagree': 58357, 'recomputed_disagree': n_disagree,
    'paper_max_disagreement_years': 2025, 'recomputed_max_disagreement_years': max_dis,
    'paper_correlation': 0.144, 'recomputed_correlation': round(corr, 6),
    'paper_agree_pct': 88.61, 'recomputed_agree_pct': round(100.0 * n_agree / n_both, 2),
    'paper_disagree_pct': 11.39, 'recomputed_disagree_pct': round(100.0 * n_disagree / n_both, 2),
}
OUT['section_D'] = D

# ===== E =====
E = {}
train = pd.read_parquet(ROOT / 'data' / 'splits' / 'all_years_train.parquet')
test = pd.read_parquet(ROOT / 'data' / 'splits' / 'all_years_test.parquet')
E['train_rows'] = {'paper': 121052, 'recomputed': int(len(train))}
E['test_rows'] = {'paper': 56723, 'recomputed': int(len(test))}
ey_dist = df['election_year'].value_counts().sort_index()
E['election_year_distribution'] = {int(yr): {'rows': int(cnt), 'pct': round(100.0 * cnt / len(df), 2)}
                                   for yr, cnt in ey_dist.items()}
# Round 10: refreshed to the post-dedup build (566,956 total rows).
E['election_year_distribution_paper'] = {2017: {'rows': 177775, 'pct': 31.4},
                                         2021: {'rows': 285240, 'pct': 50.3},
                                         2025: {'rows': 103941, 'pct': 18.3}}
n_2017 = int((df['election_year'] == 2017).sum())
E['train_plus_test'] = {'value': int(len(train) + len(test)), 'paper': 177775}
E['election_year_2017_subset'] = {'value': n_2017, 'paper': 177775}
E['identity_holds'] = (int(len(train) + len(test)) == n_2017)
E['train_sale_year_range'] = [int(train['sale_year'].min()), int(train['sale_year'].max())]
E['test_sale_year_range'] = [int(test['sale_year'].min()), int(test['sale_year'].max())]
OUT['section_E'] = E

write_json(ROOT / 'results' / 'round5_part1_DE.json', OUT)
print('WROTE results/round5_part1_DE.json')
print('D base features', D['base_feature_count'], 'political', D['political_feature_count'])
print('E train', E['train_rows'], 'test', E['test_rows'], 'identity', E['identity_holds'])

# Round 11: machine-countable status
_yb = D['yearbuilt_vs_YEARBUILT']
record_status('verify_canonical_dataset_and_split', [
    check_val('dataset_rows', 566956, D['dataset_rows']['recomputed']),
    check_val('dataset_cols_legacy_present', 53, D['dataset_cols_legacy_present']['recomputed']),
    check_val('dataset_cols_v2_new', len(V2_NEW_COLUMNS), D['dataset_cols_v2_new']['recomputed']),
    check_val('dataset_cols_v2_blacklisted_from_models', True,
              D['dataset_cols_v2_blacklisted_from_models']['recomputed']),
    check_val('base_feature_count', 24, D['base_feature_count']['from_feature_names_line']),
    check_val('political_feature_count', 25, D['political_feature_count']['from_feature_names_line']),
    check_val('yearbuilt_both_populated', 512349, _yb['recomputed_both_populated']),
    check_val('yearbuilt_agree', 453992, _yb['recomputed_agree']),
    check_val('yearbuilt_disagree', 58357, _yb['recomputed_disagree']),
    check_val('yearbuilt_correlation_3dp', 0.144, _yb['recomputed_correlation'], tol=5e-4),
    check_val('yearbuilt_agree_pct_2dp', 88.61, _yb['recomputed_agree_pct'], tol=0.005),
    check_val('train_rows', 121052, E['train_rows']['recomputed']),
    check_val('test_rows', 56723, E['test_rows']['recomputed']),
    check_val('train_plus_test_identity', True, E['identity_holds']),
    check_val('election_year_2017_rows', 177775,
              E['election_year_distribution'].get(2017, {}).get('rows')),
    check_val('election_year_2021_rows', 285240,
              E['election_year_distribution'].get(2021, {}).get('rows')),
    check_val('election_year_2025_rows', 103941,
              E['election_year_distribution'].get(2025, {}).get('rows')),
], note='Canonical schema shape, feature counts, YEAR BUILT reconciliation, '
        'split sizes and election-year counts vs manuscript targets.')
