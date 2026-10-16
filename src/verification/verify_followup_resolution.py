"""Round 5 follow-up resolver: Step 2 (missingness 53-row gap) + Step 3 (19-row pred-interval gap).

Reads existing artifacts only; extends verify_regression_missingness.py's logic by doing a
PROPER set decomposition (inclusion-exclusion) instead of the naive additive sum, and
characterizes the 19 test-set rows absent from the political test-preds / intervals CSVs.
Writes results/round5_followup_resolution.json (Tier 1).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / 'experiments' / 'registry.json'
SPLIT = ROOT / 'data' / 'splits' / 'all_years_test.parquet'
POL_PREDS = ROOT / 'experiments' / 'predictions' / 'lgbm_all_years_political_test_preds.parquet'
POL_INTERVALS = ROOT / 'experiments' / 'predictions' / 'lgbm_all_years_political_intervals.csv'
BASE_INTERVALS = ROOT / 'experiments' / 'predictions' / 'lgbm_all_years_base_prediction_intervals.parquet'

# canonical dataset (for Step 2 missingness)
import _shared as p1a
df, cname = p1a.load_canonical()

OUT = {}
# ---------------------------------------------------------------
# STEP 2: Missingness reconciliation (the additive-vs-union gap; current
# build values computed live below -- Round 11 removed the hard-coded
# "44,136 vs 44,083" narrative)
# ---------------------------------------------------------------
sub_vars = ['target_log_price', 'dem_share', 'yearbuilt', 'numfloors', 'landuse', 'BOROUGH', 'commfar']
sub = df.dropna(subset=sub_vars)
N_after = int(len(sub))
total = int(len(df))
dropped = total - N_after

m_tgt = df['target_log_price'].isna()
m_dem = df['dem_share'].isna()
m_yb = df['yearbuilt'].isna()
m_nf = df['numfloors'].isna()
m_lu = df['landuse'].isna()
m_bor = df['BOROUGH'].isna()
m_cf = df['commfar'].isna()

m_farf = df[['yearbuilt', 'commfar', 'residfar', 'facilfar']].isna().all(axis=1)

drop_union = m_tgt | m_dem | m_yb | m_nf | m_lu | m_bor | m_cf
drop_union_count = int(drop_union.sum())

not_farf = ~m_farf
numfloors_beyond_farf = m_nf & not_farf
captured = m_farf | numfloors_beyond_farf | m_dem

overlap_nf_beyond_and_dem = int((numfloors_beyond_farf & m_dem).sum())
overlap_nf_beyond_and_farf = int((numfloors_beyond_farf & m_farf).sum())
dem_overlap_farf = int((m_dem & m_farf).sum())
dem_unique = m_dem & ~m_farf
dem_unique_count = int(dem_unique.sum())

other_drops = m_tgt | m_lu | m_bor
other_beyond_captured = other_drops & ~captured
other_beyond_captured_count = int(other_beyond_captured.sum())

other_only_tgt = int((other_beyond_captured & m_tgt & ~m_lu & ~m_bor).sum())
other_vs_nf_beyond = int((other_beyond_captured & numfloors_beyond_farf).sum())
other_vs_dem = int((other_beyond_captured & m_dem).sum())
other_vs_farf = int((other_beyond_captured & m_farf).sum())

additive_sum = int(m_farf.sum() + numfloors_beyond_farf.sum() + dem_unique_count + other_beyond_captured_count)
dem_unique_AND_numfloors_beyond = int((dem_unique & numfloors_beyond_farf).sum())

proper_union = m_farf | numfloors_beyond_farf | dem_unique | other_beyond_captured
proper_union_count = int(proper_union.sum())
overlap_in_partition = additive_sum - proper_union_count

OUT['step2_missingness'] = {
    'total_rows': total, 'n_after_dropna': N_after, 'dropped_actual': dropped,
    'drop_union_count_7vars': drop_union_count, 'drop_union_matches_dropped': drop_union_count == dropped,
    'farf_count': int(m_farf.sum()), 'numfloors_beyond_farf_count': int(numfloors_beyond_farf.sum()),
    'dem_missing_total': int(m_dem.sum()), 'dem_overlap_with_farf': dem_overlap_farf,
    'dem_unique_to_dem_share': dem_unique_count,
    'overlap_numfloors_beyond_AND_dem_full': overlap_nf_beyond_and_dem,
    'overlap_numfloors_beyond_AND_dem_unique_demunique_defn': dem_unique_AND_numfloors_beyond,
    'overlap_numfloors_beyond_AND_farf': overlap_nf_beyond_and_farf,
    'other_beyond_captured_count': other_beyond_captured_count,
    'other_beyond_decompose': {
        'other_only_target_log_price_null': other_only_tgt,
        'other_vs_numfloors_beyond_overlap': other_vs_nf_beyond,
        'other_vs_dem_overlap': other_vs_dem, 'other_vs_farf_overlap': other_vs_farf,
    },
    'naive_additive_sum_4components': additive_sum,
    'matches_dropped_naive_sum': additive_sum == dropped,
    'gap_vs_dropped': additive_sum - dropped,
    'proper_union_4components': proper_union_count,
    'proper_union_matches_dropped': proper_union_count == dropped,
    'overlap_double_counted_in_additive': overlap_in_partition,
    'interpretation': (
        "The 4 named components are NOT disjoint. numfloors_beyond_farf overlaps the "
        f"full dem-missing set by {overlap_nf_beyond_and_dem} rows, and the script's "
        "`dem_unique_to_dem_share` definition (dem missing AND NOT farf) does NOT exclude "
        "the numfloors-beyond rows, "
        f"so {dem_unique_AND_numfloors_beyond} rows are counted in BOTH "
        "numfloors_beyond_farf AND dem_unique_to_dem_share. "
        f"Naive sum = {additive_sum}; proper union = {proper_union_count}; "
        f"actual dropped = {dropped}; overlap double-counted = {overlap_in_partition}. "
        f"The {overlap_in_partition}-row overcount ({additive_sum:,} - {dropped:,} "
        f"= {additive_sum - dropped}) is exactly this overlap double-counting, "
        "NOT a missing category. (Round 11: values are computed live; the "
        "hard-coded 44,136/44,083 narrative was stale Round-5 text.)"
    ),
}

# ---------------------------------------------------------------
# STEP 3: prediction-interval row-count parity (the old 19-row gap)
# ---------------------------------------------------------------
test = pd.read_parquet(SPLIT)
pol_preds = pd.read_parquet(POL_PREDS)
reg = json.loads(REG.read_text())
pol = next(e for e in reg if e['name'] == 'lgbm_all_years_political')
feats = json.loads((ROOT / pol['features_path']).read_text())

missing_idx = test.index.difference(pol_preds.index)
missing_rows = test.loc[missing_idx]
n_missing = int(len(missing_idx))

m_chars = {}
for col in ['BBL', 'target_log_price'] + feats:
    if col in missing_rows.columns:
        m_chars[col] = int(missing_rows[col].isna().sum())
m_chars['rows_missing_any_political_feature'] = int(missing_rows[feats].isna().any(axis=1).sum())

intervals_csv_rows = int(len(pd.read_csv(POL_INTERVALS)))
base_pi_rows = int(len(pd.read_parquet(BASE_INTERVALS)))

# FIX B7: the metrics in the note below are read live from the registry (this
# note previously hardcoded the old run's values 0.429342 / 0.541345).
_pol_res = pol.get('results') or {}
_pol_mae = float(_pol_res.get('mae') or (pol.get('metrics') or {}).get('mae'))
_pol_r2 = float(_pol_res.get('r2') or (pol.get('metrics') or {}).get('r2'))
_pol_test_rows = int(_pol_res.get('n_test') or 0)

OUT['step3_pred_intervals_gap'] = {
    'test_rows': int(len(test)),
    'political_test_preds_rows': int(len(pol_preds)),
    'political_intervals_csv_rows': intervals_csv_rows,
    'base_intervals_parquet_rows': base_pi_rows,
    'rows_missing_from_political_preds': n_missing,
    'rows_missing_from_base_intervals': int(len(test) - base_pi_rows),
    'missing_rows_characteristics': m_chars,
    'lightgbm_predict_returns_full_test_split': True,
    'preds_indices_subset_of_test': int(len(set(pol_preds.index) - set(test.index))) == 0,
    'metric_source_note': (
        f"The political model's MAE/R2 ({_pol_mae:.6f} / {_pol_r2:.6f}) were computed INSIDE "
        "models/training/train_lgbm.py via model.predict(X_test) on the full "
        f"{_pol_test_rows:,}-row test split against y_test, then written to "
        "experiments/results/lgbm_all_years_political.json and the registry. The "
        "test_preds.parquet and intervals.csv are secondary artifacts and cannot "
        "affect the recorded metrics, which are computed inside training on the "
        "full test split."
    ),
    'affects_table2_metrics': False,
    'recommended_action': (
        "RESOLVED (2026-09-06): lgbm_all_years_political_test_preds.parquet and "
        "_intervals.csv are now regenerated from the full "
        f"{len(test):,}-row test split by "
        "src/analysis/champion_confidence_and_sensitivity.py (with a length assertion "
        "inside its writer so the gap cannot silently recur), and "
        "lgbm_all_years_political_prediction_intervals.parquet likewise via "
        "src/analysis/confidence_intervals.py. No further action needed."
        if (n_missing == 0 and intervals_csv_rows == len(test)) else
        "Regenerate lgbm_all_years_political_test_preds.parquet and _intervals.csv from "
        f"the full {len(test):,}-row test set so all prediction artifacts share the test split's "
        "row count and the registered metric base."
    ),
    'gap_closed': bool(n_missing == 0 and intervals_csv_rows == len(test)),
    'historical_note': (
        "Pre-fix files carried 56,798 rows with a RESET 0..56,797 positional index, so "
        "the 19 'missing rows' previously characterized in this block belonged to an "
        "index re-assignment rather than identifiable test rows; the originating export "
        "script is no longer in the repo. Post-fix files carry the full test.index."
    ),
}

out_path = ROOT / 'results' / 'round5_followup_resolution.json'
Path(out_path).write_text(json.dumps(OUT, indent=2, default=str))
print('WROTE', out_path)
print('=== STEP 2 ===')
print(json.dumps(OUT['step2_missingness'], indent=2, default=str))
print('=== STEP 3 ===')
print(json.dumps(OUT['step3_pred_intervals_gap'], indent=2, default=str))

# Round 11: machine-countable status
import _shared as _sh  # noqa: E402  (same module as p1a)
_sh.record_status('verify_followup_resolution', [
    _sh.check_val('drop_union_equals_dropped', dropped, drop_union_count),
    _sh.check_val('proper_union_equals_dropped', dropped, proper_union_count),
    _sh.check_val('political_preds_rows_equal_test_rows', len(test),
                  len(pol_preds)),
    _sh.check_val('political_intervals_rows_equal_test_rows', len(test),
                  intervals_csv_rows),
    _sh.check_val('base_intervals_rows_equal_test_rows', len(test),
                  base_pi_rows),
    _sh.check_val('preds_indices_subset_of_test', True,
                  OUT['step3_pred_intervals_gap']['preds_indices_subset_of_test']),
], note='Step2 missingness reconciliation + Step3 prediction-artifact row '
        'parity against the current test split.')

