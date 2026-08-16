"""Round 5 fact-check: Section I (Table 2 model comparison). Tier 2 from registry, Tier 1 spot-check from raw predictions where possible."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from round5_factcheck_part1a import ROOT, write_json

I = {}
# Tier 2: read the canonical comparison table
comp = pd.read_csv(ROOT / 'docs' / 'model_comparison_table.csv')
I['comparison_table_source'] = str(ROOT / 'docs' / 'model_comparison_table.csv')
I['rows_in_table'] = int(len(comp))

paper_rows = {
    'lgbm_all_years_base': {'mae': 0.428972, 'r2': 0.541772},
    'lgbm_all_years_political': {'mae': 0.429342, 'r2': 0.541345},
    'rf_all_years_political_n200': {'mae': 0.43705, 'r2': 0.50663},
    'xgb_all_years_political_lr0.05_md6': {'mae': 0.43922, 'r2': 0.52869},
    'cat_all_years_political_lr0.05_d6': {'mae': 0.45391, 'r2': 0.50574},
    'ridge_all_years_political': {'mae': 0.65580, 'r2': 0.11138},
    'elasticnet_all_years_political': {'mae': 0.69037, 'r2': -0.00965},
}
out_models = {}
for name, paper_vals in paper_rows.items():
    row = comp[comp['name'] == name]
    if len(row):
        mae_csv = float(row.iloc[0]['mae'])
        r2_csv = float(row.iloc[0]['r2'])
    else:
        mae_csv = r2_csv = None
    out_models[name] = {'paper': paper_vals, 'comparison_csv': {'mae': mae_csv, 'r2': r2_csv},
                        'mae_match_rounded_5dp': round(mae_csv, 5) == round(paper_vals['mae'], 5) if mae_csv is not None else False,
                        'r2_match_rounded_5dp': round(r2_csv, 5) == round(paper_vals['r2'], 5) if r2_csv is not None else False}
I['models'] = out_models

# arithmetic check: base -> political MAE delta and R2 delta
base = comp[comp['name'] == 'lgbm_all_years_base'].iloc[0]
pol = comp[comp['name'] == 'lgbm_all_years_political'].iloc[0]
I['mae_delta_base_to_political'] = {'paper': 0.000370, 'recomputed': round(float(pol['mae']) - float(base['mae']), 6)}
I['r2_delta_base_to_political'] = {'paper': -0.000427, 'recomputed': round(float(pol['r2']) - float(base['r2']), 6)}

# Tier 1 spot-check: recompute MAE from raw prediction files where available
preds_dir = ROOT / 'experiments' / 'predictions'
spot = {}
# political LGBM test preds
pol_pp = preds_dir / 'lgbm_all_years_political_test_preds.parquet'
if pol_pp.exists():
    d = pd.read_parquet(pol_pp)
    spot['political_test_preds_columns'] = list(d.columns)
    # find prediction and actual columns
    pred_col = next((c for c in d.columns if 'pred' in c.lower() and 'interval' not in c.lower()), None)
    truth_candidates = [c for c in d.columns if c.lower() in ('target_log_price', 'y', 'actual', 'truth', 'true')]
    truth_col = truth_candidates[0] if truth_candidates else None
    if pred_col and truth_col:
        mae_recomputed = float(np.mean(np.abs(d[pred_col] - d[truth_col])))
        spot['political_mae_recomputed'] = mae_recomputed
        spot['political_pred_col'] = pred_col
        spot['political_truth_col'] = truth_col
        spot['n_pred_rows'] = int(len(d))
    else:
        spot['political_pred_or_truth_not_found'] = (pred_col, truth_col)
# also check the intervals file
pol_iv = preds_dir / 'lgbm_all_years_political_intervals.csv'
if pol_iv.exists():
    d2 = pd.read_csv(pol_iv)
    spot['intervals_csv_columns'] = list(d2.columns)
    spot['intervals_csv_rows'] = int(len(d2))
# base model prediction intervals (no point-pred file for base)
base_pi = preds_dir / 'lgbm_all_years_base_prediction_intervals.parquet'
if base_pi.exists():
    d3 = pd.read_parquet(base_pi)
    spot['base_prediction_intervals_columns'] = list(d3.columns)
    spot['base_prediction_intervals_rows'] = int(len(d3))
I['tier1_spot_label'] = ('Political MAE recomputed from raw prediction file if both pred+truth columns present; '
                         'else read from comparison CSV (Tier 2). No point-prediction file exists for the base LGBM, '
                         'so base MAE remains Tier 2 unless a paired truth file is found.')
I['tier1_spot_check'] = spot
write_json(ROOT / 'results' / 'round5_part_I.json', {'section_I_model_comparison': I})
print('WROTE results/round5_part_I.json')
print('models matched:', {k: (v['mae_match_rounded_5dp'], v['r2_match_rounded_5dp']) for k, v in out_models.items()})
print('mae_delta', I['mae_delta_base_to_political'], 'r2_delta', I['r2_delta_base_to_political'])
print('spot', spot)
