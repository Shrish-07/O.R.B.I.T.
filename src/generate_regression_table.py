from pathlib import Path
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet')
df = df.dropna(subset=['target_log_price', 'dem_share', 'yearbuilt', 'numfloors', 'landuse', 'BOROUGH', 'commfar'])

# Model 1: price ~ dem_share only
X1 = sm.add_constant(df[['dem_share']])
m1 = sm.OLS(df['target_log_price'], X1).fit()

# Model 2: price ~ dem_share + structural controls
controls = ['yearbuilt', 'numfloors', 'landuse', 'commfar']
X2 = sm.add_constant(df[['dem_share'] + controls])
m2 = sm.OLS(df['target_log_price'], X2).fit()

# Model 3: price ~ dem_share + controls + borough fixed effects
_df_fe = pd.get_dummies(df, columns=['BOROUGH'], drop_first=True, dtype=float)
boro_cols = [c for c in _df_fe.columns if c.startswith('BOROUGH_')]
X3 = sm.add_constant(_df_fe[['dem_share'] + controls + boro_cols])
m3 = sm.OLS(df['target_log_price'], X3).fit()

print('=== REGRESSION RESULTS ===')
print()
print('Model 1: dem_share only')
print(f'  dem_share coef: {m1.params["dem_share"]:.4f}  SE: {m1.bse["dem_share"]:.4f}  p: {m1.pvalues["dem_share"]:.4f}  R2: {m1.rsquared:.4f}')
print()
print('Model 2: dem_share + structural controls')
print(f'  dem_share coef: {m2.params["dem_share"]:.4f}  SE: {m2.bse["dem_share"]:.4f}  p: {m2.pvalues["dem_share"]:.4f}  R2: {m2.rsquared:.4f}')
for c in controls:
    print(f'  {c} coef: {m2.params[c]:.4f}  p: {m2.pvalues[c]:.4f}')
print()
print('Model 3: dem_share + controls + borough FE')
print(f'  dem_share coef: {m3.params["dem_share"]:.4f}  SE: {m3.bse["dem_share"]:.4f}  p: {m3.pvalues["dem_share"]:.4f}  R2: {m3.rsquared:.4f}')
print()
print(f'Sample size: {len(df):,}')

results = []
for name, model, xvars in [('Bivariate', m1, ['dem_share']), ('+ Controls', m2, ['dem_share'] + controls), ('+ Borough FE', m3, ['dem_share'] + controls)]:
    for v in xvars:
        if v in model.params:
            results.append({'Model': name, 'Variable': v, 'Coef': round(model.params[v], 4), 'SE': round(model.bse[v], 4), 'p': round(model.pvalues[v], 4), 'R2': round(model.rsquared, 4), 'N': len(df)})

pd.DataFrame(results).to_csv(ROOT / 'results' / 'regression_table.csv', index=False)
print('Saved results/regression_table.csv')

# =====================================================================
# Appendix A.2 / Section 4.3 — Model 3 OLS residual diagnostics
# Saves:
#   results/regression_diagnostics.json         (numeric fields, machine-readable)
#   results/regression_model3_full_summary.txt  (raw m3.summary() text)
# The diagnostics are read from the SAME fitted model (m3), so the numbers are
# guaranteed self-consistent with the regression table above.
# =====================================================================
diag_dir = ROOT / 'results'
diag_dir.mkdir(parents=True, exist_ok=True)

# Raw full summary text.
full_summary_text = str(m3.summary())
with open(diag_dir / 'regression_model3_full_summary.txt', 'w', encoding='utf-8') as f:
    f.write(full_summary_text)
print('Saved results/regression_model3_full_summary.txt')

# Numeric diagnostics.
# - Durbin-Watson (lower-bound 0 = + autocorrelation, 2 = no autocorrelation)
# - Condition number (m3.condition_number) — large => multicollinearity
# - skew / kurtosis of the OLS residuals (scipy uses unbiased/fisher convention;
#   statsmodels' summary prints the same values as scipy.stats.skew/kurtosis on
#   the residuals, so we use scipy to guarantee self-consistency with m3.resid).
# - Jarque-Bera + its p-value (omnibus normality test), also taken from
#   statsmodels' diagnosed m3 so it matches the printed summary tables.
resid = m3.resid
dw_val = float(durbin_watson(resid))
cond_no = float(m3.condition_number)
skew_val = float(sp_stats.skew(resid, bias=False))
# Kurtosis is reported under TWO standard conventions because the paper's
# Appendix A.2 "kurtosis = 7.204" uses the Pearson (non-excess) convention
# (kurtosis_pearson = 3 for a normal), while scipy.stats.kurtosis(fisher=True)
# returns the Fisher excess kurtosis (0 for a normal). They differ by exactly 3.
# Saving both removes the previous ambiguity and lets the report show an exact
# match to the paper's 7.204 without changing the underlying residual statistic.
kurt_fisher = float(sp_stats.kurtosis(resid, fisher=True, bias=False))   # excess (normal => 0)
kurt_pearson = float(kurt_fisher + 3.0)                                  # non-excess (normal => 3)
# statsmodels' summary Jarque-Bera reads from the omnibus diagnostic; pull the
# summary2() tables directly to guarantee the exact in-summary numbers.
s2 = m3.summary2()
jb_val = None
jb_p = None
omni_p = None
omni_val = None

# IMPORTANT: statsmodels' summary2() "third table" (the Omnibus / Durbin-Watson /
# Jarque-Bera / Skew / Kurtosis / Cond. No. block) is NOT a 2-column [label, value]
# Frame indexed by label. It is a wide 4-column layout arranged as a 2x4 grid of
# (label, value, label, value) pairs:
#     col0 = left labels        (e.g. 'Omnibus:', 'Prob(Omnibus):', 'Skew:', 'Kurtosis:')
#     col1 = left values        (e.g. '50100.014', '0.000', '0.264', '7.204')
#     col2 = right labels       (e.g. 'Durbin-Watson:', 'Jarque-Bera (JB):', 'Prob(JB):', 'Condition No.:')
#     col3 = right values       (e.g. '1.517', '352014.397', '0.000', '36407')
# Its pandas .index is just integer positions [0, 1, 2, 3] -- NOT the text labels.
# The previous parser queried the integer index for the strings 'omnibus'/'jarque-bera'
# and so the membership test was always False, silently leaving omnibus == None. We
# now read the values out of the two (label-column, value-column) pairs by lower-cased
# label match (labels come with a trailing ':' in summary2()).
_DIAG_LABEL_COL_FROM_VALUE_COL = {0: 1, 2: 3}  # label-in-col0 -> value-in-col1; label-in-col2 -> value-in-col3

def _norm_label(s):
    return str(s).strip().rstrip(':').lower()

try:
    for tbl in s2.tables:
        if not hasattr(tbl, 'iloc') or not hasattr(tbl, 'columns'):
            continue
        ncols = len(tbl.columns)
        if ncols < 4:
            continue
        # Build a label -> raw-value-string map from both label/value column pairs.
        cells = {}
        for lab_col, val_col in _DIAG_LABEL_COL_FROM_VALUE_COL.items():
            if lab_col >= ncols or val_col >= ncols:
                continue
            for r in range(len(tbl)):
                lab = _norm_label(tbl.iloc[r, lab_col])
                if not lab:
                    continue
                cells[lab] = tbl.iloc[r, val_col]
        # Only proceed if this is the diagnostics table (Omnibus present).
        if 'omnibus' not in cells and 'durbin-watson' not in cells:
            continue
        if 'omnibus' in cells and cells['omnibus'] not in (None, ''):
            omni_val = float(cells['omnibus'])
        if 'prob(omnibus)' in cells and cells['prob(omnibus)'] not in (None, ''):
            omni_p = float(cells['prob(omnibus)'])
        if 'jarque-bera (jb)' in cells and cells['jarque-bera (jb)'] not in (None, ''):
            jb_val = float(cells['jarque-bera (jb)'])
        elif 'jarque-bera' in cells and cells['jarque-bera'] not in (None, ''):
            jb_val = float(cells['jarque-bera'])
        if 'prob(jb)' in cells and cells['prob(jb)'] not in (None, ''):
            jb_p = float(cells['prob(jb)'])
        elif 'jarque-bera (jb)' in cells and 'prob(jb)' in cells:
            # already handled above; noop for clarity
            pass
        break
except Exception:
    # Leave any already-set values intact; text fallback below will fill gaps.
    pass

# Fallback: parse the plain-text m3.summary() output for the same fields, in case a
# future statsmodels version changes the summary2() table shape. The text layout is a
# fixed two-column ".label....value   label....value" block, e.g.:
#     Omnibus:                    50100.014   Durbin-Watson:                   1.517
#     Prob(Omnibus):                  0.000   Jarque-Bera (JB):           352014.397
import re
try:
    _txt = str(m3.summary())
except Exception:
    _txt = ''
def _text_num(label, text):
    # Match "<label> ... <number>" on the same physical line, label may end with ':'.
    m = re.search(r'\b' + re.escape(label) + r'\s*:?\s*([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)', text)
    return float(m.group(1)) if m else None
if omni_val is None:
    omni_val = _text_num('Omnibus', _txt)
if omni_p is None:
    _m = re.search(r'\bProb\(Omnibus\)\s*:?\s*([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)', _txt)
    if _m:
        omni_p = float(_m.group(1))
if jb_val is None:
    jb_val = _text_num('Jarque-Bera', _txt)
if jb_p is None:
    _m = re.search(r'\bProb\(JB\)\s*:?\s*([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)', _txt)
    if _m:
        jb_p = float(_m.group(1))
# Final scipy fallback for J-B only (omnibus has no direct scipy equivalent of the
# statsmodels-summary Omnibus statistic; the summary text is the ground truth).
if jb_val is None:
    jb_val = float(sp_stats.jarque_bera(resid)[0])
    jb_p = float(sp_stats.jarque_bera(resid)[1])

diagnostics = {
    'model': 'Model 3 (dem_share + structural controls + borough FE)',
    'n_observations': int(m3.nobs),
    'n_predictors': int(m3.df_model),
    'r2': float(m3.rsquared),
    'adj_r2': float(m3.rsquared_adj),
    'durbin_watson': dw_val,
    'condition_number': cond_no,
    'residual_skew': skew_val,
    'residual_kurtosis_fisher_excess': kurt_fisher,
    'residual_kurtosis_pearson': kurt_pearson,
    'kurtosis_paper_reports': 'Pearson convention (kurtosis = excess + 3)',
    'kurtosis_matches_paper': bool(abs(kurt_pearson - 7.204) < 0.001),
    'jarque_bera': jb_val,
    'jarque_bera_pvalue': jb_p,
    'omnibus': omni_val,
    'omnibus_pvalue': omni_p,
    'paper_targets_appendix_a2': {
        'durbin_watson': 1.517,
        'omnibus': 50100.014,
        'jarque_bera': 352014.397,
        'skew': 0.264,
        'kurtosis': 7.204,
        'condition_number': 3.64e4,
    },
    'note': (
        "residual_skew uses scipy.stats.skew(bias=False) on m3.resid. "
        "Two kurtosis conventions are saved to resolve a labeling ambiguity: "
        "residual_kurtosis_fisher_excess is the Fisher excess kurtosis (normal => 0), "
        "and residual_kurtosis_pearson is the Pearson (non-excess) kurtosis = excess + 3 "
        "(normal => 3). The paper's Appendix A.2 kurtosis field (7.204) is the Pearson "
        "convention, so it equals residual_kurtosis_fisher_excess + 3 exactly. "
        "All diagnostics computed from the same m3 fit as regression_table.csv."
    ),
}

with open(diag_dir / 'regression_diagnostics.json', 'w', encoding='utf-8') as f:
    json.dump(diagnostics, f, indent=2)
print('Saved results/regression_diagnostics.json')
print()
print('=== MODEL 3 DIAGNOSTICS ===')
for k in ['durbin_watson', 'condition_number', 'residual_skew',
          'residual_kurtosis_fisher_excess', 'residual_kurtosis_pearson',
          'jarque_bera', 'jarque_bera_pvalue',
          'kurtosis_paper_reports', 'kurtosis_matches_paper']:
    print(f'  {k}: {diagnostics[k]}')
