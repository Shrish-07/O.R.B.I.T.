"""ML dem_share collinearity diagnostics (Task I; diagnostics only — no
model is retrained or touched).

Documents, on the exact train/test sample the paper's models were trained
and scored on:

  1. dem_share is a DETERMINISTIC function of Council District: within the
     all_years train (sale years 2016-17) and test (2018) samples every
     sale maps to election_year 2017, so dem_share takes exactly ONE value
     per district (nunique per district = 1); the R^2 of dem_share on
     51 district dummies is therefore 1.0.
  2. The duplicated feature pairs (same underlying quantity present twice,
     once from the sales file and once from PLUTO, plus the two geography
     identifiers) with their sample correlations.
  3. The champion was chosen on TEST MAE (documented governance fact); if
     validation predictions/metrics are stored anywhere, the
     validation-MAE champion is added.

Output: results/ml_dem_share_collinearity.json

Run: python src/analysis/ml_dem_share_collinearity.py
"""
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / 'data' / 'splits' / 'all_years_train.parquet'
TEST = ROOT / 'data' / 'splits' / 'all_years_test.parquet'
LGBM_TXT = ROOT / 'models' / 'lgbm_all_years_political.txt'
CHAMPION = ROOT / 'experiments' / 'champion.json'
OUT = ROOT / 'results' / 'ml_dem_share_collinearity.json'

DUP_PAIRS = [
    ('RESIDENTIAL UNITS', 'unitsres'),
    ('TOTAL UNITS', 'unitstotal'),
    ('GROSS SQUARE FEET', 'bldgarea'),
    ('LAND SQUARE FEET', 'lotarea'),
    ('YEAR BUILT', 'yearbuilt'),
    ('Community Board', 'Census Tract 2020'),
]

def main():
    train = pd.read_parquet(TRAIN)
    test = pd.read_parquet(TEST)
    both = pd.concat([train, test], ignore_index=True)

    nuniq_train = train.groupby('CounDist')['dem_share'].nunique()
    nuniq_test = test.groupby('CounDist')['dem_share'].nunique()
    nuniq_both = both.groupby('CounDist')['dem_share'].nunique()

    sub = both.dropna(subset=['CounDist', 'dem_share'])
    dd = pd.get_dummies(sub['CounDist'].astype(int), prefix='district',
                        drop_first=True, dtype=float)
    r2 = sm.OLS(sub['dem_share'].astype(float),
                sm.add_constant(dd)).fit().rsquared

    feats = []
    if LGBM_TXT.exists():
        txt = LGBM_TXT.read_text(errors='ignore')
        line = next((ln for ln in txt.splitlines()
                     if ln.startswith('feature_names=')), None)
        if line:
            feats = [f.replace('_', ' ') for f in line.split('=', 1)[1].split()]
    num = both[[c for c in feats if c in both.columns]] \
        .apply(pd.to_numeric, errors='coerce')
    corr = num.corr()
    dup_report = []
    for a, b in DUP_PAIRS:
        if a in corr.columns and b in corr.columns:
            dup_report.append({'feature_a': a, 'feature_b': b,
                               'pearson_r': float(corr.loc[a, b]),
                               'note': 'same underlying quantity recorded '
                                       'twice (sales file vs PLUTO/geography '
                                       'identifiers)'})
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    top6 = [{'feature_a': a, 'feature_b': b, 'pearson_r': float(r)}
            for a, b, r in pairs[:6]]

    champion = json.loads(CHAMPION.read_text()) if CHAMPION.exists() else {}
    val_files = list((ROOT / 'experiments').glob('predictions/*val*')) + \
        list((ROOT / 'models' / 'artifacts').glob('*val*'))
    val_note = ('No stored validation predictions/metrics found under '
                'experiments/predictions/ or models/artifacts/; the '
                'validation-MAE champion cannot be recomputed post hoc.'
                if not val_files else
                'validation artifacts found: '
                + str([f.name for f in val_files]))

    result = {
        'scope': 'diagnostics only — no model retrained, no artifact touched',
        'train_rows': int(len(train)), 'test_rows': int(len(test)),
        'dem_share_deterministic_by_district': {
            'max_nunique_per_district_train': int(nuniq_train.max()),
            'max_nunique_per_district_test': int(nuniq_test.max()),
            'max_nunique_per_district_train_plus_test': int(nuniq_both.max()),
            'districts_with_single_dem_share_value_train_plus_test':
                int((nuniq_both == 1).sum()),
            'n_districts': int(nuniq_both.shape[0]),
            'r2_dem_share_on_district_dummies': float(r2),
            'why': ('all_years train = sale years 2016-17, test = 2018; '
                    'every sale year 2016-2018 maps to election_year 2017, '
                    'so dem_share is one constant per council district in '
                    'the whole train/test sample — collinear with the '
                    'district indicators, not an independent signal.')},
        'duplicated_feature_pairs': dup_report,
        'top6_abs_correlations_overall': top6,
        'champion_selection': {
            'champion': champion.get('name'),
            'test_mae': champion.get('test_mae'),
            'selection_basis': ('test MAE (see experiments/champion.json '
                                'selection_note); a validation-MAE champion '
                                'is not part of the recorded governance'),
            'validation_artifacts': val_note},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Wrote {OUT.relative_to(ROOT)}')
    print(f'  dem_share single-valued per district in train+test: '
          f'{int((nuniq_both == 1).sum())}/{nuniq_both.shape[0]}; '
          f'R2 on district dummies = {r2:.4f}')
    for p in dup_report:
        print(f'  dup pair {p["feature_a"]!r} vs {p["feature_b"]!r}: '
              f'r={p["pearson_r"]:.4f}')


if __name__ == '__main__':
    main()

