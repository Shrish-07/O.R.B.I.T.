from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

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
