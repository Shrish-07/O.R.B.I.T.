import json
import pandas as pd

features = json.load(open('models/artifacts/lgbm_all_years_base_features.json'))
df = pd.read_parquet('data/splits/all_years_test.parquet')
good_feats = [f for f in features if df[f].isna().sum() < 100]
print('Good features:', good_feats[:10])
if good_feats:
    sample = df.dropna(subset=good_feats[:10])
    print('Sample rows after dropna:', len(sample))
    print('NA counts (good_feats[:10]):', [df[f].isna().sum() for f in good_feats[:10]])
else:
    print('No good features found')
