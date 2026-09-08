from pathlib import Path
import json
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pandas as pd

ARTIFACT_DIR = Path('models/artifacts')
SPLIT_DIR = Path('data/splits')

for f in ARTIFACT_DIR.glob('*_features.json'):
    name = f.stem.replace('_features','')
    feats = json.loads(f.read_text())
    variant = 'year2017' if 'year2017' in name else 'all_years'
    train_path = SPLIT_DIR / f"{variant}_train.parquet"
    if not train_path.exists():
        print('Missing train split for', name)
        continue
    df = pd.read_parquet(train_path)
    # ensure feats are actual column names in the training split
    if not isinstance(feats, list) or not all(isinstance(fcol, str) for fcol in feats):
        print('Skipping', name, '- features artifact not a list of column names')
        continue
    missing = [c for c in feats if c not in df.columns]
    if missing:
        print('Skipping', name, '- missing feature columns in split:', missing)
        continue
    X = df[feats].copy()
    preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    preproc.fit(X.fillna(0))
    out = ARTIFACT_DIR / f"{name}_preproc.joblib"
    dump(preproc, out)
    print('Fitted and saved preprocessor for', name)

print('Fitting preprocessors complete')
