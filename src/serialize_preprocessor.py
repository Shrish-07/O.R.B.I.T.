from pathlib import Path
import json
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump

ARTIFACT_DIR = Path('models/artifacts')
DATA_DIR = Path('data/processed')

features_files = list(ARTIFACT_DIR.glob('*_features.json'))
if not features_files:
    print('No feature artifacts found')
    raise SystemExit(0)

for f in features_files:
    name = f.stem.replace('_features','')
    feats = json.loads(f.read_text())
    preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    # save pipeline (not fitted) so it can be fit during training or fitted to canonical dataset later
    out = ARTIFACT_DIR / f"{name}_preproc.joblib"
    dump(preproc, out)
    print('Wrote preprocessor for', name)

print('Preprocessor serialization complete')
