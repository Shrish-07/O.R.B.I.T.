import importlib
import sys
import os

# ensure repo root is on sys.path
sys.path.insert(0, os.getcwd())

FILES = [
    'src/data_loader.py',
    'src/feature_engineering.py',
    'models/training/train_lgbm.py',
    'src/auth.py',
    'app/app.py'
]

missing = [f for f in FILES if not os.path.exists(f)]
if missing:
    print('Missing expected files:', missing)
    sys.exit(2)
else:
    print('Smoke check OK — files present')
