import pandas as pd
from pathlib import Path
p = Path('data/splits/all_years_test.parquet')
if not p.exists():
    print('test split not found')
else:
    df = pd.read_parquet(p)
    print('test shape:', df.shape)
    print('index name:', df.index.name)
    print('index dtype:', df.index.dtype)
    try:
        print('index sample:', list(df.index[:5]))
    except Exception as e:
        print('index sample error:', e)
    print('BBL column present:', 'BBL' in df.columns)
