from pathlib import Path
import pandas as pd

DOCS = Path('docs')
DOCS.mkdir(exist_ok=True)

# DATA_LINEAGE.md
with open(DOCS / 'DATA_LINEAGE.md', 'w') as f:
    f.write('# Data Lineage\n\n')
    f.write('Canonical raw snapshots:\n')
    for p in sorted(Path('data/raw').glob('**/*')):
        if p.is_file():
            f.write(f'- {p}\n')
    f.write('\nProcessed datasets:\n')
    for p in sorted(Path('data/processed').glob('**/*')):
        if p.is_file():
            f.write(f'- {p}\n')

# FEATURE_DICTIONARY.md
with open(DOCS / 'FEATURE_DICTIONARY.md', 'w') as f:
    f.write('# Feature Dictionary\n\n')
    sample = None
    try:
        sample = pd.read_parquet('data/processed/modeling_dataset_fe_imputed.parquet', columns=None)
    except Exception:
        pass
    if sample is not None:
        f.write('Column | dtype | n_unique\n')
        f.write('--- | --- | ---\n')
        for c in sample.columns:
            try:
                n = sample[c].nunique()
            except Exception:
                n = ''
            f.write(f'{c} | {sample[c].dtype} | {n}\n')
    else:
        f.write('No processed modeling dataset found to enumerate features.\n')

# ARCHITECTURE.md
with open(DOCS / 'ARCHITECTURE.md', 'w') as f:
    f.write('# Project Architecture\n\n')
    f.write('- src/: ETL, feature engineering, training, experiment management, and app\n')
    f.write('- data/raw/: raw snapshots ingested from sources\n')
    f.write('- data/processed/: cleaned & feature-engineered datasets\n')
    f.write('- models/: trained model artifacts and preprocessing joblibs\n')
    f.write('- experiments/: run registry and champion selection\n')
    f.write('- app/: Streamlit UI\n')

print('Docs generated in docs/')
