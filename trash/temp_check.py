import json, pandas as pd
from pathlib import Path

print('=== CHAMPION ===')
c = json.load(open('experiments/champion.json'))
print('ID:', c['selected_experiment'])
print('MAE:', c['metric'])

print()
print('=== MODELS ===')
base_f = json.load(open('models/artifacts/lgbm_all_years_base_features.json'))
pol_f = json.load(open('models/artifacts/lgbm_all_years_political_features.json'))
base_m = json.load(open('models/artifacts/lgbm_all_years_base_metrics.json'))
pol_m = json.load(open('models/artifacts/lgbm_all_years_political_metrics.json'))
print('Base features:', len(base_f), '| dem_share:', any('dem' in f.lower() for f in base_f))
print('Political features:', len(pol_f), '| dem_share:', any('dem' in f.lower() for f in pol_f))
print('assesstot in either:', 'assesstot' in base_f or 'assesstot' in pol_f)
print('Base MAE:', base_m.get('mae_log_price', base_m.get('mae')))
print('Political MAE:', pol_m.get('mae_log_price', pol_m.get('mae')))

print()
print('=== IDEOLOGY ===')
ideo = pd.read_parquet('data/processed/ideology_by_council.parquet')
print('Rows:', len(ideo), '| Years:', sorted(ideo['election_year'].unique()))

print()
print('=== CANONICAL ===')
canon = pd.read_parquet('data/canonical/modeling_dataset_canonical_v2.parquet')
print('Rows:', len(canon), '| dem_share coverage:', canon['dem_share'].notna().mean())

print()
print('=== SCENARIOS ===')
df = pd.read_csv('results/political_scenarios/liberal_policy_by_council.csv')
print('Max pct_change:', df['mean_pct_change'].abs().max())
print('Districts with non-zero effect:', (df['mean_pct_change'].abs() > 0.01).sum())

print()
print('=== FIGURES ===')
for f in Path('docs/figures').glob('*.png'):
    print(f.name, f.stat().st_size, 'bytes')

print()
print('=== APP COMPILE ===')
import subprocess, sys
result = subprocess.run([sys.executable, '-m', 'py_compile', 'app/app.py'], capture_output=True, text=True)
print('py_compile:', 'PASS' if result.returncode == 0 else 'FAIL: ' + result.stderr)
