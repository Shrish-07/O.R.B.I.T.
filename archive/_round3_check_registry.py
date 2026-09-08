"""Throwaway Round-3 (Sep 2026) check: registry JSON validity + no backslashes."""
import json
from pathlib import Path

raw = Path('experiments/registry.json').read_text()
reg = json.loads(raw)
print('entries:', len(reg))
print('backslashes in file:', raw.count('\\'))
for e in reg[-2:]:
    print(e['name'], '|', e['model_path'], '|', e['features_path'],
          '| r2=', e['metrics']['r2'], 'mae=', e['metrics']['mae'])
for f in ['experiments/results/ridge_all_years_political.json',
          'experiments/results/elasticnet_all_years_political.json']:
    d = json.load(open(f))
    print(f, 'r2=', d.get('r2'), 'mae=', d.get('mae'), 'n_features=', d.get('n_features'))
