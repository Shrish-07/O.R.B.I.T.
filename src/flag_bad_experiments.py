import json
from pathlib import Path

REG = Path('experiments/registry.json')
if not REG.exists():
    print('No registry found')
    raise SystemExit(0)

registry = json.loads(REG.read_text())
changed = False
for exp in registry:
    metrics = exp.get('metrics', {})
    mae = None
    if 'mae_log_price' in metrics:
        mae = metrics['mae_log_price']
    elif 'mae' in metrics:
        mae = metrics['mae']
    if mae is not None:
        try:
            if float(mae) < 0.1:
                exp['tainted'] = True
                changed = True
        except Exception:
            pass

if changed:
    REG.write_text(json.dumps(registry, indent=2))
    print('Flagged suspicious experiments')
else:
    print('No suspicious experiments found')
