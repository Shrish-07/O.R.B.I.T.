import json
from pathlib import Path
from datetime import datetime

REGISTRY = Path('experiments/registry.json')
CHAMPION = Path('experiments/champion.json')
LOG = Path('logs/actions.log')

if not REGISTRY.exists():
    print('No registry found. Exiting.')
    raise SystemExit(1)

registry = json.loads(REGISTRY.read_text())

# Choose champion only from experiments created on or after this minimum date.
# Rationale: older experiments include models trained on leaky/pre-rebuild data.
# Using a hard date threshold (2026-05-18) ensures we only consider pipeline_v2 runs.
MIN_CREATED_UTC = datetime(2026, 5, 18)

best = None
best_metric = None
for exp in registry:
    # skip any experiments flagged as tainted/suspicious
    if exp.get('tainted'):
        continue
    # require created_utc and enforce minimum date
    created_str = exp.get('created_utc')
    if not created_str:
        # skip experiments without a creation timestamp
        continue
    try:
        # handle ISO timestamps that may end with 'Z'
        created = datetime.fromisoformat(created_str.rstrip('Z'))
    except Exception:
        continue
    if created < MIN_CREATED_UTC:
        # skip older experiments (pre-rebuild / possibly leaky)
        continue

    metrics = exp.get('metrics', {})
    # prefer mae_log_price then mae
    if 'mae_log_price' in metrics:
        m = metrics['mae_log_price']
    elif 'mae' in metrics:
        m = metrics['mae']
    else:
        continue
    if best is None or m < best_metric:
        best = exp
        best_metric = m

if best is None:
    print('No comparable experiments found.')
    raise SystemExit(1)

out = {
    'selected_experiment': best.get('id'),
    'metric': float(best_metric),
    'selected_utc': datetime.utcnow().isoformat() + 'Z',
    'git_commit': best.get('git_commit')
}
CHAMPION.write_text(json.dumps(out, indent=2))

LOG.parent.mkdir(parents=True, exist_ok=True)
with open(LOG, 'a') as f:
    f.write(f"{datetime.utcnow().isoformat()}Z - Champion selected: {out['selected_experiment']} (mae={out['metric']})\n")

print('Champion selected:', out)
