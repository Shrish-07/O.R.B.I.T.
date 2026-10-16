import json
from pathlib import Path
from datetime import datetime, timezone

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

def _mae_of(exp):
    metrics = exp.get('metrics', {})
    # prefer mae_log_price then mae
    if 'mae_log_price' in metrics:
        return float(metrics['mae_log_price'])
    if 'mae' in metrics:
        return float(metrics['mae'])
    return None


candidates = []
for exp in registry:
    # skip experiments flagged as tainted/suspicious OR stale
    # (stale = recorded against a superseded model/config; previously only
    #  'tainted' was consulted)
    if exp.get('tainted') or exp.get('stale'):
        continue
    # require created_utc and enforce minimum date
    created_str = exp.get('created_utc')
    if not created_str:
        continue
    try:
        created = datetime.fromisoformat(created_str.rstrip('Z'))
    except Exception:
        continue
    if created < MIN_CREATED_UTC:
        continue
    m = _mae_of(exp)
    if m is None:
        continue
    candidates.append((m, exp))

if not candidates:
    print('No comparable experiments found.')
    raise SystemExit(1)

candidates.sort(key=lambda t: (t[0], t[1].get('created_utc', '')))
best_metric, best = candidates[0]

# FIX A3: near-tie governance. A champion decided by a sub-0.001-MAE margin is
# a coin flip (the tree families routinely land within ~0.003 of each other, and
# the champion has flip-flopped between base and political across runs). Among
# candidates within EPSILON of the best, prefer the simpler base-mode spec (one
# fewer feature, no ideology surface); record the rationale in champion.json.
EPSILON_MAE = 0.001
near = [(m, e) for m, e in candidates if m <= best_metric + EPSILON_MAE]
if len(near) > 1:
    base_near = [(m, e) for m, e in near if e.get('mode') == 'base']
    if base_near:
        best_metric, best = base_near[0]
        selection_note = (
            f'TIE-BREAK (prefer simpler base spec): {len(near)} experiments within '
            f'{EPSILON_MAE} MAE of the best ({min(m for m, _ in candidates):.6f}); '
            f'selected {best.get("name")} (mae={best_metric:.6f}).')
    else:
        selection_note = (
            f'NEAR-TIE: {len(near)} experiments within {EPSILON_MAE} MAE; '
            f'selected best MAE {best.get("name")} (mae={best_metric:.6f}).')
else:
    selection_note = (
        f'OUTRIGHT WINNER: {best.get("name")} (mae={best_metric:.6f}); '
        f'no other experiment within {EPSILON_MAE} MAE.')

# Round 11: record the provenance of the REFRESH RUN itself -- current HEAD
# (this selection decision) and the dirty state of the worktree at run time --
# and preserve the winning entry's training-time provenance separately.
try:
    import subprocess
    _git = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
except Exception:
    _git = None
try:
    import subprocess as _sp
    _git_dirty = bool(_sp.check_output(['git', 'status', '--porcelain']).decode().strip())
except Exception:
    _git_dirty = None

out = {
    'selected_experiment': best.get('id'),
    'metric': float(best_metric),
    'selection_note': selection_note,
    'selection_rule': (f'lowest test MAE among untainted, non-stale, post-2026-05-18 runs; '
                       f'near-ties (<{EPSILON_MAE} MAE) resolve to the base-mode spec'),
    'selected_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    'git_commit': _git,
    'git_dirty': _git_dirty,
    'champion_model_git_commit': best.get('git_commit'),
    'champion_model_git_dirty': best.get('git_dirty'),
}
CHAMPION.write_text(json.dumps(out, indent=2))

LOG.parent.mkdir(parents=True, exist_ok=True)
with open(LOG, 'a') as f:
    f.write(f"{datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')} - Champion selected: {out['selected_experiment']} (mae={out['metric']})\n")

print('Champion selected:', out)
