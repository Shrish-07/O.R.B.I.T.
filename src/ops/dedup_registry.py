# src/ops/dedup_registry.py
# One-command registry hygiene: collapse duplicate experiment entries that
# share the same identity (script, name) — i.e. re-runs of the same trainer
# with the same config — keeping only the NEWEST one per identity.
#
# The training scripts themselves now do this replace-on-register on every
# run (see the "Replace, don't duplicate" blocks in models/training/), so this
# is only needed for legacy rows written by older versions, or after editing
# the registry by hand.
#
# NOTE on semantics (changed 2026-09-08): this used to *prefer* entries
# flagged "pipeline_v2" over newer ones, which — once train_lgbm.py started
# registering fresh entries — would have deleted the fresh entries and kept
# stale ones. It now simply keeps the newest entry per identity. If the
# previous champion's entry is removed, champion.json is NOT auto-updated
# here; run src/ops/experiment_runner.py afterwards (this script prints a
# reminder).
import json
from pathlib import Path

REGISTRY = Path('experiments/registry.json')
CHAMPION = Path('experiments/champion.json')

registry = json.loads(REGISTRY.read_text())

def _key(e):
    script, name = e.get('script'), e.get('name')
    if script or name:
        return (script, name)
    # Entry has no identifiable script/name — key on its own id so it is
    # never collapsed into an unrelated entry.
    return ('__unique__', e.get('id'))


# Keep the newest entry per (script, name) identity.
seen = {}  # key -> entry
order = []
for e in registry:
    key = _key(e)
    if key not in seen:
        seen[key] = e
        order.append(key)
    elif (e.get('created_utc', '') > seen[key].get('created_utc', '')):
        seen[key] = e

deduped = [seen[k] for k in order]

removed = [e['id'] for e in registry if e not in deduped]
REGISTRY.write_text(json.dumps(deduped, indent=2))
print(f'Registry: {len(registry)} -> {len(deduped)} entries')
print('Removed:', ', '.join(removed) if removed else '(nothing)')

champ = json.loads(CHAMPION.read_text())
if champ.get('selected_experiment') in removed:
    print('NOTE: the current champion entry was removed. Run '
          'src/ops/experiment_runner.py to re-select the champion.')

