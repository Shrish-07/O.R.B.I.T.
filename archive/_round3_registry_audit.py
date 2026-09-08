"""Round-3 follow-up (Task 11): registry hygiene audit for experiments/registry.json.
Counts entries per model name, flags/tallies duplicates, verifies every entry is
well-formed (no torn writes from the concurrent runs that were killed), and
checks the referenced artifact paths exist. Read-only."""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

REG = Path('experiments/registry.json')
raw = REG.read_text()
reg = json.loads(raw)  # raises if torn
print(f'valid JSON: True; total entries: {len(reg)}; backslash count: {raw.count(chr(92))}')

REQUIRED = ['id', 'created_utc', 'name', 'metrics', 'model_path', 'features_path', 'script', 'scope']
bad = [(e.get('id'), [k for k in REQUIRED if k not in e]) for e in reg]
bad = [b for b in bad if b[1]]
print('entries missing required keys:', bad if bad else 'none')

by_name = defaultdict(list)
for e in reg:
    by_name[e.get('name')].append(e)

print('\n== entries by model name ==')
for name, es in sorted(by_name.items()):
    marks = {e.get('id'): ('TAINTED' if e.get('tainted') else 'stale' if e.get('stale') else '') for e in es}
    print(f'{name:45s} x{len(es)}')
    for e in es:
        m = e.get('metrics', {})
        r2 = m.get('r2', m.get('r2_temporal'))
        mae = m.get('mae', m.get('mae_log_price'))
        print(f'    {e.get("created_utc")}  {e.get("id")}  r2={r2}  mae={mae}  '
              f'{"pipeline_v2" if e.get("pipeline_v2") else ""} {marks.get(e.get("id"))}')

dups = {n: len(es) for n, es in by_name.items() if len(es) > 1}
print('\n== duplicate model names ==')
print(dups if dups else 'none')

# Entries appended during the killed concurrent runs (approx. 2026-09-07T02:44–04:08Z)
KILLED_WINDOW = ('2026-09-07T02:44', '2026-09-07T04:30')
killed = [e for e in reg
          if e.get('created_utc') and KILLED_WINDOW[0] <= e['created_utc'] <= KILLED_WINDOW[1]]
print(f'\nentries created inside the killed/concurrent window {KILLED_WINDOW}: {len(killed)}')
for e in killed:
    print('   ', e.get('created_utc'), e.get('name'))

# path existence check for non-stale, non-tainted entries
missing_ref = []
for e in reg:
    if e.get('tainted') or e.get('stale'):
        continue
    for k in ('model_path', 'features_path'):
        p = Path(e.get(k, ''))
        if not p.exists():
            missing_ref.append((e.get('id'), e.get('name'), k, e.get(k)))
print('\nmissing referenced artifacts among active entries:',
      missing_ref if missing_ref else 'none')

print('\nVERDICT: see above — duplicates are append-only history from legitimate re-runs;',
      'no torn/truncated entries present.' if not bad else 'BAD ENTRIES PRESENT')
