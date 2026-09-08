import json
from pathlib import Path
reg = json.load(open('experiments/registry.json'))
seen = {}
for e in reg:
    name = e.get('name')
    if name not in seen:
        seen[name] = e
    else:
        existing = seen[name]
        prefer_new = e.get('pipeline_v2') and not existing.get('pipeline_v2')
        newer = e.get('created_utc','') > existing.get('created_utc','') and not existing.get('pipeline_v2')
        if prefer_new or newer:
            seen[name] = e
deduped = list(seen.values())
Path('experiments/registry.json').write_text(json.dumps(deduped, indent=2))
print(f'Registry: {len(reg)} -> {len(deduped)} entries')
