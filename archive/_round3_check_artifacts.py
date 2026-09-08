"""Round-3 (Sep 2026) artifact audit: PDP + sensitivity JSONs must reference
only features that are (a) in the current canonical schema and (b) NOT in
config/feature_blacklist.yaml. This is the exact check skipped when the stale
pre-blacklist artifacts slipped through."""
import glob
import json
import os
from pathlib import Path

import yaml

ART = Path('models/artifacts')
blacklist = {c.lower() for c in yaml.safe_load(open('config/feature_blacklist.yaml'))['leakage_features']}
schema = set(json.load(open(ART / 'canonical_schema.json'))['columns'])

files = sorted(glob.glob(str(ART / '*_pdp_summary.json'))) + \
        sorted(glob.glob(str(ART / '*_sensitivity*.json')))
ok_all = True
for f in files:
    data = json.load(open(f))
    feats = sorted({r.get('feature') for r in (data if isinstance(data, list)
                                               else data.get('summary', []))
                    if isinstance(r, dict)})
    blacklisted = [f2 for f2 in feats if f2.lower() in blacklist]
    not_in_schema = [f2 for f2 in feats if f2 not in schema]
    ok = not blacklisted and not not_in_schema
    ok_all &= ok
    print(f"{'OK ' if ok else 'BAD'}  {os.path.basename(f):58s} features={feats} "
          f"blacklisted={blacklisted} not_in_schema={not_in_schema}")

pngs = sorted(glob.glob(str(ART / 'pdp_*.png')))
from PIL import Image  # noqa
for p in pngs:
    im = Image.open(p)
    print('PNG ok:', os.path.basename(p), im.size)
print('ALL ARTIFACTS BLACKLIST-CLEAN:', ok_all)
