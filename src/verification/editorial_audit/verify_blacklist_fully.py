"""
Verify feature-blacklist enforcement for EVERY eligible registry model.

(Round 11 rewrite) The old version: treated the champion as a LightGBM .txt
("MISSING" when the champion became a Random Forest), checked only the two
champion slots, and printed Round-2-era text ("17/18-feature era", "EASE-MENT
is a report-generation artifact"). Now:

  - source of truth = the SAME eligible set every other stage uses
    (src/ops/champion_ranking.get_all_models -- untainted, non-stale,
    post-2026-05-18 registry entries);
  - every eligible model's trained feature list is read from
    models/artifacts/<name>_features.json -- readable for ALL families, no
    LightGBM-specific parsing;
  - each list is checked against config/feature_blacklist.yaml
    (raw and underscore->space normalized);
  - a missing feature-list artifact or ANY hit -> gate FAILS (exit 1).

Output: results/editorial_audit/blacklist_audit.json
        + results/verification/verify_blacklist_fully.json (Round 11 status).

Run: python src/verification/editorial_audit/verify_blacklist_fully.py
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src' / 'verification'))
sys.path.insert(0, str(ROOT))
from _shared import record_status, check_val  # noqa: E402
from src.ops.champion_ranking import get_all_models  # noqa: E402

RESULTS_DIR = ROOT / "results" / "editorial_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

blacklist_path = ROOT / "config" / "feature_blacklist.yaml"
with open(blacklist_path) as f:
    blacklist_cfg = yaml.safe_load(f)
blacklisted = set()
for cat, items in blacklist_cfg.items():
    for item in items:
        blacklisted.add(str(item).strip())
print(f"Loaded {len(blacklisted)} blacklisted features")


def norm_label(name):
    """Normalize a feature name for blacklist comparison: model files store
    underscored names; the blacklist YAML uses spaces."""
    return str(name).replace("_", " ").strip()


checks = []
per_model = {}
for ent in get_all_models(write_audit=False):
    name = ent['name']
    rel = f'models/artifacts/{name}_features.json'
    fs = ROOT / rel
    if not fs.exists():
        per_model[name] = {'status': 'MISSING feature-list artifact',
                           'features_path': rel}
        checks.append(check_val(f'{name}.features_json_exists', True, False))
        print(f"  {name}: MISSING {rel} -- gate FAILS (refuse a vacuous pass)")
        continue
    feats = json.loads(fs.read_text())
    bad = [f for f in feats
           if str(f) in blacklisted or norm_label(f) in blacklisted]
    per_model[name] = {
        'features_path': rel,
        'n_features': len(feats),
        'blacklist_violations': [str(f) for f in bad],
        'blacklist_clean': not bad,
        'rank': ent.get('rank'), 'family': ent.get('family'),
        'mode': ent.get('mode'), 'test_mae': ent.get('test_mae'),
    }
    checks.append(check_val(f'{name}.blacklist_clean', True, not bad))
    print(f"  {name} (rank {ent.get('rank')}, {ent.get('family')}): "
          f"{len(feats)} features, violations: {bad if bad else 'none'}")

all_ok = all(c['match'] for c in checks)

audit = {
    'blacklist_path': 'config/feature_blacklist.yaml',
    'blacklisted_feature_count': len(blacklisted),
    'n_eligible_models_checked': len(per_model),
    'per_model': per_model,
    'enforcement_passed': bool(all_ok),
}
AUDIT_PATH = RESULTS_DIR / 'blacklist_audit.json'
with open(AUDIT_PATH, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)
print(f"Wrote {AUDIT_PATH}")

record_status('verify_blacklist_fully', checks,
              note='Every eligible registry model models/artifacts/'
                   '<name>_features.json vs config/feature_blacklist.yaml; '
                   'missing artifact or any hit fails.')

if not all_ok:
    print('BLACKLIST ENFORCEMENT FAILURE: at least one eligible model '
          'carries a blacklisted feature or has no feature-list artifact.')
    raise SystemExit(1)
print(f'Blacklist enforcement: PASS for all {len(per_model)} '
      'eligible models.')
