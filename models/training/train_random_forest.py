import sys

# Allow emoji/unicode prints when this script is run directly on a cp1252 Windows
# console. When launched via main.py this is already handled (main.py sets
# PYTHONUTF8/PYTHONIOENCODING for stages).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

"""RandomForest trainer -- Round 10 rewrite: complexity-constrained,
validation-tuned, SHAP-tractable.

Why this replaced the unpruned trainer
--------------------------------------
The previous trainer grew 200 fully unpruned trees (max_depth=None), producing
a ~990 MB model on which exact TreeSHAP could not finish inside the shap_tiers
90-minute per-tier budget (Round 9, 2026-09-28: tier (a) and tier (c) both hit
the hard 5,400 s timeout at ~99% CPU; the tier-(b) depth-25 stand-in was
rejected on MAE fidelity, so the forest only ever received gain importance --
method='feature_importance_fallback'). A model family whose importances can
only be reported as gain cannot appear in the paper's SHAP tables on equal
footing, and the 180+ minutes of burned SHAP budget dominated every full run.

Methodology (pre-specified, defensible in a paper)
--------------------------------------------------
(a) The test set is NEVER used to choose hyperparameters. It is scored exactly
    once, after selection, on the refitted winner.
(b) Selection uses the SAME time-ordered validation slice of the training years
    that models/training/train_lgbm.py uses (its FIX A1): sort the training
    split by sale_date, last 15% = validation, earlier 85% = fit.
(c) A pre-specified tractability cap (MAX_MEAN_LEAVES below) disqualifies any
    config whose fitted forest averages more leaves per tree than the cap,
    BEFORE its validation MAE is looked at.
(d) Small explicit grid: max_depth x min_samples_leaf x max_features below,
    n_estimators=200, random_state=42, n_jobs=-1, fillna(0) inputs exactly as
    this trainer has always used.
(e) Best validation MAE among cap-satisfying configs wins; ties resolve to
    fewer mean leaves. The winner is REFIT on the full training split, then
    the test set is scored once.
(f) The full grid (config, mean leaves, validation MAE, eligible /
    disqualified-by-cap) is persisted to results/rf_tuning_grid.json, and the
    chosen config / cap / selection rule are recorded in the registry entry
    and the experiments/results JSON.
If no config satisfies the cap, the grid widens toward smaller trees
(min_samples_leaf 50, 100) rather than relaxing the cap.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------------------------
# Pre-specified SHAP-tractability cap.
#
# Exact TreeSHAP on a sklearn forest scales with tree size; the unpruned
# 200-tree forest (max_depth=None, ~990 MB) exceeded the shap_tiers 90-minute
# per-tier budget in Round 9 without ever finishing, so it could only be
# reported with gain importance. Any config whose fitted forest averages more
# than MAX_MEAN_LEAVES leaves per tree (sum(t.get_n_leaves() for t in
# model.estimators_)/len(model.estimators_)) is DISQUALIFIED before its
# validation MAE is even compared. The cap is fixed BEFORE training; if no
# config satisfies it, the grid widens toward smaller trees, never the other
# way.
# ---------------------------------------------------------------------------
MAX_MEAN_LEAVES = 3000

GRID_MAX_DEPTH = [10, 14, 18, 25]
GRID_MIN_SAMPLES_LEAF = [1, 5, 20]
GRID_MAX_FEATURES = [1.0, 0.5, 0.33]
# Fallback stage if (and only if) NO config above satisfies the cap: widen
# toward smaller leaves rather than relaxing the cap.
WIDENED_MIN_SAMPLES_LEAF = [50, 100]

VALIDATION_FRACTION = 0.15  # identical to train_lgbm.py FIX A1

RETIRED_RF_NAMES = ('rf_all_years_political_n200',)

parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['all_years', 'year2017'], required=True)
parser.add_argument('--mode', choices=['base', 'political'], required=True)
parser.add_argument('--n_estimators', type=int, default=200)
args = parser.parse_args()

SPLIT_DIR = Path('data/splits')
MODEL_DIR = Path('models')
ARTIFACT_DIR = Path('models/artifacts')
RESULTS_DIR = Path('experiments/results')
REGISTRY = Path('experiments/registry.json')
GRID_OUT = Path('results/rf_tuning_grid.json')

MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRID_OUT.parent.mkdir(parents=True, exist_ok=True)

TARGET = 'target_log_price'

TRAIN_PATH = SPLIT_DIR / f"{args.variant}_train.parquet"
TEST_PATH = SPLIT_DIR / f"{args.variant}_test.parquet"

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# load leakage blacklist
blacklist_path = Path('config/feature_blacklist.yaml')
if blacklist_path.exists():
    with open(blacklist_path) as f:
        blacklist = {c.lower() for c in yaml.safe_load(f).get('leakage_features', [])}
else:
    blacklist = set()

# select numeric features while excluding blacklist and target (unchanged from
# the pre-Round-10 trainer -- same feature list, same fillna(0) convention)
features = [c for c in train.select_dtypes(include='number').columns
            if c != TARGET and c.lower() not in blacklist]

if args.mode == 'political':
    ideology_features = [c for c in ['dem_share', 'rep_share', 'turnout'] if c in train.columns]
    features += [c for c in ideology_features if c not in features]
features = list(dict.fromkeys(features))

X_train = train[features].fillna(0)
y_train = train[TARGET]
X_test = test[features].fillna(0)
y_test = test[TARGET]

# ---------------------------------------------------------------------------
# Time-ordered validation slice -- mirrors train_lgbm.py FIX A1 exactly, so
# the RF selection protocol is IDENTICAL to the LightGBM trainers'. The test
# set is not touched until the final, single scoring below.
# ---------------------------------------------------------------------------
if 'sale_date' in train.columns:
    _order = train.sort_values('sale_date').index
    _cutoff = int(len(train) * 0.85)
    _fit_idx, _val_idx = _order[:_cutoff], _order[_cutoff:]
else:
    _cutoff = int(len(train) * 0.85)
    _fit_idx, _val_idx = train.index[:_cutoff], train.index[_cutoff:]
X_fit, y_fit = X_train.loc[_fit_idx], y_train.loc[_fit_idx]
X_val, y_val = X_train.loc[_val_idx], y_train.loc[_val_idx]
print(f'Selection on time-ordered train slice: fit={len(X_fit):,} rows (earlier '
      f'sales), val={len(X_val):,} rows (most recent in-support sales); '
      f'test untouched until scoring')


def mean_leaves_of(model):
    return float(sum(t.get_n_leaves() for t in model.estimators_)
                 / len(model.estimators_))


def run_grid(min_samples_leaf_values, stage_label):
    """Fit every (depth, min_samples_leaf, max_features) config on the
    fit-only portion; return the full result rows. Cap eligibility is
    computed here and applied BEFORE any MAE comparison downstream."""
    rows = []
    for depth in GRID_MAX_DEPTH:
        for msl in min_samples_leaf_values:
            for mf in GRID_MAX_FEATURES:
                cfg = dict(max_depth=depth, min_samples_leaf=msl,
                           max_features=mf, n_estimators=args.n_estimators)
                model = RandomForestRegressor(random_state=42, n_jobs=-1, **cfg)
                model.fit(X_fit, y_fit)
                ml = mean_leaves_of(model)
                val_mae = mean_absolute_error(y_val, model.predict(X_val))
                eligible = ml <= MAX_MEAN_LEAVES
                rows.append({
                    'stage': stage_label,
                    'config': cfg,
                    'mean_leaves_per_tree': ml,
                    'validation_mae': float(val_mae),
                    'eligible': bool(eligible),
                    'status': ('eligible' if eligible else
                               'disqualified_by_cap (mean leaves '
                               f'{ml:.0f} > {MAX_MEAN_LEAVES})'),
                })
                print(f'  [{stage_label}] depth={depth} min_leaf={msl} '
                      f'max_features={mf} -> mean_leaves={ml:.0f} '
                      f'val_mae={val_mae:.6f} '
                      f'{"ELIGIBLE" if eligible else "DISQUALIFIED (cap)"}')
    return rows


print(f'Grid: {len(GRID_MAX_DEPTH)} x {len(GRID_MIN_SAMPLES_LEAF)} x '
      f'{len(GRID_MAX_FEATURES)} = '
      f'{len(GRID_MAX_DEPTH) * len(GRID_MIN_SAMPLES_LEAF) * len(GRID_MAX_FEATURES)} '
      f'configs, n_estimators={args.n_estimators}, cap={MAX_MEAN_LEAVES} '
      f'mean leaves/tree')
grid_rows = run_grid(GRID_MIN_SAMPLES_LEAF, 'base_grid')

if not any(r['eligible'] for r in grid_rows):
    print('\n*** NO config in the pre-specified grid satisfied the '
          f'{MAX_MEAN_LEAVES}-mean-leaves cap. Widening toward smaller trees '
          f'(min_samples_leaf {WIDENED_MIN_SAMPLES_LEAF}) -- the cap itself is '
          'NOT relaxed. ***\n')
    grid_rows += run_grid(WIDENED_MIN_SAMPLES_LEAF, 'widened_grid')

eligible_rows = [r for r in grid_rows if r['eligible']]
if not eligible_rows:
    raise SystemExit(
        f'ERROR: no grid config (base or widened) satisfied the '
        f'{MAX_MEAN_LEAVES}-mean-leaves tractability cap; refusing to relax '
        'the cap. Report this -- the grid needs manual revision.')

# Selection rule: lowest validation MAE among cap-satisfying configs; ties ->
# fewer mean leaves. Test set was not consulted.
eligible_rows.sort(key=lambda r: (r['validation_mae'], r['mean_leaves_per_tree']))
best_row = eligible_rows[0]
best_cfg = best_row['config']
print(f"\nSelected config (validation data only): {best_cfg} "
      f"(val_mae={best_row['validation_mae']:.6f}, "
      f"mean_leaves={best_row['mean_leaves_per_tree']:.0f})")

# ---------------------------------------------------------------------------
# Refit the winner on the FULL training split (exactly the way this trainer
# always fit), then score the test set ONCE.
# ---------------------------------------------------------------------------
name = f"rf_{args.variant}_{args.mode}_tuned"
model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_cfg)
model.fit(X_train, y_train)

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
final_mean_leaves = mean_leaves_of(model)

SELECTION_RULE = ('lowest validation MAE among cap-satisfying configs '
                  f'(mean leaves per tree <= {MAX_MEAN_LEAVES}); ties resolve '
                  'to fewer mean leaves; test set scored once on the refit '
                  'winner. Validation slice identical to train_lgbm.py FIX A1 '
                  '(time-ordered, last 15% of the training years).')

results = {
    'model': 'RandomForest',
    'name': name,
    'variant': args.variant,
    'mode': args.mode,
    'n_train': int(len(train)),
    'n_test': int(len(test)),
    'n_features': int(len(features)),
    'r2': float(r2),
    'mae': float(mae),
    'tuning': {
        'mean_leaves_cap': MAX_MEAN_LEAVES,
        'selection_rule': SELECTION_RULE,
        'chosen_config': best_cfg,
        'chosen_mean_leaves_per_tree': final_mean_leaves,
        'validation_mae_of_chosen_config': best_row['validation_mae'],
        'grid_results_file': GRID_OUT.as_posix(),
        'widened_grid_used': any(r['stage'] == 'widened_grid' for r in grid_rows),
    },
}

with open(RESULTS_DIR / f"{name}.json", 'w') as f:
    json.dump(results, f, indent=2)

model_path = MODEL_DIR / f"{name}.joblib"
dump(model, model_path)

with open(ARTIFACT_DIR / f"{name}_features.json", 'w') as f:
    json.dump(features, f, indent=2)
with open(ARTIFACT_DIR / f"{name}_metrics.json", 'w') as f:
    json.dump({'r2': r2, 'mae': mae}, f, indent=2)

# ---------------------------------------------------------------------------
# Full grid record: every config, its mean leaves, its validation MAE, and
# whether it was eligible or disqualified by the cap. This file is the audit
# trail for the paper's complexity-constrained RF.
# ---------------------------------------------------------------------------
grid_payload = {
    'generated_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    'variant': args.variant,
    'mode': args.mode,
    'n_estimators': args.n_estimators,
    'mean_leaves_cap': MAX_MEAN_LEAVES,
    'cap_rationale': ('pre-specified SHAP-tractability cap; the unpruned '
                      'Round-9 forest exceeded the shap_tiers 90-minute '
                      'per-tier TreeSHAP budget, so configs above this cap '
                      'are disqualified before validation MAE is compared'),
    'selection_rule': SELECTION_RULE,
    'validation_protocol': ('time-ordered slice of the TRAINING years only '
                            '(sort by sale_date, last 15% = validation, '
                            'earlier 85% = fit) -- identical to '
                            'models/training/train_lgbm.py FIX A1; the test '
                            'set was scored exactly once, after selection'),
    'test_used_for_selection': False,
    'chosen_config': best_cfg,
    'chosen_mean_leaves_per_tree_final_refit': final_mean_leaves,
    'chosen_config_validation_mae': best_row['validation_mae'],
    'final_test_mae': float(mae),
    'final_test_r2': float(r2),
    'n_eligible': len(eligible_rows),
    'n_disqualified_by_cap': len(grid_rows) - len(eligible_rows),
    'grid': grid_rows,
}
with open(GRID_OUT, 'w') as f:
    json.dump(grid_payload, f, indent=2)

print('RandomForest (tuned) training complete:', results['r2'], results['mae'])
print('\n\U0001F4E6 Artifacts written:')
print('Model:   ', model_path)
print('Metrics: ', ARTIFACT_DIR / f"{name}_metrics.json")
print('Features:', ARTIFACT_DIR / f"{name}_features.json")
print('Results: ', RESULTS_DIR / f"{name}.json")
print('Grid:    ', GRID_OUT)

# register in experiments/registry.json
try:
    git = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
except Exception:
    git = None
try:
    git_dirty = bool(subprocess.check_output(
        ['git', 'status', '--porcelain']).decode().strip())
except Exception:
    git_dirty = None

exp = {
    'id': f"exp-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    'created_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    'git_commit': git,
    'git_dirty': git_dirty,
    'script': 'models/training/train_random_forest.py',
    'name': name,
    'mode': args.mode,
    'scope': '2017' if args.variant == 'year2017' else 'all',
    'hypothesis': ('Complexity-constrained, validation-tuned RandomForest '
                   '(SHAP-tractable replacement for the unpruned forest)'),
    'reasoning': ('Round 10: the unpruned forest was intractable for exact '
                  'TreeSHAP; this entry is selected on validation data only '
                  'under a pre-specified mean-leaves cap'),
    'metrics': {'r2': float(r2), 'mae': float(mae)},
    'results': results,
    # .as_posix(): str(Path) on Windows emits backslashes, which silently
    # breaks registry.json portability (this bug recurred twice -- see
    # VERIFICATION_REPORT.md Rounds 1-2). as_posix() is the structural fix.
    'features_path': (ARTIFACT_DIR / f"{name}_features.json").as_posix(),
    'model_path': model_path.as_posix(),
    'tuning': results['tuning'],
}

if REGISTRY.exists():
    registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
else:
    registry = []
# Replace, don't duplicate: a re-run of this exact script+config produces the
# identical (script, name) identity, so drop the prior entry instead of
# accumulating stale duplicate rows in the registry.
registry = [e for e in registry
             if not (e.get('script') == exp['script'] and e.get('name') == exp['name'])]

# Retire the old unpruned forest (Round 10): it must never re-enter champion
# selection or downstream analysis. The eligibility filters in
# src/ops/experiment_runner.py and src/ops/champion_ranking.py both skip
# entries flagged stale; this makes the retirement durable across re-runs
# even if registry.json is restored from git history.
for e in registry:
    if e.get('name') in RETIRED_RF_NAMES and not e.get('stale'):
        e['stale'] = True
        e['stale_reason'] = (
            'Retired in Round 10: unpruned forest (max_depth=None, ~990 MB) -- '
            'exact TreeSHAP intractable (tiers a/c exceeded the 90-minute '
            'per-tier budget on 2026-09-28; the tier-b depth-25 variant was '
            'rejected on MAE fidelity), so it only ever received gain '
            'importance. Replaced by rf_all_years_political_tuned, selected '
            'on validation data only under the pre-specified 3000-mean-leaves '
            'tractability cap. Kept for audit history; excluded from champion '
            'selection and all downstream analysis via the existing stale '
            'filter.')
        print('Marked retired entry stale:', e.get('name'))

registry.append(exp)
REGISTRY.write_text(json.dumps(registry, indent=2), encoding='utf-8')

print('Registered experiment:', exp['id'])

# Keep champion.json pointing at a live entry: replacing an entry above could
# otherwise orphan the previous champion id until the next full pipeline run.
subprocess.run([sys.executable, str(Path('src/ops/experiment_runner.py'))], check=False)
