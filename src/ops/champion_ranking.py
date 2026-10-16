"""Ranked interpretation-model resolution -- the single shared place every
downstream analysis stage imports so 'which models are we interpreting?' has
exactly one answer per run.

Why this exists (Round 9)
-------------------------
The interpreted-model passes (SHAP, PDP, confidence intervals, sensitivity,
counterfactual scenarios and their verifier) previously covered 'the
champion' (whatever experiments/champion.json selects) plus the two LightGBM
paper models, and political_scenarios.py additionally hard-wired
lgbm_all_years_political for its mixed_governance scenario. This module
generalizes that to full ranked registry coverage:

  rank 1  = exactly what experiments/champion.json already designates. That
            file's selection logic (src/ops/experiment_runner.py) is NOT
            re-litigated or duplicated here; this module reads the pointer
            and validates it refers to an ELIGIBLE registry entry.
  rank 2+ = the remaining eligible registry entries in ascending test-MAE
            order, recomputed FRESH from experiments/registry.json on every
            call -- never a hard-coded model name.

The eligibility filter is identical to src/ops/experiment_runner.py
(refresh_champion): untainted, non-stale, created_utc >= 2026-05-18. No
ranked slot can therefore be a disqualified run that happens to carry a low
MAE.

Audit trail: every get_* call with write_audit=True rewrites
results/rank1_rank2_selection.json with the full eligible ranking -- every
model, its MAE/R2/family/mode/paths. Idempotent (same registry -> same
content, fresh selected_utc).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = _REPO_ROOT / 'experiments' / 'registry.json'
_CHAMPION = _REPO_ROOT / 'experiments' / 'champion.json'
_AUDIT_OUT = _REPO_ROOT / 'results' / 'rank1_rank2_selection.json'

# Identical to src/ops/experiment_runner.py: runs older than this include
# models trained on leaky / pre-rebuild data and are excluded from selection.
MIN_CREATED_UTC = datetime(2026, 5, 18)

# The two LightGBM paper models. NOT used to pick ranks -- exported only so
# stages can ask whether a model is one of the paper-comparison pair (their
# analyses have paper-continuity constraints of their own).
PAPER_MODEL_NAMES = ('lgbm_all_years_political', 'lgbm_all_years_base')

# File suffix -> model family (mirrors src/ops/model_io.load_model).
_SUFFIX_FAMILY = {'.txt': 'lightgbm', '.cbm': 'catboost',
                  '.model': 'xgboost', '.joblib': 'sklearn'}


def _mae_of(exp):
    metrics = exp.get('metrics') or {}
    if metrics.get('mae_log_price') is not None:
        return float(metrics['mae_log_price'])
    if metrics.get('mae') is not None:
        return float(metrics['mae'])
    return None


def _r2_of(exp):
    metrics = exp.get('metrics') or {}
    for key in ('r2_log_price', 'r2_temporal', 'r2'):
        if metrics.get(key) is not None:
            return float(metrics[key])
    return None


def eligible_experiments():
    """All registry entries passing the refresh_champion eligibility filter
    (untainted, non-stale, created_utc >= 2026-05-18, has MAE), sorted by
    (test MAE, created_utc). Returns the raw registry dicts."""
    if not _REGISTRY.exists():
        raise SystemExit('No registry found (experiments/registry.json).')
    registry = json.loads(_REGISTRY.read_text())
    candidates = []
    for exp in registry:
        if exp.get('tainted') or exp.get('stale'):
            continue
        created_str = exp.get('created_utc')
        if not created_str:
            continue
        try:
            created = datetime.fromisoformat(str(created_str).rstrip('Z'))
        except Exception:
            continue
        if created < MIN_CREATED_UTC:
            continue
        if _mae_of(exp) is None:
            continue
        candidates.append(exp)
    candidates.sort(key=lambda e: (_mae_of(e), e.get('created_utc', '')))
    return candidates


def _family_of(exp):
    suffix = Path(str(exp.get('model_path', ''))).suffix.lower()
    fam = _SUFFIX_FAMILY.get(suffix, 'sklearn')
    # The joblib bucket splits further by trainer: the linear baselines ship
    # a separate models/artifacts/{name}_preproc.joblib (median SimpleImputer
    # + StandardScaler, models/training/train_linear_baselines.py) that MUST
    # be applied before scoring, so interpret them as their own family.
    if fam == 'sklearn':
        name = str(exp.get('name', ''))
        if 'linear' in str(exp.get('script', '')) or name.startswith(('ridge', 'elasticnet')):
            return 'linear'
    return fam


def _resolve(p):
    """Round 11: emit REPO-RELATIVE POSIX paths in every audit JSON (the rank
    audit + scenario_run_metadata read from these); absolute machine-local
    paths break cross-machine comparability. Consumers resolve relative paths
    against the repo root (see src/ops/predictors._as_path)."""
    p = Path(str(p))
    if p.is_absolute():
        try:
            return p.relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            return p.as_posix()
    return p.as_posix()


def _entry(rank, exp):
    d = {
        'rank': rank,
        'experiment_id': exp.get('id'),
        'name': exp.get('name'),
        'family': _family_of(exp),
        'mode': exp.get('mode'),
        'test_mae': _mae_of(exp),
        'test_r2': _r2_of(exp),
        'model_path': _resolve(exp.get('model_path')),
        'features_path': _resolve(exp.get('features_path')),
        'created_utc': exp.get('created_utc'),
    }
    if d['family'] == 'linear':
        d['preproc_path'] = ('models/artifacts/'
                                 + str(exp.get('name')) + '_preproc.joblib')
    return d


def _champion():
    if not _CHAMPION.exists():
        raise SystemExit('No champion file (experiments/champion.json). '
                         'Run refresh_champion before any analysis stage.')
    return json.loads(_CHAMPION.read_text())


def _rank_all(champ_id):
    eligible = eligible_experiments()
    if not eligible:
        raise SystemExit('No eligible experiments in experiments/registry.json.')
    by_id = {e.get('id'): e for e in eligible}
    if champ_id not in by_id:
        raise SystemExit(
            'champion.json points at ' + str(champ_id) + ', which is not an '
            'eligible (untainted, non-stale, post-2026-05-18) registry entry. '
            'Run the refresh_champion stage before any analysis stage.')
    ordered = [by_id[champ_id]] + [e for e in eligible if e.get('id') != champ_id]
    return [_entry(i + 1, e) for i, e in enumerate(ordered)]


def get_all_models(write_audit=True):
    """Every eligible model as a ranked entry (rank 1..K). Rank 1 is exactly
    experiments/champion.json's selection; ranks 2..K follow ascending test
    MAE. Each entry: rank/experiment_id/name/family/mode/test_mae/test_r2/
    model_path/features_path/created_utc (+ preproc_path for linear)."""
    champ = _champion()
    ranked = _rank_all(champ.get('selected_experiment'))
    if write_audit:
        _write_audit(champ, ranked)
    return ranked




def _write_audit(champ, ranked):
    top2 = ranked[:2]
    names = ', '.join('rank{r}={n} (mae={m:.6f})'.format(
        r=e['rank'], n=e['name'], m=e['test_mae']) for e in top2)
    note = ('Rank 1 is exactly what experiments/champion.json designates (its '
            'own selection logic is untouched); ranks 2..N follow ascending '
            'test MAE among the remaining eligible entries, recomputed fresh '
            'from experiments/registry.json on this run -- never hard-coded. '
            'Current interpretation slots: ' + names + '.')
    if len(top2) > 1 and top2[1]['test_mae'] < top2[0]['test_mae']:
        note += (' NOTE: rank 2 carries a LOWER test MAE than rank 1. This is '
                 'not an error: champion.json applies a near-tie (<0.001 MAE) '
                 'base-spec preference that this ranking deliberately does not '
                 're-litigate; champion.json stays authoritative for rank 1.')
    payload = {
        'selection_rule': champ.get('selection_rule'),
        'champion_selection_note': champ.get('selection_note'),
        'champion_selected_experiment': champ.get('selected_experiment'),
        'champion_selected_utc': champ.get('selected_utc'),
        'selected_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'eligibility_filter': ('untainted, non-stale, created_utc >= '
                               '2026-05-18 -- identical to '
                               'src/ops/experiment_runner.py (refresh_champion)'),
        'selection_note': note,
        'interpretation_coverage': ('every eligible model below is run through '
                                    'SHAP / PDP / confidence intervals / '
                                    'sensitivity / counterfactual scenarios '
                                    'each pipeline run (full coverage)'),
        'ranked_eligible_models': ranked,
    }
    _AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    _AUDIT_OUT.write_text(json.dumps(payload, indent=2))
    print('[champion_ranking] ' + note)
