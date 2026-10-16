"""Tiered, family-optimized SHAP computation for every interpreted model
(Round 9).

TreeSHAP tractability on the unpruned 200-tree RandomForest is the hard case
(the repo's Round-8 record: >40 CPU-min unfinished even at n=300/2000). The
ladder below runs tiers in order and stops at the FIRST one that completes
inside a hard 90-minute budget PER TIER (wall clock; a single TreeSHAP job is
effectively single-CPU, so wall and CPU minutes coincide):

  (a) exact TreeSHAP on the standard sampled test slice (up to 2,000 rows,
      seed 42, sampled after dropna on the 10 features with <100 NAs -- the
      exact sampling rule generate_shap_summary.py has always used);
  (b) RandomForest only: retrain a max_depth=25 variant of THAT specific
      forest (every other hyperparameter read from the fitted model's own
      get_params()), confirm its test MAE is within 0.001 of the original's
      (both MAEs recomputed identically on fillna(0) test inputs -- the same
      convention train_random_forest.py used), save it as
      models/{name}_depth25_for_shap.joblib for the record (it is NOT
      registered in experiments/registry.json -- a SHAP stand-in must never
      compete for champion selection), then exact TreeSHAP on the variant
      inside its own fresh 90-minute budget. If the MAE fidelity check
      FAILS the variant is flagged loudly and NOT used; the ladder continues
      on the original model;
  (c) exact SHAP on a fixed random sample of 5,000 test rows (seed 42),
      computed on the depth-capped variant when a faithful one exists,
      otherwise on the original model;
  (d) gain-based feature importance ONLY, labeled
      method='feature_importance_fallback' (the repo's existing label, not a
      new name). Those are importance scores, NOT SHAP values.

Linear models (ridge/elasticnet) take their own exact path: shap
.LinearExplainer on the model's OWN preprocessed feature space
(median-imputed + standardized, exactly as trained) -- tier label
'linear_exact' -- with the same (c)/(d) fallbacks behind it.

Every payload self-documents which tier produced it and why; there is no
silent fallback. Consumer-compatible keys: 'model', 'method', 'summary' with
each summary row carrying 'feature'+'mean_abs_shap' (tier-d fallback rows
also carry 'gain', mirroring the pre-Round-9 fallback contract that
generate_figures.py / generate_research_summary.py depend on).

Implementation detail: tiers (a)-(c) run in a SPAWNED child process so the
hard timeout can actually be enforced on Windows. The worker target must be
a module-level function HERE, never a function defined inside an analysis
script, because spawn re-imports the module that defines the target -- the
analysis scripts execute work at import time and a child would re-run the
whole pipeline. (The caller also wraps its body in `if __name__ ==
'__main__':` for the same reason.)
"""

import multiprocessing
import pickle
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_DIR = _REPO_ROOT / 'data' / 'splits'
_MODEL_DIR = _REPO_ROOT / 'models'


_ONEDRIVE_WARNED = False


def _warn_once_onedrive():
    """Print the OneDrive startup warning once per stage (first
    compute_shap_payload call). OneDrive sync can lock files during
    spawn and stall SHAP children at interpreter start."""
    global _ONEDRIVE_WARNED
    if _ONEDRIVE_WARNED:
        return
    _ONEDRIVE_WARNED = True
    if 'OneDrive' in str(_REPO_ROOT):
        print('=' * 74)
        print('WARNING (SHAP stage start): the repository lives under'
              ' OneDrive:')
        print('    ' + str(_REPO_ROOT))
        print('  OneDrive sync can lock files during spawn and stall child'
              ' processes at interpreter start (observed 2026-09-28: the'
              ' tier-(a) child sat at ~0% CPU / 4 MB RSS for its whole'
              ' 90-minute budget without finishing its imports).')
        print('  RECOMMENDED: move the repository out of OneDrive, or pause'
              ' OneDrive sync, before long SHAP tiers. A 5-minute startup'
              ' watchdog (kill + one respawn with a fresh, unspent budget)'
              ' is armed, but it only DETECTS the stall.')
        print('=' * 74)

TIER_BUDGET_SECONDS = 90 * 60
TARGET = 'target_log_price'
SAMPLE_SEED = 42
TIER_A_ROWS = 2000
TIER_C_ROWS = 5000
MAE_FIDELITY_TOL = 0.001
DEPTH_CAP = 25

# Round 10: mean-leaves-per-tree tractability cap (mirrors MAX_MEAN_LEAVES
# in models/training/train_random_forest.py). Tier (b)'s depth-capped
# retrain only makes sense for a forest that is BOTH uncapped
# (max_depth=None) AND above this leaf budget; a complexity-constrained
# forest goes straight through tier (a).
RF_LEAF_TRACTABILITY_CAP = 3000

# Round 10 Part 3 (startup watchdog): a spawned SHAP child writes a
# heartbeat file right after its imports complete. If the parent sees no
# heartbeat within this window of spawn, the child stalled at startup
# (observed 2026-09-28: the tier-(a) child sat at ~0% CPU / 4 MB RSS for
# the whole 90-minute budget without finishing its imports -- likely a
# OneDrive file lock at interpreter start). The stall is NOT charged to
# the tier's budget; the child is killed and respawned exactly once.
HEARTBEAT_TIMEOUT_SECONDS = 5 * 60


# --------------------------------------------------------------------------
# child-process worker. MUST stay module-level (spawn pickles it by qualname)
# and this module MUST remain import-safe (no side effects at import time).
# --------------------------------------------------------------------------
def _shap_worker(model_path, preproc_path, family, sample_pickle, out_npy,
                 err_txt, heartbeat_path=None):
    """Child: compute exact per-feature mean|SHAP| on the pickled sample.

    Writes np.save(out_npy, mean_abs) on success; an error string to err_txt
    on failure. When heartbeat_path is given the child writes it
    IMMEDIATELY after imports finish; that file is what the parent's
    startup watchdog arms against (no heartbeat within HEARTBEAT_TIMEOUT_SECONDS
    of spawn = stalled spawn, the 2026-09-28 OneDrive-lock incident).''
    """
    try:
        import sys
        sys.path.insert(0, str(_REPO_ROOT))
        import numpy as _np
        import shap
        if heartbeat_path:
            with open(str(heartbeat_path), 'w') as _hb:
                _hb.write(str(time.time()))
        with open(sample_pickle, 'rb') as f:
            X = pickle.load(f)
        if family == 'linear':
            from joblib import load as _jl
            model = _jl(str(model_path))
            preproc = _jl(str(preproc_path))
            Xp = preproc.transform(X)          # median impute + standardize
            expl = shap.LinearExplainer(model, Xp,
                                        feature_perturbation='interventional')
            vals = expl.shap_values(Xp)
        else:
            from src.ops.model_io import load_model
            model = load_model(model_path)
            expl = shap.TreeExplainer(model)
            vals = expl.shap_values(X)
        if isinstance(vals, list):
            vals = vals[0]
        mean_abs = _np.asarray(_np.mean(_np.abs(vals), axis=0),
                               dtype=_np.float64)
        _np.save(str(out_npy), mean_abs)
    except Exception as e:
        try:
            with open(err_txt, 'w') as ferr:
                ferr.write(f'{type(e).__name__}: {e}')
        except Exception:
            pass


def _run_shap_with_timeout(entry, X, budget_seconds, tier_label,
                           attempts=None):
    """Spawn a child running _shap_worker; enforce a hard wall-clock budget.

    Returns (mean_abs ndarray, None) on success or (None, reason) where
    reason is 'timeout ...' or the child's error text.

    Round 10 startup watchdog: the child writes a heartbeat file right after
    its imports finish. If no heartbeat arrives within HEARTBEAT_TIMEOUT_SECONDS
    (5 minutes) of spawn, the child stalled at startup (OneDrive file lock), is
    killed, and is respawned exactly once; the tier's budget then runs from the
    heartbeat, so the stall is NOT charged against the tier. When `attempts`
    is provided, 'child stalled at startup, respawned' is appended to it (the
    tier_attempts audit list).''
    """
    tmp = Path(tempfile.mkdtemp(prefix='shap_tier_'))
    pkl = tmp / 'sample.pkl'
    out = tmp / 'mean_abs.npy'
    err = tmp / 'err.txt'
    hb = tmp / 'heartbeat.txt'
    with open(pkl, 'wb') as f:
        pickle.dump(X, f)

    stalled_once = False
    while True:
        for _stale in (out, err, hb):
            if _stale.exists():
                _stale.unlink()
        ctx = multiprocessing.get_context('spawn')
        proc = ctx.Process(target=_shap_worker,
                           args=(str(entry['model_path']),
                                 str(entry.get('preproc_path') or ''),
                                 entry.get('family'),
                                 str(pkl), str(out), str(err), str(hb)))
        t0 = time.time()
        proc.start()
        # --- startup watchdog: wait for the child's import-finished heartbeat;
        # the wait itself is capped at min(5 minutes, remaining budget)
        while (proc.is_alive() and not hb.exists()
               and (time.time() - t0) < min(HEARTBEAT_TIMEOUT_SECONDS,
                                            budget_seconds)):
            time.sleep(1.0)
        if proc.is_alive() and not hb.exists():
            proc.terminate()
            proc.join()
            if stalled_once:
                return None, ('child stalled at startup twice (no heartbeat '
                              'within 5 minutes of either spawn); not '
                              'respawned again')
            stalled_once = True
            if attempts is not None:
                attempts.append(tier_label + ': child stalled at startup '
                                '(no heartbeat within 5 minutes of spawn), '
                                'respawned')
            continue
        # The budget runs from the heartbeat (start of real work); the startup
        # window before the heartbeat is deliberately NOT charged to the tier.
        t_live = time.time()
        proc.join(budget_seconds)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return None, ('time-out: exceeded the {m}-minute hard budget '
                          '(terminated at {s:.0f}s after heartbeat)').format(
                              m=int(budget_seconds // 60), s=time.time() - t_live)
        if out.exists():
            return np.load(out), None
        msg = (err.read_text() if err.exists()
               else 'child exited without output')
        return None, 'error: ' + msg[:2000]


def _standard_sample(test_df, features):
    """The repo's pre-existing SHAP sampling rule, preserved exactly: keep
    the first 10 features with <100 NAs, dropna on those, then sample up to
    TIER_A_ROWS rows with seed 42."""
    na_counts = test_df[features].isna().sum()
    good = [f for f in features if na_counts[f] < 100]
    if not good:
        raise ValueError('No features with <100 NA for SHAP sampling.')
    sample = test_df.dropna(subset=good[:10])
    if len(sample) == 0:
        raise ValueError('No rows left after dropna on low-NA features.')
    return sample.sample(n=min(TIER_A_ROWS, len(sample)),
                         random_state=SAMPLE_SEED)[features]


def _shap_input(entry, sample):
    """SHAP input frame matching the model's trained input convention."""
    if entry.get('family') == 'lightgbm':
        return sample.copy()                     # native NaN, as trained
    if entry.get('family') == 'linear':
        return sample.copy()                     # preproc imputes median
    return sample.fillna(0)                      # cat/xgb/sklearn = fillna(0)


def _summary_rows(mean_abs, features):
    rows = [{'feature': f, 'mean_abs_shap': float(v)}
            for f, v in zip(features, mean_abs)]
    rows.sort(key=lambda r: -r['mean_abs_shap'])
    return rows


def _retrain_depth_capped_rf(entry, model, features):
    """Tier (b): refit THIS forest with max_depth=DEPTH_CAP; every other
    hyperparameter inherited from the fitted model's own get_params()
    (train_random_forest.py's spec: n_estimators=200, max_depth=None,
    n_jobs=-1, random_state=42 with only max_depth changed). NEVER registers
    the variant in experiments/registry.json -- a SHAP stand-in must not
    compete for champion selection.

    Returns (variant_model, info_dict).''
    """
    from joblib import dump
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    params = model.get_params()
    params['max_depth'] = DEPTH_CAP
    variant = RandomForestRegressor(**params)
    train = pd.read_parquet(_SPLIT_DIR / 'all_years_train.parquet')
    test = pd.read_parquet(_SPLIT_DIR / 'all_years_test.parquet')
    variant.fit(train[features].fillna(0), train[TARGET])
    Xte = test[features].fillna(0)
    mae_variant = float(mean_absolute_error(test[TARGET], variant.predict(Xte)))
    mae_orig = float(mean_absolute_error(test[TARGET], model.predict(Xte)))
    vpath = _MODEL_DIR / (str(entry['name']) + '_depth' + str(DEPTH_CAP)
                          + '_for_shap.joblib')
    dump(variant, str(vpath))
    info = {
        'max_depth': DEPTH_CAP,
        'n_estimators': params.get('n_estimators'),
        'other_hyperparameters': 'unchanged -- inherited from the fitted model get_params()',
        'variant_model_file': str(vpath).replace(chr(92), '/'),
        'test_mae_original': mae_orig,
        'test_mae_depth_capped': mae_variant,
        'abs_mae_delta': abs(mae_variant - mae_orig),
        'within_0p001': abs(mae_variant - mae_orig) <= MAE_FIDELITY_TOL,
        'registered_in_experiments_registry': False,
        # Round 10 Part 2: the 'used as stand-in' wording is only true when
        # the fidelity check PASSED and tier (b) actually ran on the variant.
        # A REJECTED variant must be labeled rejected -- the old unconditional
        # wording mislabeled rejected variants in the audit (e.g. the former
        # rf_all_years_political_n200: |MAE delta| 0.006363 > 0.001).
        'note': (
            'SHAP computed on a depth-capped variant for tractability; MAE '
            'confirmed within 0.001 of the original uncapped model.'
            if abs(mae_variant - mae_orig) <= MAE_FIDELITY_TOL else
            ('variant REJECTED as SHAP stand-in (|MAE delta| = '
             + ('%.6f' % abs(mae_variant - mae_orig)) + ' > '
             + str(MAE_FIDELITY_TOL) + '); NO SHAP was computed on it')),
    }
    return variant, info


def _payload(base, tier, mean_abs, features, sample_rows, attempts,
             shap_model_path, tier_note, extra=None):
    p = {
        'model': base['model'],
        'family': base['family'],
        'registry_test_mae': base['registry_test_mae'],
        'method': 'shap',
        'shap_fallback_tier': tier,
        'shap_fallback_note': tier_note,
        'shap_model_path': shap_model_path,
        'budget_minutes_per_tier': base['budget_minutes_per_tier'],
        'computed_utc': base['computed_utc'],
        'sample_n': sample_rows,
        'tier_attempts': list(attempts),
        'summary': _summary_rows(mean_abs, features),
    }
    if extra:
        p.update(extra)
    return p


_TIER_NOTES = {
    'a': ('exact TreeSHAP on the standard sampled test slice (n<=' +
          str(TIER_A_ROWS) + ', seed 42, low-NA feature sampling).'),
    'linear_exact': ('exact SHAP for a linear model (shap.LinearExplainer, '
                     "interventional, on the model's own median-imputed + "
                     'standardized feature space as trained), standard '
                     'sampled test slice (n<=' + str(TIER_A_ROWS) + ', seed 42).'),
    'b': ('SHAP computed on a depth-capped variant for tractability; MAE '
          'confirmed within 0.001 of the original uncapped model.'),
    'c': 'sampled SHAP, N=5,000, not the full test set (fixed sample, seed=42).',
    'd': ('feature_importance_fallback: gain-based feature importance ONLY '
          '-- these are importance scores, NOT SHAP values; every exact-SHAP '
          'tier above failed or timed out inside the per-tier budget.'),
}


def compute_shap_payload(entry, model, test_df, budget_seconds=TIER_BUDGET_SECONDS):
    """Run the tier ladder for one model and return the payload dict (the
    caller writes models/artifacts/{name}_shap_summary.json).''
    """
    _warn_once_onedrive()
    from src.ops.predictors import features_of
    features = features_of(entry)
    family = entry.get('family')
    attempts = []
    base = {
        'model': entry['name'],
        'family': family,
        'registry_test_mae': entry.get('test_mae'),
        'budget_minutes_per_tier': int(budget_seconds // 60),
        'computed_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    }
    is_rf = (family == 'sklearn'
             and type(model).__name__ == 'RandomForestRegressor')

    # ---------------- tier (a): exact SHAP, standard sample ----------------
    Xa = None
    try:
        sample_a = _standard_sample(test_df, features)
        Xa = _shap_input(entry, sample_a)
        mean_abs, terr = _run_shap_with_timeout(entry, Xa, budget_seconds,
                                                'tier a', attempts)
        if mean_abs is not None:
            tier = 'linear_exact' if family == 'linear' else 'a'
            return _payload(base, tier, mean_abs, features, int(len(Xa)),
                            attempts, entry['model_path'], _TIER_NOTES[tier])
        attempts.append('tier a (exact ' + ('LinearExplainer' if family == 'linear' else 'TreeSHAP')
                        + ', n<=' + str(TIER_A_ROWS) + '): ' + str(terr))
    except Exception as e:
        attempts.append('tier a: sample setup failed: '
                        + type(e).__name__ + ': ' + str(e))
    if Xa is None:
        # sample setup failed entirely -- fall back to a plain seed-42 sample
        sample_a = test_df.sample(n=min(TIER_A_ROWS, len(test_df)),
                                  random_state=SAMPLE_SEED)[features]
        Xa = _shap_input(entry, sample_a)

    # ---------------- tier (b): depth-capped RandomForest stand-in --------
    variant_info = None
    active_entry = entry
    _tier_b_applies = False
    if is_rf:
        # Round 10 Part 1(h): tier (b) (depth-capped retrain) only makes
        # sense for an UNPRUNED forest. A forest that already has a depth
        # cap or already satisfies the mean-leaves tractability cap goes
        # straight through tier (a); tiers (c)/(d) remain safety nets.
        _rf_max_depth = model.get_params().get('max_depth')
        _rf_mean_leaves = float(
            sum(t.get_n_leaves() for t in model.estimators_)
            / len(model.estimators_))
        _tier_b_applies = (_rf_max_depth is None
                           and _rf_mean_leaves > RF_LEAF_TRACTABILITY_CAP)
        if not _tier_b_applies:
            attempts.append(
                'tier b: skipped -- this forest is already '
                'complexity-constrained (max_depth='
                + str(_rf_max_depth) + ', mean leaves/tree = %.0f'
                % _rf_mean_leaves + '); the depth-capped retrain tier'
                ' only applies to an unpruned forest above the '
                + str(RF_LEAF_TRACTABILITY_CAP)
                + ' mean-leaves cap, and exact SHAP runs directly via'
                ' tier (a).')
    if is_rf and _tier_b_applies:
        try:
            variant, variant_info = _retrain_depth_capped_rf(entry, model,
                                                             features)
        except Exception as e:
            attempts.append('tier b (depth-capped variant retrain): '
                            + type(e).__name__ + ': ' + str(e))
            variant = None
        if variant is not None:
            if not variant_info['within_0p001']:
                attempts.append(
                    'tier b: depth-capped variant REJECTED -- MAE fidelity '
                    'check failed (abs delta '
                    + ('%.6f' % variant_info['abs_mae_delta']) + ' > '
                    + str(MAE_FIDELITY_TOL) + '); the variant was saved to '
                    + variant_info['variant_model_file'] + ' for the record '
                    'but was NOT used as a SHAP stand-in; the ladder '
                    'continues on the original model.')
            else:
                ventry = dict(entry)
                ventry['model_path'] = variant_info['model_path'] = (
                    variant_info['variant_model_file'])
                mean_abs, terr = _run_shap_with_timeout(ventry, Xa,
                                                        budget_seconds,
                                                        'tier b', attempts)
                if mean_abs is not None:
                    payload = _payload(base, 'b', mean_abs, features,
                                       int(len(Xa)), attempts,
                                       variant_info['variant_model_file'],
                                       _TIER_NOTES['b'],
                                       extra={'depth_capped_variant': variant_info})
                    return payload
                attempts.append('tier b (exact TreeSHAP on the depth-capped '
                                'variant, n<=' + str(TIER_A_ROWS) + '): '
                                + str(terr))
                active_entry = ventry
    else:
        attempts.append('tier b: skipped -- tier b only applies to an '
                        'sklearn RandomForestRegressor (this model is '
                        + str(family) + '/' + type(model).__name__ + ').')

    # ---------------- tier (c): fixed 5,000-row sample SHAP ---------------
    try:
        n_c = min(TIER_C_ROWS, len(test_df))
        sample_c = test_df.sample(n=n_c, random_state=SAMPLE_SEED)[features]
        Xc = _shap_input(active_entry, sample_c)
        mean_abs, terr = _run_shap_with_timeout(active_entry, Xc,
                                                budget_seconds, 'tier c',
                                                attempts)
        if mean_abs is not None:
            extra = {
                'sampled_shap': {
                    'n': int(n_c),
                    'seed': SAMPLE_SEED,
                    'note': 'sampled SHAP, N=5,000, not the full test set',
                    'computed_on': ('depth-capped variant' if active_entry is not entry
                                    else 'original model'),
                }
            }
            if variant_info is not None:
                extra['depth_capped_variant'] = variant_info
            payload = _payload(base, 'c', mean_abs, features, int(n_c),
                               attempts, active_entry['model_path'],
                               _TIER_NOTES['c'], extra=extra)
            return payload
        attempts.append('tier c (exact SHAP, fixed 5,000-row seed-42 '
                        'sample): ' + str(terr))
    except Exception as e:
        attempts.append('tier c: ' + type(e).__name__ + ': ' + str(e))

    # ---------------- tier (d): gain-based importance fallback ------------
    from src.ops.model_io import gain_importance
    rows = []
    for fname, v in gain_importance(model, features):
        if fname in features:
            # Both keys are emitted (mean_abs_shap mirrors gain) because
            # downstream consumers (generate_figures.py,
            # generate_research_summary.py) plot mean_abs_shap
            # unconditionally; the method field keeps provenance honest.
            rows.append({'feature': fname, 'gain': float(v),
                         'mean_abs_shap': float(v)})
    extra = {}
    if variant_info is not None:
        extra['depth_capped_variant'] = variant_info
    payload = {
        'model': base['model'],
        'family': base['family'],
        'registry_test_mae': base['registry_test_mae'],
        'method': 'feature_importance_fallback',
        'shap_fallback_tier': 'd',
        'shap_fallback_note': _TIER_NOTES['d'],
        'shap_model_path': entry['model_path'],
        'budget_minutes_per_tier': base['budget_minutes_per_tier'],
        'computed_utc': base['computed_utc'],
        'tier_attempts': list(attempts),
        'summary': sorted(rows, key=lambda r: -r.get('gain', 0.0)),
    }
    payload.update(extra)
    return payload
