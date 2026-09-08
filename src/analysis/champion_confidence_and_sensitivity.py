import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

REG = Path('experiments/registry.json')
CHAMP = Path('experiments/champion.json')
SPLIT_DIR = Path('data/splits')
PRED_DIR = Path('experiments/predictions')
ARTIFACT_DIR = Path('models/artifacts')

PRED_DIR.mkdir(parents=True, exist_ok=True)

champ = json.loads(CHAMP.read_text())
registry = json.loads(REG.read_text())
champ_id = champ.get('selected_experiment')
exp = next((e for e in registry if e.get('id') == champ_id), None)
if exp is None:
    print('Champion not found')
    raise SystemExit(1)

name = exp.get('name')
model_path = Path(exp.get('model_path'))
features_path = Path(exp.get('features_path'))
features = json.loads(features_path.read_text())

# load splits
train = pd.read_parquet(SPLIT_DIR / 'all_years_train.parquet')
test = pd.read_parquet(SPLIT_DIR / 'all_years_test.parquet')


def predict_frame(booster, feats, df):
    """Predict with LightGBM's native NaN handling over ALL rows (no dropna)."""
    return booster.predict(df[feats])


def write_test_preds(exp_name, preds, df):
    """{name}_test_preds.parquet: BBL + pred_log_price, index = test.index.

    NOTE (2026-09-06): the political model's historical test_preds.parquet
    carried 56,798 rows with a RESET 0..56,797 index (19 rows short of the
    56,817-row test split; the exporting script is no longer in the repo and
    the reset index means the missing rows are not even identifiable from the
    artifact). Regenerated here from the full split, with an assertion so the
    gap cannot silently recur.
    """
    out = pd.DataFrame({'BBL': df['BBL'].values, 'pred_log_price': preds},
                       index=df.index)
    assert len(out) == len(df), f'{exp_name}: {len(out)} rows vs {len(df)} test rows'
    path = PRED_DIR / f"{exp_name}_test_preds.parquet"
    out.to_parquet(path)
    print(f'Saved {len(out)}-row test preds to {path}')
    return path


def write_intervals(exp_name, preds, resid_q, n_test):
    """{name}_intervals.csv: pred/lower/upper via additive residual quantiles."""
    lower_q, upper_q = resid_q
    intervals = pd.DataFrame({'pred': preds})
    intervals['lower'] = intervals['pred'] + lower_q
    intervals['upper'] = intervals['pred'] + upper_q
    assert len(intervals) == n_test, f'{exp_name}: {len(intervals)} rows vs {n_test} test rows'
    path = PRED_DIR / f"{exp_name}_intervals.csv"
    intervals.to_csv(path, index=False)
    print(f'Saved {len(intervals)}-row intervals to {path}')
    return path

TARGET = 'target_log_price'

# load model (assume lightgbm champion)
booster = lgb.Booster(model_file=str(model_path))

# base predictions
preds_test = booster.predict(test[features])
preds_train = booster.predict(train[features])

# residual quantiles on training set
residuals = train[TARGET] - preds_train
lower_q, upper_q = np.percentile(residuals, [2.5, 97.5])

write_test_preds(name, preds_test, test)
intervals_path = write_intervals(name, preds_test, (lower_q, upper_q), len(test))

# sensitivity analysis for top features (use earlier PDP summary or feature importance)
fi = booster.feature_importance(importance_type='gain')
names = booster.feature_name()
feat_imp = sorted(zip(names, fi), key=lambda x: x[1], reverse=True)
top_features = [f for f,_ in feat_imp if f in features][:5]

summaries = []
base_mean = preds_test.mean()
for feat in top_features:
    for pct in [0.10, 0.25]:
        for sign in [1, -1]:
            factor = 1 + sign * pct
            X_mod = test[features].copy()
            if feat in X_mod.columns:
                X_mod[feat] = X_mod[feat] * factor
            preds_mod = booster.predict(X_mod)
            delta = preds_mod - preds_test
            summaries.append({
                'feature': feat,
                'perturb': f"{sign*pct*100:.0f}%",
                'mean_delta': float(delta.mean()),
                'median_delta': float(np.median(delta)),
                'mean_pct_change': float(delta.mean() / base_mean) if base_mean != 0 else None,
            })

with open(ARTIFACT_DIR / f"{name}_sensitivity.json", 'w') as f:
    json.dump(summaries, f, indent=2)

print('Saved intervals to', intervals_path)
print('Saved sensitivity summary to', ARTIFACT_DIR / f"{name}_sensitivity.json")

# ---------------------------------------------------------------------------
# Paper's political model: regenerate its prediction artifacts from the FULL
# 56,817-row test split (closes the 19-row gap recorded in
# results/round5_followup_resolution.json; the files were previously produced
# by a secondary export script that is no longer in this repo).
# ---------------------------------------------------------------------------
POL_NAME = 'lgbm_all_years_political'
pol = next((e for e in registry if e.get('name') == POL_NAME), None)
if pol is None:
    print(f'NOTE: {POL_NAME} not in registry; political test-preds/intervals not regenerated')
elif pol.get('id') == champ_id:
    print(f'NOTE: {POL_NAME} IS the champion; artifacts already written above')
else:
    pol_features = json.loads(Path(pol.get('features_path')).read_text())
    pol_booster = lgb.Booster(model_file=str(pol.get('model_path')))
    pol_preds_test = predict_frame(pol_booster, pol_features, test)
    write_test_preds(POL_NAME, pol_preds_test, test)
    pol_resid = train[TARGET] - predict_frame(pol_booster, pol_features, train)
    write_intervals(POL_NAME, pol_preds_test, tuple(np.percentile(pol_resid, [2.5, 97.5])), len(test))
    print(f'Regenerated {POL_NAME} artifacts at full {len(test)}-row test coverage '
          f'(closes the 19-row gap)')
