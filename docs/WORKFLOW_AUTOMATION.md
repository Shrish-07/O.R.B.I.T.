O.R.B.I.T — Automation & Reproducibility Summary
===============================================

This document summarizes automated actions taken by the agent to validate and run the canonical pipeline, produce reproducible artifacts, and provide an interactive explorer.

Artifacts created
- data/canonical/LOCK.json — canonical dataset snapshot metadata (sha256, rows, columns)
- data/canonical/modeling_dataset_canonical.parquet — copied canonical dataset snapshot
- data/splits/* — canonical temporal train/test splits (all_years, year2017)
- models/lgbm_all_years_political.txt — trained LightGBM model artifact
- models/artifacts/lgbm_all_years_political_metrics.json — metrics
- models/artifacts/lgbm_all_years_political_features.json — features used
- models/artifacts/lgbm_all_years_political_importance.json — feature importance (gain)
- models/artifacts/lgbm_all_years_political_shap_summary.json — SHAP summary
- models/artifacts/lgbm_all_years_political_shap_sample.parquet — sampled SHAP values
- experiments/results/lgbm_all_years_political.json — experiment result record
- experiments/registry.json — centralized experiment registry (appended entry)
- experiments/predictions/lgbm_all_years_political_test_preds.parquet — test predictions
- logs/actions.log — chronological action log

Key commands used (run inside repository venv)

```powershell
# Run pipeline sanity checks
& ".venv\Scripts\python.exe" src\pipeline_sanity_check.py

# Produce temporal splits
& ".venv\Scripts\python.exe" src\split_temporal.py

# Train canonical LightGBM (numeric features + ideology)
& ".venv\Scripts\python.exe" models\training\train_lgbm.py --variant all_years --mode political

# Start interactive explorer (Streamlit)
& ".venv\Scripts\python.exe" -m streamlit run app\app.py
```

Assumptions made
- Where preprocessing left object-typed numeric fields, the agent coerced values to numeric and filled missing values with `0` for prediction/simulation. This is conservative and documented in the app; a production system should use consistent preprocessing pipelines and imputation strategies.
- The representative experiment hypothesis used: "Including ideology features improves temporal MAE by capturing political variation." This was recorded in `experiments/registry.json`.

Next recommended steps
- Add automated CI that runs `pipeline_sanity_check.py` and a short training smoke-test on push.
- Formalize experiment scheduler and champion selection in `src/` (small orchestration script or use MLflow).
- Harden the Streamlit app to accept arbitrary uploaded portfolios and to persist user scenarios.
