check_intervals.py, check_test_split.py, write_log.py, src/smoke_test_app_8507.py — temporary diagnostic scripts accidentally committed during development. Moved to trash 2026-06-05.
Moved files
===========

This file documents files moved into `trash/` on automated cleanup pass.

Moved items:

- src/preprocess_DEPRECATED.py
  - Reason: Explicitly marked DEPRECATED in filename. Legacy preprocessing script superseded by newer, modular pipeline in `src/` and `models/training`.
  - Replaced by: current pipeline scripts under `src/` (e.g., `build_sales_pluto.py`, `merge_sales_pluto_ideology.py`, `feature_engineering.py`).

- notebooks/counterfactual_sim.py
  - Reason: Empty file (no content). Likely leftover placeholder.

- src/_debug_minimal.py
  - Reason: Minimal debug utility used for local imports & quick checks. Not part of production pipeline; noisy in repo root.

- src/_trace_imports.py
  - Reason: Development helper to trace imports. Not part of production pipeline.

- package-lock.json
  - Reason: Node lockfile present in a Python-focused repo and empty packages; likely accidental. Kept for recovery but moved out of main tree.

Notes:

- Nothing was deleted permanently. Originals are preserved in this `trash/` folder so recovery is straightforward.
- If you want these restored to their original locations, move them back and commit.
- NOTE: mixed_governance scenario remains unresponsive to dem_share mutation even at ±0.35. This is a known limitation of the current LightGBM political model. Documented for reproducibility. Proceeding with remaining fixes.

- check_app_response.py, check_political_feature_importance.py, check_shap_sample.py, script_temp.py, temp_check.py — temporary diagnostic scripts created during debugging sessions. Moved 2026-05-25.
lgbm_all_years_features.json � legacy pre-v2 feature list containing leakage features (assesstot, EASE-MENT). Superseded by lgbm_all_years_base_features.json and lgbm_all_years_political_features.json. Moved 2026-05-25.
