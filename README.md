# O.R.B.I.T v2 — Forecast Explorer

Quick start

- Create and activate your Python virtual environment.
- Install dependencies: `pip install -r requirements.txt`
- Run the Streamlit app: `streamlit run app/app.py`

Notes

- The app requires a test split at `data/splits/all_years_test.parquet` and a trained champion model at `models/lgbm_all_years_political.txt` plus its feature list in `models/artifacts/`.
- A simple SQLite-backed account system is available at `src/auth.py`. Sign up via the app UI, then log in to persist predictions and counterfactuals to `data/users.db`.
- CI: a lightweight GitHub Actions workflow runs `src/ci_smoke_check.py` to verify repository files and basic imports.
