# O.R.B.I.T. – Optimized Real Estate Bayesian-Integrated Transformer

A Streamlit app for NYC property price prediction using a stacked ensemble of XGBoost, LightGBM, and Bayesian Ridge Regression.

## Files
- `orbit.py`: main Streamlit app
- `requirements.txt`: Python dependencies
- (`orbit_model.pkl` — the trained model — is not shipped in this snapshot; the legacy pipeline must be re-run to produce it)

## Features
- Upload & clean NYC housing data
- Predict property prices (single/batch)
- SHAP + PDP explainability
- Deployment-ready pipeline

## How to Run
1. Clone repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run Streamlit: `streamlit run orbit.py`

## Authors
TODO: Add author name(s)
