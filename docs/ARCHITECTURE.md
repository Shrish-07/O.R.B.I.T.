# O.R.B.I.T. — System Architecture

## System overview

O.R.B.I.T. (Observational Real-estate Behavior & Intelligence Toolkit) is an integrated data and model platform for forecasting property-level outcomes across New York City. It ingests public and internal raw sources, builds a canonical modeling dataset, trains machine learning models, and exposes an interactive Streamlit application for single-property exploration, counterfactual analysis, and batch portfolio scoring.

At its core, O.R.B.I.T. separates data ingestion, canonicalization, modeling, and application layers so that experiments, model artifacts, and prediction-serving remain reproducible and auditable.

## Data pipeline

- Raw sources: raw data lands in `data/raw/` and may include PAD/PLUTO extracts, election result CSVs, sales records, and geospatial crosswalks.
- Preprocessing: ingestion scripts under `src/` (e.g., `build_sales_pluto.py`, `normalize_ed_results.py`) standardize column names, handle coordinate reprojection, and correct encoding/format issues.
- Canonical dataset: the canonical modeling table is stored as `data/canonical/modeling_dataset_canonical_v2.parquet`. This is the authoritative feature table used for training and downstream inference. Canonicalization includes leakage removal, feature imputation, and the application of the feature blacklist.
- Splits: temporal train/test splits are persisted under `data/splits/` (for example `all_years_test.parquet`). Splits are constructed to respect temporal holdouts and avoid information leakage from future data.
- Outputs: intermediate and processed data are under `data/processed/`, while final split files and canonical artifacts are versioned inside `data/canonical/` and checked into experiments and artifacts folders where appropriate.

## ML pipeline

- Feature blacklisting: a curated `config/feature_blacklist.yaml` drives the removal of leaky features (for example `assesstot`, `EASE-MENT`) before model training.
- Training entry points: training code is found in `src/train_model.py` and `src/train_baseline.py`. Typical runs: preprocess canonical table → create features → train LightGBM models (`train_lgbm.py` semantics captured via `train_model.py`).
- Registry: experiments are registered in `experiments/registry.json`, and the current champion is recorded at `experiments/champion.json`. Each experiment entry lists `model_path`, `features_path`, and metadata.
- Champion selection: evaluation metrics (MAE, log-MAE) and OOF performance are used to select the champion. A lightweight selection procedure marks the champion and writes SHAP/summary artifacts under `models/artifacts/`.

## Political modeling pipeline

- Sources: raw election CSVs (multiple years) are processed to compute vote shares at the council and precinct level.
- Ideology derivation: the `src/compute_ideology.py` and `src/impute_ideology.py` scripts create `ideology_by_council.parquet` and the `dem_share` feature used by the political model. Ideology is inferred across three election years to improve robustness.
- Political model: a dedicated LightGBM political model (`models/lgbm_all_years_political.txt`) is trained on features tailored to political sensitivity (ideology features, demographic indicators) and stored with accompanying feature lists.
- Scenarios: `src/political_scenarios.py` applies scenario transformations (liberal_policy, conservative_policy, mixed_governance) and uses the political model for the mixed governance variant where appropriate. Scenario outputs are stored in `results/political_scenarios/` for aggregation and visualization.

## Application architecture

- Streamlit frontend: `app/app.py` is the single entry point that renders multiple pages: Home, Individual Property Analysis, Portfolio Analysis, Political Scenarios, Model Explorer, Research Outputs, and User Dashboard.
- Authentication: a minimal auth layer exists under `src/auth.py` which supports lightweight session storage and user history persistence.
- Geocoding: `src/geocoder.py` integrates the NYC GeoSearch API to accept freeform addresses and map them to BBLs for direct property lookup.
- Model loading: the app resolves the champion using `experiments/champion.json` and loads the corresponding LightGBM model and feature list from `models/` and `models/artifacts/`.
- Prediction serving: predictions are computed in-memory using the loaded LightGBM booster and optional serialized preprocessor; prediction intervals (when present) are loaded from `experiments/predictions/`.

## Artifact inventory

- `models/`: trained model binaries (LightGBM text dumps) and `models/artifacts/` contains feature lists, SHAP summaries, and the frozen canonical schema (e.g., `canonical_schema.json`).
- `experiments/`: registry, champion pointer, prediction outputs, and per-experiment result JSONs.
- `data/`: raw, processed, canonical, and split datasets. Canonical v2 is the authoritative modeling table.
- `results/`: scenario outputs and aggregated results used by the app and figures.
- `docs/`: frozen schema (`modeling_schema.csv`), architecture documentation, and figures for publication.

## Key design decisions

- Temporal splits: training uses a time-based holdout to minimize information leakage from future sales; this choice favors generalization to future periods.
- Leakage blacklist: known leaky features (assessed totals and easements) are explicitly blacklisted and removed prior to training; this is logged in `config/feature_blacklist.yaml` to preserve auditability.
- Ideology signal: ideology is derived using multiple election years (e.g., 2017/2021/2025) to smooth temporal variance and impute missing council-level data.
- Mixed governance: the `mixed_governance` scenario intentionally uses the political model for certain features (e.g., `dem_share`) rather than the base model, because political shifts are modeled with a separate, politically-focused training regime.

## Reproducibility instructions

To reproduce the canonical dataset, train models, and run the app from a clean environment, execute these steps (assumes Python 3.11, virtualenv):

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Build canonical dataset and splits:

```powershell
python src/build_sales_pluto.py
python src/compute_ideology.py
python src/freeze_schema.py
```

4. Train a model (example):

```powershell
python src/train_model.py --config configs/lgbm_base.yaml
```

5. Run app locally:

```powershell
python -m streamlit run app/app.py --server.headless true --server.port 8501
```

6. Run smoke/CI checks:

```powershell
python src/pipeline_sanity_check.py
```

Maintainers should pin package versions in `requirements.txt` and record experiment metadata in `experiments/registry.json` to enable deterministic rebuilds.

---

This document is intended to provide a compact but actionable overview of O.R.B.I.T.'s architecture for engineers and auditors. For detailed developer notes, see `src/` scripts and `docs/` subfiles.
# Project Architecture

- src/: ETL, feature engineering, training, experiment management, and app
- data/raw/: raw snapshots ingested from sources
- data/processed/: cleaned & feature-engineered datasets
- models/: trained model artifacts and preprocessing joblibs
- experiments/: run registry and champion selection
- app/: Streamlit UI
