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
- Ideology derivation: the **production** `dem_share` feature comes from `src/build_ideology_scores.py`, which writes `data/processed/ideology_by_council.parquet` (a custom line-by-line parser of the raw ED `ed_results_{year}_mayor.csv` files, aggregated to council+year). `src/rebuild_canonical.py` merges it per `(CounDist, election_year)` into the canonical dataset, and it is aggregated across three election years (2017/2021/2025) for robustness. A SEPARATE abandoned pipeline — `src/compute_ideology.py` → `ed_ideology.parquet` → `src/build_district_ideology.py` → `district_ideology.parquet` (+ `src/impute_ideology.py`) — is **blacklisted** in `config/feature_blacklist.yaml` and feeds no model.
- Political model: a dedicated LightGBM political model (`models/lgbm_all_years_political.txt`) is trained on features tailored to political sensitivity (ideology features, demographic indicators) and stored with accompanying feature lists.
- Scenarios: `src/political_scenarios.py` applies scenario transformations (liberal_policy, conservative_policy, mixed_governance) and uses the political model for the mixed governance variant where appropriate. Scenario outputs are stored in `results/political_scenarios/` for aggregation and visualization.

## Spatial statistics pipeline

- Moran's I (paper Section 4.2) is computed by `src/compute_morans_i.py`. Unlike earlier internal versions that erroneously built the spatial-weights neighborhoods from the mean lat/lon of *individual sale transactions* grouped by council district, the production path now:
  - Loads the **official NYC City Council District boundary shapefile** at `data/raw/election_districts/NYC_City_Council_Districts.shp` (note: the co-located `geo_export_9895bb0a-*.shp` is the *Election District* boundary file used by the ED-to-council crosswalk's area-overlap column — they must not be confused).
  - Merges polygon geometries onto council-district aggregates (`CounDist`, in both the shapefile's attribute table and the canonical dataset).
  - Builds the spatial weights from **polygon centroids** (`geometry.centroid`) using `libpysal.weights.KNN` with `k=5` nearest neighbours, row-standardized.
  - Computes Moran's I via `esda.Moran` with 999 conditional permutations (seed=42), for both the median `target_log_price` and the mean `dem_share` per council district.
- This script declares its own dependencies (`esda`, `libpysal`) in `requirements.txt` and writes a self-documenting artifact at `results/morans_i_results.json` (including the shapefile path, join key, CRS, and weights construction). OLS regression diagnostics for Section 4.3/Appendix A.2 are emitted by `src/generate_regression_table.py` into `results/regression_diagnostics.json` and `results/regression_model3_full_summary.txt`.

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

3. Build the canonical dataset and splits (production pipeline; see "Rebuild the modeling pipeline" in `README.md` for the full sequence):

```powershell
python src/rebuild_canonical.py
python src/split_temporal.py
```

   Notes:
   - `rebuild_canonical.py` is the production canonicalizer. It pulls the production `dem_share` from `src/build_ideology_scores.py` → `data/processed/ideology_by_council.parquet`. Do **not** run the abandoned `src/compute_ideology.py` pipeline — it is blacklisted in `config/feature_blacklist.yaml` and feeds no model (see the "Political modeling pipeline" section above).
   - The earlier `build_sales_pluto.py` / `freeze_schema.py` steps are folded into `rebuild_canonical.py`; they are not a required standalone sequence.

4. Train a model (example; training config/overrides come from experiment entries in `experiments/registry.json`, not a top-level YAML):

```powershell
python models/training/train_lgbm.py
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
