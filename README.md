# O.R.B.I.T. — Observational Real-estate Behavior & Intelligence Toolkit

O.R.B.I.T. is a reproducible research platform and forecasting toolkit designed to produce property-level price forecasts and scenario analyses for New York City. It integrates public datasets (PLUTO, PAD, sales, election returns), canonicalizes and cleans features, trains LightGBM models, and exposes an interactive Streamlit application for single-property exploration, batch scoring, and political scenario simulation.

This README documents the current system status, how to run the app, how to rebuild the canonical dataset and models, and where to find key artifacts and research outputs.

## System status

- Champion model: `lgbm_all_years_base` (LightGBM). Empirical test MAE (log-price) ≈ 0.429.
- Available features: canonical modeling table at `data/canonical/modeling_dataset_canonical_v2.parquet` with the frozen schema recorded in `models/artifacts/canonical_schema.json`.
- App: Streamlit frontend `app/app.py` with 7 pages (Home, Individual Property Analysis, Portfolio Analysis, Political Scenarios, Model Explorer, Research Outputs, User Dashboard).
- Functionality added: address lookup (free-text addresses → BBL via NYC GeoSearch), prediction intervals (per-property lower/upper bounds loaded from `experiments/predictions/*.parquet`), and a 5-year price forecast panel (borough-specific appreciation rates).
- Risk indicators: per-property risk panel reporting overvaluation vs borough median, prediction uncertainty (interval width), and building age risk.
- Political scenarios: three scenario variants (`liberal_policy`, `conservative_policy`, `mixed_governance`) are implemented and saved under `results/political_scenarios/`.

## How to run (developer / local)

All commands assume Windows PowerShell and a Python 3.11 virtual environment. From the project root:

1. Create and activate the virtual environment:

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the Streamlit app (headless on a specific port):

```powershell
python -m streamlit run app/app.py --server.headless true --server.port 8501
```

Open `http://localhost:8501` to use the app.

## Rebuild the modeling pipeline (from raw to artifacts)

To rebuild the canonical table, retrain models, and regenerate research outputs, run the scripts in order (example sequence):

```powershell
python src/rebuild_canonical.py
python src/split_temporal.py
python models/training/train_lgbm.py
python src/political_scenarios.py
python src/compute_morans_i.py
python src/generate_regression_table.py
python src/generate_figures.py
python src/generate_research_summary.py
python src/generate_political_figures.py
```

Notes:
- Each script writes versioned artifacts under `data/`, `data/canonical/`, `data/splits/`, `experiments/`, `models/artifacts/`, and `results/`.
- `src/compute_morans_i.py` produces `results/morans_i_results.json` (paper Section 4.2 Moran's I) using the official NYC City Council District shapefile polygon centroids.
- `src/generate_regression_table.py` produces the OLS coefficients table (`results/regression_table.csv`) AND the Appendix A.2 / Section 4.3 residual diagnostics (`results/regression_diagnostics.json`, `results/regression_model3_full_summary.txt`).
- Training scripts accept configuration overrides through experiment entries in `experiments/registry.json`.

## Data sources (what belongs in `data/raw/`)

- NYC sales CSV exports (DOF / ACRIS sales records)
- PLUTO CSV (tax lot attributes and building characteristics)
- PAD / PLUTO addendum crosswalks (for BBL mappings)
- Election result CSVs (council, precinct results used to compute ideology / `dem_share`)
- Shapefiles and geospatial exports for council / precinct boundaries (stored in `data/raw/election_districts/`)

Populate `data/raw/` with the above sources before running the rebuild steps.

## Model summary

- Champion model: LightGBM (`models/lgbm_all_years_base.txt` or path recorded in `experiments/champion.json`).
- Performance: test MAE (log-price) ≈ 0.429 as measured against the canonical test split.
- Features: full feature lists are in `models/artifacts/` (e.g., `lgbm_all_years_base_features.json`). A frozen canonical schema is available at `models/artifacts/canonical_schema.json`.
- Leakage: known leaky variables (e.g., assessed totals like `assesstot`, easement indicators such as `EASE-MENT`) were explicitly blacklisted and removed prior to training using `config/feature_blacklist.yaml`. This was done to ensure temporal integrity and prevent information from future sales or administrative values from contaminating model training.

## Political modeling

- Scenarios: three scenario families are supported:
	- `liberal_policy`: simulates policy shifts that generally increase values in certain council districts.
	- `conservative_policy`: simulates more restrictive policy outcomes.
	- `mixed_governance`: uses the political model and scenario-specific adjustments for `dem_share` and related features to reflect governance-driven changes.
- `dem_share` is computed by `src/build_ideology_scores.py`, which writes `data/processed/ideology_by_council.parquet` (a custom line-by-line parser of the raw ED `ed_results_{year}_mayor.csv` election files, aggregated to council+year). `src/rebuild_canonical.py` merges this per `(CounDist, election_year)` into the canonical dataset; this is the production `dem_share` feature the political model trains on.
- A SEPARATE, abandoned pipeline — `src/compute_ideology.py` → `ed_ideology.parquet` → `src/build_district_ideology.py` → `district_ideology.parquet` (+ `src/impute_ideology.py` for imputation) — is **blacklisted** in `config/feature_blacklist.yaml` (`district_ideology`, `weighted_dem`, `weighted_rep`, `dem_vote_share`, `rep_vote_share`, `turnout` are all listed). It feeds NO model and must not be confused with the `dem_share` production path.
- The `mixed_governance` path selectively applies the political model where political features have higher predictive power, while the base price model handles structural and market signals.

## Address lookup

- Free-text address lookup is provided by `src/geocoder.py`. It queries the NYC GeoSearch service:

	`https://geosearch.planninglabs.nyc/v2/search` (query param `text`)

- The geocoder extracts the PAD/BBL when available (`properties.addendum.pad.bbl`) and attempts to match the returned BBL to the canonical/test split. If a BBL is not present in the test split the app presents a fallback slider-based estimator built from borough defaults and feature sliders.

## Key files

- `app/app.py` — Streamlit app entrypoint and UI logic (address lookup, predictions, SHAP, scenarios).
- `src/geocoder.py` — Geosearch client and BBL extraction helper.
- `src/political_scenarios.py` — Scenario definitions and application logic.
- `experiments/registry.json` & `experiments/champion.json` — Experiment registry and selected champion.
- `models/lgbm_all_years_base.txt` — Champion LightGBM model dump (text format).
- `models/artifacts/*` — Feature lists, SHAP summaries, canonical schema JSON.
- `data/canonical/modeling_dataset_canonical_v2.parquet` — The authoritative canonical dataset used for training.

## Research outputs

- Figures and the research summary are generated into `docs/figures/` and `docs/research_summary.md`.
- Regenerate with: `python src/generate_figures.py` (SHAP / model comparison / scenario-by-district figures), `python src/generate_research_summary.py` (the markdown research summary), and `python src/generate_political_figures.py` (political ideology vs price + Figure 2 temporal ideology shift).
- `docs/figures/shap_top10.png` / `shap_top15.png` always render the political model (`lgbm_all_years_political`, the model the paper's Table 3 / Figure 4 reports); `shap_top10_champion.png` / `shap_top15_champion.png` separately render the current champion selected by `experiments/champion.json`, so the two never silently overwrite each other.
- `docs/research_summary.md` carries the political-model SHAP table as its primary (paper-matching) table and the current-champion SHAP table as a clearly-labeled secondary table.

## License & Disclaimer

This repository contains a research-grade forecasting and analysis toolkit. It is not financial or investment advice. Forecasts and scenario outputs should be treated as research outputs and verified independently before operational use. Use at your own risk.

---

If you need a condensed quick-start or CI/CD instructions (Dockerfile, GitHub Actions push, or experiment promotion), open a new issue or request and I will add them to this README and `docs/`.

