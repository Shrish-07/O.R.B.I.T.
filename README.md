# O.R.B.I.T. — Observational Real-estate Behavior & Intelligence Toolkit

O.R.B.I.T. is a reproducible research platform and forecasting toolkit built
to support "Political Ideology and Property Values in New York City." It
integrates public datasets (PLUTO, PAD, sales, election returns),
canonicalizes and cleans features, trains LightGBM and comparison models,
independently re-verifies every number the paper reports, and exposes an
interactive Streamlit application for single-property exploration, batch
scoring, and political scenario simulation.

**New here?** Read `PAPER_TO_CODE_MAP.md` to go from any table/figure/claim
in the paper to the exact script that produced it and the exact script that
independently verifies it. Read `ACTION_PLAN_FOR_SUBMISSION.md` for a
prioritized list of what to strengthen before submission, including a
methodological finding already run against the real data.

## Quick start

```bash
pip install -r requirements.txt
# populate data/ per "Data sources" below and docs/DATA_LINEAGE.md
python main.py --check-data      # confirm data/ is populated correctly
python main.py                   # run the ENTIRE pipeline end to end:
                                  # data build -> all model training ->
                                  # all analysis -> all figures -> the full
                                  # independent fact-check suite
```

`python main.py --list` shows every stage in order, with the paper section
it feeds. `python main.py --help` shows all flags (run a single stage,
resume from a stage, skip heavy training, stop on first failure, etc).
`main.py` does not duplicate any script's logic — it sequences the existing
scripts described below, times each one, and writes a consolidated
`results/PIPELINE_RUN_REPORT.md` at the end.

## System status

- Champion model: `lgbm_all_years_base` (LightGBM). Empirical test MAE (log-price) &asymp; 0.429.
- Champion selection is **automated, not manual**: every training script appends its run (id, metrics, feature count, artifact paths) to `experiments/registry.json`, and a full `python main.py` run finishes its training phase with the `refresh_champion` stage (`src/ops/experiment_runner.py`), which re-selects the champion (lowest MAE among untainted, post-2026-05-18 runs) and rewrites `experiments/champion.json` with a fresh `selected_utc`. Note: running a single training stage via `--stage` does **not** refresh the champion by itself — add `refresh_champion` explicitly or run the full pipeline.
- Available features: canonical modeling table at `data/canonical/modeling_dataset_canonical_v2.parquet` with the frozen schema recorded in `models/artifacts/canonical_schema.json`.
- App: Streamlit frontend `app/app.py` with 7 pages (Home, Individual Property Analysis, Portfolio Analysis, Political Scenarios, Model Explorer, Research Outputs, User Dashboard).
- Functionality: address lookup (free-text addresses &rarr; BBL via NYC GeoSearch), prediction intervals (per-property lower/upper bounds loaded from `experiments/predictions/*.parquet`), and a 5-year price forecast panel (borough-specific appreciation rates).
- Risk indicators: per-property risk panel reporting overvaluation vs borough median, prediction uncertainty (interval width), and building age risk.
- Political scenarios: three scenario variants (`liberal_policy`, `conservative_policy`, `mixed_governance`), saved under `results/political_scenarios/`.

## Repository layout

```
main.py                          # single-command pipeline orchestrator (see Quick start)
PAPER_TO_CODE_MAP.md             # paper section <-> code file index
ACTION_PLAN_FOR_SUBMISSION.md    # prioritized next steps for this year's paper
config/                          # config.yaml, feature_blacklist.yaml
data/                            # gitignored; see "Data sources" below
src/
  data_pipeline/                 # raw sources -> canonical dataset
  analysis/                      # Moran's I, OLS regression, SHAP, scenarios, robustness checks
  figures/                       # everything that writes to docs/figures/
  verification/                  # independent fact-check suite, one script per paper section
    editorial_audit/             # feature-blacklist compliance audit
  causal/                        # causal_analysis.py — unfinished, see ACTION_PLAN Priority 1
  ops/                           # training automation, CI smoke checks, preprocessing utilities
app/                             # Streamlit app + auth.py
models/                          # training/ (scripts) + artifacts/ + trained model binaries
experiments/                     # registry.json, champion.json, results/, predictions/
results/                         # every generated report/JSON, including FULL_PAPER_FACT_CHECK.md
docs/                            # ARCHITECTURE.md, DATA_LINEAGE.md, FEATURE_DICTIONARY.md, figures/
archive/                         # superseded/dead scripts, kept for history, not run by main.py
```

## Running the app directly

```bash
python -m streamlit run app/app.py --server.headless true --server.port 8501
```

Open `http://localhost:8501`.

## Rebuilding individual pipeline stages

`python main.py` runs everything; to run one piece by hand (the underlying
commands `main.py` itself calls), from the project root:

```bash
python src/data_pipeline/rebuild_canonical.py
python src/data_pipeline/split_temporal.py
python models/training/train_lgbm.py --variant all_years --mode base
python models/training/train_lgbm.py --variant all_years --mode political
python src/analysis/political_scenarios.py
python src/analysis/compute_morans_i.py
python src/analysis/generate_regression_table.py
python src/analysis/generate_robustness_checks.py
python src/figures/generate_figures.py
python src/figures/generate_research_summary.py
python src/figures/generate_political_figures.py
```

Notes:
- Each script writes versioned artifacts under `data/`, `data/canonical/`, `data/splits/`, `experiments/`, `models/artifacts/`, and `results/`.
- `src/analysis/compute_morans_i.py` produces `results/morans_i_results.json` (paper Section 4.2) using the official NYC City Council District shapefile polygon centroids.
- `src/analysis/generate_regression_table.py` produces the OLS coefficients table AND the Appendix A.2 / Section 4.3 residual diagnostics (`results/regression_diagnostics.json`, `results/regression_model3_full_summary.txt`).
- `src/analysis/generate_robustness_checks.py` is new (added during this reorganization) — see `ACTION_PLAN_FOR_SUBMISSION.md` Priority 0 for what it found.
- Training scripts accept configuration overrides through experiment entries in `experiments/registry.json`.

## Data sources (what belongs in `data/raw/`)

- NYC sales CSV exports (DOF / ACRIS sales records) &rarr; `data/raw/sales/`
- PLUTO CSV (tax lot attributes and building characteristics) &rarr; `data/raw/pluto/`
- Election result CSVs (used to compute ideology / `dem_share`) &rarr; `data/raw/election_results/`
- Shapefiles for council / election district boundaries &rarr; `data/raw/election_districts/`, `data/raw/council_districts/`

Populate `data/raw/` with the above sources before running the data-pipeline
stages. Full raw &rarr; processed &rarr; canonical provenance is in
`docs/DATA_LINEAGE.md`.

## Model summary

- Champion model: LightGBM (`models/lgbm_all_years_base.txt`, or the path recorded in `experiments/champion.json`).
- Performance: test MAE (log-price) &asymp; 0.429 against the canonical test split.
- Features: full feature lists in `models/artifacts/` (e.g., `lgbm_all_years_base_features.json`). Frozen canonical schema at `models/artifacts/canonical_schema.json`.
- Leakage: known leaky variables (e.g., assessed totals like `assesstot`, easement indicators such as `EASE-MENT`) are explicitly blacklisted in `config/feature_blacklist.yaml` and audited by `src/verification/editorial_audit/verify_blacklist_fully.py`.

## Political modeling

- Three scenario families, implemented in `src/analysis/political_scenarios.py`:
  - `liberal_policy` — simulates policy shifts that generally increase values in certain council districts.
  - `conservative_policy` — simulates more restrictive policy outcomes.
  - `mixed_governance` — uses the political model and scenario-specific adjustments for `dem_share` and related features to reflect governance-driven changes.
- `dem_share` is computed by `src/data_pipeline/build_ideology_scores.py`, which writes `data/processed/ideology_by_council.parquet` (a line-by-line parser of the raw ED `ed_results_{year}_mayor.csv` election files, aggregated to council+year). `src/data_pipeline/rebuild_canonical.py` merges this per `(CounDist, election_year)` into the canonical dataset — this is the production `dem_share` feature the political model trains on, and the *only* ideology feature that reaches any model.
- A separate, already-abandoned ideology pipeline (`district_ideology`, `weighted_dem`, `weighted_rep`, `dem_vote_share`, `rep_vote_share`, `turnout`) is **blacklisted** in `config/feature_blacklist.yaml`. It feeds no model. The scripts that once produced it are no longer in the repository; only historical intermediate parquet files remain, purely as an audit trail — see `docs/DATA_LINEAGE.md`.
- `mixed_governance` selectively applies the political model where political features have higher predictive power, while the base price model handles structural and market signals.

## Address lookup

`src/data_pipeline/geocoder.py` queries the NYC GeoSearch service
(`https://geosearch.planninglabs.nyc/v2/search`, query param `text`),
extracts the PAD/BBL when available (`properties.addendum.pad.bbl`), and
matches it to the canonical/test split. If a BBL isn't present in the test
split, the app falls back to a slider-based estimator built from borough
defaults.

## Key files

- `app/app.py` — Streamlit app entrypoint and UI logic (address lookup, predictions, SHAP, scenarios).
- `app/auth.py` — local user accounts for the app (not part of the research pipeline).
- `src/data_pipeline/geocoder.py` — GeoSearch client and BBL extraction helper.
- `src/analysis/political_scenarios.py` — Scenario definitions and application logic.
- `experiments/registry.json` & `experiments/champion.json` — Experiment registry and selected champion.
- `models/lgbm_all_years_base.txt` — Champion LightGBM model dump (text format).
- `models/artifacts/*` — Feature lists, SHAP summaries, canonical schema JSON.
- `data/canonical/modeling_dataset_canonical_v2.parquet` — The authoritative canonical dataset used for training.

## Research outputs

- Figures and the research summary are generated into `docs/figures/` and `docs/research_summary.md`.
- Regenerate with `python src/figures/generate_figures.py` (SHAP / model comparison / scenario-by-district figures), `python src/figures/generate_research_summary.py` (the markdown research summary), and `python src/figures/generate_political_figures.py` (political ideology vs. price + temporal ideology shift).
- `docs/figures/shap_top10.png` / `shap_top15.png` always render the political model (`lgbm_all_years_political`, the model the paper's Table 3 reports); `shap_top10_champion.png` / `shap_top15_champion.png` separately render the current champion selected by `experiments/champion.json`, so the two never silently overwrite each other.
- `docs/research_summary.md` carries the political-model SHAP table as its primary (paper-matching) table and the current-champion SHAP table as a clearly-labeled secondary table.

## Verification / fact-check suite

`src/verification/` independently re-derives every numeric claim in the
paper directly from the data and model artifacts, one script per paper
section (see `PAPER_TO_CODE_MAP.md` for the full index). The
already-finalized, human-reviewed synthesis of every check is
`results/FULL_PAPER_FACT_CHECK.md`. Run the whole suite with:

```bash
python main.py --category verification
```

## License & Disclaimer

This repository contains a research-grade forecasting and analysis toolkit.
It is not financial or investment advice. Forecasts and scenario outputs
should be treated as research outputs and verified independently before
operational use. Use at your own risk.
