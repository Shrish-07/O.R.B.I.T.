# O.R.B.I.T. — Observational Real-estate Behavior & Intelligence Toolkit

O.R.B.I.T. is a reproducible research platform and forecasting toolkit built
to support the paper "Political Ideology and Property Values in New York
City." It integrates public datasets (NYC property sales, PLUTO, mayoral and
council election returns, district geography), canonicalizes and cleans
features, trains the two paper-cited LightGBM models plus five comparison
models (Random Forest, CatBoost, XGBoost, Ridge, ElasticNet), independently
re-verifies every number the paper reports, and exposes an interactive
Streamlit application for single-property exploration, batch scoring, and
political scenario simulation.

**New here?**

- `PAPER_TO_CODE_MAP.md` — go from any table/figure/claim in the paper to the
  exact script that produced it and the exact script that independently
  verifies it.
- `VERIFICATION_REPORT.md` — the independent re-verification record of the
  whole repository: Rounds 1–4 (reorganization & repair, 2026-09-01..08),
  Round 5 (2026-09-13: data-refresh reconciliation + causal-ID build),
  Round 6 (2026-09-14: IV rescue + City of Yes DiD), Round 7 (2026-09-22:
  the `BBL_pluto` blacklist), Round 8 (2026-09-24/25: champion flip to Random
  Forest, duplicate dedup, and the 24/25-feature spec), plus the full
  2026-09-10 data-refresh impact record appended as **Appendix A**.
- "Where the paper stands" below — the condensed submission strategy.
  (The standalone `ACTION_PLAN_FOR_SUBMISSION.md` was folded into this README
  on 2026-09-25; the 2026-09-10 data-refresh report, the architecture /
  data-lineage / feature-dictionary / data-contract docs, and the delivery
  notes were likewise merged into README.md / VERIFICATION_REPORT.md the same
  day — their full texts remain in git history.)

## Quick start

```bash
pip install -r requirements.txt      # Python 3.12+ required (esda==2.10.0
                                     # needs >=3.12); CI and local run 3.13
# populate data/ per "Data sources & provenance" below
python main.py --check-data          # confirm data/ is populated correctly
python main.py                       # run the ENTIRE 53-stage pipeline:
                                     # data build -> training -> analysis ->
                                     # figures -> independent fact-check suite
```

`python main.py --list` shows every stage in order with the paper section it
feeds. `python main.py --help` shows all flags: run a single stage
(`--stage`), resume (`--from`), skip stages (`--skip`), skip heavy training
(`--fast`), run one category (`--category`), or abort on first failure
(`--stop-on-error`; default is to continue).

`main.py` does not duplicate any script's logic — it sequences the existing
scripts, times them, captures each stage's output to
`results/logs/pipeline/<stage_id>.log`, and writes a consolidated
`results/PIPELINE_RUN_REPORT.md` at the end. A genuinely full run (no filter
flags) additionally writes a dated, git-tracked snapshot
(`results/PIPELINE_RUN_REPORT_full_YYYYMMDD.md`) so the always-overwritten
report can't be mistaken for evidence of a full pass.

## System status (verified full run 2026-09-25: 53/53 stages PASS)

- **Canonical dataset:** `data/canonical/modeling_dataset_canonical_v2.parquet`
  — **566,956 rows × 53 columns**, sale years 2016–2025 (built 2026-09-25;
  provenance in `data/DATA_MANIFEST.json`). The build dropped **1,041 exact
  duplicate transactions** vs the 2026-09-10 refresh build (567,997 rows) —
  FIX C13: `rebuild_canonical.py` now ends with `drop_duplicates()`. The raw
  input is the 2026-09-10 sales pull (845,607 rows, 2016–2025).
- **Modeling spec (derived, not hard-coded):** every trainer takes the
  numeric canonical columns minus `config/feature_blacklist.yaml` (and the
  target) at train time, so the spec tracks the data build and the blacklist
  automatically. The current spec is **24 base / 25 political** features
  (`dem_share` is the political delta). The 2026-09-24 rebuild widened the
  previous 18/19 spec by five+ columns because `LAND SQUARE FEET`,
  `GROSS SQUARE FEET`, `lotarea`, `bldgarea`, `unitsres`, `unitstotal` were
  coerced from strings to numeric (they had been silently excluded from every
  model). The exact per-model feature lists regenerate each run at
  `models/artifacts/lgbm_all_years_{base,political}_features.json`.
  (`BBL_pluto` — a sales/PLUTO merge-suffix parcel-identifier artifact — is
  blacklisted; see `VERIFICATION_REPORT.md` Round 7.)
- **Champion model and selection rule:** the champion is whichever run
  `experiments/champion.json` currently points at — that file carries the
  identity, metric, and a human-readable `selection_note`. At last selection
  (2026-09-25) it is **`rf_all_years_political_n200`**
  (`exp-20260925T041001Z`, test **MAE 0.4167477**). Selection is **automated
  with near-tie governance**: `refresh_champion` (`src/ops/experiment_runner.py`)
  picks the lowest test MAE among untainted, non-stale runs created after
  2026-05-18, and when candidates land within 0.001 MAE of each other the
  simpler base-mode spec wins. The current selection recorded a **NEAR-TIE**
  (three political tree ensembles inside 0.001 MAE: rf 0.416748 /
  catboost 0.417364 / xgboost 0.417695) with no base-mode candidate inside
  the epsilon, so the outright best MAE won.
- **Champions may be any model family** (LightGBM `.txt`, sklearn `.joblib`,
  CatBoost `.cbm`, XGBoost `.model`). Every champion-loading script resolves
  the model through `src/ops/model_io.py` (family-agnostic load / predict /
  gain-importance helpers, added 2026-09-24 after the champion flipped from
  LightGBM to the joblib-pickled Random Forest and five pipeline stages that
  assumed `lgb.Booster` crashed). sklearn-family predictions go through
  `fillna(0)` exactly as those models were trained; the tree boosters predict
  with native NaN handling.
- **Manuscript mismatch warning:** the paper text still cites numbers from the
  514,618-row / 16–17-feature build (MAE 0.429, N 470,535, old Table 1–5
  values). The repo has since been through the 2026-09-10 refresh (567,997
  rows; 19/20 features), the Round-7 `BBL_pluto` blacklist (18/19), and the
  2026-09-24/25 dedup + numeric-coercion rebuild (566,956; 24/25). The
  manuscript rewrite is a separate, pending task; the fact-check harness
  intentionally *prints* these mismatches rather than being edited to hide
  them — see "Verification / fact-check suite."
- **App:** Streamlit frontend `app/app.py` with 7 pages (Home, Individual
  Property Analysis, Portfolio Analysis, Political Scenarios, Model Explorer,
  Research Outputs, User Dashboard): NYC GeoSearch address lookup → BBL,
  per-property prediction intervals (from `experiments/predictions/*.parquet`),
  a 5-year price forecast panel (borough-specific appreciation rates),
  per-property risk indicators (overvaluation vs borough median, interval
  width, building age), and the three political-scenario variants
  (`liberal_policy`, `conservative_policy`, `mixed_governance`).

## How the pipeline hangs together

```
data/raw/  -->  data/processed/            -->  data/canonical/          -->  data/splits/
(sales CSV,      ed_results_clean,              modeling_dataset_             all_years_train  = 2016--17 (121,052)
 PLUTO CSV,      ideology_by_council,           canonical_v2                  all_years_test   = 2018     (56,723)
 mayoral ED      council_election_margins,      (566,956 x 53)                year2017_{train,test}
 CSVs,           city_of_yes_roll_calls,                                            |
 council Recap   ed_to_council_crosswalk)                                              v
 pdf/csv,                  ^-- the margins + roll-call builders are run BY HAND,  models/ (7 trainers, champion)
 shapefiles)               not by main.py                                       src/analysis/ (stats, SHAP, scenarios, causal)
                                                                                src/figures/  -> docs/figures/, docs/research_summary.md
                                                                                src/verification/ (independent re-derivation)
```

- The canonical build (`src/data_pipeline/rebuild_canonical.py`) discovers its
  raw inputs by glob and **refuses to run on zero or multiple matches**, so a
  stale duplicate can never be silently preferred; every build records exactly
  which raw files it consumed in `data/DATA_MANIFEST.json`. The build also
  coerces the PLUTO structural numerics (`bldgarea`/`lotarea`/`unitsres`/
  `unitstotal`) and the sales `LAND/GROSS SQUARE FEET` columns to numeric
  (they historically shipped as strings and were silently excluded from every
  model — this is what produced the 24/25-feature spec), and ends with an
  exact-duplicate `drop_duplicates()` (1,041 rows on the current build).
- The ideology feature: `src/data_pipeline/build_ideology_scores.py` parses
  the raw ED mayoral-race CSVs line-by-line and writes
  `data/processed/ideology_by_council.parquet`; `rebuild_canonical.py` merges
  it per `(CounDist, election_year)` as `dem_share` — the **only** ideology
  feature any model trains on. `election_year` maps sale years onto election
  cycles: 2016–18 → 2017, 2019–23 → 2021, 2024–25 → 2025. (A separate,
  abandoned ideology pipeline's columns — `district_ideology`,
  `weighted_dem`, … — are blacklisted in `config/feature_blacklist.yaml`; its
  output parquets remain in `data/processed/` purely as an audit trail, and
  two verifiers explicitly confirm no model or script reads them.)
- The ML split is temporal (`src/data_pipeline/split_temporal.py`,
  `TEST_YEAR=2018`): the 2025-sales refresh did not extend the train/test
  years; any row-count movement since comes from cleaning (exact duplicate
  rows are dropped by the build) and the fixed early-stopping protocol, not
  from more training years. **Current split: 121,052 train (2016–17) /
  56,723 test (2018).**
- Early stopping never touches the test split (FIX A1): trainers carve a
  time-ordered validation slice (last 15% of the training years by
  `sale_date`) and score the held-out 2018 test set exactly once.

- The feature set is not hard-coded anywhere in the wired pipeline: all seven
  trainers select numeric, non-blacklisted, non-target columns from the splits
  at train time, so editing `config/feature_blacklist.yaml` and re-running
  training is the *only* change path to the spec. (The one exception is
  `models/training/train_smoke.py`, a CI smoke test not wired into `main.py`;
  it now also filters the same blacklist — FIX C4.)
- Two derived datasets are **not** built by `main.py` stages; run their
  builders by hand after populating raw inputs (the causal stages fail loudly
  and name the missing file otherwise):

```bash
python src/data_pipeline/build_council_election_margins.py   # -> data/processed/council_election_margins.parquet
python src/data_pipeline/build_city_of_yes_roll_calls.py     # -> data/processed/city_of_yes_roll_calls.parquet
```

## Design decisions

- **Temporal holdout over random split:** training uses a time-based holdout
  to minimize information leakage from future sales; this favors
  generalization to future periods over in-sample accuracy.
- **Blacklist as the single leakage gate:** known leaky columns (PLUTO 2026
  snapshot assessed totals, easements, post-sale identifiers, derived target
  encodings, 100%-null columns) live in exactly one place —
  `config/feature_blacklist.yaml` — and every trainer derives its feature set
  from it. `rebuild_canonical.py`'s separate hard-coded `BAN_COLUMNS` set
  drops PLUTO assessment columns *from the canonical build* — a different
  job; `BBL_pluto` correctly remains a canonical column and is only dropped
  at model feature-selection time.
- **One ideology feature:** `dem_share` is the production ideology signal
  (mayoral two-party vote share, council district × election year, built from
  multiple election years to smooth temporal variance). The abandoned
  `compute_ideology.py → district_ideology` lineage is blacklisted, and the
  two parsers disagree non-trivially at council level (CD1 2017: 0.715908 vs
  0.721132) — recorded as a known methodological confound.
- **`mixed_governance` scenario runs the political model**, not the base
  model: political shifts are modeled with a separate, politically-focused
  training spec, so the dem_share mutation flows only to the model that
  actually trains on it.
- **Experiments are registry-governed:** every trainer appends-or-replaces
  its entry in `experiments/registry.json` (identity = script+name; replace
  on re-run, never accumulate), then calls `refresh_champion` so
  `champion.json` always points at a live entry. `logs/actions.log` is the
  append-only audit trail of selections.
- **Fail-loud data hygiene:** missing inputs raise with the exact filename;
  the sanity check enforces target integrity, sale-year plausibility, and
  ideology coverage up front; raw-file discovery refuses ambiguous matches.

## Repository layout

```
main.py                            # single-entry 53-stage orchestrator
PAPER_TO_CODE_MAP.md               # paper section <-> producer script <-> verifier index
VERIFICATION_REPORT.md             # 8-round independent re-verification record + Appendix A
Dockerfile                         # app-only deploy: Streamlit on :8501 (python:3.13-slim)
requirements.txt                   # pinned env (lightgbm 4.6.0, linearmodels 7.0, spreg 1.9.1, pypdf, ...)
config/                            # config.yaml, feature_blacklist.yaml  (leakage gate)
data/                              # gitignored except DATA_MANIFEST.json
src/
  data_pipeline/                   # raw -> processed -> canonical -> splits; geocoder;
                                   #   by-hand council-margins + City-of-Yes builders
  analysis/                        # Moran's I, OLS + robustness, spatial models, SHAP/PDP,
                                   #   counterfactual scenarios, comparison tables, intervals
  causal/                          # candidate causal-ID stack (close-election IV + rescue + City of Yes DiD)
  figures/                         # everything that writes docs/figures/ + research summary
  verification/                    # independent fact-check suite, one script per paper section
    editorial_audit/               # feature-blacklist enforcement (gates the pipeline)
  ops/                             # experiment_runner (champion), model_io (family-agnostic model IO),
                                   #   sanity/CI checks, registry hygiene (dedup, MLflow export)
app/                               # Streamlit app (app.py, auth.py)
models/                            # training/ scripts; 2 tracked LightGBM text dumps;
                                   # artifacts/ (features/metrics/SHAP) are gitignored, regenerated
experiments/                       # registry.json, champion.json, results/; predictions/ gitignored
results/                           # gitignored except PIPELINE_RUN_REPORT_full_*.md snapshots;
                                   # causal/, logs/, qa evidence, FULL_PAPER_FACT_CHECK.md are local
docs/                              # figures/, research_summary.md (pipeline-generated),
                                   # modeling_schema.csv, feature_manifest.json, model_comparison_table.csv
tables/                            # tracked LaTeX tables regenerated from results/
archive/                           # superseded/dead scripts + archived stale artifacts (history)
legacy/                            # the original pre-reorganization ORBIT (self-contained)
notebooks/                         # 01_validate_raw_data.ipynb (glob-based raw-data spot check)
```

## Data sources & provenance

**Unit of analysis:** one row = one NYC property sale with political context
at time of sale. **Join keys:** sales ↔ PLUTO on BBL (internal `BBL_join`);
ideology merged on `(CounDist, election_year)`; election districts ↔ council
districts via the precomputed `ElectDist` → `CounDist` crosswalk.

`data/` is gitignored (the provenance file `data/DATA_MANIFEST.json` is
tracked). On a fresh clone, `python main.py --check-data` reports which
inputs are missing. Required raw inputs:

- NYC Citywide Annualized Calendar Sales export (DOF/ACRIS; the current build
  consumed `NYC_Citywide_Annualized_Calendar_Sales_Update_20260910.csv`,
  845,607 rows, 2016–2025) → `data/raw/sales/` — keep **exactly one** file
  matching the glob there; old pulls live in `data/raw/sales/_archive/`.
- PLUTO CSV (tax-lot/building attributes) → `data/raw/pluto/` — same
  exact-one-file rule; currently the 20260105 snapshot.
- Election-district mayoral-race CSVs (2017/2021/2025, for `dem_share`) →
  `data/raw/election_results/`.
- Election-district + council-district shapefiles →
  `data/raw/election_districts/` and `data/raw/council_districts/`.
- NYC BOE council general-election Recap files (2013 PDFs; 2017/2021/2025
  CSVs — 204 files, 51 districts × 4 cycles) → `data/raw/council_elections/<year>/`,
  consumed by the by-hand builders above.

**Refresh history** (full before/after tables are in `VERIFICATION_REPORT.md`
Appendix A): the 2026-01-05 snapshot (2016–2024, the 514,618-row build era;
archived at `data/raw/sales/_archive/`) → the **2026-09-10 refresh**
(`w2pb-icbu` pull: 845,607 rows, 2016–2025; 84,693 rows in 2025; canonical
567,997 rows; 19/20-feature spec after admitting the 3 UNITS columns) → the
**2026-09-24/25 rebuild** (exact-duplicate dedup, −1,041 rows → 566,956;
numeric coercion of the string-typed square-footage/area/unit columns → 24/25
features; full 53/53 retrain with the champion flip to Random Forest).

**Provenance record:** `data/DATA_MANIFEST.json` — rewritten on every
canonical build with exactly which raw files were consumed, their row counts,
file mtimes, and the output row count.

**Reconstituting `data/` from the original delivery zips:** `data/raw/` from
`Data 1.zip`'s `raw/` + `data/splits/` from its `splits/`; `data/processed/`
from `Data 2.zip`'s `processed/`; `data/canonical/` from `Data 3.zip`'s
`canonical/`. Then `python main.py --check-data` confirms the layout.

The roll-call builder covers City of Yes for Housing Opportunity (2024-12-05,
31 Yes / 20 No) from a verified reference table; the Carbon Neutrality
(Dec 2023) and Economic Opportunity (June 2024) member-level votes were not
programmatically retrievable (Legistar API 403 without a token, JS-only HTML
search) and are flagged in the script's docstring for manual retrieval.

## The modeling feature set

Derived, not hard-coded: every trainer selects the numeric canonical columns
minus `config/feature_blacklist.yaml` minus the target at train time. Current
spec (regenerated each run at `models/artifacts/*_features.json`; dtypes from
`docs/modeling_schema.csv`):

| # | Feature | Dtype | What it is |
|---|---------|-------|------------|
| 1 | `BOROUGH` | int64 | NYC borough code 1–5 at time of sale |
| 2 | `ZIP CODE` | float64 | sales ZIP |
| 3 | `RESIDENTIAL UNITS` | float64 | sales-record unit count |
| 4 | `COMMERCIAL UNITS` | float64 | sales-record unit count |
| 5 | `TOTAL UNITS` | float64 | sales-record unit count |
| 6 | `LAND SQUARE FEET` | float64 | lot size (numeric since 2026-09-24) |
| 7 | `GROSS SQUARE FEET` | float64 | building size (numeric since 2026-09-24) |
| 8 | `YEAR BUILT` | float64 | sales-record year built |
| 9 | `TAX CLASS AT TIME OF SALE` | int64 | NYC tax class |
| 10 | `Community Board` | float64 | NYC community board id |
| 11 | `Council District` | float64 | council district (sales record) |
| 12 | `Census Tract 2020` | float64 | 2020 census tract id |
| 13 | `landuse` | float64 | PLUTO land-use code |
| 14 | `lotarea` | float64 | PLUTO lot area (numeric since 2026-09-24) |
| 15 | `bldgarea` | float64 | PLUTO building area (numeric since 2026-09-24) |
| 16 | `numfloors` | float64 | PLUTO number of floors |
| 17 | `unitsres` | float64 | PLUTO residential units (numeric since 2026-09-24) |
| 18 | `unitstotal` | float64 | PLUTO total units (numeric since 2026-09-24) |
| 19 | `yearbuilt` | float64 | PLUTO year built |
| 20 | `residfar` / `commfar` / `facilfar` | float64 | PLUTO floor-area ratios |
| 21 | `election_year` | int64 | mapped election cycle (2017/2021/2025) |
| 22 | `CounDist` | float64 | council district id (lowercase form) |
| +1 | `dem_share` | float64 | **political model only** — Democratic two-party mayoral vote share at (district, election year) |

(The three PLUTO FAR columns (`residfar` / `commfar` / `facilfar`) share one
row above; counting each individually, the full base list is 24 columns,
political = base + `dem_share` = 25.)

**Blacklist** (`config/feature_blacklist.yaml`, 50 entries, case-insensitive
match): the target and its direct variants (`target_log_price`, `sale price`,
…); PLUTO assessment leakage (`assesstot`, `exempttot`, `assessland`,
`appbbl`, `taxmap`, `plutomapid` — 2026-snapshot values encode historical
appreciation); temporal identifiers (split-only); administrative date fields;
post-sale row keys (`bin`, `bbl`, `lot`, `block`, `taxlot`, …) plus
`BBL_pluto` (the merge-suffix artifact, added Round 7); raw geo coordinates;
raw election columns (`district_ideology`, `weighted_dem`, … — the abandoned
pipeline); derived target encodings; the 100%-null `EASE-MENT`. Enforcement:
`src/verification/editorial_audit/verify_blacklist_fully.py` **exits 1** if
either paper model's trained feature list contains a blacklisted column and
writes `results/editorial_audit/blacklist_audit.json` every run.

## Models & experiment governance

Seven models trained in the last full run (2026-09-25; all comparison models
are political-mode; LightGBM also runs base). Current test metrics — 121,052
train (2016–17) / 56,723 test (2018) rows; 24 base / 25 political features:

| Model | Binary (git status) | Test MAE | Test R² | Features |
|---|---|---|---|---|
| **Random Forest (n=200) — champion** | `models/rf_all_years_political_n200.joblib` (gitignored, ~1.0 GB) | **0.41675** | 0.53017 | 25 |
| CatBoost (lr .05, d 6) | `.cbm` (gitignored) | 0.41736 | 0.58128 | 25 |
| XGBoost (lr .05, md 6) | `.model` (gitignored) | 0.41770 | 0.57020 | 25 |
| LightGBM political (paper's Table 2/3 model) | `models/lgbm_all_years_political.txt` (tracked) | 0.43994 | 0.54336 | 25 |
| LightGBM base | `models/lgbm_all_years_base.txt` (tracked) | 0.44183 | 0.53504 | 24 |
| Ridge (RidgeCV) | `.joblib` (gitignored) | 0.62934 | 0.17915 | 25 |
| ElasticNet (ElasticNetCV) | `.joblib` (gitignored) | 0.63127 | 0.17708 | 25 |

- Only the two LightGBM text dumps are tracked; other binaries regenerate
  from training (the RF joblib is unusually large for a joblib — ~1 GB of
  leaf nodes — regenerate with `python main.py --stage train_random_forest`).
  Feature lists / metrics / SHAP summaries live in `models/artifacts/`
  (gitignored) and `experiments/results/*.json` (tracked);
  `docs/model_comparison_table.csv` is regenerated by the
  `generate_research_summary` stage.
- Every trainer **replaces** its `experiments/registry.json` entry in place
  (same script+name identity) rather than duplicating, then calls
  `src/ops/experiment_runner.py` so `champion.json` never goes stale mid-run;
  the registry currently holds exactly the 7 latest runs (one per model).
- The linear baselines use cross-validated regularization (`RidgeCV` on a
  `logspace(-4, 7, 40)` grid / `ElasticNetCV`, chosen α/l1_ratio recorded in
  results + registry) because default-alpha Ridge collapsed (R² → −4.97) on
  the refreshed near-collinear UNITS columns — see `VERIFICATION_REPORT.md`
  Round 7.
- **Champion-SHAP caveat (2026-09-24):** for the current Random-Forest
  champion, `models/artifacts/rf_all_years_political_n200_shap_summary.json`
  is a **gain-importance fallback**, not TreeSHAP — TreeSHAP is intractable on
  this repo's unpruned forest (observed >40 CPU-min at n=2000) and
  `generate_shap_summary.py` skips straight to labeled
  `method: feature_importance_fallback` output. The paper-cited
  `lgbm_all_years_*` SHAP summaries are true TreeSHAP and are regenerated on
  every full run regardless of which model is champion (FIX B1).
- Leakage: known leaky/identifier columns (assessed totals, `EASE-MENT`,
  the `BBL_pluto` merge-suffix artifact, post-sale identifiers, abandoned
  ideology columns, …) are blacklisted in `config/feature_blacklist.yaml`
  and enforced by `src/verification/editorial_audit/verify_blacklist_fully.py`,
  which **fails the pipeline** (exit 1) if either paper model contains one.

## Analysis suite (`src/analysis/`)

- `compute_morans_i.py` — global Moran's I for price and dem_share over the 51
  council districts from the official Council District shapefile's polygon
  centroids (k=5 row-standardized KNN, 999 permutations, seed 42) →
  `results/morans_i_results.json`, Section 4.2. Current build: I = 0.5942
  (price) / 0.5705 (dem_share), both p = 0.001 (paper: 0.5959 / 0.5797; the
  script prints a live discrepancy flag when its values drift from the paper).
- `generate_regression_table.py` — three nested OLS specifications (bivariate
  → + structural controls → + borough FE) + full Model-3 residual diagnostics
  → `results/regression_table.csv`, `regression_diagnostics.json`,
  `regression_model3_full_summary.txt`; Table 1 / Appendix A.2. Current
  build: N = 520,365; dem_share 0.9449 → 0.6146 → 0.2883 (SE 0.0087).
- `generate_robustness_checks.py` — district-clustered SEs, 51-district FE,
  borough + sale-year FE, winsorized and Cook's-D-trimmed re-estimates →
  `results/robustness_checks.json` + tracked `tables/table_robustness.tex`;
  Table 4. Current build: Model 3 clustered-by-CounDist p = 0.0936; Model 4
  51-district-FE dem_share 0.7764 (clustered p ≈ 4.1e-17); Model 5
  borough+year-FE 0.0667 (clustered p 0.7427); winsorized 0.2860; Cook's-D
  trim (28,833 rows) 0.2288.
- `spatial_regression_model.py` — district-level (N=51) spatial lag / spatial
  error MLE models (spreg), identical weights construction to Moran's I →
  `results/spatial_regression.json` + tracked `tables/table_spatial_regression.tex`;
  Table 5. Current build: ρ = 0.7029 (SLM), λ = 0.7809 (SEM), dem_share
  insignificant in all three (p 0.327 / 0.143 / 0.323), mean_commfar
  significant throughout.
- `political_scenarios.py` — the three counterfactual simulations
  (`liberal_policy`: R3→R7 zoning substring + residfar ×1.5;
  `conservative_policy`: R7→R3 + residfar ×0.7; `mixed_governance`:
  dem_share + 0.15 clipped to [0,1] through the political model, with an
  on-disk extrapolation-support diagnostic) → per-district CSVs and
  all-properties parquets under `results/political_scenarios/`; Section 4.7 /
  Appendix A.1.
- `generate_shap_summary.py` — mean |SHAP| for the champion **and** explicit
  passes for both paper models (`lgbm_all_years_{base,political}`) on every
  run, no skipping (FIX B1); falls back to a labeled gain-importance payload
  when TreeSHAP is unavailable/intractable.
- `generate_pdp.py`, `confidence_intervals.py`, `sensitivity_analysis.py`,
  `champion_confidence_and_sensitivity.py` — champion-model diagnostics that
  are **not currently cited in the manuscript** (candidate additions). These
  stages also regenerate the two paper models' PDP / interval / sensitivity
  artifacts every run; row-count assertions keep the prediction artifacts at
  full test-split coverage (the old 19-row export gap cannot recur).
- `compare_models.py` — reconciles the two result-JSON schemas and tabulates
  every `experiments/results/*.json` entry (Table 2 cross-check —
  `model_comparison_raw.csv` + `ideology_ablation_summary.csv`).

## Causal identification (candidate design — not yet in the manuscript)

Wired into `main.py` as four analysis + three verification stages. These
stages **exit 0 even when an identification verdict fails**: a thin
instrument or a null effect is a reported finding, not a pipeline crash.

- `src/causal/build_close_election_instrument.py` →
  `results/causal/close_election_coverage.json`: coverage diagnostic run
  *before* any instrument is fitted — of the 204 district-year council races
  (2013/2017/2021/2025), close races (margin ≤ bandwidth) are ±5pp: 4,
  ±10pp: 7, ±15pp: 12, ±20pp: 17 (4 / 6 / 9 / 13 usable as prior-cycle
  instruments; 2025 races have no downstream canonical year). A
  plain-language thin-coverage warning is printed and stored.
- `src/causal/iv_district_fe_regression.py` → `iv_district_fe_results.json`:
  2SLS (`linearmodels.iv.IV2SLS`) instrumenting `dem_share` with a prior-cycle
  close-Democratic-win indicator, district + sale-year FE, Model 4 controls +
  the UNITS columns. **Verdict: not identified at any of ±5/10/15/20pp** —
  the instrument lies exactly in the fixed-effects column space given only
  4–13 close cells (`election_year` is a deterministic grouping of
  `sale_year`, so district+year FE span the instrument).
- `src/causal/iv_district_fe_rescue.py` → `iv_district_fe_rescue_results.json`:
  the full rescue trail — a borough-FE variant (identifies at ±15/20pp but
  first-stage F = 1.04 / 0.59, with sign-unstable, implausibly scaled
  second-stage coefficients −14.7 / +18.3 vs the +0.78 non-instrumented
  Model-4 benchmark) and a wider-bandwidth district-FE sweep (identifies from
  ±25pp; all first-stage F < 10: 2.44 / 7.40 / 5.13 / 7.30; second-stage
  point estimates inconsistent in sign). Honest read: **no close-council-race
  instrument on this dataset currently supports a causal claim** — present as
  attempted-and-not-identified, or find an office with many close races.
- `src/causal/city_of_yes_did.py` → `city_of_yes_did_results.json`: an
  ideology/preference-proxy DiD around the 2024-12-05 Housing Opportunity
  vote. Because that action is a city-wide text amendment, the Yes/No split
  measures a district's **revealed development stance, not the policy's
  effect** — the caveat is written into every output. Current build:
  pre-trend event study **passes** (joint Wald p = 0.2209, n = 355,480);
  main DiD +0.0793 (p = 0.1318, n = 333,670; 2024 transition year excluded);
  sensitivity +0.0740 (p = 0.1383) — a suggestive null. Do not cite as a
  policy-application estimate.
- Each analysis has an independent verifier (`verify_iv_district_fe.py`,
  `verify_iv_district_fe_rescue.py`, `verify_city_of_yes_did.py`) that
  re-derives everything from the raw parquets *without importing the analysis
  script* and compares to 4 decimal places; all three currently report
  full agreement.

## Verification / fact-check suite

`src/verification/` holds 19 scripts — one per paper section — that
**independently re-derive** the paper's numbers from raw artifacts (not from
pipeline outputs where avoidable) and diff them against the published values:

| Verifier | Section covered |
|---|---|
| `verify_abstract_and_intro.py` | Abstract, Section 1 |
| `verify_ideology_construction.py` | 3.2 Ideology construction (dem_share) |
| `verify_cd1_dem_share.py` | 3.2 footnote worked example (CD1 2017) |
| `verify_canonical_dataset_and_split.py` | 3.3 Canonical dataset, 3.4 split |
| `verify_spatial_pattern.py` | 4.1 spatial descriptive claims |
| `verify_regression_missingness.py` / `verify_numfloors_missingness.py` | 4.3 missingness paragraph |
| `verify_regression_ols.py` | 4.3, Table 1, Appendix A.2 |
| `verify_spatial_regression.py` | 4.5, Table 5 |
| `verify_ml_performance.py` | 4.6.1, Table 2 |
| `verify_shap_importance.py` | 4.6.2, Table 3 (18-row comparison file) |
| `verify_scenario_mechanics.py` | 4.7 mechanics + support diagnostic |
| `verify_appendix_a1_scenarios.py` | Appendix A.1 (153-value diff) |
| `verify_borough_map_and_crosscutting.py` | Appendix A.3 + blacklist/PNG/tex cross-cut |
| `verify_followup_resolution.py` | Round-5 follow-up findings |
| `editorial_audit/verify_blacklist_fully.py` | Enforcement gate (exits 1 on failure) |
| `verify_iv_district_fe.py` / `verify_iv_district_fe_rescue.py` / `verify_city_of_yes_did.py` | the causal-ID stages |

Together they re-verify every Abstract/Intro headline number, the CD1 worked
example, canonical row/column counts (vs frozen `models/artifacts/
canonical_schema.json` + `docs/feature_manifest.json`), every Table 1
coefficient, the Moran's I quartet, the Table 2 MAE/R² seven-row match,
all 51-row Appendix A.1 percentages, the A.2 diagnostics, and the causal-ID
artifacts.

Two things to know about this suite:

1. **They are mismatch detectors, not enforcement gates**: they exit 0 and
   print/write `results/round5_*.json` files even when values drift from the
   paper's numbers (by design — those reference values must not be edited to
   force matches). On the current build, every verifier runs clean; several
   intentionally show the paper-vs-current drift, so treat their prints as
   diagnosis rather than failure signals.
2. **`verify_blacklist_fully.py` is the one structural gate**: it exits 1
   (fails the pipeline) if either champion model's trained feature list
   contains a blacklisted column, and writes the machine-readable
   `results/editorial_audit/blacklist_audit.json` every run (currently:
   50 blacklist entries; champion + political models both clean;
   `enforcement_passed: true`).

The human-reviewed synthesis of a full verification pass is
`results/FULL_PAPER_FACT_CHECK.md` (local artifact — `results/` is gitignored
except the dated `PIPELINE_RUN_REPORT_full_*.md` snapshots; that file is a
frozen Round-5 record written against the 514,618-row build).

## The app, CI, and deployment

```bash
python -m streamlit run app/app.py --server.headless true --server.port 8501
# -> http://localhost:8501
```

`app/auth.py` provides local accounts (SQLite + bcrypt at `data/users.db`;
app-only, not part of the research pipeline). Address lookup goes through
`src/data_pipeline/geocoder.py` (NYC GeoSearch → BBL) with a borough-default
slider fallback when a BBL is not in the test split. The app resolves the
champion from `experiments/champion.json` unconditionally and renders the
registry's untainted, non-stale runs in Model Explorer.

- **CI** (`.github/workflows/ci.yml`, push/PR to main/master): Python 3.13,
  `pip install -r requirements.txt`, then `src/ops/ci_smoke_check.py`; a
  second data-conditional job runs `pipeline_sanity_check.py`,
  `split_temporal.py`, and the blacklist-filtered `train_smoke.py` when the
  canonical parquet is present (it is gitignored and absent in CI).
- **Dockerfile**: app-only deploy (`python:3.13-slim`, Streamlit on :8501;
  bumped from 3.11 because esda 2.10.0 requires Python ≥ 3.12).
- `archive/` holds superseded scripts and historical result snapshots
  (referenced by the verification report); `legacy/` is the original
  pre-reorganization ORBIT, self-contained.

## Where the paper stands (submission strategy, condensed)

*Condensed 2026-09-25 from the merged `ACTION_PLAN_FOR_SUBMISSION.md`; the
full text remains in git history (`git show d0cdd12:ACTION_PLAN_FOR_SUBMISSION.md`).*

**Competitive positioning** (vs. all 9 unique accepted papers in the CER 2025
journal pool):

| | This paper | Accepted pool (n=9) |
|---|---|---|
| Length | 6,210 words, 34 pages | 9,290–22,944 words (avg ~13,000) |
| Tables | 12 | typically 4–9, embedded in longer text |
| Figures | 6 | comparable |
| Explicit "Robustness" language | 0 mentions (pre-refresh draft) | present in most; several have a full subsection |
| Causal identification | none (stated as the paper's own #1 limitation) | DiD, panel FE, IV in most of the pool |
| Independent numerical fact-check of every claim | yes — the whole `src/verification/` suite | not observed in any accepted paper |

**The honest read:** the paper is unusually well-audited but shorter and more
purely associational than the accepted pool; "no causal identification" was
already its own stated top limitation, and that is the axis reviewers grade
hardest. The repo-side work to close that gap has been built and run —
cluster-robustness section (§4.4), formal spatial regression (§4.5), and the
close-election IV + City of Yes DiD stack — **with negative or null results
throughout on the causal side** (see "Causal identification"): the reviewer
hardening stands on §4.4/§4.5 robustness, not on an identified IV.

**What remains, in priority order:**
1. **Rewrite the manuscript to the refreshed numbers** (the single biggest
   outstanding item): every pipeline/regenerated number — canonical counts,
   split, 24/25 features, Table 1–5, Moran's I, SHAP, appendix — is computed
   and verified in this repo; the manuscript text itself was never edited and
   is now two rebuilds behind.
2. Fold in the pending manuscripts patches the verifiers drafted: numfloors
   missingness 11,918 → **11,930** (paper's Section 4.3 value is 12 rows
   short); Moran's I z-convention (paper 7.58/7.69 vs recomputed 7.46/7.99 —
   coefficients match; reconcile which z the paper quotes); the missingness
   four-component overlap note (53 rows double-counted; union = 44,083
   exactly).
3. Present the causal work as honest negative results (IV not identified at
   any bandwidth/configuration; City of Yes DiD null under the
   preference-proxy framing) or source an instrument from an office with many
   close races.
4. Retrieve the two missing City of Yes roll calls manually from Legistar
   (Carbon Neutrality Dec 2023, Economic Opportunity June 2024) if a real
   policy-application DiD is wanted.
5. Decide on the unused-but-computed assets (PDP panels, prediction intervals,
   sensitivity analyses — regenerate-clean under the 24/25 spec) as paper
   appendices or app-only.

## Documentation index

| File | Purpose |
|---|---|
| `PAPER_TO_CODE_MAP.md` | paper section ↔ producer script ↔ verifier index |
| `VERIFICATION_REPORT.md` | 8-round independent re-verification record + Appendix A (2026-09-10 data-refresh impact) |
| `docs/research_summary.md` | pipeline-generated champion/comparison/SHAP digest (rewritten every full run; never edit by hand) |
| `archive/legacy_results_pre_canonical_v2/README.md` | what the archived pre-canonical-v2 result files are |
| `legacy/README.md` | the original pre-reorganization ORBIT snapshot |
| `results/FULL_PAPER_FACT_CHECK.md` | frozen Round-5 human-reviewed fact-check record (local, gitignored) |

## Known caveats

- **The manuscript is two rebuilds behind.** All paper-cited numbers (Tables
  1–5, Moran's I, Appendix A.1, 16/17 feature counts) predate both the
  2026-09-10 refresh and the 2026-09-24/25 rebuild; regeneration is done,
  rewriting is pending (top priority above).
- The council-district shapefile still describes **pre-2023 boundaries** —
  all CounDist-level results (Moran's I, Table 5, Appendix A.1) inherit that
  (VERIFICATION_REPORT.md Appendix A §12).
- **Champion MAE margin is thin:** the top three political tree ensembles sit
  within 0.001 MAE (rf 0.416748 / catboost 0.417364 / xgboost 0.417695) —
  champion identity may flip on future retrains; nothing preserves it by hand,
  and `model_io.py` exists precisely so any family can serve as champion.
- For the current Random-Forest champion, its "SHAP summary" artifact is a
  **labeled gain-importance fallback**, not TreeSHAP (intractable on the
  unpruned forest); the paper-cited LightGBM SHAP summaries remain true
  TreeSHAP — do not conflate them.
- The verification suite intentionally prints paper-era constants (from the
  514,618-row build) as references — seeing a mismatch printed is not a
  pipeline failure; it is the point of the harness.
- CPU budget: `verify_scenario_mechanics` (many full-dataset predicts) and
  `sensitivity_analysis` each take several minutes on a typical laptop; this
  is expected, not a bug.

## License & Disclaimer

This repository contains a research-grade forecasting and analysis toolkit.
It is not financial or investment advice. Forecasts and scenario outputs
should be treated as research outputs and verified independently before
operational use. Use at your own risk.
