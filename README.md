# Political Ideology and Property Values in New York City — the O.R.B.I.T. reproducible pipeline

This repository is the single-command pipeline that reproduces the paper
"Political Ideology and Property Values in New York City". The O.R.B.I.T.
name is kept for identity. The repository is a single command
research pipeline that reproduces every number, table and figure in the
paper, and then re-derives all of them independently.

- **One command, 56 stages, about 100 minutes.** `python main.py` builds the
  canonical dataset from public NYC sources (845,607 property sales for 2016
  to 2025, merged with PLUTO and Board of Elections returns, giving 566,956
  canonical rows), trains all 7 registered model experiments (the champion is
  the complexity constrained, validation tuned Random Forest with test MAE
  0.4016, alongside LightGBM base and political, CatBoost, XGBoost, Ridge and
  ElasticNet), runs the full analysis suite (Moran's I, nested OLS,
  robustness, district level spatial regression, exact SHAP for every model,
  PDPs, counterfactual scenarios, close election IV and the City of Yes
  DiD), regenerates every manuscript table and figure from repo artifacts in
  `docs/paper_assets/`, and re-verifies everything. Each run records
  `git_commit` and `git_dirty` provenance in the registry, in
  `experiments/champion.json` and in the run report.
- **Verification is machine-countable and enforced.** 19 independent
  verifiers each write a per check status file. The final
  `verification_summary` stage aggregates them and exits non zero if any
  verifier reports a mismatch, is missing, or carries a stale status file.
  The last verification gate (Round 13, 2026-10-19) reported 840 checks,
  0 mismatches and 26 of 26 verifiers reporting. The audit history is in
  `VERIFICATION_REPORT.md` (Rounds 1 to 11c, Round 13 appended).
- **Null causal results are reported as findings.** The close election IV
  instrument is not identified at any bandwidth and the City of Yes DiD is a
  null under the preference proxy framing. Every attempted specification is
  tabulated.
- **The small paper relevant artifacts are tracked in git** (results JSONs,
  per model scenario tables, SHAP and feature manifests, the dated full run
  report), so the reported numbers can be inspected without a re-run.
- A Streamlit app in `app/` covers single property exploration, batch
  scoring and scenario simulation on top of the trained models.

**New here?**

- `PAPER_TO_CODE_MAP.md` maps any table, figure or claim in the paper to the
  exact script that produced it and the exact script that independently
  verifies it.
- `VERIFICATION_REPORT.md` is the re verification record of the whole
  repository. It covers Rounds 1 to 4 (reorganization and repair,
  2026-09-01 to 08), Round 5 (2026-09-13, data refresh reconciliation and
  causal ID build), Round 6 (2026-09-14, IV rescue and City of Yes DiD),
  Round 7 (2026-09-22, the `BBL_pluto` blacklist), Round 8 (2026-09-24/25,
  champion flip to Random Forest, duplicate dedup and the 24/25 feature
  spec), Round 9 (2026-09-27, full model interpretation coverage), Round 10
  (2026-09-28 to 30, tuned complexity constrained Random Forest, SHAP audit
  fixes and startup watchdog, paper asset generation, refreshed verifier
  targets), Round 11 (2026-10-10, paper asset LaTeX escape and layout repairs
  with a standalone compile check, machine countable verifier status and the
  verification_summary gate, git_commit and git_dirty provenance, repo
  relative POSIX paths, PDP pruning and a stale artifact audit, gitignore
  whitelisting of paper relevant artifacts), Round 11b (2026-10-12, stale
  verifier targets corrected, with the 5dp delta tolerance set at its exact
  rounding bound, yearbuilt SEs compared at the printed "(0.0000)", uncited
  adj R2 and F checks deleted, and post refresh District 1 targets), Round
  11c (2026-10-16, first full 56 of 56 run with 647 checks and 0 mismatches
  in 101.7 minutes, plus stale file cleanup), and the 2026-09-10 data
  refresh impact record appended as Appendix A.
- `docs/paper_assets/` holds the paper's Tables 1 to 5, Appendix A.1 and
  Figures 3 to 7, generated from repo artifacts with no hand assembled
  numbers.
- `docs/FEATURE_DICTIONARY.md` documents the current 24 feature base and 25
  feature political specification.

## Quick start

```bash
pip install -r requirements.txt      # Python 3.12+ required (esda==2.10.0
                                     # needs >=3.12); CI and local run 3.13
# populate data/ per the Data sources section below
python main.py --check-data          # confirm data/ is populated correctly
python main.py                       # run the entire 56-stage pipeline
                                     # (data build, training, analysis,
                                     # figures, fact-check suite)
```

`python main.py --list` shows every stage in order with the paper section it
feeds. `python main.py --help` shows all flags, including single stage runs
(`--stage`), resume (`--from`), skipping stages (`--skip`), skipping heavy
training (`--fast`), running one category (`--category`), and aborting on
first failure (`--stop-on-error`; the default is to continue).

`main.py` does not duplicate any script's logic. It sequences the existing
scripts, times them, captures each stage's output to
`results/logs/pipeline/<stage_id>.log`, and writes a consolidated
`results/PIPELINE_RUN_REPORT.md` at the end. A genuinely full run (no filter
flags) additionally writes a dated, git-tracked snapshot
(`results/PIPELINE_RUN_REPORT_full_YYYYMMDD.md`) so the always-overwritten
report can't be mistaken for evidence of a full pass.


## System status

- **Canonical dataset.** `data/canonical/modeling_dataset_canonical_v2.parquet`
  holds 566,956 rows and 53 columns covering sale years 2016 to 2025
  (provenance in `data/DATA_MANIFEST.json`). The 2026-09-24/25 build ended
  with `drop_duplicates()`, which removed 1,041 exact duplicate transactions
  relative to the 2026-09-10 refresh, and coerced the string typed square
  footage, area and unit columns to numeric. That widened the modeling spec
  to 24 base and 25 political features.
- **Regression sample (Table 1).** 520,365 rows after the missingness drop
  (46,591 dropped, made up of the 32,823 row PLUTO join failure "farf"
  group, 13,277 numfloors, 137 dem_share and 407 other, with a 53 row
  overlap). The train and test split is 121,052 and 56,723 rows, time
  ordered, and the all_years split always covers the 2017 election cycle
  subset of 177,775 rows.
- **Last recorded full run.** 2026-10-19, 84 of 84 stages passed in 91.7
  minutes (`results/PIPELINE_RUN_REPORT.md`, with the durable snapshot
  `results/PIPELINE_RUN_REPORT_full_20261019.md`). The report footer carries
  the run provenance (git_commit and git_dirty) and the verification totals
  (840 checks, 0 mismatches, 26 of 26 verifiers in
  `results/verification_summary.json`). Wall time is dominated
  by the RF tuning grid (about 14 min), the exact SHAP ladder across all 7
  models (about 35 min, all tier a or linear exact, with no timeouts and no
  fallbacks), the v2/v3 robustness suites (about 16 min combined), the
  political scenario scoring (about 7 min), and the pre-2024 suite
  (about 7 min).
- **Random Forest (Round 10).** The unpruned forest
  `rf_all_years_political_n200` (about 990 MB) was retired because exact
  TreeSHAP could not finish on it within the 90 minute tier budget, so it
  only ever received labeled gain importance. Its replacement
  `rf_all_years_political_tuned` is selected from a pre specified 36 config
  grid (max_depth in 10, 14, 18 or 25, min_samples_leaf in 1, 5 or 20, and
  max_features in 1.0, 0.5 or 0.33) on the time ordered 15 percent
  validation slice of the training years only, under a pre specified
  tractability cap of at most 3,000 mean leaves per tree applied before any
  MAE comparison. The test set is scored exactly once, after selection. The
  full audit is `results/rf_tuning_grid.json`.
- **SHAP ladder (Round 10).** Every eligible model goes through
  `src/ops/shap_tiers.py`, which tries exact TreeSHAP on the standard
  sampled test slice, then a depth capped variant for unpruned Random
  Forests only, then a fixed 5,000 row sample, then a labeled gain fallback.
  The depth capped tier skips itself for forests that are already
  complexity constrained. A 5 minute startup watchdog kills and respawns a
  stalled SHAP child exactly once without charging the stall to the tier
  budget. Rejected depth capped variants are labeled as such and no SHAP
  claim is made on them. The stage prints a OneDrive warning when the repo
  lives under OneDrive.
- **Round 13 (split-ED fix + new robustness).** The legacy ED-to-council
  aggregation sent each ED's full votes to every council district it
  touched, inflating summed CD votes to 1.29-1.31x raw votes
  (`results/crosswalk_audit.json`). Round 13 adds nine allocation-corrected
  ideology measures — six `_aw` area-weighted (each ED split across its
  districts by normalized intersection-area weights, overcount = 0.00% and
  loss = 0.00% per year, `results/crosswalk_audit_v3.json`) and three `_dom`
  dominant-district variants — in
  `data/canonical/modeling_dataset_canonical_v3.parquet` (all 64 legacy
  columns byte-identical, all nine new columns blacklisted from models; no
  retraining). New suites: `results/robustness_checks_v3.json` (126 fits),
  `results/robustness_pre2024_v3.json` (Cuomo-free pre-2024 sample),
  `results/time_trend_placebo_v3.json` (wave-mean vs demeaned placebo),
  `results/hedonic_v3.json` (N=23,789 audit + land/units variant),
  `results/morans_i_results_v3.json` + `results/spatial_regression_v3.json`,
  and `results/causal/city_of_yes_did_results_v3.json` (wild-bootstrap p
  for the DiD window/trends specs + the continuous
  `dem_share_pres_vintage_aw` x post arm). Two verified bug fixes:
  `compute_morans_i_v2.py` now includes the measure among the residual
  regressors (legacy dem_share pooled residual I reproduces 0.3950, spreg
  z = 6.14; previously every measure showed 0.3825), and the IV
  local-linear RD fits in `iv_district_fe_regression_v2.py` now cluster
  the covariance by council district (pre-fix first-stage F of 3,000-77,000
  ignored the clustering; singular clustered fits are marked
  "not estimable" with no unadjusted fallback).

### Verification commands (for auditors)

You can rerun every piece in isolation.

| Command (relative to repo root) | Purpose |
|---|---|
| `python src/data_pipeline/rebuild_canonical.py` | rebuild the modeling dataset |
| `python src/analysis/compute_morans_i.py` | Moran's I (official shapefile, k=5 KNN, seed 42) |
| `python src/analysis/generate_regression_table.py` | Table 1 (`results/regression_table.csv`) |
| `python src/analysis/generate_robustness_checks.py` | Table 4 (`results/robustness_checks.json`) |
| `python src/analysis/spatial_regression_model.py` | Table 5 (district-level spatial lag/error) |
| `python src/analysis/generate_shap_summary.py` | SHAP Table 3 (per-model, tiered ladder, every eligible model) |
| `python src/analysis/political_scenarios.py` | Appendix A.1 counterfactuals (every eligible ranked model; name-keyed by_model copies) |
| `python src/verification/verify_canonical_dataset_and_split.py` | dataset shape + split (24/25 features; 121,052 / 56,723) |
| `python src/verification/verify_spatial_pattern.py` | district correlation (r = 0.175, 51 districts) |
| `python src/verification/verify_regression_ols.py` + `python src/verification/verify_regression_missingness.py` + `python src/verification/verify_numfloors_missingness.py` | Table 1 re-derived from scratch (β 0.9449 / 0.6146 / 0.2883) + missingness reconciliation |
| `python src/verification/verify_ml_performance.py` | Table 2 MAE/R² vs the registry (resolved by model name; stale entries excluded) |
| `python src/verification/verify_shap_importance.py` | SHAP Table 3 headline cells (Community Board 0.32088; dem_share 0.04685, rank 11 of 25) |
| `python src/verification/verify_appendix_a1_scenarios.py` | all 51 by 3 Appendix A.1 scenario percentages vs the `lgbm_all_years_political` outputs resolved by name |
| `python src/verification/verify_scenario_mechanics.py` | scenario mechanics (filters, clip bounds, support diagnostics) for every eligible ranked model |
| `python src/verification/verify_spatial_regression.py` | Table 5 independently re-fit and compared (ρ 0.7029, λ 0.7809) |
| `python src/verification/verify_borough_map_and_crosscutting.py` | borough map; computed blacklist status; artifact sanity |
| `python src/verification/verify_followup_resolution.py` | resolves the missingness overlap note and the pred-interval gap |
| `python src/verification/verify_ideology_construction.py` + `python src/verification/verify_cd1_dem_share.py` | ideology construction + CD1 worked example (0.715908) |
| `python src/verification/editorial_audit/verify_blacklist_fully.py` | fail-loud leakage gate on every trained model + artifact |
| `python src/verification/verify_abstract_and_intro.py` | abstract/intro claims |
| `python src/causal/iv_district_fe.py` / `iv_district_fe_rescue.py` / `city_of_yes_did.py` | causal-ID analyses (specification search + event-study DiD) |
| `python src/verification/verify_iv_district_fe.py` / `verify_iv_district_fe_rescue.py` / `verify_city_of_yes_did.py` | causal-ID verifiers |
| `python src/figures/generate_paper_assets.py` | regenerate every paper table/figure from repo artifacts (`docs/paper_assets/`) |

These stages print comparisons to the manuscript's current values, write
detailed JSON deltas under `results/`, and write a per verifier status under
`results/verification/`. Since Round 11 the suite is enforced by the final
`verification_summary` stage, which aggregates all statuses and exits non
zero on any mismatch beyond tolerance, on any missing verifier report, or on
any stale status file. The last verification gate (Round 13, 2026-10-19)
reported 840 checks, 0 mismatches and 26 of 26 verifiers reporting.

## Champion selection and interpretability coverage

7 model experiments are registered, one per trainer per mode through
replace-on-register in `models/training/`. They are LightGBM base and
political, Random Forest political (tuned), CatBoost political, XGBoost
political, and Ridge and ElasticNet political.

On every full run the `refresh_champion` stage
(`src/ops/experiment_runner.py`) picks the champion as the lowest test MAE
experiment among untainted, non-stale registry entries dated on or after
2026-05-18. A documented near tie rule applies when several experiments sit
within 0.001 MAE of the best, in which case the simpler base mode spec is
preferred, and `experiments/champion.json` records the note. The stale flag
is the governance hook. Entries recorded against a superseded model or
config can be flagged `stale`, which makes them invisible to champion
selection and to every ranked model downstream stage. Round 10 used this
flag to retire `rf_all_years_political_n200`, and the selection rule itself
is unchanged.

Interpretability artifacts are computed for every eligible ranked model, not
just the champion. Ranks come from `src/ops/champion_ranking.py`, where
rank 1 is the champion.json selection and ranks 2 to N follow ascending
test MAE, with the audit trail in `results/rank1_rank2_selection.json`. The
per model passes are `generate_shap_summary.py` (tiered SHAP),
`generate_pdp.py` (PDPs), `confidence_intervals.py` and
`champion_confidence_and_sensitivity.py` (prediction intervals and
perturbation), `sensitivity_analysis.py`, and `political_scenarios.py`
(three counterfactual scenarios per model, with name keyed copies under
`results/political_scenarios/by_model/<model>/`).

All paper facing assets are resolved by experiment name, never by rank
number, because rank assignments shift whenever the eligible set changes.

## Data sources and provenance

The canonical build's raw inputs are recorded in `data/DATA_MANIFEST.json`
(filenames, row counts and timestamps) and in section 14 and Appendix A of
`VERIFICATION_REPORT.md`. Neither file asserts download URLs or dataset IDs
beyond that, so where a URL cannot be confirmed from those records the
table below says **ASK AUTHOR** rather than guessing.

| Raw input | Exact public source (as recorded or confirmable) |
|---|---|
| Property sales from `data/raw/sales/NYC_Citywide_Annualized_Calendar_Sales_Update_20260910.csv` (845,607 rows, sale years 2016 to 2025, 2025 complete) | NYC Open Data dataset **"NYC Citywide Annualized Calendar Sales Update"** (data.cityofnewyork.us). **ASK AUTHOR** for the exact dataset ID and download URL, which are not recorded in the provenance files. |
| PLUTO map from `data/raw/pluto/Primary_Land_Use_Tax_Lot_Output_(PLUTO)_20260105.csv` | NYC Department of City Planning **Primary Land Use Tax Lot Output (PLUTO)**, 2026-01-05 vintage (NYC Open Data PLUTO dataset). **ASK AUTHOR** for the exact dataset ID and URL. |
| Election returns (mayoral and council, election district level) from `data/raw/election_results/…/recap` files, 2013 PDFs and 2017/2021/2025 CSVs (204 BOE Recap files incl. the council margin build) | **NYC Board of Elections** election results (public Recap PDFs and CSVs. RCV years 2021 and 2025 use certified first choice totals with fusion lines aggregated, see the `build_council_election_margins.py` docstring). **ASK AUTHOR** for the exact download URLs. |
| Council roll calls (City of Yes) from `data/processed/city_of_yes_roll_calls.parquet` | NYC Council **Legistar** API at `https://webapi.legistar.com/` with the `council.nyc.gov` site. Per Round 6 section 14 of `VERIFICATION_REPORT.md`, the Carbon Neutrality (Dec 2023) and Economic Opportunity (June 2024) roll calls could not be retrieved without a token and are absent. |
| Council district boundaries from `data/raw/election_districts/NYC_City_Council_Districts.shp` | official NYC Council District shapefile. **ASK AUTHOR** for the download URL. Known limitation, the boundaries are pre-2023 (`VERIFICATION_REPORT.md` Appendix A section 12). |
| `data/processed/ed_to_council_crosswalk.parquet` boundary crosswalk | built in repo from NYC Planning Geosupport and the ED shapefile (`Election_Districts_20260105.csv` is an off path input), see `src/data_pipeline/build_district_crosswalk.py`. |

The pipeline is source reproducible but not source frozen. Two runs differ
only if the underlying NYC files are republished. The analysis figures come
only from the canonical parquet and the split parquets. `data/` is not in
git (see `.gitignore`), and on a fresh clone use `python main.py --check-data`.


## Project structure

```
main.py                            # single-entry 56-stage orchestrator
PAPER_TO_CODE_MAP.md               # maps paper sections to producer and verifier scripts
VERIFICATION_REPORT.md             # 11-round independent re-verification record + Appendix A
requirements.txt                   # pinned, py3.12+ CI-tested
.github/workflows/ci.yml           # build-canonical + 8-column audit stages

app/                               # Streamlit app (run_app.py / dashboard engine / explainability/)
archive/                           # superseded verification harness scripts (read-only; git history)
config/
  config.yaml                      # OLS regression spec (Table 1)
  feature_blacklist.yaml           # leakage + null columns hard-removed from ALL training
data/                              # not in git (raw, intermediate, processed, canonical, splits)
docs/
  modeling_schema.csv              # frozen 53-column schema (column, dtype)
  feature_manifest.json            # mirrored manifest (same build)
  FEATURE_DICTIONARY.md            # the 24-feature base / 25-feature political specification
  model_comparison_table.csv       # pipeline-written table of every eligible registry model
  figures/                         # exploratory PNGs (dark-theme, auto-generated)
  paper_assets/                    # the paper's LaTeX tables + PNG figures + manifest.json
  research_summary.md              # LAST RUN's digest (rewritten every run; never edit)
models/
  lgbm_all_years_base.txt          # LightGBM base (the paper's paper-model; 24 features)
  lgbm_all_years_political.txt     # LightGBM political (Table 3 source; 25 features)
  rf_all_years_political_tuned.joblib      # tuned RF (comparator + possible champion)
  cat_all_years_political_lr0.05_d6.cbm    # CatBoost comparator
  xgb_all_years_political_lr0.05_md6.model # XGBoost comparator
  ridge_all_years_political.joblib
  elasticnet_all_years_political.joblib
  artifacts/                       # *_features/_metrics/_shap/_pdp/_sensitivity per model
  training/                        # one script per model family (fresh retrains)
experiments/
  registry.json                    # every registered run (tainted/stale flags exclude)
  champion.json                    # the pointer refresh_champion owns
  results/                         # raw per-model metrics JSONs
notebooks/                         # docs/reference code (not part of main.py)
tables/                            # pre-existing drop-in .tex tables (Table 4, Table 5)
results/
  regression_table.csv             # Table 1 (OLS)
  robustness_checks.json           # Table 4
  spatial_regression.json          # Table 5 (joint-lag, null-hetero Row regress)
  morans_i_results.json            # Moran's I audit + methodology record
  rf_tuning_grid.json              # the full RF tuning grid (config/leaves/val MAE/eligibility)
  political_scenarios/             # per-rank + by_model scenario outputs (Appendix A.1)
  causal/                          # IV + City-of-Yes DiD results and their verification
  logs/pipeline/                   # per-stage logs from main.py runs
src/
  data_pipeline/                   # normalize -> crosswalk -> ideology -> rebuild -> freeze -> split
  analysis/                        # Moran, regressions, robustness, SHAP, PDP, sensitivity, scenarios
  causal/                          # close-election IV (incl. rescue) + City of Yes DiD
  figures/                         # figure writers + generate_paper_assets.py
  ops/                             # sanity check, champions, model_io, predictors, shap_tiers
  verification/                    # the independent fact-check harness (each script re-derives)
    editorial_audit/               # fail-loud blacklist gate
legacy/                            # the original pre-reorganization ORBIT snapshot (read-only)
```

## Reporting & testing

- **CI** (`.github/workflows/ci.yml`) keeps a slim build-canonical +
  8-column audit pass on every push (Ubuntu, Python 3.13).
- **Full pipeline tests** are the 56-stage `python main.py` run. Every
  verifier stage writes a machine readable status under
  `results/verification/`, which the `verification_summary` stage aggregates
  into `results/verification_summary.json` and enforces by exiting non zero
  on any mismatch. Each verifier also keeps its human readable
  `results/logs/pipeline/*.log` and JSON delta files.
- **Report pages.** `results/PIPELINE_RUN_REPORT.md` is the last run,
  `results/PIPELINE_RUN_REPORT_full_YYYYMMDD.md` is the durable full run
  snapshot, `docs/research_summary.md` is the human digest, and
  `docs/paper_assets/` holds the manuscript inserts.

## Privacy & access

- **No personal data.** Sales/PLUTO records are public records; all model
  inputs are property-level public attributes.
- The Streamlit app (`app/app.py`) ignores `data/raw` entirely and reads only
  `models/` + printed artifact JSONs inside a sandboxed `data/processed`
  snapshot; it does not ship voter records.


## Causal identification (post-descriptive add-on, null results)

Two designs support the paper's causal section, and both produce **null
results** that are reported as such.

1. **Close election IV and RDD flavored designs**
   (`src/causal/iv_district_fe.py`, `iv_district_fe_rescue.py`) instrument
   district ideology with the previous cycle's close council elections in a
   full and escalating specification search. Every recorded arm is
   tabulated by `generate_paper_assets.py` (`docs/paper_assets/
   table_iv_spec_search.tex`, `fig6_iv_spec_search.png`), identified or not.
2. **City of Yes Housing Opportunity DiD**
   (`src/causal/city_of_yes_did.py`) runs an event study with a pre trend
   joint Wald test plus a main and a sensitivity DiD
   (`docs/paper_assets/table_did_event_study.tex`, `fig7_did_event_study.png`).

The two Carbon Neutrality and Economic Opportunity roll calls could not be
retrieved without a legistar token and are absent (see the Round 6 note in
`VERIFICATION_REPORT.md`).

## Where the paper stands

Every pipeline regenerated number (canonical counts, split, 24/25 features,
Tables 1 to 5, Moran's I, SHAP, Appendix A.1) is computed and verified in
this repo against the manuscript's current values, and the manuscript
facing tables and figures are generated rather than hand copied by
`generate_paper_assets.py` into `docs/paper_assets/`. Current state.

1. The old identified out discrepancies are closed. The former `numfloors`
   11,918 versus 11,930 gap is obsolete, since 13,277 is what the current
   build produces and what the manuscript uses, and the Moran's I z
   convention is now aligned (z 7.4264 for price and 7.8625 for dem_share,
   permutation based).
2. The causal work is presented as negative results. The close election
   instrument is not identified at any bandwidth or configuration (the
   plus or minus 20pp district FE spec appears in both the primary search
   and the wider bandwidth rescue arm, and the paper asset table and figure
   list it once with a footnote), and the City of Yes DiD is a null under
   the preference proxy framing.
3. Remaining action item, only if a real policy application DiD is wanted.
   Retrieve the two missing City of Yes roll calls manually from Legistar.
4. The unused but computed assets (PDP panels, prediction intervals,
   sensitivity analyses, all regenerated each run) are optional paper
   appendices or app only surfaces.
5. **Current state (Round 11).** There are 7 registered model experiments
   (LightGBM base and political, the tuned Random Forest as champion per
   `experiments/champion.json`, CatBoost, XGBoost, Ridge and ElasticNet),
   56 pipeline stages, and a full run of about 100 minutes on this machine
   (the RF tuning grid at about 27 min, the exact SHAP ladder at about
   45 min, and scenario scoring plus verification at about 19 min are the
   long parts). Every verifier reports through
   `src/verification/_shared.record_status`, and the `verification_summary`
   stage prints and records the total checks and mismatches, exiting non
   zero on any mismatch or missing report. All registry entries and
   `champion.json` record `git_commit` and `git_dirty`, and
   `PIPELINE_RUN_REPORT.md` shows both.


## Documentation index

| File | Purpose |
|---|---|
| `PAPER_TO_CODE_MAP.md` | paper section ↔ producer script ↔ verifier index |
| `VERIFICATION_REPORT.md` | 11-round independent re-verification record + Appendix A (2026-09-10 data-refresh impact) |
| `docs/FEATURE_DICTIONARY.md` | per-feature provenance for the 24/25 specification |
| `docs/paper_assets/` | the manuscript's tables/figures (+ `manifest.json` output->source map) |
| `docs/research_summary.md` | pipeline-generated champion/comparison/SHAP digest (rewritten every full run; never edit by hand) |
| `results/rf_tuning_grid.json` | the full Random-Forest tuning grid and cap/selection audit |
| `archive/legacy_results_pre_canonical_v2/README.md` | what the archived pre-canonical-v2 result files are |
| `legacy/README.md` | the original pre-reorganization ORBIT snapshot |
| `results/FULL_PAPER_FACT_CHECK.md` | frozen Round-5 human-reviewed fact-check record (local, gitignored) |

## Known caveats

- **Council-district vintage mismatch (flagged).** The close-election IV
  instrument maps prior-cycle (2013/2017/2021) winners, elected under the
  PRE-2022 council district plan, onto sales rows keyed to CURRENT
  (post-2022) council districts, so district identities do not line up
  across vintages wherever an old-plan district does not map cleanly onto
  its 2023-plan successor. The flag is recorded in
  `results/causal/iv_district_fe_results_v2.json`
  (`council_district_vintage_mismatch`) and in `results/crosswalk_audit.json`
  (`council_district_vintage_note`); the areal correspondence needed to fix
  it is already built (`data/processed/council_2013_to_current.parquet`).
  The original 12 IV arms in
  `results/causal/iv_district_fe_results.json` are untouched.
- The council district shapefile still describes **pre-2023 boundaries**,
  so all CounDist level results (Moran's I, Table 5, Appendix A.1) inherit
  that (`VERIFICATION_REPORT.md` Appendix A section 12).
- **Champion MAE margin is thin.** The political tree ensembles routinely
  sit within about 0.003 MAE of each other. The `champion.json` documented
  near tie rule (prefer the simpler base spec within 0.001) resolves those
  margins deterministically, but the champion identity may shift on
  retrains, nothing preserves it by hand, and `model_io.py` and
  `predictors.py` exist so any family can serve as champion.
- A SHAP payload whose `method` is `feature_importance_fallback` is a
  labeled gain importance fallback, not TreeSHAP, so it should not be
  reported as SHAP values. The tiered ladder emits those only when every
  exact tier failed its budget, and the paper asset cross model table marks
  them "Gain (not SHAP)".
- If the repo lives under OneDrive, `generate_shap_summary` prints a startup
  warning. OneDrive sync can lock files during process spawn and stall SHAP
  children. The 5 minute watchdog respawns a stalled child once, but
  pausing sync or moving the repo is the real fix.
- CPU budget. The RF tuning grid (36 capped fits plus the final refit)
  costs roughly 15 to 45 minutes, and `verify_scenario_mechanics` and
  `sensitivity_analysis` each take several minutes on a typical laptop.
  This is expected, not a bug. (The unpruned RF's 180 minute SHAP burn is
  gone, and the tuned forest completes exact TreeSHAP inside the tier a
  budget.)

## License & Disclaimer

This repository contains a research-grade, independently verified analysis
pipeline and its manuscript artifacts.
It is not financial or investment advice. Forecasts and scenario outputs
should be treated as research outputs and verified independently before
operational use. Use at your own risk.

