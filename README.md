# O.R.B.I.T. — Observational Real-estate Behavior & Intelligence Toolkit

O.R.B.I.T. is a reproducible research platform and forecasting toolkit built
to support the paper "Political Ideology and Property Values in New York
City." It integrates public datasets (NYC property sales, PLUTO, mayoral and
council election returns, district geography), canonicalizes and cleans
features, trains the two paper-cited LightGBM models plus five comparison
models (a complexity-constrained, validation-tuned Random Forest, CatBoost,
XGBoost, Ridge, ElasticNet), independently re-verifies every number the paper
reports, generates the paper's tables and figures straight from repo
artifacts, and exposes an interactive Streamlit application for
single-property exploration, batch scoring, and political scenario
simulation.

**New here?**

- `PAPER_TO_CODE_MAP.md` — go from any table/figure/claim in the paper to the
  exact script that produced it and the exact script that independently
  verifies it.
- `VERIFICATION_REPORT.md` — the independent re-verification record of the
  whole repository: Rounds 1–4 (reorganization & repair, 2026-09-01..08),
  Round 5 (2026-09-13: data-refresh reconciliation + causal-ID build),
  Round 6 (2026-09-14: IV rescue + City of Yes DiD), Round 7 (2026-09-22:
  the `BBL_pluto` blacklist), Round 8 (2026-09-24/25: champion flip to Random
  Forest, duplicate dedup, and the 24/25-feature spec), Round 9 (2026-09-27:
  full-model interpretation coverage), Round 10 (2026-09-28..30: tuned
  complexity-constrained Random Forest, SHAP audit fixes + startup watchdog,
  paper-asset generation, refreshed verifier targets), Round 11 (2026-10-10:
  paper-asset LaTeX escape/layout repairs with standalone-compile check,
  machine-countable verifier status + verification_summary gate, provenance
  git_commit/git_dirty, repo-relative POSIX paths, PDP pruning + stale-artifact
  audit, .gitignore whitelisting of paper-relevant artifacts), Round 11b
  (2026-10-12: stale verifier TARGETS corrected — 5dp-delta tolerance set at
  its exact rounding bound, yearbuilt SEs compared at the printed "(0.0000)",
  uncited adj-R2/F-stat checks deleted, post-refresh District-1 targets),
  Round 11c (2026-10-16: first full 56/56 run — 647 checks / 0 mismatches in
  101.7 min — plus stale-file cleanup), plus the full
  2026-09-10 data-refresh impact record appended as **Appendix A**.
- `docs/paper_assets/` — the paper's Tables 1–5, Appendix A.1, and
  Figures 3–7, generated from repo artifacts (no hand-assembled numbers).
- `docs/FEATURE_DICTIONARY.md` — the current 24-feature base / 25-feature
  political specification.

## Quick start

```bash
pip install -r requirements.txt      # Python 3.12+ required (esda==2.10.0
                                     # needs >=3.12); CI and local run 3.13
# populate data/ per "Data sources & provenance" below
python main.py --check-data          # confirm data/ is populated correctly
python main.py                       # run the ENTIRE 56-stage pipeline:
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


## System status

- **Canonical dataset:** `data/canonical/modeling_dataset_canonical_v2.parquet`
  — **566,956 rows × 53 columns**, sale years 2016–2025 (provenance in
  `data/DATA_MANIFEST.json`). The 2026-09-24/25 build ended with
  `drop_duplicates()` (−1,041 exact-duplicate transactions vs the 2026-09-10
  refresh) and coerced the string-typed square-footage/area/unit columns to
  numeric — that widened the modeling spec to **24 base / 25 political**
  features.
- **Regression sample** (Table 1): **520,365** rows after the missingness
  drop (46,591 dropped = 32,823 PLUTO join-failure "farf" group + 13,277
  numfloors + 137 dem_share + 407 other, −53 overlap). Train/test split:
  **121,052 / 56,723** (time-ordered; the all_years split always covers the
  2017 election-cycle subset, 177,775 rows).
- **Last recorded full run:** 2026-10-16, **56/56 stages PASS in 101.7 min**
  (`results/PIPELINE_RUN_REPORT.md`; durable snapshot
  `results/PIPELINE_RUN_REPORT_full_20261016.md`). The report footer carries
  the run provenance (git_commit / git_dirty) and the machine-countable
  verification totals — **647 checks, 0 mismatches, 19/19 verifiers reporting**
  (`results/verification_summary.json`). Wall time is dominated by the RF
  tuning grid (~27 min), the exact-SHAP ladder across all 7 models (~45 min;
  all tier-a / linear_exact, no timeouts, no fallbacks, no watchdog fires),
  and the 7-model x 3-scenario counterfactual scoring plus its independent
  re-derivation verifier (~19 min combined).
- **Random Forest (Round 10):** the unpruned forest
  (`rf_all_years_political_n200`, ~990 MB) was retired — exact TreeSHAP could
  not finish on it within the 90-minute tier budget, so it only ever received
  labeled gain importance. Its replacement, `rf_all_years_political_tuned`,
  is selected from a pre-specified 36-config grid
  (max_depth {10, 14, 18, 25} × min_samples_leaf {1, 5, 20} ×
  max_features {1.0, 0.5, 0.33}) on the time-ordered 15% validation slice of
  the training years **only** under a pre-specified **≤ 3,000 mean
  leaves/tree** tractability cap (cap applied before any MAE comparison; the
  test set is scored exactly once, after selection). Full audit:
  `results/rf_tuning_grid.json`.
- **SHAP ladder hardening (Round 10):** every eligible model goes through
  `src/ops/shap_tiers.py` (exact TreeSHAP on the standard sampled test slice →
  (unpruned RandomForest only) depth-capped variant → fixed 5,000-row sample →
  labeled gain fallback). The depth-capped-variant tier auto-skips for forests
  that are already complexity-constrained; a 5-minute startup watchdog kills
  and respawns a stalled SHAP child exactly once without charging the stall
  to the tier budget; REJECTED depth-capped variants are labeled as such (no
  SHAP claim is made on them); the stage prints a OneDrive warning when the
  repo lives under OneDrive.

### Verification commands (for auditors)

You can rerun every piece in isolation:

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
| `python src/verification/verify_appendix_a1_scenarios.py` | all 51×3 Appendix A.1 scenario percentages vs the `lgbm_all_years_political` outputs resolved by NAME |
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

These stages print comparisons to the manuscript's current values and write
detailed JSON deltas under `results/`. The verification suite exits 0 on
mismatch by design — it REPORTS instead of silently overwriting.

## How the champion model is selected — and where interpretability comes from

7 model experiments are registered (one per trainer per mode, via
replace-on-register in `models/training/`): LightGBM base & political,
Random Forest political (tuned), CatBoost political, XGBoost political, Ridge
& ElasticNet political.

On every full run, the `refresh_champion` stage
(`src/ops/experiment_runner.py`) picks the champion as the **lowest test-MAE
experiment among untainted, non-stale, post-2026-05-18 registry entries**,
with a documented near-tie rule: if several experiments sit within 0.001 MAE
of the best, the simpler *base-mode* spec is preferred
(`experiments/champion.json` records the selection note). "Non-stale" is the
governance hook: entries recorded against a superseded model/config can be
flagged `stale` and are then invisible to champion selection and to every
ranked-model downstream stage. (Round 10 used exactly this flag to retire
`rf_all_years_political_n200`; the champion-selection rule itself is
unchanged.)

Interpretability artifacts are computed for **every eligible ranked model**
(resolved by `src/ops/champion_ranking.py`: rank 1 = champion.json's
selection, ranks 2..N = ascending test MAE; audit trail in
`results/rank1_rank2_selection.json`), not just the champion:
`generate_shap_summary.py` (tiered SHAP), `generate_pdp.py` (PDPs),
`confidence_intervals.py` / `champion_confidence_and_sensitivity.py`
(prediction intervals + perturbation), `sensitivity_analysis.py`, and
`political_scenarios.py` (three counterfactual scenarios per model, with
name-keyed copies under `results/political_scenarios/by_model/<model>/`).

All paper-facing assets are resolved by experiment NAME, never by rank
number — rank assignments shift whenever the eligible set changes.

## Data sources & provenance

The canonical build's raw inputs are recorded in `data/DATA_MANIFEST.json`
(filenames, row counts, timestamps) and `VERIFICATION_REPORT.md` (§14 +
Appendix A). Nothing else about URLs/dataset IDs is asserted in either file,
so where a URL cannot be confirmed from those records it says **ASK AUTHOR**
rather than guessing.

| Raw input | Exact public source (as recorded / confirmable) |
|---|---|
| Property sales: `data/raw/sales/NYC_Citywide_Annualized_Calendar_Sales_Update_20260910.csv` (845,607 rows, sale years 2016–2025; 2025 complete) | NYC Open Data, dataset **"NYC Citywide Annualized Calendar Sales Update"** (data.cityofnewyork.us). **ASK AUTHOR** for the exact dataset ID / download URL (not recorded in the provenance files). |
| PLUTO: `data/raw/pluto/Primary_Land_Use_Tax_Lot_Output_(PLUTO)_20260105.csv` | NYC Dept. of City Planning **Primary Land Use Tax Lot Output (PLUTO)**, 2026-01-05 vintage (NYC Open Data PLUTO dataset). **ASK AUTHOR** for the exact dataset ID/URL. |
| Election returns (mayoral + council, election-district level): `data/raw/election_results/…/recap` files, 2013 PDFs + 2017/2021/2025 CSVs (204 BOE Recap files incl. the council-margin build) | **NYC Board of Elections** election results (public Recap PDFs/CSVs; RCV years 2021/2025 use certified first-choice totals with fusion lines aggregated — see `build_council_election_margins.py` docstring). **ASK AUTHOR** for the exact download URLs. |
| Council roll calls (City of Yes): `data/processed/city_of_yes_roll_calls.parquet` | NYC Council **Legistar** API — `https://webapi.legistar.com/` with `council.nyc.gov` site (see VERIFICATION_REPORT.md Round 6 §14: Carbon Neutrality (Dec 2023) and Economic Opportunity (June 2024) roll calls were NOT retrievable without a token and are deliberately absent). |
| Council district boundaries: `data/raw/election_districts/NYC_City_Council_Districts.shp` | official NYC Council District shapefile. **ASK AUTHOR** for the download URL. NOTE (known limitation): boundaries are pre-2023 (VERIFICATION_REPORT.md Appendix A §12). |
| `data/processed/ed_to_council_crosswalk.parquet` boundary crosswalk | built in-repo from NYC Planning Geosupport + the ED shapefile (`Election_Districts_20260105.csv` is off-path input) — see `src/data_pipeline/build_district_crosswalk.py`. |

The pipeline is *source-reproducible but not source-frozen*: two runs differ
only if the underlying NYC files are republished. The analysis figures come
only from the canonical parquet + split parquets. `data/` is NOT in git (see
`.gitignore`); on a fresh clone use `python main.py --check-data`.


## Project structure

```
main.py                            # single-entry 56-stage orchestrator
PAPER_TO_CODE_MAP.md               # paper section <-> producer script <-> verifier index
VERIFICATION_REPORT.md             # 11-round independent re-verification record + Appendix A
requirements.txt                   # pinned, py3.12+ CI-tested
.github/workflows/ci.yml           # build-canonical + 8-column audit stages

app/                               # Streamlit app (run_app.py / dashboard engine / explainability/)
archive/                           # superseded verification harness scripts (read-only; git history)
config/
  config.yaml                      # OLS regression spec (Table 1)
  feature_blacklist.yaml           # leakage + null columns hard-removed from ALL training
data/                              # NOT in git: raw -> intermediate -> processed -> canonical -> splits
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
  verification/                    # the independent fact-check harness (each scripts re-derives)
    editorial_audit/               # fail-loud blacklist gate
legacy/                            # the original pre-reorganization ORBIT snapshot (read-only)
```

## Reporting & testing

- **CI** (`.github/workflows/ci.yml`) keeps a slim build-canonical +
  8-column audit pass on every push (Ubuntu, Python 3.13).
- **Full pipeline tests** are the 56-stage `python main.py` run; every
  verifier stage writes a machine-readable status (`results/verification/
  <verifier>.json`, aggregated by the `verification_summary` stage into
  `results/verification_summary.json`, which fails loudly on any mismatch)
  in addition to its human-readable `results/logs/pipeline/*.log` and JSON
  delta files.
- **Report pages**: `results/PIPELINE_RUN_REPORT.md` (last run),
  `results/PIPELINE_RUN_REPORT_full_YYYYMMDD.md` (durable full-run snapshot),
  `docs/research_summary.md` (human digest), `docs/paper_assets/` (manuscript
  inserts).

## Privacy & access

- **No personal data.** Sales/PLUTO records are public records; all model
  inputs are property-level public attributes.
- The Streamlit app (`app/app.py`) ignores `data/raw` entirely and reads only
  `models/` + printed artifact JSONs inside a sandboxed `data/processed`
  snapshot; it does not ship voter records.


## Causal identification (post-descriptive add-on, honest nulls)

Two designs strengthen the paper's causal section; both are honest **null
results** and are presented as such:

1. **Close-election IV / RDD-flavored designs**
   (`src/causal/iv_district_fe.py`, `iv_district_fe_rescue.py`) — instrument
   district ideology with the *previous cycle's* close council elections, in a
   full (and escalating) specification search. Every recorded arm is
   tabulated by `generate_paper_assets.py` (`docs/paper_assets/
   table_iv_spec_search.tex`, `fig6_iv_spec_search.png`), identified or not.
2. **City of Yes Housing Opportunity DiD**
   (`src/causal/city_of_yes_did.py`) — event-study with a pre-trend joint
   Wald test plus a main and a sensitivity DiD
   (`docs/paper_assets/table_did_event_study.tex`, `fig7_did_event_study.png`).

The two Carbon Neutrality / Economic Opportunity roll calls could not be
retrieved without a legistar token and are deliberately not fabricated (see
the Round-6 note in `VERIFICATION_REPORT.md`).

## Where the paper stands

Every pipeline-regenerated number — canonical counts, split, 24/25 features,
Tables 1–5, Moran's I, SHAP, Appendix A.1 — is computed and verified in this
repo against the manuscript's current values, and the manuscript-facing
tables/figures are *generated* (not hand-copied) by `generate_paper_assets.py`
into `docs/paper_assets/`. Known final state:

1. The old identified-out discrepancies are closed: the former `numfloors`
   11,918-vs-11,930 gap is obsolete (the 13,277 figure is what the current
   build produces and what the manuscript uses), and the Moran's-I z convention
   is now aligned (z 7.4264 price / 7.8625 dem_share; permutation-based).
2. The causal work is presented as honest negative results: the close-election
   instrument is not identified at any bandwidth/configuration (the ±20pp
   district-FE spec appears in both the primary search and the wider-bandwidth
   rescue arm; the paper-asset table/figure list it once with a footnote), and
   the City of Yes DiD is a null under the preference-proxy framing.
3. Remaining action item (only if a real policy-application DiD is wanted):
   retrieve the two missing City of Yes roll calls manually from Legistar.
4. The unused-but-computed assets (PDP panels, prediction intervals,
   sensitivity analyses — all regenerated each run) are optional paper
   appendices or app-only surfaces.
5. **Final state (Round 11):** 7 registered model experiments (LightGBM
   base/political, tuned RF = champion per `experiments/champion.json`,
   CatBoost, XGBoost, Ridge, ElasticNet); 56 pipeline stages; a full run takes
   ~100 min on this machine (RF tuning grid ~27 min + exact-SHAP ladder
   ~45 min + scenario scoring/verification ~19 min are the long poles);
   every verifier reports its status through
   `src/verification/_shared.record_status` and the `verification_summary`
   stage prints + records total checks/mismatches (exits non-zero on any
   mismatch or missing report). All registry entries and `champion.json`
   record `git_commit` AND `git_dirty`; `PIPELINE_RUN_REPORT.md` shows both.


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

- The council-district shapefile still describes **pre-2023 boundaries** —
  all CounDist-level results (Moran's I, Table 5, Appendix A.1) inherit that
  (VERIFICATION_REPORT.md Appendix A §12).
- **Champion MAE margin is thin.** The political tree ensembles routinely
  sit within ~0.003 MAE of each other; `champion.json`'s documented
  near-tie rule (prefer the simpler base spec within 0.001) resolves those
  margins deterministically, but the champion identity may shift on retrains
  — nothing preserves it by hand, and `model_io.py`/`predictors.py` exist so
  any family can serve as champion.
- A SHAP payload whose `method` is `feature_importance_fallback` is a
  labeled gain-importance fallback, NOT TreeSHAP — never report those as
  SHAP values. (The tiered ladder emits them only when every exact tier
  failed its budget; the paper-asset cross-model table marks them
  "Gain (not SHAP)".)
- If the repo lives under OneDrive, `generate_shap_summary` prints a startup
  warning: OneDrive sync can lock files during process spawn and stall SHAP
  children; the 5-minute watchdog respawns a stalled child once, but pausing
  sync (or moving the repo) is the real fix.
- CPU budget: the RF tuning grid (36 capped fits + final refit) costs
  roughly 15–45 minutes, and `verify_scenario_mechanics` /
  `sensitivity_analysis` each take several minutes on a typical laptop;
  this is expected, not a bug. (The unpruned RF's 180-minute SHAP burn is
  gone — the tuned forest completes exact TreeSHAP inside the tier-(a)
  budget.)

## License & Disclaimer

This repository contains a research-grade forecasting and analysis toolkit.
It is not financial or investment advice. Forecasts and scenario outputs
should be treated as research outputs and verified independently before
operational use. Use at your own risk.

