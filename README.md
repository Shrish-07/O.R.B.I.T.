# O.R.B.I.T. — Observational Real-estate Behavior & Intelligence Toolkit

O.R.B.I.T. is a reproducible research platform and forecasting toolkit built
to support the paper "Political Ideology and Property Values in New York
City." It integrates public datasets (NYC property sales, PLUTO, mayoral and
council election returns, district geography), canonicalizes and cleans
features, trains LightGBM and comparison models, independently re-verifies
every number the paper reports, and exposes an interactive Streamlit
application for single-property exploration, batch scoring, and political
scenario simulation.

**New here?**

- `PAPER_TO_CODE_MAP.md` — go from any table/figure/claim in the paper to the
  exact script that produced it and the exact script that independently
  verifies it.
- `ACTION_PLAN_FOR_SUBMISSION.md` — prioritized list of what to strengthen
  before the paper is submitted.
- `VERIFICATION_REPORT.md` — the seven-round independent re-verification
  record of the whole repository (covers the 2026-09 data refresh, the
  causal-identification work, and the `BBL_pluto` blacklist).
- `DATA_REFRESH_IMPACT_REPORT.md` — exactly what changed when the sales data
  was refreshed from a January-2026 snapshot (2016–2024) to the full
  2016–2025 pull, table by table.

## Quick start

```bash
pip install -r requirements.txt      # Python 3.12+ required (esda==2.10.0
                                     # needs >=3.12); CI and local run 3.13
# populate data/ per "Data sources" below and docs/DATA_LINEAGE.md
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

## System status (verified full run 2026-09-23: 53/53 stages PASS)

- **Canonical dataset:** `data/canonical/modeling_dataset_canonical_v2.parquet`
  — 567,997 rows × 53 columns, sale years 2016–2025 (refreshed 2026-09-10;
  provenance in `data/DATA_MANIFEST.json`).
- **Permanent modeling spec: 18 base / 19 political features.** The 17
  original political features (16 base + `dem_share`) plus the three UNITS
  columns (`RESIDENTIAL UNITS`, `COMMERCIAL UNITS`, `TOTAL UNITS`, admitted
  when the 2026-09 sales pull made them numeric), minus `BBL_pluto` —
  blacklisted 2026-09-22 as a sales/PLUTO merge-suffix parcel-identifier
  artifact (`config/feature_blacklist.yaml`; see `VERIFICATION_REPORT.md`
  Round 7).
- **Champion model: `lgbm_all_years_political`** (LightGBM): test MAE
  ≈ 0.41195, R² ≈ 0.5650 on the 56,817-row 2018 test split — vs the base
  model's 0.41218, a ~0.00023 log-price margin (the political spec's first
  champion selection; treat "ideology helps MAE" as at best marginal).
  Selection is **automated**: every training script registers its run in
  `experiments/registry.json` and the `refresh_champion` stage
  (`src/ops/experiment_runner.py`) re-selects the champion — the lowest-MAE
  untainted run created after 2026-05-18 — and rewrites
  `experiments/champion.json`. Running one training stage via `--stage` does
  **not** refresh the champion by itself.
- **Manuscript mismatch warning:** the paper text still cites pre-refresh
  numbers (16/17 features, MAE 0.429, N 470,535, old Table 1–5 values). The
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
(sales CSVs,     ed_results_clean,              modeling_dataset_             all_years_train  = 2016--17 (121,622)
 PLUTO CSV,      ideology_by_council,           canonical_v2                  all_years_test   = 2018     (56,817)
 mayoral ED      council_election_margins,      (567,997 x 53)                year2017_{train,test}
 CSVs,           city_of_yes_roll_calls)
 council Recap
 + pdf/csv,
 shapefiles)
                     ^-- the margins + roll-call builders are run BY HAND,
                         not by main.py                                    -->  models/ (7 trainers, champion)
                                                                                src/analysis/ (stats, SHAP, scenarios, causal)
                                                                                src/figures/  -> docs/figures/, docs/research_summary.md
                                                                                src/verification/ (independent re-derivation)
```

- The canonical build (`src/data_pipeline/rebuild_canonical.py`) discovers its
  raw inputs by glob and **refuses to run on zero or multiple matches**, so a
  stale duplicate can never be silently preferred; every build records exactly
  which raw files it consumed in `data/DATA_MANIFEST.json`.
- The ideology feature: `src/data_pipeline/build_ideology_scores.py` parses the
  raw ED mayoral-race CSVs and writes `data/processed/ideology_by_council.parquet`;
  `rebuild_canonical.py` merges it per `(CounDist, election_year)` as
  `dem_share` — the **only** ideology feature any model trains on. (A separate,
  abandoned ideology pipeline's columns are blacklisted; see
  `config/feature_blacklist.yaml` and `docs/DATA_LINEAGE.md`.)
- The ML split is temporal (`src/data_pipeline/split_temporal.py`,
  `TEST_YEAR=2018`): the post-2022 data refresh that added 2025 sales did **not**
  touch the train/test rows — Table 2 metric changes since came from the
  feature set, not from more training data.
- The feature set is not hard-coded anywhere in the wired pipeline: all five
  trainers select numeric, non-blacklisted, non-target columns from the splits
  at train time, so editing `config/feature_blacklist.yaml` and re-running
  `python main.py` is the complete propagation path.

## Repository layout

```
main.py                          # single-command 53-stage pipeline orchestrator
PAPER_TO_CODE_MAP.md             # paper section <-> code file index
ACTION_PLAN_FOR_SUBMISSION.md    # prioritized next steps for the paper
VERIFICATION_REPORT.md           # 7-round independent re-verification record
DATA_REFRESH_IMPACT_REPORT.md    # before/after numbers for the 2026-09 sales refresh
DELIVERY_NOTES.md                # how to reconstitute data/ from the delivery zips
Dockerfile                       # app-only deploy: Streamlit on :8501 (python:3.11-slim)
requirements.txt                 # pinned env (lightgbm 4.6, linearmodels 7.0, pypdf, ...)
config/                          # config.yaml, feature_blacklist.yaml  (leakage gate)
data/                            # gitignored except DATA_CONTRACT.md + DATA_MANIFEST.json
src/
  data_pipeline/                 # raw -> processed -> canonical -> splits; geocoder
  analysis/                      # Moran's I, OLS + robustness, spatial models, SHAP/PDP,
                                 # counterfactual scenarios, comparison tables, intervals
  causal/                        # candidate causal-ID stack (IV + rescue + City of Yes DiD);
                                 # causal_analysis.py = superseded X-learner scaffolding
  figures/                       # everything that writes docs/figures/ + research summary
  verification/                  # independent fact-check suite, one script per paper section
    editorial_audit/             # feature-blacklist enforcement (gates the pipeline)
  ops/                           # experiment_runner (champion), sanity/CI checks, automation
app/                             # Streamlit app (app.py, auth.py)
models/                          # training/ scripts; 2 tracked LightGBM text dumps;
                                 # artifacts/ (features/metrics/SHAP) are gitignored, regenerated
experiments/                     # registry.json, champion.json, results/; predictions/ gitignored
results/                         # gitignored except PIPELINE_RUN_REPORT_full_*.md snapshots;
                                 # causal/, logs/, qa evidence, FULL_PAPER_FACT_CHECK.md are local
docs/                            # ARCHITECTURE / DATA_LINEAGE / FEATURE_DICTIONARY /
                                 # WORKFLOW_AUTOMATION / figures / generated summaries
tables/                          # tracked LaTeX tables regenerated from results/
archive/                         # superseded/dead scripts + archived stale artifacts (history)
legacy/                          # the original pre-reorganization ORBIT (self-contained)
notebooks/                       # raw-data validation + EDA audit notebooks
```

## Data sources (what belongs in `data/`)

`data/` is gitignored (except `data/DATA_CONTRACT.md` and the provenance file
`data/DATA_MANIFEST.json`). On a fresh clone, `python main.py --check-data`
reports which inputs are missing. Required raw inputs:

- NYC Citywide Annualized Calendar Sales export (DOF/ACRIS; the current build
  consumed `NYC_Citywide_Annualized_Calendar_Sales_Update_20260910.csv`,
  845,607 rows, 2016–2025) → `data/raw/sales/` — keep **exactly one** file
  matching the glob there; old pulls live in `data/raw/sales/_archive/`.
- PLUTO CSV (tax-lot/building attributes) → `data/raw/pluto/` — same
  exact-one-file rule; currently the 20260105 snapshot.
- Election-district mayoral-race CSVs (2017/2021/2025, for `dem_share`) →
  `data/raw/election_results/`.
- Election-district + council-district shapefiles → `data/raw/election_districts/`
  and `data/raw/council_districts/`.
- NYC BOE council general-election Recap files (2013 PDFs; 2017/2021/2025
  CSVs) → `data/raw/council_elections/<year>/` — consumed by
  `src/data_pipeline/build_council_election_margins.py`.

Two derived datasets are **not** built by `main.py` stages; run their builders
by hand after populating raw inputs (the causal stages fail loudly and name
the missing file otherwise):

```bash
python src/data_pipeline/build_council_election_margins.py   # -> data/processed/council_election_margins.parquet
python src/data_pipeline/build_city_of_yes_roll_calls.py     # -> data/processed/city_of_yes_roll_calls.parquet
```

The roll-call builder covers City of Yes for Housing Opportunity (2024-12-05,
31 Yes / 20 No) from a verified reference table; the Carbon Neutrality
(Dec 2023) and Economic Opportunity (June 2024) member-level votes were not
programmatically retrievable (Legistar API) and are flagged for manual
retrieval — see its docstring. Full raw → processed → canonical provenance:
`docs/DATA_LINEAGE.md`.

## Models & experiment governance

Seven models were trained in the last full run (all political-mode; LightGBM
also runs base). Current test metrics — 121,622 train (2016–17) / 56,817 test
(2018) rows; 18 base / 19 political features:

| Model | Binary (git status) | Test MAE | Test R² | Features |
|---|---|---|---|---|
| **LightGBM political (champion)** | `lgbm_all_years_political.txt` (tracked) | **0.41195** | 0.56497 | 19 |
| LightGBM base | `lgbm_all_years_base.txt` (tracked) | 0.41218 | 0.56548 | 18 |
| XGBoost | `.model` (gitignored) | 0.41414 | 0.56478 | 19 |
| Random Forest (n=200) | `.joblib` (gitignored) | 0.42624 | 0.48696 | 19 |
| CatBoost | `.cbm` (gitignored) | 0.42753 | 0.54688 | 19 |
| Ridge (RidgeCV, α≈55,410) | `.joblib` (gitignored) | 0.65161 | 0.13022 | 19 |
| ElasticNet (α≈0.0428, l1_ratio=0.7) | `.joblib` (gitignored) | 0.65350 | 0.12981 | 19 |

- Only the two LightGBM text dumps are tracked; other binaries regenerate
  from training. Feature lists / metrics / SHAP summaries live in
  `models/artifacts/` (gitignored) and `experiments/results/*.json` (tracked).
- Every trainer **replaces** its `experiments/registry.json` entry in place
  (same script+name identity) rather than duplicating; `refresh_champion`
  re-selects the champion from the registry each full run;
  `logs/actions.log` is the append-only audit trail of selections.
- The linear baselines use cross-validated regularization (`RidgeCV` /
  `ElasticNetCV`, chosen α/l1_ratio recorded in results + registry) because
  default-alpha Ridge collapsed (R² → −4.97) on the refreshed
  near-collinear UNITS columns — see `VERIFICATION_REPORT.md` Round 7.
- Leakage: known leaky/identifier columns (assessed totals, `EASE-MENT`,
  the `BBL_pluto` merge-suffix artifact, post-sale identifiers, abandoned
  ideology columns, ...) are blacklisted in `config/feature_blacklist.yaml`
  and enforced by `src/verification/editorial_audit/verify_blacklist_fully.py`,
  which now **fails the pipeline** (exit 1) if either champion model contains
  one, and writes `results/editorial_audit/blacklist_audit.json` on every run.

## Analysis suite (`src/analysis/`)

- `compute_morans_i.py` — global Moran's I for price and dem_share over the 51
  council districts (k=5 KNN weights, 999 permutations, seed 42) → Section 4.2.
- `generate_regression_table.py` — three nested OLS specifications + residual
  diagnostics → Table 1 / Appendix A.2.
- `generate_robustness_checks.py` — district-clustered SEs, 51-district FE,
  borough + sale-year FE, winsorized and Cook's-D-trimmed re-estimates →
  Table 4 (`tables/table_robustness.tex` is tracked).
- `spatial_regression_model.py` — district-level (N=51) spatial lag / spatial
  error MLE models → Table 5 (`tables/table_spatial_regression.tex`).
- `political_scenarios.py` — the three counterfactual simulations
  (liberal / conservative / mixed governance) → Section 4.7 and the
  Appendix A.1 per-district CSVs under `results/political_scenarios/`.
- `generate_shap_summary.py` — mean |SHAP| for the champion **and** an explicit
  pass for `lgbm_all_years_political` (the model Table 3 / Figure 4 cite),
  skipping the political pass when the champion already is that model; falls
  back to LightGBM gain importance if `shap` is unavailable.
- `generate_pdp.py`, `confidence_intervals.py`, `sensitivity_analysis.py`,
  `champion_confidence_and_sensitivity.py` — champion-model diagnostics that
  are **not currently cited in the manuscript** (candidate additions; see the
  action plan).
- `compare_models.py` — side-by-side tabulation of every
  `experiments/results/*.json` entry (Table 2 cross-check).

## Causal identification (candidate design — not yet in the manuscript)

Wired into `main.py` as four analysis + three verification stages, all tagged
"candidate — see ACTION_PLAN_FOR_SUBMISSION.md; not yet adopted". These stages
**exit 0 even when an identification verdict fails**: a thin instrument or a
null effect is a reported finding, not a pipeline crash.

- `src/causal/build_close_election_instrument.py` → `results/causal/
  close_election_coverage.json`: coverage diagnostic run *before* any
  instrument is fitted — of the 204 district-year council races (2013/2017/
  2021/2025), close races (margin ≤ bandwidth) are ±5pp: 4, ±10pp: 7,
  ±15pp: 12, ±20pp: 17 (4/6/9/13 usable as prior-cycle instruments). A
  plain-language thin-coverage warning is printed and stored.
- `src/causal/iv_district_fe_regression.py` → `iv_district_fe_results.json`:
  2SLS (`linearmodels.iv.IV2SLS`) instrumenting `dem_share` with a prior-cycle
  close-Democratic-win indicator, district + sale-year FE, Model 4 controls +
  the UNITS columns. **Verdict: not identified at any of ±5/10/15/20pp** — the
  instrument lies exactly in the fixed-effects column space given only 4–13
  close cells.
- `src/causal/iv_district_fe_rescue.py` → `iv_district_fe_rescue_results.json`:
  the full rescue trail — a borough-FE variant (identifies at ±15/20pp but
  first-stage F = 1.04 / 0.59, with sign-unstable, implausibly scaled
  second-stage coefficients −14.7 / +18.3 vs the +0.803 non-instrumented
  benchmark) and a wider-bandwidth district-FE sweep (identifies from ±25pp;
  all first-stage F < 10: 2.44 / 7.40 / 5.13 / 7.30). Honest read: **no
  close-council-race instrument on this dataset currently supports a causal
  claim**.
- `src/causal/city_of_yes_did.py` → `city_of_yes_did_results.json`: an
  ideology/preference-proxy DiD around the 2024-12-05 Housing Opportunity
  vote. Because that action is a city-wide text amendment, the Yes/No split
  measures a district's **revealed development stance, not the policy's
  effect** — the caveat is written into every output. Pre-trend event study
  PASSes (Wald p = 0.2416, n = 355,997); main DiD +0.0812 (p = 0.1283);
  sensitivity +0.0762 (p = 0.1331) — a suggestive null.
- Each analysis has an independent verifier (`verify_iv_district_fe.py`,
  `verify_iv_district_fe_rescue.py`, `verify_city_of_yes_did.py`) that
  re-derives every number fresh from the parquets — never importing the
  analysis script — and compares field-by-field to 4 decimal places.
- `src/causal/causal_analysis.py` (the original X-learner / `causalml`
  script) is unfinished, superseded by the IV design as the recommended path,
  and kept only as historical scaffolding; it is not wired into `main.py`,
  and its `data/intermediate/` inputs are not part of the delivered repo.

## Figures & generated docs

Everything under `docs/figures/` is regenerated (tracked in git):

- `generate_political_figures.py` → ideology-vs-price scatter + temporal
  shift (**Figures 1, 2**; `political_ideology_vs_price.png`,
  `temporal_ideology_shift.png`).
- `generate_figures.py` → model-comparison bars (**Figure 3**) and the
  political-model SHAP bar (**Figure 4**; `shap_top15.png`), plus champion
  variants (`shap_top*_champion.png`) and a scenario-by-district bar chart.
  `shap_top10/15.png` always render the political model (the model Table 3
  reports); the `*_champion.png` files render whatever `champion.json`
  currently selects, so the two never silently overwrite each other.
- `generate_fig5_scenario_impact.py` → three-panel mean-scenario-impact
  figure (**Figure 5**; `scenario_impact_all_three.png`), and
  `generate_scenario_map.py` → the choropleth style variant
  (`scenario_impact_all_three_FIXED.png`; not currently cited). Both apply a
  zero-straddling two-slope color norm **only when the panel's values cross
  zero**, falling back to a plain linear norm on one-sided panels (the
  refreshed mixed-governance scenario is 50-of-51 negative — a hard crash
  under the old code).
- `generate_research_summary.py` → `docs/research_summary.md` (plain-language
  digest; political-model SHAP table is the primary paper-matching one).

## Verification / fact-check suite

`src/verification/` independently re-derives every numeric claim in the paper
from raw data and current artifacts — one script per paper section (the full
index is `PAPER_TO_CODE_MAP.md`). Run them all with:

```bash
python main.py --category verification
```

Two behavioral contracts to understand before reading a run report:

1. **Paper-value verifiers are mismatch detectors, not gates.** About ten of
   them (`verify_ml_performance`, `verify_regression_ols`, `verify_shap_
   importance`, ...) hardcode the *manuscript's* pre-refresh numbers as their
   comparison targets. After the 2026-09 refresh they correctly print
   genuine mismatches **while exiting 0**, so the pipeline reports PASS —
   that is the harness logging "manuscript ≠ current artifacts" for the
   pending manuscript rewrite, not a bug. Their reference values must not be
   edited to force matches.
2. **`verify_blacklist_fully.py` is the one structural gate**: it exits 1
   (fails the pipeline) if either champion model's trained feature list
   contains a blacklisted column, and writes the machine-readable
   `results/editorial_audit/blacklist_audit.json` every run.

The human-reviewed synthesis of a full verification pass is
`results/FULL_PAPER_FACT_CHECK.md` (local artifact — `results/` is gitignored
except the dated `PIPELINE_RUN_REPORT_full_*.md` snapshots).

## The app, CI, and deployment

```bash
python -m streamlit run app/app.py --server.headless true --server.port 8501
# -> http://localhost:8501
```

`app/auth.py` provides local accounts (app-only, not part of the research
pipeline). Address lookup goes through `src/data_pipeline/geocoder.py`
(NYC GeoSearch → BBL) with a borough-default slider fallback when a BBL is not
in the test split.

- **CI** (`.github/workflows/ci.yml`, push/PR to main/master): Python 3.13,
  `pip install -r requirements.txt`, then `src/ops/ci_smoke_check.py` and the
  pipeline sanity check.
- **Dockerfile**: app-only deploy (`python:3.11-slim`, Streamlit on :8501).
- `archive/` holds superseded scripts and historical result snapshots
  (referenced by the verification report); `legacy/` is the original
  pre-reorganization ORBIT, self-contained.

## Documentation index

| File | Purpose |
|---|---|
| `PAPER_TO_CODE_MAP.md` | paper section ↔ producer script ↔ verifier index |
| `ACTION_PLAN_FOR_SUBMISSION.md` | prioritized pre-submission work |
| `VERIFICATION_REPORT.md` | 7-round independent re-verification record |
| `DATA_REFRESH_IMPACT_REPORT.md` | before/after numbers of the 2026-09 refresh |
| `docs/ARCHITECTURE.md` | system architecture |
| `docs/DATA_LINEAGE.md` | raw → processed → canonical provenance |
| `docs/FEATURE_DICTIONARY.md` | hand-maintained feature dictionary (**known stale** — describes the 16/17-feature era; fix during the manuscript-rewrite pass) |
| `docs/WORKFLOW_AUTOMATION.md` | automation notes |
| `DELIVERY_NOTES.md` | how to reconstitute `data/` from the delivery zips |
| `data/DATA_CONTRACT.md` | data-schema contract |

## Known caveats

- **The manuscript is pre-refresh.** All paper-cited numbers (Tables 1–5,
  Moran's I, Appendix A.1, 16/17 feature counts) predate the 2026-09-10
  refresh and the 18/19 spec; regeneration is done, rewriting is pending.
- A few **narrative literals** went stale with the retrain and are flagged
  rather than silently edited: `generate_research_summary.py` claims
  "dem_share appears at rank 12" (now rank 11 in the 19-feature table), and
  `PAPER_TO_CODE_MAP.md`'s pipeline table still says 514,618 canonical rows
  (now 567,997). See `VERIFICATION_REPORT.md` Round 7, "Still needs attention".
- `models/training/train_smoke.py` (not wired into `main.py`, no paper
  artifact) selects features without consulting the blacklist.
- The council-district shapefile still describes **pre-2023 boundaries** —
  all CounDist-level results (Moran's I, Table 5, Appendix A.1) inherit that
  (`DATA_REFRESH_IMPACT_REPORT.md` §12).
- Champion MAE margin (political vs base) is ~0.00023 log-price — champion
  identity may flip on future retrains; nothing preserves it by hand.

## License & Disclaimer

This repository contains a research-grade forecasting and analysis toolkit.
It is not financial or investment advice. Forecasts and scenario outputs
should be treated as research outputs and verified independently before
operational use. Use at your own risk.

