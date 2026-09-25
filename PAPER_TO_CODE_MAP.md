# Paper &lrarr; Code Map

This file is the single index a reader should use to go from a claim, table,
or figure in `orbit_paper_CER.tex` to the exact script(s) that produced it,
and the exact script that independently re-derives it. Every path below is
relative to the repository root. Run `python main.py --list` for the same
information with live pass/fail status.

Legend: **Builds** = produces the number/figure. **Verifies** = independently
re-derives it from raw artifacts as a check (see `results/FULL_PAPER_FACT_CHECK.md`
for the consolidated, already-reviewed report).

---

## Abstract & Section 1 (Introduction)

| Claim | Builds | Verifies |
|---|---|---|
| Sample size, date range, headline correlation numbers | `src/data_pipeline/rebuild_canonical.py` | `src/verification/verify_abstract_and_intro.py` |

## Section 3.1 — Data Sources and Construction

| Step | Script | Output |
|---|---|---|
| Clean raw BOE election-district results | `src/data_pipeline/normalize_ed_results.py` | `data/processed/ed_results_clean.parquet` |
| Validate ED&rarr;Council-District crosswalk | `src/data_pipeline/build_district_crosswalk.py` | (assertion-only; no file output) |
| Merge sales + PLUTO + ideology into one table | `src/data_pipeline/rebuild_canonical.py` | `data/canonical/modeling_dataset_canonical_v2.parquet` (row/column counts regenerate with each raw refresh; the frozen record is `docs/modeling_schema.csv`) |
| Build council general-election margins (winner/runner-up per district, 2013/2017/2021/2025; party labels consolidated to Democratic/Republican/Other — Round 5 fix) | `src/data_pipeline/build_council_election_margins.py` | `data/processed/council_election_margins.parquet` |
| Build City of Yes roll-call votes (Housing Opportunity 2024-12-05 only; Carbon Neutrality / Economic Opportunity pending manual retrieval — see script docstring) | `src/data_pipeline/build_city_of_yes_roll_calls.py` | `data/processed/city_of_yes_roll_calls.parquet` |
| Freeze the modeling schema | `src/data_pipeline/freeze_schema.py` | `docs/modeling_schema.csv` |

Full raw&rarr;processed&rarr;canonical provenance, including every abandoned
intermediate table, is in `README.md` (Data sources & provenance).

## Section 3.2 — Ideology Construction (`dem_share`)

| Claim | Builds | Verifies |
|---|---|---|
| Production `dem_share` pipeline | `src/data_pipeline/build_ideology_scores.py` &rarr; `data/processed/ideology_by_council.parquet` | `src/verification/verify_ideology_construction.py` |
| CD1 2017 worked example (0.715908 / 0.711309) | (same) | `src/verification/verify_cd1_dem_share.py` |

**Important:** `dem_share` is the *only* ideology feature that reaches any
model. `district_ideology` (area-weighted) and the entire
`compute_ideology.py &rarr; ed_ideology.parquet &rarr; build_district_ideology.py`
lineage are blacklisted in `config/feature_blacklist.yaml` and feed no model
— `verify_cd1_dem_share.py` explicitly checks that no script accidentally
reads the abandoned parser. Both of those abandoned scripts were already
removed from the codebase before this reorganization; only their output
parquet files remain, purely as an audit trail.

## Section 3.3 — Canonical Dataset and Feature Construction

| Claim | Builds | Verifies |
|---|---|---|
| Row/column counts, feature list | `src/data_pipeline/rebuild_canonical.py` | `src/verification/verify_canonical_dataset_and_split.py` |
| Blacklisted-feature exclusion | `config/feature_blacklist.yaml` | `src/verification/editorial_audit/verify_blacklist_fully.py`, cross-cut in `src/verification/verify_borough_map_and_crosscutting.py` |

## Section 3.4 — Train/Test Split

| Claim | Builds | Verifies |
|---|---|---|
| `all_years` and `year2017` temporal splits | `src/data_pipeline/split_temporal.py` &rarr; `data/splits/*.parquet` | `src/verification/verify_canonical_dataset_and_split.py` |

## Section 4.1 — Spatial Pattern of Ideology and Price

| Claim | Builds | Verifies |
|---|---|---|
| Figures 1–2 (ideology vs. price scatter, temporal shift) | `src/figures/generate_political_figures.py` &rarr; `docs/figures/political_ideology_vs_price.png`, `docs/figures/temporal_ideology_shift.png` | — |
| 51-district bivariate correlation (r = 0.163) | (computed inline in the figure script) | `src/verification/verify_spatial_pattern.py` |

## Section 4.2 — Spatial Autocorrelation (Moran's I)

| Claim | Builds | Verifies |
|---|---|---|
| I = 0.5959 (price), I = 0.5797 (dem_share), both p = 0.001 | `src/analysis/compute_morans_i.py` &rarr; `results/morans_i_results.json` | re-run the same script — it is itself the independent, from-scratch recomputation the paper cites (k=5 KNN, 999 permutations, seed=42) |

Confirmed reproducible from a clean reorganized checkout in this session:
`moran_i_price = 0.59589`, `moran_i_dem_share = 0.57966`, matching the
manuscript to the reported precision.

## Section 4.3 — Regression Results

| Claim | Builds | Verifies |
|---|---|---|
| Table 1 (3 nested OLS specs), Appendix A.2 diagnostics | `src/analysis/generate_regression_table.py` &rarr; `results/regression_table.csv`, `results/regression_diagnostics.json`, `results/regression_model3_full_summary.txt` (the final `tables/table1_ols_regression.tex` is **not** script-generated — see the note under Appendix A.1) | `src/verification/verify_regression_ols.py` |
| 514,618 &rarr; 470,535 missingness decomposition | (documented in-paper) | `src/verification/verify_regression_missingness.py`, focused re-check `src/verification/verify_numfloors_missingness.py` |

## Section 4.4 — Robustness to Spatial Clustering and Alternative Specifications

| Claim | Builds | Verifies |
|---|---|---|
| Table 4 (Model 3 clustered SE; Model 4 district FE; Model 5 year FE; winsorized; Cook's-D-trimmed) | `src/analysis/generate_robustness_checks.py` &rarr; `results/robustness_checks.json` + `tables/table_robustness.tex` (this `.tex` **is** script-generated; the manuscript-sourced `table4_robustness.tex` in the paper's own `tables/` folder is hand-reformatted from it — same numbers, different layout) | the script's own internal cross-check (it re-fits Table 1's Model 3 and confirms the classical-SE row against it); Table-1 numbers themselves re-verified by `src/verification/verify_regression_ols.py` |

Headline (computed on the real canonical dataset): clustering Model 3's
standard errors by council district (51 clusters, the natural unit given how
Moran's I was computed) moves the dem_share p-value from ~1e-87 to **0.318** —
not significant at conventional levels under that correction. A specification
with full council-district fixed effects produces a *larger* coefficient that
*does* survive district clustering (p &asymp; 8e-06); a specification with
year fixed effects instead of district FE is smaller and non-significant once
clustered (p &asymp; 0.71). This three-way tension is a genuine substantive
finding (now Section 4.4 of the manuscript), not a bug.

## Section 4.5 — Formal Spatial Regression (District-Level Spatial Lag / Spatial Error)

| Claim | Builds | Verifies |
|---|---|---|
| Table 5 (+ Section 4.5 prose: &rho; = 0.709, &lambda; = 0.787, both p &lt; 0.001; dem_share p = 0.385 / 0.170 / 0.374; OLS-anchor residual Moran's I = 0.389, z = 6.05) | `src/analysis/spatial_regression_model.py` &rarr; `results/spatial_regression.json` (integrated into this repo from the `CER Papers 5` code addendum; the published `tables/table5_spatial_regression.tex` is **hand-formatted from that JSON, not auto-generated** — same status as Tables 1/3/A.1; the script additionally emits a draft `tables/table_spatial_regression.tex`) | `src/verification/verify_spatial_regression.py` (independently re-fits all three models from the canonical parquet + shapefile — no import of the analysis script — and cross-checks a pure-numpy OLS against the labeled spreg output, the regression test guarding the coefficient-labeling bug described in the analysis script's `extract()` docstring) |

## Section 4.6 — Machine Learning and Feature Importance

| Claim | Builds | Verifies |
|---|---|---|
| Champion LightGBM (base, 16 feat.) | `models/training/train_lgbm.py --variant all_years --mode base` &rarr; `models/lgbm_all_years_base.txt` | — |
| Champion LightGBM (political, 17 feat.) | `models/training/train_lgbm.py --variant all_years --mode political` &rarr; `models/lgbm_all_years_political.txt` | — |
| Table 2 (MAE/R² for LightGBM + 5 comparison families) | `models/training/train_{random_forest,linear_baselines,catboost,xgboost}.py`, tabulated by `src/analysis/compare_models.py` | `src/verification/verify_ml_performance.py` |
| Figure 3 (`model_comparison.png`) | `src/figures/generate_figures.py` | — |
| Table 3 (SHAP importance) | `src/analysis/generate_shap_summary.py` &rarr; `models/artifacts/lgbm_all_years_political_shap_summary.json` (the final `tables/table3_shap_importance.tex` is **not** script-generated — see the note under Appendix A.1); Figure 4 (`shap_top15.png`) via `src/figures/generate_figures.py` | `src/verification/verify_shap_importance.py` |

## Section 4.7 — Counterfactual Policy Simulation

| Claim | Builds | Verifies |
|---|---|---|
| Liberal / conservative / mixed-governance scenarios | `src/analysis/political_scenarios.py` | `src/verification/verify_scenario_mechanics.py` (re-derives which columns each scenario touches and re-runs the LightGBM inference — this stage alone takes several minutes; see the note in `README.md` (Where the paper stands)) |
| Figure 5 (`scenario_impact_all_three.png`) | `src/figures/generate_fig5_scenario_impact.py` (added 2026-09-06, Round-4 follow-up: until then, **no script in this repo produced that file**, and the manuscript's copy lived only in the paper's own `figures/` folder) — three 51-district panels straight from `results/political_scenarios/*_by_council.csv` | — |
| Scenario-impact figure variants | `src/figures/generate_scenario_map.py` (choropleth-style, `scenario_impact_all_three_FIXED.png`), `src/figures/generate_figures.py` (grouped-bar, `scenario_impact_by_district.png`) | — |

## Appendix A.1 — Full 51 &times; 3 Scenario Table

| Claim | Builds | Verifies |
|---|---|---|
| Table A.1 (51 &times; 3 scenario table) | `src/analysis/political_scenarios.py` &rarr; `results/political_scenarios/{liberal_policy,conservative_policy,mixed_governance}_by_council.csv` (+ `*_all_properties.parquet`) | `src/verification/verify_appendix_a1_scenarios.py` |

**Note (corrected during Round 2 verification):** the final `.tex` for
Tables 1, 3, and A.1 (and Table 2) is **not currently automated**. No script
in this repo writes a manuscript-ready `.tex` file —
`generate_regression_table.py`, `generate_shap_summary.py`, and
`political_scenarios.py` write the CSV/JSON/TXT data behind those tables
(shown in the rows above), while the `tables/table1_ols_regression.tex`,
`table3_shap_importance.tex`, and `tableA1_scenario_comparison.tex` files
the paper `\input`s live in the paper's own `tables/` folder and are
hand-formatted. An earlier draft of this map listed those `.tex` paths as
script outputs; the rows above now state the real data files each script
actually writes.

## Appendix A.2 — Regression Diagnostics

See Section 4.3 row above (same scripts).

## Appendix A.3 — Borough / Council-District Crosswalk

| Claim | Builds | Verifies |
|---|---|---|
| Borough&harr;district mapping table | (documented in-paper) | `src/verification/verify_borough_map_and_crosscutting.py` |

## Causal identification — Round 5 build (not yet in the manuscript)

Added 2026-09-13 as the candidate causal-identification design (the paper's #1
stated gap; see `README.md` "Where the paper stands"). **None of these appears in the
manuscript** — `paper_ref` in `main.py` is tagged accordingly.

| Step | Script | Output |
|---|---|---|
| Close-election instrument coverage diagnostic (how many of the 204 council district-year races fall within +/-5/10/15/20pp, by cycle and winner party, with a plain-language thin-coverage warning) | `src/causal/build_close_election_instrument.py` | `results/causal/close_election_coverage.json` |
| IV / instrumented district-FE 2SLS (prior-cycle close-race indicator instruments `dem_share`; first-stage F-stat + per-bandwidth programmatic verdict, side-by-side with the non-instrumented Model 4 district-FE result from `results/robustness_checks.json`) | `src/causal/iv_district_fe_regression.py` | `results/causal/iv_district_fe_results.json` |
| Independent re-derivation of the above (fresh from the canonical parquet + margins parquet; matches to 4 dp; never imports the analysis script) | `src/verification/verify_iv_district_fe.py` | `results/causal/iv_district_fe_verification.json` |

Both causal stages print and persist the weak-instrument finding rather than
failing the pipeline (`main.py` stages exit 0 by design — a thin instrument is
a reported finding, not a crash). With the current margins data the instrument
is **not identified at any bandwidth** (too few close NYC council races for the
district-FE design to have residual instrument variation) — see
`VERIFICATION_REPORT.md` Round 5 for the full readout.

**Superseded:** `src/causal/causal_analysis.py` — the original X-learner
(`causalml`) script — remains unfinished and is superseded by the IV design
above as the recommended path. It is kept in place only as historical
scaffolding unless the author decides otherwise; it is not wired into `main.py`.

**Feature-spec note (permanent, final):** the modeling feature set is fixed at
the 19-feature base / 20-feature political spec — i.e. the 17 original
political features (16 base + `dem_share`) **plus the 3 UNITS columns**
(`RESIDENTIAL UNITS`, `COMMERCIAL UNITS`, `TOTAL UNITS`), admitted in the
2026-09-10/11 data refresh and now accepted as the final, non-negotiable spec
going forward.

**Round 7 (2026-09-22) supersession:** the permanent spec is now **18-feature
base / 19-feature political** — `BBL_pluto` is blacklisted in
`config/feature_blacklist.yaml` as a sales/PLUTO merge-suffix artifact (a
parcel identifier, not a genuine feature), superseding the 19/20-feature spec
above. See `VERIFICATION_REPORT.md` Round 7.

## Cross-cutting fact-check output

`results/FULL_PAPER_FACT_CHECK.md` is the already-finalized, human-reviewed
synthesis of every `verify_*.py` script above. `src/verification/` does not
currently include a script that mechanically regenerates that Markdown file
from scratch — the original assembler (`archive/assemble_full_factcheck_report.py`)
read intermediate "chunk" files from a scratch directory that no longer
exists on disk, so it has been archived as dead code rather than fixed. If
you want an automated end-to-end regeneration of `FULL_PAPER_FACT_CHECK.md`
itself (not just its underlying JSON audits), that assembler needs to be
rewritten to read directly from the `results/*.json` files each `verify_*.py`
script above already writes — see `README.md` (Where the paper stands) for a scoped description
of that task.

## Not currently cited in the manuscript (existing, computed, unused assets)

These already run and already produce real output in this repository, but
nothing in `orbit_paper_CER.tex` references them yet. See `README.md` (Where the paper stands) for
what to do with each.

| Asset | Script | Output |
|---|---|---|
| Partial dependence plots, top-5 features (champion base + political models) | `src/analysis/generate_pdp.py` | `models/artifacts/pdp_lgbm_all_years_{base,political}_*.png` + `*_pdp_summary.json` (regenerated on every full run; blacklist gate = verify_blacklist_fully) |
| Per-property prediction intervals (both LightGBM models, full 56,817-row test split) | `src/analysis/confidence_intervals.py` (bootstrap `{name}_prediction_intervals.parquet`), `src/analysis/champion_confidence_and_sensitivity.py` (residual-quantile `{name}_intervals.csv` + `{name}_test_preds.parquet`) | `experiments/predictions/*` |
| Feature-perturbation sensitivity analysis | `src/analysis/sensitivity_analysis.py` | `models/artifacts/lgbm_all_years_{base,political}_sensitivity_analysis.json` (regenerated on every full run; blacklist gate = verify_blacklist_fully) |
| Combined confidence + sensitivity pass | `src/analysis/champion_confidence_and_sensitivity.py` | `experiments/predictions/*_intervals.csv` + `*_test_preds.parquet`, `models/artifacts/*_sensitivity.json` |

## Robustness checks — origin note

`src/analysis/generate_robustness_checks.py` was first added during this
reorganization (it did not exist in earlier versions of this repo) and its
findings were initially flagged here as a "candidate new subsection." The
manuscript has since adopted them as **Section 4.4** (and the addendum's
district-level spatial lag/error models as **Section 4.5**), so the mapping
entries above are the canonical home for both — see **## Section 4.4 —
Robustness to Spatial Clustering and Alternative Specifications** and
**## Section 4.5 — Formal Spatial Regression** above. The three-way
specification tension (clustered SE p ≈ 0.32 in the pre-refresh run - the current value is in results/robustness_checks.json; district-FE coefficient larger
and significant; year-FE coefficient small and insignificant once clustered)
is a genuine substantive finding, not a bug — see `README.md` (Where the paper stands) for how to
think about it.

## Not part of the reproducible pipeline (historical / superseded)

Moved to `archive/` during this reorganization; kept for audit history only,
not wired into `main.py`:

| File | Why archived |
|---|---|
| `archive/_truth_assemble.py`, `_truth_crosscut.py`, `_truth_write.py`, `_truth_assemble_data.json` | Pre-`round5` scratch verification harness; the repo's own `.gitignore` already labeled these "transient... not source" before this reorganization. |
| `archive/final_verification.py`, `archive/verify_paper_numbers.py` | Earlier, superseded rounds of manual fact-checking. Print to stdout only, produce no persisted artifact, and are not cited anywhere in `results/FULL_PAPER_FACT_CHECK.md`. |
| `archive/assemble_full_factcheck_report.py` | Reads from a chunk directory (`results/_fpc_chunks/`) that no longer exists on disk; would raise on first run. |
| `archive/create_model_performance_table.py` | Reads `experiments/results/model_comparison_raw.csv` and writes to a top-level `figures/` directory — neither exists; superseded by `src/analysis/compare_models.py` and `docs/model_comparison_table.csv`. |
| `archive/feature_engineering.py` | Reads `data/processed/modeling_dataset_with_target.parquet`, an intermediate table from the pre-canonical-v2 pipeline that the README's data-provenance table already lists as superseded and no longer produced by any script. |
