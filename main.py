#!/usr/bin/env python3
"""
main.py — O.R.B.I.T. single-command pipeline orchestrator.

Runs the full research pipeline behind "Political Ideology and Property Values
in New York City" end to end: builds the canonical dataset from raw sources,
trains every model reported in the paper, computes every statistic (Moran's I,
OLS regression, SHAP, counterfactual scenarios), regenerates every figure and
table, and re-runs the independent fact-check harness that verifies the
paper's numbers against freshly recomputed values.

Usage
-----
    python main.py                 # run everything, in order
    python main.py --list          # show every stage, its status, and its paper section
    python main.py --check-data    # verify data/ is populated correctly, then exit
    python main.py --category verification   # run only one category
    python main.py --stage compute_morans_i   # run one stage by id
    python main.py --from train_lgbm_base     # resume from a given stage
    python main.py --skip train_catboost train_xgboost   # skip specific stages
    python main.py --stop-on-error            # abort on first failure (default: continue)

Every stage is a pre-existing script in this repository; main.py does not
duplicate their logic, it sequences them, times them, captures their output to
results/logs/pipeline/<stage_id>.log, and writes a consolidated
results/PIPELINE_RUN_REPORT.md at the end mapping every stage to the paper
table/figure it feeds. See PAPER_TO_CODE_MAP.md for the full section-by-section
correspondence between the manuscript and this codebase.
"""

import argparse
import os
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# --------------------------------------------------------------------------
# Stage registry
# --------------------------------------------------------------------------
# Each stage is one existing script, run as `python <script> <args>` with
# cwd=ROOT (every script in this repo resolves its paths relative to the repo
# root, exactly as documented in each subfolder). "paper_ref" is the section
# of orbit_paper_CER.tex the stage's output feeds — see PAPER_TO_CODE_MAP.md.

STAGES = [
    # ---- 0. Pre-flight -----------------------------------------------
    dict(id="sanity_check", category="preflight",
         script="src/ops/pipeline_sanity_check.py", args=[],
         desc="Confirms the canonical dataset exists and has the expected shape.",
         paper_ref="N/A (pre-flight check)"),

    # ---- 1. Data pipeline (raw -> canonical) --------------------------
    dict(id="normalize_ed_results", category="data_pipeline",
         script="src/data_pipeline/normalize_ed_results.py", args=[],
         desc="Cleans ALL raw NYC BOE election-district CSVs (mayoral 2017/2021/2025 AND presidential 2016/2020/2024, both BOE row layouts via the shared boe_ed_reader) into one tidy table with a year/office/class columns, reconciled against the production ideology builder -> data/processed/ed_results_clean.parquet",
         paper_ref="3.2 Ideology Construction"),
    dict(id="build_district_crosswalk", category="data_pipeline",
         script="src/data_pipeline/build_district_crosswalk.py", args=[],
         desc="Validates the ED-to-Council-District crosswalk (range/key checks).",
         paper_ref="3.2 Ideology Construction"),
    dict(id="build_ideology_scores", category="data_pipeline",
         script="src/data_pipeline/build_ideology_scores.py", args=[],
         desc="Production dem_share pipeline -> data/processed/ideology_by_council.parquet",
         paper_ref="3.2 Ideology Construction (dem_share, the ONLY ideology feature the models train on)"),
    dict(id="build_legacy_ed_crosswalk", category="data_pipeline",
         script="src/data_pipeline/build_legacy_ed_crosswalk.py", args=[],
         desc="Builds the vintage-matched ED->council crosswalk (pre-2022 DCP ED boundary files nyed_16a/18a/20a overlaid on current council districts) -> data/processed/ed_to_council_crosswalk_legacy.parquet + the 2013-plan council correspondence. Recovers the 17.8-24.1% of 2016-2021 ED-level candidate votes whose pre-redistricting keys are missing from the 2026 crosswalk (see crosswalk_audit).",
         paper_ref="3.2 Ideology Construction (vintage audit; candidate robustness extension)"),
    dict(id="build_ideology_scores_v2", category="data_pipeline",
         script="src/data_pipeline/build_ideology_scores_v2.py", args=[],
         desc="v2 ideology measures (legacy dem_share untouched): dem_share_mayor_allcand (Dem / ALL candidate-line votes, mayoral 2017/2021/2025) and dem_share_pres = (Dem+WFP)/(Dem+WFP+Rep+Conservative) for President 2016/2020/2024, each also in a vintage-corrected variant using the legacy ED crosswalk -> data/processed/ideology_by_council_v2.parquet + ideology_pres_by_council.parquet. Reconciles ED-level Dem/Rep totals against the production builder and aborts on any mismatch.",
         paper_ref="3.2 Ideology Construction (v2 measures for the rewritten paper)"),
    dict(id="ideology_line_audit", category="data_pipeline",
         script="src/data_pipeline/ideology_line_audit.py", args=[],
         desc="Mayoral ballot-line audit 2017/2021/2025: citywide votes by ballot line (parenthetical in Unit Name), Dem/Rep/Other classification, Other's share of candidate votes, 2025 'Fight and Deliver' flag -> results/ideology_line_audit.json + .csv.",
         paper_ref="3.2 Ideology Construction (ballot-line audit)"),
    dict(id="crosswalk_audit", category="data_pipeline",
         script="src/data_pipeline/crosswalk_audit.py", args=[],
         desc="ED/crosswalk vintage audit for every raw ED file (2016-2025): candidate votes on missing crosswalk keys, COMBINED-INTO handling, borough-level losses, CD-level totals vs raw candidate sums (legacy full-to-each assignment vs area-weighted benchmark), and the vintage-matched recovery -> results/crosswalk_audit.json. Exits non-zero only if a year-level loss is NOT recovered to <=1% by the vintage crosswalk.",
         paper_ref="3.2 Ideology Construction (vintage audit)"),
    dict(id="rebuild_canonical", category="data_pipeline",
         script="src/data_pipeline/rebuild_canonical.py", args=[],
         desc="Merges sales + PLUTO + ideology -> data/canonical/modeling_dataset_canonical_v2.parquet. Coerces PLUTO structural numerics (bldgarea/lotarea/unitsres/unitstotal) and the sales LAND/GROSS SQUARE FEET columns to numeric (they shipped as strings and were silently excluded from every model), and drops exact duplicate rows. Exact counts and provenance are recorded in data/DATA_MANIFEST.json. v2: also merges the new ideology columns (dem_share_mayor_allcand, dem_share_pres, dem_share_vintage + the two *_vintage variants) and the Task-D flags (sample_pre2024, is_resid_tc12, price_ge_100k, multi_parcel_flag, building_class_category) as NEW columns; legacy columns are byte-identical and all new columns are blacklisted from every model.",
         paper_ref="3.1 Data Sources and Construction, 3.3 Canonical Dataset"),
    dict(id="freeze_schema", category="data_pipeline",
         script="src/data_pipeline/freeze_schema.py", args=[],
         desc="Writes docs/modeling_schema.csv, a frozen record of the canonical schema.",
         paper_ref="3.3 Canonical Dataset and Feature Construction"),
    dict(id="split_temporal", category="data_pipeline",
         script="src/data_pipeline/split_temporal.py", args=[],
         desc="Writes data/splits/{all_years,year2017}_{train,test}.parquet",
         paper_ref="3.4 Train/Test Split"),

    # ---- 2. Model training ---------------------------------------------
    dict(id="train_lgbm_base", category="training",
         script="models/training/train_lgbm.py", args=["--variant", "all_years", "--mode", "base"],
         desc="LightGBM base spec (no dem_share; features = numeric canonical columns minus config/feature_blacklist.yaml). Early stopping monitors a time-ordered validation slice of the training years; the held-out 2018 test set is scored exactly once, never tuned on.",
         paper_ref="4.6 Machine Learning and Feature Importance, Table 2"),
    dict(id="train_lgbm_political", category="training",
         script="models/training/train_lgbm.py", args=["--variant", "all_years", "--mode", "political"],
         desc="LightGBM political spec (base features + dem_share). Same test-untouched early-stopping protocol as train_lgbm_base.",
         paper_ref="4.6 Machine Learning and Feature Importance, Table 2 & 3 (SHAP)"),
    dict(id="train_random_forest", category="training", heavy=True,
         script="models/training/train_random_forest.py",
         args=["--variant", "all_years", "--mode", "political", "--n_estimators", "200"],
         desc="Complexity-constrained, validation-tuned Random Forest (Round 10): explicit grid max_depth {10,14,18,25} x min_samples_leaf {1,5,20} x max_features {1.0,0.5,0.33}, n_estimators=200; pre-specified SHAP-tractability cap (<=3000 mean leaves/tree) applied before MAE comparison; selection on the time-ordered training validation slice (same protocol as train_lgbm) with the test set scored exactly once; retires the unpruned rf_all_years_political_n200 (stale flag) and writes results/rf_tuning_grid.json.",
         paper_ref="4.6, Table 2 (comparison model family)"),
    # NOTE: production Table 2 Ridge/ElasticNet numbers come from
    # models/training/train_linear_baselines.py, NOT the standalone
    # train_ridge.py / train_elasticnet.py scripts below (confirmed by
    # matching experiments/results/ridge_all_years_political.json's exact
    # filename and metrics — the standalone scripts write differently-named
    # files with a different, blacklist-incomplete feature set; see
    # README.md ("Where the paper stands") and the independent verification report
    # for how this was caught). One stage trains both models.
    dict(id="train_linear_baselines", category="training", heavy=True,
         script="models/training/train_linear_baselines.py", args=["--variant", "all_years", "--mode", "political"],
         desc="Ridge + ElasticNet comparison models (trained together; this is the script that actually produced the recorded Table 2 Ridge/ElasticNet rows).",
         paper_ref="4.6, Table 2 (comparison model family)"),
    dict(id="train_catboost", category="training", heavy=True,
         script="models/training/train_catboost.py",
         args=["--variant", "all_years", "--mode", "political", "--learning_rate", "0.05", "--depth", "6"],
         desc="CatBoost comparison model (lr=0.05, depth=6).",
         paper_ref="4.6, Table 2 (comparison model family)"),
    dict(id="train_xgboost", category="training", heavy=True,
         script="models/training/train_xgboost.py",
         args=["--variant", "all_years", "--mode", "political", "--learning_rate", "0.05", "--max_depth", "6"],
         desc="XGBoost comparison model (lr=0.05, max_depth=6).",
         paper_ref="4.6, Table 2 (comparison model family)"),
    # Champion selection is deliberately automated: every training script above
    # appends its run to experiments/registry.json, and this stage then
    # re-selects the champion (lowest MAE among untainted, post-2026-05-18
    # runs) and rewrites experiments/champion.json. Running a training stage
    # via --stage alone does NOT refresh the champion — either run the full
    # pipeline or pass refresh_champion alongside.
    dict(id="refresh_champion", category="training",
         script="src/ops/experiment_runner.py", args=[],
         desc="Re-selects the champion from experiments/registry.json and rewrites experiments/champion.json: lowest test MAE among untainted, non-stale, post-2026-05-18 runs; near-ties within 0.001 MAE resolve to the base-mode spec; selection rationale is recorded in champion.json selection_note; updates selected_utc on every run.",
         paper_ref="Experiments governance (experiments/champion.json)"),

    # ---- 3. Analysis / statistics --------------------------------------
    dict(id="compute_morans_i", category="analysis",
         script="src/analysis/compute_morans_i.py", args=[],
         desc="Global Moran's I for price and dem_share (k=5 KNN weights, 999 permutations, seed=42).",
         paper_ref="4.2 Spatial Autocorrelation (Moran's I)"),
    dict(id="sample_funnel", category="analysis",
         script="src/analysis/sample_funnel.py", args=[],
         desc="Row counts from the 845,607 raw sales through every rebuild filter to the 566,956 canonical rows, the analysis base sample and the restricted analysis sample, plus Task-D flag shares and the BUILDING CLASS CATEGORY normalization effect -> results/sample_funnel.json + .csv.",
         paper_ref="3.3 Canonical Dataset (sample construction)"),
    dict(id="robustness_checks_v2", category="analysis",
         script="src/analysis/robustness_checks_v2.py", args=[],
         desc="v2 robustness suite: Models 3/4/5 recomputed with t(G-1) and wild-cluster-bootstrap inference (Webb weights, 9,999 reps, seed 42) PLUS Models 6 (district+year FE), 7 (borough x year FE), 8 (district-specific trends) and an upgraded hedonic spec — every spec x every ideology measure (legacy/allcand/pres + vintage variants) x full and restricted samples -> results/robustness_checks_v2.json.",
         paper_ref="4.3 Regression + robustness (rewritten-paper results)"),
    dict(id="compute_morans_i_v2", category="analysis",
         script="src/analysis/compute_morans_i_v2.py", args=[],
         desc="Moran's I for every ideology measure + residual Moran's I (with and without borough FE) -> results/morans_i_results_v2.json.",
         paper_ref="4.2 Spatial Autocorrelation (v2 measures)"),
    dict(id="spatial_regression_model_v2", category="analysis",
         script="src/analysis/spatial_regression_model_v2.py", args=[],
         desc="District-level spatial lag/error regressions for every ideology measure, pooled and with borough FE -> results/spatial_regression_v2.json.",
         paper_ref="4.5 Spatial regression (v2 measures)"),
    dict(id="ml_dem_share_collinearity", category="analysis",
         script="src/analysis/ml_dem_share_collinearity.py", args=[],
         desc="Diagnostics only (no retraining): dem_share is deterministic in Council District on the train/test sample (nunique per district = 1; R2 on district dummies = 1), the six duplicated feature pairs with correlations, and the test-MAE champion-selection note -> results/ml_dem_share_collinearity.json.",
         paper_ref="4.6 ML caveats (collinearity diagnostics)"),
    dict(id="iv_district_fe_regression_v2", category="analysis",
         script="src/causal/iv_district_fe_regression_v2.py", args=[],
         desc="Task-G IV additions: spec-MATCHED OLS benchmark per ideology measure at every bandwidth (the old benchmark pointed at a mismatched Table 2 Model 4), G and treated/control cell counts with a G<30 flag, and a local-linear RD-style spec at bandwidths 10-40 with first-stage F -> results/causal/iv_district_fe_results_v2.json (original 12 arms untouched).",
         paper_ref="(candidate causal-identification design, v2 benchmark)"),
    dict(id="generate_regression_table", category="analysis",
         script="src/analysis/generate_regression_table.py", args=[],
         desc="Three nested OLS specifications + residual diagnostics.",
         paper_ref="4.3 Regression Results, Table 1, Appendix A.2"),
    dict(id="generate_robustness_checks", category="analysis",
         script="src/analysis/generate_robustness_checks.py", args=[],
         desc="District-clustered SE, district FE, year FE, winsorized + Cook's-D-trimmed re-estimates of Table 1.",
         paper_ref="4.4 Robustness to Spatial Clustering and Alternative Specifications, Table 4"),
    dict(id="generate_spatial_regression", category="analysis",
         script="src/analysis/spatial_regression_model.py", args=[],
         desc="District-level spatial lag and spatial error MLE models (N=51), "
              "reusing the identical shapefile/weights construction as compute_morans_i.",
         paper_ref="4.5 Formal Spatial Regression, Table 5"),
    dict(id="political_scenarios", category="analysis",
         script="src/analysis/political_scenarios.py", args=[],
         desc="Runs the liberal/conservative/mixed-governance counterfactual simulations for EVERY eligible ranked model (rank 1 = champion.json selection; ranks 2+ = ascending test MAE via src/ops/champion_ranking.py), writing per-rank outputs plus the legacy rank-1 filenames AND (Round 10) name-keyed results/political_scenarios/by_model/<model>/<scenario>_by_council.csv copies so consumers can resolve models by name, never by rank; mixed_governance is no longer hard-wired to lgbm_all_years_political.",
         paper_ref="4.7 Counterfactual Policy Simulation"),
    dict(id="generate_shap_summary", category="analysis",
         script="src/analysis/generate_shap_summary.py", args=[],
         desc='Mean |SHAP| values for EVERY eligible ranked model via the tiered SHAP ladder in src/ops/shap_tiers.py (exact TreeSHAP on the standard sample / depth-capped RF variant for unpruned forests only / fixed 5,000-row sample / labeled gain fallback); the tier used is recorded in each payload. Round 10: 5-minute startup watchdog with one child respawn (unspent budget), tier (b) auto-skipped for complexity-constrained forests, and an honest REJECTED-via-fidelity-check note.',
         paper_ref="4.6.2 SHAP Feature Importance, Table 3"),
    dict(id="generate_pdp", category="analysis",
         script="src/analysis/generate_pdp.py", args=[],
         desc='Partial dependence plots for top features of EVERY eligible ranked model (top-5 by per-family importance (gain or |coef|); predictions via src/ops/predictors.py). Round 11: after writing this run\'s summaries, DELETES any stale models/artifacts/pdp_*.png (model/feature not in the run\'s summaries -- the pre-blacklist BBL_pluto leftover class). NOT currently cited in the paper - see PAPER_TO_CODE_MAP.md. Regenerated on every full run.',
         paper_ref="(unused — candidate addition, see action plan)"),
    dict(id="confidence_intervals", category="analysis",
         script="src/analysis/confidence_intervals.py", args=[],
         desc='Per-property prediction intervals (residual bootstrap) for EVERY eligible ranked model on the full test split (family-optimized prediction conventions via src/ops/predictors.py). NOT currently cited in the paper - see PAPER_TO_CODE_MAP.md.',
         paper_ref="(unused — candidate addition, see action plan)"),
    dict(id="sensitivity_analysis", category="analysis",
         script="src/analysis/sensitivity_analysis.py", args=[],
         desc='Feature-perturbation sensitivity for EVERY eligible ranked model (top features from the models own freshly regenerated SHAP summary). NOT currently cited in the paper - see PAPER_TO_CODE_MAP.md.',
         paper_ref="(unused — candidate addition, see action plan)"),
    dict(id="champion_confidence_and_sensitivity", category="analysis",
         script="src/analysis/champion_confidence_and_sensitivity.py", args=[],
         desc='Combined confidence+sensitivity pass for EVERY eligible ranked model; writes test_preds.parquet and intervals.csv at full test-split coverage for all of them (rank 1 = champion).',
         paper_ref="(unused — candidate addition, see action plan)"),
    dict(id="compare_models", category="analysis",
         script="src/analysis/compare_models.py", args=[],
         desc="Tabulates every experiments/results/*.json entry side by side.",
         paper_ref="4.6.1 Predictive Performance, Table 2 (cross-check)"),
    dict(id="build_close_election_instrument", category="analysis",
         script="src/causal/build_close_election_instrument.py", args=[],
         desc="Honest coverage diagnostic for the close-election instrument: counts close "
              "council-race cells by bandwidth (5/10/15/20pp), cycle, and winner party, with a "
              "plain-language thin-coverage warning. Writes results/causal/close_election_coverage.json.",
         paper_ref="(candidate causal-identification design — see README.md §Causal identification and VERIFICATION_REPORT.md Round 5; not yet adopted into the manuscript)"),
    dict(id="iv_district_fe_regression", category="analysis",
         script="src/causal/iv_district_fe_regression.py", args=[],
         desc="IV / instrumented district-FE 2SLS: prior-cycle close-council-race indicator "
              "instruments dem_share; first-stage F + per-bandwidth verdicts vs the existing "
              "non-instrumented Model 4 benchmark. Writes results/causal/iv_district_fe_results.json.",
         paper_ref="(candidate causal-identification design — see README.md §Causal identification and VERIFICATION_REPORT.md Round 5; not yet adopted into the manuscript)"),
    dict(id="iv_district_fe_rescue", category="analysis",
         script="src/causal/iv_district_fe_rescue.py", args=[],
         desc="Round-6 rescue spec search: (1a) borough-FE variant of the close-race IV; "
              "(1b) wider bandwidths on the original district-FE IV. Every attempt recorded, "
              "identified or not, each with a credibility note. Writes "
              "results/causal/iv_district_fe_rescue_results.json.",
         paper_ref="(candidate causal-identification design, Round 6 — see README.md §Causal identification and VERIFICATION_REPORT.md Round 6; not yet adopted into the manuscript)"),
    dict(id="city_of_yes_did", category="analysis",
         script="src/causal/city_of_yes_did.py", args=[],
         desc="City of Yes preference-proxy DiD: event study + main + sensitivity DiD, clustered by district. Round 12: the pre-trend Wald field previously mislabeled a chi-square(3) as an F — now stored as pre_trend_wald_chi2/_p with the implied F(3,G-1) added; the run also writes the v2 specs (wild-bootstrap main estimate, 2022-25 window, district trends, continuous dem_share_pres x post, leave-one-borough-out) to results/causal/city_of_yes_did_results_v2.json.",
         paper_ref="(candidate causal-identification design, Round 6; v2 outputs feed the rewritten paper)"),

    # ---- 4. Figures & tables --------------------------------------------
    dict(id="generate_figures", category="figures",
         script="src/figures/generate_figures.py", args=[],
         desc="SHAP / model-comparison / scenario-by-district figures -> docs/figures/ "
              "(verified: writes model_comparison.png = Fig 3, shap_top15.png = Fig 4; "
              "also writes unused scenario_impact_by_district.png and shap_top10/champion variants).",
         paper_ref="Figures 3, 4 (Table 2/3 companion figures)"),
    dict(id="generate_political_figures", category="figures",
         script="src/figures/generate_political_figures.py", args=[],
         desc="Ideology-vs-price scatter + temporal ideology shift figures.",
         paper_ref="Figures 1, 2 (Section 4.1)"),
    dict(id="generate_scenario_map", category="figures",
         script="src/figures/generate_scenario_map.py", args=[],
         desc="Choropleth of scenario impact by council district (writes scenario_impact_all_three_FIXED.png).",
         paper_ref="(unused — this choropleth does not appear in the current manuscript; see PAPER_TO_CODE_MAP.md)"),
    dict(id="generate_fig5_scenario_impact", category="figures",
         script="src/figures/generate_fig5_scenario_impact.py", args=[],
         desc="Three-panel mean-scenario-impact figure by council district; writes docs/figures/scenario_impact_all_three.png under the exact filename the manuscript includes for Figure 5 (added 2026-09-06 — previously no script produced Fig 5).",
         paper_ref="Figure 5 (Section 4.7)"),
    dict(id="generate_research_summary", category="figures",
         script="src/figures/generate_research_summary.py", args=[],
         desc="Writes docs/research_summary.md, a plain-language digest of all results.",
         paper_ref="N/A (supporting documentation, not cited in the manuscript)"),
    dict(id="generate_paper_assets", category="figures",
         script="src/figures/generate_paper_assets.py", args=[],
         desc="Round 11: builds the paper's LaTeX tables + PNG figures entirely from repo artifacts (no hard-coded numbers) into docs/paper_assets/; models resolved by experiment NAME via src/ops/champion_ranking.py; headers/captions/notes emitted verbatim (only data cells escaped); Appendix A.1 = two-block longtable with CSV-derived borough names; wide scenario summary; IV table numeric-sorted with the +-20pp district-FE duplicate footnoted; Fig 6 horizontal bars; Fig 3 readable names + champion (via champion.json) + LightGBM marks; every generated .tex wrapped in a tiny standalone doc (booktabs/amsmath/float/longtable) and pdflatex-checked when available (SKIPPED with a clear message otherwise); any compile error fails the stage.",
         paper_ref="Tables 1, 2, 4, 5; SHAP Table 3 + cross-model provenance table; Appendix A.1; Figures 3, 4, 5, 6, 7"),
    dict(id="stale_artifact_audit", category="figures",
         script="src/ops/stale_artifact_audit.py", args=[],
         desc="Round 11: LISTS (never deletes) files in docs/figures, docs/paper_assets and models/artifacts whose mtime predates this run's start -- i.e. artifacts NOT regenerated by the current run (leftover candidates). Writes results/stale_artifacts.json. Informational only; always passes.",
         paper_ref="N/A (repository hygiene audit)"),

    # ---- 5. Independent fact-check / verification harness ---------------
    # These re-derive every numeric claim in the paper directly from the
    # data/artifacts above, independent of any earlier run's output.
    dict(id="verify_abstract_and_intro", category="verification",
         script="src/verification/verify_abstract_and_intro.py", args=[],
         desc="Re-derives every Abstract- and Introduction-level number.",
         paper_ref="Abstract, Section 1 (Introduction)"),
    dict(id="verify_ideology_construction", category="verification",
         script="src/verification/verify_ideology_construction.py", args=[],
         desc="Re-derives dem_share/district_ideology worked examples and coverage stats.",
         paper_ref="3.2 Ideology Construction"),
    dict(id="verify_cd1_dem_share", category="verification",
         script="src/verification/verify_cd1_dem_share.py", args=[],
         desc="Focused re-check of the CD1 2017 dem_share worked example (0.715908).",
         paper_ref="3.2 Ideology Construction (footnote worked example)"),
    dict(id="verify_canonical_dataset_and_split", category="verification",
         script="src/verification/verify_canonical_dataset_and_split.py", args=[],
         desc="Re-derives canonical dataset shape and the train/test split sizes.",
         paper_ref="3.3 Canonical Dataset, 3.4 Train/Test Split"),
    dict(id="verify_spatial_pattern", category="verification",
         script="src/verification/verify_spatial_pattern.py", args=[],
         desc="Re-derives the 51-district bivariate dem_share/price correlation (r=0.175).",
         paper_ref="4.1 Spatial Pattern of Ideology and Price"),
    dict(id="verify_regression_missingness", category="verification",
         script="src/verification/verify_regression_missingness.py", args=[],
         desc='Re-derives the regression dropna row-count decomposition (current counts land in results/round5_part_H_missingness.json).',
         paper_ref="4.3 Regression Results (missingness paragraph)"),
    dict(id="verify_numfloors_missingness", category="verification",
         script="src/verification/verify_numfloors_missingness.py", args=[],
         desc="Focused re-check of the numfloors missingness component.",
         paper_ref="4.3 Regression Results (missingness paragraph)"),
    dict(id="verify_regression_ols", category="verification",
         script="src/verification/verify_regression_ols.py", args=[],
         desc="Re-derives all 3 OLS specifications and residual diagnostics to 4 dp.",
         paper_ref="4.3 Regression Results, Table 1, Appendix A.2"),
    dict(id="verify_spatial_regression", category="verification",
         script="src/verification/verify_spatial_regression.py", args=[],
         desc="Independently re-fits the N=51 district spatial lag/error MLE models (fresh from canonical + shapefile, no import of the analysis script) and compares against Table 5 and the pipeline's own JSON.",
         paper_ref="4.5 Formal Spatial Regression, Table 5"),
    dict(id="verify_ml_performance", category="verification",
         script="src/verification/verify_ml_performance.py", args=[],
         desc="Re-derives base vs. political MAE/R^2 (Table 2).",
         paper_ref="4.6.1 Predictive Performance"),
    dict(id="verify_shap_importance", category="verification",
         script="src/verification/verify_shap_importance.py", args=[],
         desc="Re-derives SHAP ranks and the Community-Board base-vs-political comparison.",
         paper_ref="4.6.2 SHAP Feature Importance, Table 3"),
    dict(id="verify_scenario_mechanics", category="verification",
         script="src/verification/verify_scenario_mechanics.py", args=[],
         desc="Re-derives which columns each counterfactual scenario actually modifies, plus clip rates, independently for EVERY ranked model (same ranking as the producer: src/ops/champion_ranking.py); writes per-rank results/round5_part_K_mechanics_rank{N}.json plus the legacy rank-1 file. Round 11: no paper-era 109/9/816; clip counts recomputed fresh each run and cross-checked against the producer's rank diag files.",
         paper_ref="4.7 Counterfactual Policy Simulation"),
    dict(id="verify_appendix_a1_scenarios", category="verification",
         script="src/verification/verify_appendix_a1_scenarios.py", args=[],
         desc="Re-derives all 51x3 Appendix A.1 district means INDEPENDENTLY from the rank-suffixed property-level parquets (lgbm_all_years_political resolved BY NAME via scenario_run_metadata.json; producer filters replicated: base_pred >= 9.0, pct_change clipped [-100,100], district mean) and compares per cell against (i) the name-keyed by_model CSVs and (ii) docs/paper_assets/appendix_a1_scenarios.tex (two-block longtable parsed); tolerance 0.005 pct points, not loosened. The stale embedded old-manuscript table (152 bogus exceptions) is GONE.",
         paper_ref="Appendix A.1"),
    dict(id="verify_borough_map_and_crosscutting", category="verification",
         script="src/verification/verify_borough_map_and_crosscutting.py", args=[],
         desc="Re-derives the borough<->council-district map and cross-cutting blacklist/figure checks.",
         paper_ref="Appendix A.3; cross-cutting"),
    dict(id="verify_blacklist_fully", category="verification",
         script="src/verification/editorial_audit/verify_blacklist_fully.py", args=[],
         desc="Round 11: checks EVERY eligible registry model's models/artifacts/<name>_features.json against config/feature_blacklist.yaml (all families, not LightGBM .txt only); missing feature list or any hit fails the gate. Stale '17/18-feature era' text removed.",
         paper_ref="3.3 Canonical Dataset (feature blacklist paragraph)"),
    dict(id="verify_followup_resolution", category="verification",
         script="src/verification/verify_followup_resolution.py", args=[],
         desc="Resolves the two known loose ends (additive-vs-union missingness gap; prediction-interval row-count parity). Round 11: the hard-coded 44,136/44,083 narrative and 'needs_patch_sentence' flag are removed; all values computed live.",
         paper_ref="results/FULL_PAPER_FACT_CHECK.md ('NEW findings' section)"),
    dict(id="verify_iv_district_fe", category="verification",
         script="src/verification/verify_iv_district_fe.py", args=[],
         desc="Independently re-derives the close-election IV instrument, sample counts, and "
              "identification verdicts (and, only where identified, re-fits the 2SLS) from the "
              "canonical parquet + margins parquet against results/causal/iv_district_fe_results.json.",
         paper_ref="(candidate causal-identification design — see README.md §Causal identification and VERIFICATION_REPORT.md Round 5; not yet adopted into the manuscript)"),
    dict(id="verify_iv_district_fe_rescue", category="verification",
         script="src/verification/verify_iv_district_fe_rescue.py", args=[],
         desc="Independently re-derives both Round-6 IV rescue arms and compares every field to "
              "results/causal/iv_district_fe_rescue_results.json to 4dp.",
         paper_ref="(candidate causal-identification design, Round 6 — see README.md §Causal identification and VERIFICATION_REPORT.md Round 6; not yet adopted into the manuscript)"),
    dict(id="verify_city_of_yes_did", category="verification",
         script="src/verification/verify_city_of_yes_did.py", args=[],
         desc="Independently re-fits the City of Yes DiD event study + main spec + sensitivity "
              "from the canonical parquet + roll-call parquet and compares every field to "
              "results/causal/city_of_yes_did_results.json to 4dp.",
         paper_ref="(candidate causal-identification design, Round 6 — see README.md §Causal identification and VERIFICATION_REPORT.md Round 6; not yet adopted into the manuscript)"),
    dict(id="verify_analysis_v2", category="verification",
         script="src/verification/verify_analysis_v2.py", args=[],
         desc="Independently re-derives the sample funnel counts, the v2 robustness suite (statsmodels cross-check of Model 4 vs the file AND the paper's recorded values, deterministic wild-bootstrap p, hedonic R2) and the ML collinearity diagnostics.",
         paper_ref="Rewritten-paper results (v2 robustness + funnel + ML caveats)"),
    dict(id="verify_causal_spatial_v2", category="verification",
         script="src/verification/verify_causal_spatial_v2.py", args=[],
         desc="Independently re-derives the DiD v2 specs and the Wald-field bug-fix identities, the IV matched-OLS benchmark with G/treated/control counts, and the per-measure spatial statistics (Moran's I re-derived with esda).",
         paper_ref="Rewritten-paper results (v2 causal + spatial)"),
    dict(id="verify_ideology_v2", category="verification",
         script="src/verification/verify_ideology_v2.py", args=[],
         desc="Independently re-derives every v2 ideology number (allcand/pres and their vintage variants, canonical merge mappings incl. election_year_pres, ballot-line audit, crosswalk-audit vintage resolution, Task-D flags, building-class normalization) WITHOUT importing the v2 builder, and compares to the written artifacts.",
         paper_ref="3.2 Ideology Construction (v2 measures for the rewritten paper)"),
    dict(id="verification_summary", category="verification",
         script="src/ops/verification_summary.py", args=[],
         desc="Round 11: aggregates every verifier's machine-countable status (results/verification/<verifier>.json, written via src/verification/_shared.record_status) into results/verification_summary.json, prints the per-verifier table, and EXITS NON-ZERO if any verifier reports a mismatch beyond its tolerance or failed to report at all. This is the gate that keeps verifiers from hiding mismatches.",
         paper_ref="N/A (verification harness governance)"),
]

STAGE_BY_ID = {s["id"]: s for s in STAGES}
CATEGORIES = list(dict.fromkeys(s["category"] for s in STAGES))  # preserve order, dedupe

DATA_CHECKS = [
    ("data/raw/sales", "NYC Citywide Annualized Calendar Sales CSV"),
    ("data/raw/pluto", "PLUTO CSV"),
    ("data/raw/election_results", "Election Districts mayoral-race CSVs (2017/2021/2025)"),
    ("data/raw/election_districts", "Election District + Council District shapefiles"),
    ("data/processed", "Intermediate parquet files (ed_results_clean, ideology_by_council, ...)"),
    ("data/canonical", "modeling_dataset_canonical_v2.parquet"),
    ("data/splits", "Temporal train/test splits"),
]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def check_data(verbose=True):
    ok = True
    for rel, label in DATA_CHECKS:
        p = ROOT / rel
        exists = p.exists() and any(p.iterdir()) if p.is_dir() else p.exists()
        status = "OK" if exists else "MISSING"
        if not exists:
            ok = False
        if verbose:
            print(f"  [{status:7s}] {rel:30s} {label}")
    if not ok and verbose:
        print(
            "\nMissing data. Populate data/raw/, data/processed/, data/canonical/, and "
            "data/splits/ per README.md 'Data sources & provenance' before running the "
            "data_pipeline/training/analysis stages. Verification stages that only read "
            "already-built artifacts will still fail without data/canonical/."
        )
    return ok


def run_stage(stage, log_dir):
    script = ROOT / stage["script"]
    if not script.exists():
        return dict(status="MISSING_SCRIPT", duration=0.0, log=None,
                     error=f"Script not found: {stage['script']}")

    log_path = log_dir / f"{stage['id']}.log"
    cmd = [PY, str(script)] + stage.get("args", [])
    # Force UTF-8 in children: several scripts print non-ASCII characters
    # (e.g. emoji in pipeline_sanity_check.py), which crash under the cp1252
    # codec Windows assigns to a redirected stdout by default.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"$ {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env)
    duration = time.time() - t0
    status = "PASS" if proc.returncode == 0 else "FAIL"
    return dict(status=status, duration=duration, log=str(log_path.relative_to(ROOT)),
                returncode=proc.returncode)


def write_report(results, out_path):
    lines = []
    lines.append("# O.R.B.I.T. Pipeline Run Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    # Round 11: provenance (git head + dirty state at report time)
    git_commit = None
    git_dirty = None
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT).decode().strip()
    except Exception:
        pass
    try:
        git_dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT).decode().strip())
    except Exception:
        pass
    lines.append(f"Provenance: git_commit={git_commit}, git_dirty={git_dirty}")
    # Round 11: machine-countable verification totals (when available)
    vs_path = ROOT / 'results' / 'verification_summary.json'
    if vs_path.exists():
        try:
            vs = json.loads(vs_path.read_text(encoding='utf-8'))
            lines.append(
                f"Verification: {vs.get('total_checks')} checks / "
                f"{vs.get('total_mismatches')} mismatches / "
                f"{vs.get('n_problem_verifiers')} problem verifier(s) "
                f"(see results/verification_summary.json + results/verification/*.json).")
        except Exception as _e:
            lines.append(f"Verification summary unreadable: {_e}")
    lines.append("")
    n_pass = sum(1 for r in results if r["result"]["status"] == "PASS")
    n_fail = sum(1 for r in results if r["result"]["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["result"]["status"] == "SKIPPED")
    n_missing = sum(1 for r in results if r["result"]["status"] == "MISSING_SCRIPT")
    lines.append(f"**{n_pass} passed, {n_fail} failed, {n_skip} skipped, {n_missing} missing script** "
                 f"out of {len(results)} stages.")
    lines.append("")
    lines.append("| Stage | Category | Status | Time (s) | Paper section | Log |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        s, res = r["stage"], r["result"]
        log = res.get("log") or "-"
        lines.append(
            f"| `{s['id']}` | {s['category']} | {res['status']} | "
            f"{res.get('duration', 0):.1f} | {s['paper_ref']} | {log} |"
        )
    lines.append("")
    if n_fail or n_missing:
        lines.append("## Failures")
        lines.append("")
        for r in results:
            if r["result"]["status"] in ("FAIL", "MISSING_SCRIPT"):
                s, res = r["stage"], r["result"]
                lines.append(f"### `{s['id']}` ({s['script']})")
                lines.append(f"- Status: {res['status']}")
                if res.get("log"):
                    lines.append(f"- See `{res['log']}` for full output.")
                if res.get("error"):
                    lines.append(f"- {res['error']}")
                lines.append("")
    lines.append("---")
    lines.append("Full paper-section <-> code correspondence: see `PAPER_TO_CODE_MAP.md`.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="O.R.B.I.T. single-command pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List all stages and exit.")
    parser.add_argument("--check-data", action="store_true", help="Check data/ prerequisites and exit.")
    parser.add_argument("--stage", nargs="+", metavar="ID", help="Run only these stage id(s).")
    parser.add_argument("--category", nargs="+", metavar="CAT",
                         help=f"Run only these categories: {', '.join(CATEGORIES)}")
    parser.add_argument("--skip", nargs="+", metavar="ID", default=[], help="Skip these stage id(s).")
    parser.add_argument("--from", dest="from_stage", metavar="ID",
                         help="Resume the full pipeline starting at this stage id.")
    parser.add_argument("--fast", action="store_true",
                         help="Skip 'heavy' comparison-model training stages (random forest, ridge, "
                              "elasticnet, catboost, xgboost) — keeps the two LightGBM models the "
                              "paper's headline numbers depend on.")
    parser.add_argument("--stop-on-error", action="store_true",
                         help="Abort immediately on the first failing stage (default: keep going and "
                              "report all failures at the end).")
    args = parser.parse_args()

    if args.list:
        print(f"{'ID':38s} {'CATEGORY':12s} PAPER SECTION")
        print("-" * 100)
        for s in STAGES:
            print(f"{s['id']:38s} {s['category']:12s} {s['paper_ref']}")
        return

    if args.check_data:
        print("Checking data/ prerequisites...\n")
        ok = check_data()
        sys.exit(0 if ok else 1)

    # Build the run list
    run_list = list(STAGES)

    if args.stage:
        unknown = [i for i in args.stage if i not in STAGE_BY_ID]
        if unknown:
            print(f"Unknown stage id(s): {unknown}\nRun --list to see valid ids.")
            sys.exit(2)
        run_list = [STAGE_BY_ID[i] for i in args.stage]

    elif args.category:
        unknown = [c for c in args.category if c not in CATEGORIES]
        if unknown:
            print(f"Unknown categories: {unknown}\nValid categories: {CATEGORIES}")
            sys.exit(2)
        run_list = [s for s in STAGES if s["category"] in args.category]

    elif args.from_stage:
        if args.from_stage not in STAGE_BY_ID:
            print(f"Unknown stage id: {args.from_stage}\nRun --list to see valid ids.")
            sys.exit(2)
        idx = [s["id"] for s in STAGES].index(args.from_stage)
        run_list = STAGES[idx:]

    if args.fast:
        run_list = [s for s in run_list if not s.get("heavy")]

    if args.skip:
        run_list = [s for s in run_list if s["id"] not in args.skip]

    # Pre-flight data check (informational only; stages will fail loudly on
    # their own if a specific file is missing, but this gives a fast summary
    # up front for a first-time clone).
    print("=" * 100)
    print("O.R.B.I.T. pipeline — data prerequisite check")
    print("=" * 100)
    data_ok = check_data()
    print()
    if not data_ok:
        print("Continuing anyway — stages needing missing data will fail individually and be "
              "reported at the end. Ctrl-C now if you'd rather populate data/ first.\n")

    log_dir = ROOT / "results" / "logs" / "pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Round 11: run-start marker, used by the stale_artifact_audit and
    # verification_summary stages to reference THIS run's regenerations.
    (log_dir / ".run_start_utc").write_text(str(time.time()))

    print("=" * 100)
    print(f"Running {len(run_list)} stage(s)")
    print("=" * 100)

    results = []
    t_start = time.time()
    for i, stage in enumerate(run_list, 1):
        print(f"\n[{i}/{len(run_list)}] {stage['id']}  —  {stage['desc']}")
        res = run_stage(stage, log_dir)
        results.append(dict(stage=stage, result=res))
        symbol = {"PASS": "OK", "FAIL": "FAILED", "MISSING_SCRIPT": "MISSING"}.get(res["status"], res["status"])
        print(f"    -> {symbol} ({res.get('duration', 0):.1f}s)"
              + (f"  [log: {res['log']}]" if res.get("log") else ""))
        if res["status"] != "PASS" and args.stop_on_error:
            print("\n--stop-on-error set: aborting.")
            break

    total_time = time.time() - t_start

    report_path = ROOT / "results" / "PIPELINE_RUN_REPORT.md"
    write_report(results, report_path)

    # Full-pipeline runs (no --stage/--category/--from/--skip/--fast filter)
    # also write a dated, distinctly-named copy. The always-overwritten
    # PIPELINE_RUN_REPORT.md gets clobbered by the next single-stage debug
    # run, so this dated snapshot is the durable evidence that a genuine
    # full pass happened (and when).
    is_full_run = not (args.stage or args.category or args.from_stage
                       or args.skip or args.fast)
    dated_report = None
    if is_full_run:
        dated_report = ROOT / "results" / (
            f"PIPELINE_RUN_REPORT_full_{datetime.now().strftime('%Y%m%d')}.md")
        write_report(results, dated_report)

    n_pass = sum(1 for r in results if r["result"]["status"] == "PASS")
    n_fail = len(results) - n_pass
    print("\n" + "=" * 100)
    print(f"Done in {total_time/60:.1f} min. {n_pass}/{len(results)} stages passed.")
    print(f"Full report: {report_path.relative_to(ROOT)}")
    if dated_report is not None:
        print(f"Full-run snapshot: {dated_report.relative_to(ROOT)}")
    if n_fail:
        print(f"{n_fail} stage(s) failed — see the report and results/logs/pipeline/*.log for details.")
    print("=" * 100)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
