# O.R.B.I.T. Pipeline Run Report

Generated: 2026-09-23T22:01:47

**47 passed, 6 failed, 0 skipped, 0 missing script** out of 53 stages.

| Stage | Category | Status | Time (s) | Paper section | Log |
|---|---|---|---|---|---|
| `sanity_check` | preflight | PASS | 2.2 | N/A (pre-flight check) | results\logs\pipeline\sanity_check.log |
| `normalize_ed_results` | data_pipeline | PASS | 1.1 | 3.2 Ideology Construction | results\logs\pipeline\normalize_ed_results.log |
| `build_district_crosswalk` | data_pipeline | PASS | 0.6 | 3.2 Ideology Construction | results\logs\pipeline\build_district_crosswalk.log |
| `build_ideology_scores` | data_pipeline | PASS | 3.0 | 3.2 Ideology Construction (dem_share, the ONLY ideology feature the models train on) | results\logs\pipeline\build_ideology_scores.log |
| `rebuild_canonical` | data_pipeline | PASS | 14.6 | 3.1 Data Sources and Construction, 3.3 Canonical Dataset | results\logs\pipeline\rebuild_canonical.log |
| `freeze_schema` | data_pipeline | PASS | 1.5 | 3.3 Canonical Dataset and Feature Construction | results\logs\pipeline\freeze_schema.log |
| `split_temporal` | data_pipeline | PASS | 2.5 | 3.4 Train/Test Split | results\logs\pipeline\split_temporal.log |
| `train_lgbm_base` | training | PASS | 9.8 | 4.6 Machine Learning and Feature Importance, Table 2 | results\logs\pipeline\train_lgbm_base.log |
| `train_lgbm_political` | training | PASS | 21.3 | 4.6 Machine Learning and Feature Importance, Table 2 & 3 (SHAP) | results\logs\pipeline\train_lgbm_political.log |
| `train_random_forest` | training | PASS | 83.3 | 4.6, Table 2 (comparison model family) | results\logs\pipeline\train_random_forest.log |
| `train_linear_baselines` | training | PASS | 26.8 | 4.6, Table 2 (comparison model family) | results\logs\pipeline\train_linear_baselines.log |
| `train_catboost` | training | PASS | 33.3 | 4.6, Table 2 (comparison model family) | results\logs\pipeline\train_catboost.log |
| `train_xgboost` | training | PASS | 13.3 | 4.6, Table 2 (comparison model family) | results\logs\pipeline\train_xgboost.log |
| `refresh_champion` | training | PASS | 0.2 | Experiments governance (experiments/champion.json) | results\logs\pipeline\refresh_champion.log |
| `compute_morans_i` | analysis | PASS | 5.2 | 4.2 Spatial Autocorrelation (Moran's I) | results\logs\pipeline\compute_morans_i.log |
| `generate_regression_table` | analysis | PASS | 4.3 | 4.3 Regression Results, Table 1, Appendix A.2 | results\logs\pipeline\generate_regression_table.log |
| `generate_robustness_checks` | analysis | PASS | 22.3 | 4.4 Robustness to Spatial Clustering and Alternative Specifications, Table 4 | results\logs\pipeline\generate_robustness_checks.log |
| `generate_spatial_regression` | analysis | PASS | 2.7 | 4.5 Formal Spatial Regression, Table 5 | results\logs\pipeline\generate_spatial_regression.log |
| `political_scenarios` | analysis | FAIL | 2.5 | 4.7 Counterfactual Policy Simulation | results\logs\pipeline\political_scenarios.log |
| `generate_shap_summary` | analysis | FAIL | 1.9 | 4.6.2 SHAP Feature Importance, Table 3 | results\logs\pipeline\generate_shap_summary.log |
| `generate_pdp` | analysis | FAIL | 2.3 | (unused — candidate addition, see action plan) | results\logs\pipeline\generate_pdp.log |
| `confidence_intervals` | analysis | PASS | 34.4 | (unused — candidate addition, see action plan) | results\logs\pipeline\confidence_intervals.log |
| `sensitivity_analysis` | analysis | PASS | 305.4 | (unused — candidate addition, see action plan) | results\logs\pipeline\sensitivity_analysis.log |
| `champion_confidence_and_sensitivity` | analysis | FAIL | 3.0 | (unused — candidate addition, see action plan) | results\logs\pipeline\champion_confidence_and_sensitivity.log |
| `compare_models` | analysis | PASS | 1.1 | 4.6.1 Predictive Performance, Table 2 (cross-check) | results\logs\pipeline\compare_models.log |
| `build_close_election_instrument` | analysis | PASS | 0.8 | (candidate causal-identification design — see ACTION_PLAN_FOR_SUBMISSION.md; not yet adopted into the manuscript) | results\logs\pipeline\build_close_election_instrument.log |
| `iv_district_fe_regression` | analysis | PASS | 4.7 | (candidate causal-identification design — see ACTION_PLAN_FOR_SUBMISSION.md; not yet adopted into the manuscript) | results\logs\pipeline\iv_district_fe_regression.log |
| `iv_district_fe_rescue` | analysis | PASS | 53.7 | (candidate causal-identification design, Round 6 — see ACTION_PLAN_FOR_SUBMISSION.md; not yet adopted into the manuscript) | results\logs\pipeline\iv_district_fe_rescue.log |
| `city_of_yes_did` | analysis | PASS | 24.1 | (candidate causal-identification design, Round 6 — see ACTION_PLAN_FOR_SUBMISSION.md; not yet adopted into the manuscript) | results\logs\pipeline\city_of_yes_did.log |
| `generate_figures` | figures | PASS | 5.2 | Figures 3, 4 (Table 2/3 companion figures) | results\logs\pipeline\generate_figures.log |
| `generate_political_figures` | figures | PASS | 2.5 | Figures 1, 2 (Section 4.1) | results\logs\pipeline\generate_political_figures.log |
| `generate_scenario_map` | figures | PASS | 1.9 | (unused — this choropleth does not appear in the current manuscript; see PAPER_TO_CODE_MAP.md) | results\logs\pipeline\generate_scenario_map.log |
| `generate_fig5_scenario_impact` | figures | PASS | 1.9 | Figure 5 (Section 4.7) | results\logs\pipeline\generate_fig5_scenario_impact.log |
| `generate_research_summary` | figures | PASS | 2.4 | N/A (supporting documentation, not cited in the manuscript) | results\logs\pipeline\generate_research_summary.log |
| `verify_abstract_and_intro` | verification | PASS | 1.2 | Abstract, Section 1 (Introduction) | results\logs\pipeline\verify_abstract_and_intro.log |
| `verify_ideology_construction` | verification | PASS | 1.2 | 3.2 Ideology Construction | results\logs\pipeline\verify_ideology_construction.log |
| `verify_cd1_dem_share` | verification | PASS | 0.8 | 3.2 Ideology Construction (footnote worked example) | results\logs\pipeline\verify_cd1_dem_share.log |
| `verify_canonical_dataset_and_split` | verification | PASS | 1.5 | 3.3 Canonical Dataset, 3.4 Train/Test Split | results\logs\pipeline\verify_canonical_dataset_and_split.log |
| `verify_spatial_pattern` | verification | PASS | 1.4 | 4.1 Spatial Pattern of Ideology and Price | results\logs\pipeline\verify_spatial_pattern.log |
| `verify_regression_missingness` | verification | PASS | 1.4 | 4.3 Regression Results (missingness paragraph) | results\logs\pipeline\verify_regression_missingness.log |
| `verify_numfloors_missingness` | verification | PASS | 1.3 | 4.3 Regression Results (missingness paragraph) | results\logs\pipeline\verify_numfloors_missingness.log |
| `verify_regression_ols` | verification | PASS | 3.4 | 4.3 Regression Results, Table 1, Appendix A.2 | results\logs\pipeline\verify_regression_ols.log |
| `verify_spatial_regression` | verification | PASS | 2.7 | 4.5 Formal Spatial Regression, Table 5 | results\logs\pipeline\verify_spatial_regression.log |
| `verify_ml_performance` | verification | PASS | 0.8 | 4.6.1 Predictive Performance | results\logs\pipeline\verify_ml_performance.log |
| `verify_shap_importance` | verification | FAIL | 0.7 | 4.6.2 SHAP Feature Importance, Table 3 | results\logs\pipeline\verify_shap_importance.log |
| `verify_scenario_mechanics` | verification | FAIL | 2.4 | 4.7 Counterfactual Policy Simulation | results\logs\pipeline\verify_scenario_mechanics.log |
| `verify_appendix_a1_scenarios` | verification | PASS | 0.6 | Appendix A.1 | results\logs\pipeline\verify_appendix_a1_scenarios.log |
| `verify_borough_map_and_crosscutting` | verification | PASS | 2.8 | Appendix A.3; cross-cutting | results\logs\pipeline\verify_borough_map_and_crosscutting.log |
| `verify_blacklist_fully` | verification | PASS | 1.8 | 3.3 Canonical Dataset (feature blacklist paragraph) | results\logs\pipeline\verify_blacklist_fully.log |
| `verify_followup_resolution` | verification | PASS | 1.4 | results/FULL_PAPER_FACT_CHECK.md ('NEW findings' section) | results\logs\pipeline\verify_followup_resolution.log |
| `verify_iv_district_fe` | verification | PASS | 3.1 | (candidate causal-identification design — see ACTION_PLAN_FOR_SUBMISSION.md; not yet adopted into the manuscript) | results\logs\pipeline\verify_iv_district_fe.log |
| `verify_iv_district_fe_rescue` | verification | PASS | 63.3 | (candidate causal-identification design, Round 6 — see ACTION_PLAN_FOR_SUBMISSION.md; not yet adopted into the manuscript) | results\logs\pipeline\verify_iv_district_fe_rescue.log |
| `verify_city_of_yes_did` | verification | PASS | 25.1 | (candidate causal-identification design, Round 6 — see ACTION_PLAN_FOR_SUBMISSION.md; not yet adopted into the manuscript) | results\logs\pipeline\verify_city_of_yes_did.log |

## Failures

### `political_scenarios` (src/analysis/political_scenarios.py)
- Status: FAIL
- See `results\logs\pipeline\political_scenarios.log` for full output.

### `generate_shap_summary` (src/analysis/generate_shap_summary.py)
- Status: FAIL
- See `results\logs\pipeline\generate_shap_summary.log` for full output.

### `generate_pdp` (src/analysis/generate_pdp.py)
- Status: FAIL
- See `results\logs\pipeline\generate_pdp.log` for full output.

### `champion_confidence_and_sensitivity` (src/analysis/champion_confidence_and_sensitivity.py)
- Status: FAIL
- See `results\logs\pipeline\champion_confidence_and_sensitivity.log` for full output.

### `verify_shap_importance` (src/verification/verify_shap_importance.py)
- Status: FAIL
- See `results\logs\pipeline\verify_shap_importance.log` for full output.

### `verify_scenario_mechanics` (src/verification/verify_scenario_mechanics.py)
- Status: FAIL
- See `results\logs\pipeline\verify_scenario_mechanics.log` for full output.

---
Full paper-section <-> code correspondence: see `PAPER_TO_CODE_MAP.md`.