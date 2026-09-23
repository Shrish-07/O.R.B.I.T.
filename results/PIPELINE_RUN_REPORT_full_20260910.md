# O.R.B.I.T. Pipeline Run Report

Generated: 2026-09-10T23:38:27

**46 passed, 0 failed, 0 skipped, 0 missing script** out of 46 stages.

| Stage | Category | Status | Time (s) | Paper section | Log |
|---|---|---|---|---|---|
| `sanity_check` | preflight | PASS | 3.0 | N/A (pre-flight check) | results\logs\pipeline\sanity_check.log |
| `normalize_ed_results` | data_pipeline | PASS | 2.3 | 3.2 Ideology Construction | results\logs\pipeline\normalize_ed_results.log |
| `build_district_crosswalk` | data_pipeline | PASS | 1.1 | 3.2 Ideology Construction | results\logs\pipeline\build_district_crosswalk.log |
| `build_ideology_scores` | data_pipeline | PASS | 12.1 | 3.2 Ideology Construction (dem_share, the ONLY ideology feature the models train on) | results\logs\pipeline\build_ideology_scores.log |
| `rebuild_canonical` | data_pipeline | PASS | 53.2 | 3.1 Data Sources and Construction, 3.3 Canonical Dataset | results\logs\pipeline\rebuild_canonical.log |
| `freeze_schema` | data_pipeline | PASS | 2.7 | 3.3 Canonical Dataset and Feature Construction | results\logs\pipeline\freeze_schema.log |
| `split_temporal` | data_pipeline | PASS | 11.1 | 3.4 Train/Test Split | results\logs\pipeline\split_temporal.log |
| `train_lgbm_base` | training | PASS | 157.3 | 4.6 Machine Learning and Feature Importance, Table 2 | results\logs\pipeline\train_lgbm_base.log |
| `train_lgbm_political` | training | PASS | 114.8 | 4.6 Machine Learning and Feature Importance, Table 2 & 3 (SHAP) | results\logs\pipeline\train_lgbm_political.log |
| `train_random_forest` | training | PASS | 248.1 | 4.6, Table 2 (comparison model family) | results\logs\pipeline\train_random_forest.log |
| `train_linear_baselines` | training | PASS | 9.4 | 4.6, Table 2 (comparison model family) | results\logs\pipeline\train_linear_baselines.log |
| `train_catboost` | training | PASS | 110.5 | 4.6, Table 2 (comparison model family) | results\logs\pipeline\train_catboost.log |
| `train_xgboost` | training | PASS | 65.6 | 4.6, Table 2 (comparison model family) | results\logs\pipeline\train_xgboost.log |
| `refresh_champion` | training | PASS | 0.3 | Experiments governance (experiments/champion.json) | results\logs\pipeline\refresh_champion.log |
| `compute_morans_i` | analysis | PASS | 18.0 | 4.2 Spatial Autocorrelation (Moran's I) | results\logs\pipeline\compute_morans_i.log |
| `generate_regression_table` | analysis | PASS | 15.2 | 4.3 Regression Results, Table 1, Appendix A.2 | results\logs\pipeline\generate_regression_table.log |
| `generate_robustness_checks` | analysis | PASS | 61.3 | 4.4 Robustness to Spatial Clustering and Alternative Specifications, Table 4 | results\logs\pipeline\generate_robustness_checks.log |
| `generate_spatial_regression` | analysis | PASS | 11.9 | 4.5 Formal Spatial Regression, Table 5 | results\logs\pipeline\generate_spatial_regression.log |
| `political_scenarios` | analysis | PASS | 603.6 | 4.7 Counterfactual Policy Simulation | results\logs\pipeline\political_scenarios.log |
| `generate_shap_summary` | analysis | PASS | 91.5 | 4.6.2 SHAP Feature Importance, Table 3 | results\logs\pipeline\generate_shap_summary.log |
| `generate_pdp` | analysis | PASS | 86.2 | (unused — candidate addition, see action plan) | results\logs\pipeline\generate_pdp.log |
| `confidence_intervals` | analysis | PASS | 64.9 | (unused — candidate addition, see action plan) | results\logs\pipeline\confidence_intervals.log |
| `sensitivity_analysis` | analysis | PASS | 572.0 | (unused — candidate addition, see action plan) | results\logs\pipeline\sensitivity_analysis.log |
| `champion_confidence_and_sensitivity` | analysis | PASS | 217.2 | (unused — candidate addition, see action plan) | results\logs\pipeline\champion_confidence_and_sensitivity.log |
| `compare_models` | analysis | PASS | 1.7 | 4.6.1 Predictive Performance, Table 2 (cross-check) | results\logs\pipeline\compare_models.log |
| `generate_figures` | figures | PASS | 14.3 | Figures 3, 4 (Table 2/3 companion figures) | results\logs\pipeline\generate_figures.log |
| `generate_political_figures` | figures | PASS | 10.1 | Figures 1, 2 (Section 4.1) | results\logs\pipeline\generate_political_figures.log |
| `generate_scenario_map` | figures | PASS | 8.0 | (unused — this choropleth does not appear in the current manuscript; see PAPER_TO_CODE_MAP.md) | results\logs\pipeline\generate_scenario_map.log |
| `generate_fig5_scenario_impact` | figures | PASS | 7.7 | Figure 5 (Section 4.7) | results\logs\pipeline\generate_fig5_scenario_impact.log |
| `generate_research_summary` | figures | PASS | 7.0 | N/A (supporting documentation, not cited in the manuscript) | results\logs\pipeline\generate_research_summary.log |
| `verify_abstract_and_intro` | verification | PASS | 3.7 | Abstract, Section 1 (Introduction) | results\logs\pipeline\verify_abstract_and_intro.log |
| `verify_ideology_construction` | verification | PASS | 3.9 | 3.2 Ideology Construction | results\logs\pipeline\verify_ideology_construction.log |
| `verify_cd1_dem_share` | verification | PASS | 1.9 | 3.2 Ideology Construction (footnote worked example) | results\logs\pipeline\verify_cd1_dem_share.log |
| `verify_canonical_dataset_and_split` | verification | PASS | 4.9 | 3.3 Canonical Dataset, 3.4 Train/Test Split | results\logs\pipeline\verify_canonical_dataset_and_split.log |
| `verify_spatial_pattern` | verification | PASS | 4.3 | 4.1 Spatial Pattern of Ideology and Price | results\logs\pipeline\verify_spatial_pattern.log |
| `verify_regression_missingness` | verification | PASS | 4.0 | 4.3 Regression Results (missingness paragraph) | results\logs\pipeline\verify_regression_missingness.log |
| `verify_numfloors_missingness` | verification | PASS | 3.7 | 4.3 Regression Results (missingness paragraph) | results\logs\pipeline\verify_numfloors_missingness.log |
| `verify_regression_ols` | verification | PASS | 12.7 | 4.3 Regression Results, Table 1, Appendix A.2 | results\logs\pipeline\verify_regression_ols.log |
| `verify_spatial_regression` | verification | PASS | 11.8 | 4.5 Formal Spatial Regression, Table 5 | results\logs\pipeline\verify_spatial_regression.log |
| `verify_ml_performance` | verification | PASS | 2.1 | 4.6.1 Predictive Performance | results\logs\pipeline\verify_ml_performance.log |
| `verify_shap_importance` | verification | PASS | 1.7 | 4.6.2 SHAP Feature Importance, Table 3 | results\logs\pipeline\verify_shap_importance.log |
| `verify_scenario_mechanics` | verification | PASS | 472.1 | 4.7 Counterfactual Policy Simulation | results\logs\pipeline\verify_scenario_mechanics.log |
| `verify_appendix_a1_scenarios` | verification | PASS | 1.1 | Appendix A.1 | results\logs\pipeline\verify_appendix_a1_scenarios.log |
| `verify_borough_map_and_crosscutting` | verification | PASS | 10.0 | Appendix A.3; cross-cutting | results\logs\pipeline\verify_borough_map_and_crosscutting.log |
| `verify_blacklist_fully` | verification | PASS | 3.1 | 3.3 Canonical Dataset (feature blacklist paragraph) | results\logs\pipeline\verify_blacklist_fully.log |
| `verify_followup_resolution` | verification | PASS | 2.2 | results/FULL_PAPER_FACT_CHECK.md ('NEW findings' section) | results\logs\pipeline\verify_followup_resolution.log |

---
Full paper-section <-> code correspondence: see `PAPER_TO_CODE_MAP.md`.