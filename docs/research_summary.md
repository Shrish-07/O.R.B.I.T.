# Research Summary

## Champion
- Selected champion: exp-20261016T161155Z (mae=0.40161417910660113, name=rf_all_years_political_tuned)

## Model comparison

name,mae,r2
rf_all_years_political_tuned,0.40161417910660113,0.5900387792297042
cat_all_years_political_lr0.05_d6,0.41736433913362825,0.5812828576992348
xgb_all_years_political_lr0.05_md6,0.41769515091835496,0.5702048188828388
lgbm_all_years_political,0.43994123422895204,0.5433591928764994
lgbm_all_years_base,0.44182670516761613,0.5350382066966075
ridge_all_years_political,0.629341791343628,0.1791541147004576
elasticnet_all_years_political,0.6312658045356286,0.1770844213769116


## SHAP top-10 — Political model (PAPER Table 3; model=lgbm_all_years_political)

This table matches the paper's Table 3 / Figure 4. `dem_share` currently appears at rank 11 of 24 non-zero features (computed live from the artifact).

                  feature  mean_abs_shap  rank
          Community Board       0.320876     1
        GROSS SQUARE FEET       0.151545     2
         LAND SQUARE FEET       0.149027     3
                numfloors       0.107309     4
              TOTAL UNITS       0.092508     5
        RESIDENTIAL UNITS       0.088492     6
         Council District       0.077458     7
                 ZIP CODE       0.069502     8
               unitstotal       0.064327     9
                yearbuilt       0.053045    10
                dem_share       0.046846    11
TAX CLASS AT TIME OF SALE       0.046803    12
        Census Tract 2020       0.041098    13
               YEAR BUILT       0.037387    14
                 facilfar       0.036598    15
                 unitsres       0.035623    16
         COMMERCIAL UNITS       0.029497    17
                  landuse       0.025257    18
                 residfar       0.018472    19
                  BOROUGH       0.018472    20
                  commfar       0.018381    21
                 CounDist       0.012615    22
                 bldgarea       0.006318    23
                  lotarea       0.001068    24
            election_year       0.000000    25

## SHAP top-10 — Current champion (secondary; model=rf_all_years_political_tuned)

Secondary table for the experiment currently selected in `experiments/champion.json`. Shown separately so it never silently overwrites the paper's political-model result.

                  feature  mean_abs_shap  rank
                  BOROUGH       0.172686     1
        RESIDENTIAL UNITS       0.149263     2
                 ZIP CODE       0.115395     3
              TOTAL UNITS       0.113047     4
          Community Board       0.107521     5
                 unitsres       0.076338     6
         COMMERCIAL UNITS       0.058728     7
               unitstotal       0.055924     8
                 facilfar       0.045574     9
                numfloors       0.042216    10
                yearbuilt       0.038424    11
         Council District       0.038285    12
        Census Tract 2020       0.035139    13
                 CounDist       0.035049    14
                dem_share       0.031892    15
               YEAR BUILT       0.028664    16
                  commfar       0.014629    17
TAX CLASS AT TIME OF SALE       0.013906    18
                  landuse       0.012717    19
                 residfar       0.011755    20
        GROSS SQUARE FEET       0.004264    21
                 bldgarea       0.003998    22
         LAND SQUARE FEET       0.000858    23
                  lotarea       0.000800    24
            election_year       0.000000    25

## SHAP top-10 -- all other eligible interpreted models (Round 9)

Rank ordering recomputed fresh via src/ops/champion_ranking.py; audit in results/rank1_rank2_selection.json. Each table carries the tier recorded in its own payload.

### cat_all_years_political_lr0.05_d6 (method=shap, tier a)
*exact TreeSHAP on the standard sampled test slice (n<=2000, seed 42, low-NA feature sampling).*

          feature  mean_abs_shap
RESIDENTIAL UNITS       0.158406
          BOROUGH       0.132069
  Community Board       0.120588
        numfloors       0.114632
      TOTAL UNITS       0.106643
         ZIP CODE       0.089900
       unitstotal       0.070195
 COMMERCIAL UNITS       0.062128
        dem_share       0.061169
         facilfar       0.057509

### elasticnet_all_years_political (method=shap, tier linear_exact)
*exact SHAP for a linear model (shap.LinearExplainer, interventional, on the model's own median-imputed + standardized feature space as trained), standard sampled test slice (n<=2000, seed 42).*

                  feature  mean_abs_shap
          Community Board       0.175594
                 unitsres       0.105339
                 residfar       0.076313
                numfloors       0.056240
                  landuse       0.043478
TAX CLASS AT TIME OF SALE       0.029310
                  commfar       0.021308
        Census Tract 2020       0.013910
                yearbuilt       0.010056
        RESIDENTIAL UNITS       0.005125

### lgbm_all_years_base (method=shap, tier a)
*exact TreeSHAP on the standard sampled test slice (n<=2000, seed 42, low-NA feature sampling).*

          feature  mean_abs_shap
  Community Board       0.316491
 LAND SQUARE FEET       0.157802
GROSS SQUARE FEET       0.150241
        numfloors       0.111749
 Council District       0.092893
RESIDENTIAL UNITS       0.090888
      TOTAL UNITS       0.090483
         ZIP CODE       0.089384
       unitstotal       0.057063
        yearbuilt       0.052723

### ridge_all_years_political (method=shap, tier linear_exact)
*exact SHAP for a linear model (shap.LinearExplainer, interventional, on the model's own median-imputed + standardized feature space as trained), standard sampled test slice (n<=2000, seed 42).*

                  feature  mean_abs_shap
          Community Board       0.106658
                  landuse       0.071076
                 residfar       0.070004
                 unitsres       0.061152
                  BOROUGH       0.054687
                numfloors       0.054293
TAX CLASS AT TIME OF SALE       0.051830
               unitstotal       0.047919
                  commfar       0.030017
        Census Tract 2020       0.023620

### xgb_all_years_political_lr0.05_md6 (method=shap, tier a)
*exact TreeSHAP on the standard sampled test slice (n<=2000, seed 42, low-NA feature sampling).*

          feature  mean_abs_shap
          BOROUGH       0.228510
      TOTAL UNITS       0.150244
RESIDENTIAL UNITS       0.140284
        numfloors       0.110285
  Community Board       0.103401
         ZIP CODE       0.086108
       unitstotal       0.069116
         unitsres       0.068441
 COMMERCIAL UNITS       0.064479
         facilfar       0.059704

## Political scenario summaries
- conservative_policy_by_council: mean abs delta by council = 0.00809901372769
- liberal_policy_by_council: mean abs delta by council = 0.00803336466980062
- mixed_governance_by_council: mean abs delta by council = 0.036416168926460386
