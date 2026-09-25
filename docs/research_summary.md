# Research Summary

## Champion
- Selected champion: exp-20260925T041001Z (mae=0.4167477032408586, name=rf_all_years_political_n200)

## Model comparison

name,mae,r2
rf_all_years_political_n200,0.4167477032408586,0.530168999422612
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

## SHAP top-10 — Current champion (secondary; model=rf_all_years_political_n200)

Secondary table for the experiment currently selected in `experiments/champion.json`. Shown separately so it never silently overwrites the paper's political-model result.

                  feature     gain  mean_abs_shap  rank
              TOTAL UNITS 0.143442       0.143442     1
                  BOROUGH 0.137268       0.137268     2
               YEAR BUILT 0.097945       0.097945     3
        RESIDENTIAL UNITS 0.077114       0.077114     4
        Census Tract 2020 0.074982       0.074982     5
          Community Board 0.061242       0.061242     6
                 ZIP CODE 0.051376       0.051376     7
               unitstotal 0.050725       0.050725     8
                yearbuilt 0.046660       0.046660     9
                numfloors 0.042686       0.042686    10
                 unitsres 0.032234       0.032234    11
         COMMERCIAL UNITS 0.029444       0.029444    12
                dem_share 0.028856       0.028856    13
TAX CLASS AT TIME OF SALE 0.021495       0.021495    14
                 CounDist 0.020675       0.020675    15
         Council District 0.018398       0.018398    16
                 facilfar 0.017678       0.017678    17
                  landuse 0.012919       0.012919    18
                 residfar 0.012520       0.012520    19
                  commfar 0.009887       0.009887    20
        GROSS SQUARE FEET 0.004629       0.004629    21
         LAND SQUARE FEET 0.003897       0.003897    22
                 bldgarea 0.002857       0.002857    23
                  lotarea 0.001072       0.001072    24
            election_year 0.000000       0.000000    25

## Political scenario summaries
- conservative_policy_by_council: mean abs delta by council = 0.013694610134087194
- liberal_policy_by_council: mean abs delta by council = 0.013280182382994032
- mixed_governance_by_council: mean abs delta by council = 0.10404138464344531
