# Research Summary

## Champion
- Selected champion: exp-20260923T165845Z (mae=0.4119535913641316, name=lgbm_all_years_political)

## Model comparison

name,mae,r2
lgbm_all_years_political,0.4119535913641316,0.5649699897075158
lgbm_all_years_base,0.4121843889444273,0.565476709183065
xgb_all_years_political_lr0.05_md6,0.41413507725020726,0.5647777233002675
rf_all_years_political_n200,0.42623686132633337,0.4869638995464056
cat_all_years_political_lr0.05_d6,0.42752815890453694,0.5468754109513326
ridge_all_years_political,0.6516142609044911,0.13021880106466954
elasticnet_all_years_political,0.6535018444530415,0.12980985154024438


## SHAP top-10 — Political model (PAPER Table 3; model=lgbm_all_years_political)

This table matches the paper's Table 3 / Figure 4. `dem_share` appears at rank 12.

                  feature  mean_abs_shap  rank
          Community Board       0.270062     1
              TOTAL UNITS       0.173424     2
        RESIDENTIAL UNITS       0.143835     3
                  landuse       0.091388     4
                numfloors       0.082551     5
                 ZIP CODE       0.081280     6
         Council District       0.079521     7
TAX CLASS AT TIME OF SALE       0.067480     8
                yearbuilt       0.064195     9
         COMMERCIAL UNITS       0.060936    10
                dem_share       0.046431    11
        Census Tract 2020       0.044857    12
               YEAR BUILT       0.039614    13
                 facilfar       0.035427    14
                 CounDist       0.026463    15
                 residfar       0.022658    16
                  BOROUGH       0.022223    17
                  commfar       0.019794    18
            election_year       0.000000    19

## Political scenario summaries
- conservative_policy_by_council: mean abs delta by council = 0.01681119409404517
- liberal_policy_by_council: mean abs delta by council = 0.05089990775084258
- mixed_governance_by_council: mean abs delta by council = 0.1140975619109605
