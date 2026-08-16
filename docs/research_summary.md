# Research Summary

## Champion
- Selected champion: exp-20260518T000001Z (mae=0.428972, name=lgbm_all_years_base)

## Model comparison

name,mae,r2
lgbm_all_years_base,0.428972,0.541772
lgbm_all_years_political,0.429342,0.541345
rf_all_years_political_n200,0.4370477634963115,0.5066343950498919
xgb_all_years_political_lr0.05_md6,0.4392201484272884,0.5286943913095352
cat_all_years_political_lr0.05_d6,0.4539147018162985,0.5057437362685357
ridge_all_years_political,0.6558020430385062,0.11137927403013992
elasticnet_all_years_political,0.6903724545495283,-0.00964726048578668


## SHAP top-10 — Political model (PAPER Table 3; model=lgbm_all_years_political)

This table matches the paper's Table 3 / Figure 4. `dem_share` appears at rank 12.

                  feature  mean_abs_shap  rank
          Community Board       0.274723     1
                  landuse       0.116880     2
                yearbuilt       0.105997     3
                numfloors       0.097216     4
TAX CLASS AT TIME OF SALE       0.087283     5
                BBL_pluto       0.076537     6
               YEAR BUILT       0.065516     7
                 ZIP CODE       0.058227     8
         Council District       0.054185     9
                 facilfar       0.047731    10
        Census Tract 2020       0.045003    11
                dem_share       0.034731    12
                 residfar       0.026623    13
                  commfar       0.019828    14
                  BOROUGH       0.010986    15
                 CounDist       0.004461    16
                EASE-MENT       0.000000    17
            election_year       0.000000    18

## SHAP top-10 — Current champion (secondary; model=lgbm_all_years_base)

Secondary table for the experiment currently selected in `experiments/champion.json`. Shown separately so it never silently overwrites the paper's political-model result.

                  feature  mean_abs_shap  rank
          Community Board       0.301007     1
                  landuse       0.118490     2
                yearbuilt       0.109071     3
                numfloors       0.097980     4
TAX CLASS AT TIME OF SALE       0.089214     5
                BBL_pluto       0.079787     6
               YEAR BUILT       0.075002     7
                 ZIP CODE       0.052608     8
        Census Tract 2020       0.046549     9
         Council District       0.043619    10
                 facilfar       0.039053    11
                 residfar       0.025907    12
                  commfar       0.019370    13
                  BOROUGH       0.017735    14
                 CounDist       0.009397    15
            election_year       0.000000    16

## Political scenario summaries
- conservative_policy_by_council: mean abs delta by council = 0.011072865873208562
- liberal_policy_by_council: mean abs delta by council = 0.06420467186656258
- mixed_governance_by_council: mean abs delta by council = 0.06617431688194697
