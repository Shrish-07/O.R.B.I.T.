# Research Summary

## Champion
- Selected champion: exp-20260518T000001Z (mae=0.428972)

## Model comparison

name,mae,r2
lgbm_all_years_base,0.428972,0.541772
lgbm_all_years_political,0.429342,0.541345
rf_all_years_political_n200,0.4370477634963115,0.5066343950498919
xgb_all_years_political_lr0.05_md6,0.4392201484272884,0.5286943913095352
cat_all_years_political_lr0.05_d6,0.4539147018162985,0.5057437362685357
ridge_all_years_political,0.6558020430385062,0.11137927403013992
elasticnet_all_years_political,0.6903724545495283,-0.00964726048578668


## SHAP top-10

                  feature  mean_abs_shap
          Community Board       0.301007
                  landuse       0.118490
                yearbuilt       0.109071
                numfloors       0.097980
TAX CLASS AT TIME OF SALE       0.089214
                BBL_pluto       0.079787
               YEAR BUILT       0.075002
                 ZIP CODE       0.052608
        Census Tract 2020       0.046549
         Council District       0.043619

## Political scenario summaries
- conservative_policy_by_council: mean abs delta by council = 0.011072865873208562
- liberal_policy_by_council: mean abs delta by council = 0.06420467186656258
- mixed_governance_by_council: mean abs delta by council = 0.06617431688194697
