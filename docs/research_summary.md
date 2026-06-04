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
          Community Board       0.274723
                  landuse       0.116880
                yearbuilt       0.105997
                numfloors       0.097216
TAX CLASS AT TIME OF SALE       0.087283
                BBL_pluto       0.076537
               YEAR BUILT       0.065516
                 ZIP CODE       0.058227
         Council District       0.054185
                 facilfar       0.047731

## Political scenario summaries
- conservative_policy_by_council: mean abs delta by council = 0.011072865873208562
- liberal_policy_by_council: mean abs delta by council = 0.06420467186656258
- mixed_governance_by_council: mean abs delta by council = 0.06617431688194697
