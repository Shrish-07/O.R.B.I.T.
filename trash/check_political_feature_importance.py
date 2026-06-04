import lightgbm as lgb
import json
model = lgb.Booster(model_file='models/lgbm_all_years_political.txt')
features = json.load(open('models/artifacts/lgbm_all_years_political_features.json'))
imp = model.feature_importance()
for f, v in zip(features, imp):
    print(f, v)
