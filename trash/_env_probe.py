import importlib.util as u
for m in ['esda','libpysal','geopandas','shapely','pyproj','numpy','pandas','shap']:
    print(m, 'OK' if u.find_spec(m) else 'MISSING')