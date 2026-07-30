import pandas as pd, geopandas as gpd
df = pd.read_parquet('data/canonical/modeling_dataset_canonical_v2.parquet')
print('CANON shape', df.shape)
print('CANON cols:', list(df.columns))
for c in ['CounDist','Council District','COUNCIL_DISTRICT','dem_share','target_log_price','district_ideology']:
    print(c, 'present' if c in df.columns else 'ABSENT')
gdf = gpd.read_file('data/raw/election_districts/NYC_City_Council_Districts.shp')
print('SHP shape', gdf.shape, 'crs', gdf.crs)
print('SHP cols:', list(gdf.columns))
print(gdf.head(3).to_string())