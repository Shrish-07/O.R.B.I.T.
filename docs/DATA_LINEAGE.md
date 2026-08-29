# Data Lineage

Canonical raw snapshots:
- data\raw\council_districts\NYC_City_Council_Districts.csv
- data\raw\election_districts\Election_Districts_20260105.csv
- data\raw\election_districts\geo_export_9895bb0a-b735-497a-82d6-3c045e905b87.cpg
- data\raw\election_districts\geo_export_9895bb0a-b735-497a-82d6-3c045e905b87.dbf
- data\raw\election_districts\geo_export_9895bb0a-b735-497a-82d6-3c045e905b87.prj
- data\raw\election_districts\geo_export_9895bb0a-b735-497a-82d6-3c045e905b87.shp
- data\raw\election_districts\geo_export_9895bb0a-b735-497a-82d6-3c045e905b87.shx
- data\raw\election_districts\NYC_City_Council_Districts.cpg
- data\raw\election_districts\NYC_City_Council_Districts.csv
- data\raw\election_districts\NYC_City_Council_Districts.dbf
- data\raw\election_districts\NYC_City_Council_Districts.prj
- data\raw\election_districts\NYC_City_Council_Districts.shp
- data\raw\election_districts\NYC_City_Council_Districts.shx
- data\raw\election_districts\NYC_City_Council_Districts.xml
- data\raw\election_results\ed_results_2017_mayor.csv
- data\raw\election_results\ed_results_2021_mayor.csv
- data\raw\election_results\ed_results_2025_mayor.csv
- data\raw\pluto\Primary_Land_Use_Tax_Lot_Output_(PLUTO)_20260105.csv
- data\raw\sales\NYC_Citywide_Annualized_Calendar_Sales_Update_20260105.csv

Processed datasets:
- data\processed\district_ideology.parquet
- data\processed\ed_ideology.parquet
- data\processed\ed_results_clean.parquet
- data\processed\ed_to_council_crosswalk.parquet
- data\processed\ideology_by_council.parquet

Superseded / removed processed modeling artifacts (no longer on disk; replaced by the canonical dataset below):
- data\processed\modeling_dataset.parquet
- data\processed\modeling_dataset_fe.parquet
- data\processed\modeling_dataset_fe_imputed.parquet
- data\processed\modeling_dataset_with_target.parquet
- data\processed\sales_pluto_ideology.parquet

Authoritative canonical modeling dataset (used by pipeline_sanity_check.py, freeze_schema.py, split_temporal.py, political_scenarios.py):
- data\canonical\modeling_dataset_canonical_v2.parquet
