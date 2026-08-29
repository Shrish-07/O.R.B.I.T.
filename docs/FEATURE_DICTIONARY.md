# Feature Dictionary

The champion **base** model uses 16 features; the champion **political** model uses the same 16 plus `dem_share` (17 total). Both are sourced from `data/canonical/modeling_dataset_canonical_v2.parquet`.

| Feature name | Dtype | Description | Used in |
|--------------|-------|-------------|---------|
| BOROUGH | int64 | Borough identifier (1-5) at time of sale | both |
| ZIP CODE | float64 | Sales ZIP code | both |
| YEAR BUILT | float64 | Year built recorded on the sales record | both |
| TAX CLASS AT TIME OF SALE | int64 | NYC tax class at the time of sale | both |
| Community Board | float64 | NYC Community Board identifier | both |
| Council District | float64 | NYC Council District identifier | both |
| Census Tract 2020 | float64 | 2020 Census tract identifier | both |
| landuse | float64 | Land-use code | both |
| numfloors | float64 | Number of floors (PLUTO) | both |
| yearbuilt | float64 | Year built (PLUTO) | both |
| residfar | float64 | Residential floor-area ratio (PLUTO) | both |
| commfar | float64 | Commercial floor-area ratio (PLUTO) | both |
| facilfar | float64 | Facility floor-area ratio (PLUTO) | both |
| BBL_pluto | float64 | Borough-Block-Lot identifier (PLUTO) | both |
| election_year | int64 | Election year for the ideological feature merge | both |
| CounDist | float64 | Council district identifier (lowercase form) | both |
| dem_share | float64 | Democratic two-party vote share, aggregated across election years | political |

## Excluded (blacklisted) features

Complete exclusion list from `config/feature_blacklist.yaml`. Reasons are the comments already given in that file.

### Target & direct leakage
*Reason from `config/feature_blacklist.yaml`: Target & direct leakage*

- target_log_price
- sale price
- sale_price
- sale_price_num
- log_sale_price

### PLUTO assessment leakage
*Reason from `config/feature_blacklist.yaml`: PLUTO assessment leakage — 2026 snapshot values encode historical price appreciation. These fields originate from the PLUTO 2026 snapshot and leak future assessment information into historical sales features. They must be excluded from model training.*

- assesstot
- exempttot
- assessland
- appbbl
- taxmap
- plutomapid

### Temporal identifiers (split-only)
*Reason from `config/feature_blacklist.yaml`: Temporal identifiers (split-only)*

- sale date
- sale_date
- sale_year
- sale_month
- sale_quarter
- sale_day
- sale_timestamp

### Administrative date fields
*Reason from `config/feature_blacklist.yaml`: Administrative date fields*

- basempdate
- dcasdate
- edesigdate
- landmkdate
- masdate
- polidate
- rpaddate
- zoningdate
- appdate

### Post-sale identifiers / row keys
*Reason from `config/feature_blacklist.yaml`: Post-sale identifiers / row keys*

- bin
- bbl
- lot
- block
- taxlot
- record_id
- sale_id

### Duplicate / raw geo fields
*Reason from `config/feature_blacklist.yaml`: Duplicate / raw geo fields*

- latitude
- longitude
- xcoord
- ycoord

### Raw ideology / election variables
*Reason from `config/feature_blacklist.yaml`: Raw ideology / election variables (ONLY included in political models)*

- district_ideology
- weighted_dem
- weighted_rep
- dem_vote_share
- rep_vote_share
- turnout

### Derived target encodings (forbidden)
*Reason from `config/feature_blacklist.yaml`: Derived target encodings (forbidden)*

- mean_price
- borough_mean_price
- council_district_mean_price
- building class category_mean_price

### Null features
*Reason from `config/feature_blacklist.yaml`: 100% null features — contribute zero signal, waste model capacity*

- EASE-MENT

