# Data Contract

## Canonical Unit
One row = one NYC property sale with political context at time of sale.

## Required Datasets
- Citywide Sales (CSV)
- PLUTO (CSV)
- Council Districts (SHP)
- Election Districts (SHP)
- Election Results (CSV)

## Join Keys
- Sales ↔ PLUTO: BBL
- PLUTO ↔ Council: council
- Election Results ↔ ED: ElectDist
- ED ↔ Council: CounDist

## Required Fields (Final Table)
- BBL
- SALE_DATE
- SALE_PRICE (log)
- GROSS_SQFT
- LAND_SQFT
- YEAR_BUILT
- BUILDING_CLASS
- ZONING_CODE
- COUNCIL
- IDEOLOGY_SCORE
- POLICY_INDICATOR

## Mutability
- Mutable: ZONING_CODE, POLICY_INDICATOR
- Fixed: YEAR_BUILT, LOCATION, BUILDING_CLASS

No model may be trained unless all required fields exist.
