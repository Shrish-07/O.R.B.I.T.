# Feature Dictionary — current specification (Round 10)

The models train on **24 base features**; the political spec adds
**`dem_share`** for **25**. The list is derived at training time (numeric
columns of the canonical training split minus `target_log_price` minus
`config/feature_blacklist.yaml`; political mode re-adds the ideology features
that exist), so the authoritative machine-readable lists are:

- `models/artifacts/lgbm_all_years_base_features.json` (24)
- `models/artifacts/lgbm_all_years_political_features.json` (25)

This document is the human-readable dictionary for the current build
(566,956 rows × 53 columns; splits 121,052 / 56,723). (Round 8 note: the old
dictionary, absorbed into README.md in 2026-09-25 doc consolidation, described
a 16/17-feature era and listed `BBL_pluto` "used in both" — both statements
were stale and are corrected here; this file is the replacement.)

## Base features (24)

| Feature | Source | Meaning | Notes |
|---|---|---|---|
| `BOROUGH` | Sales roll | Borough code (1=Manhattan … 5=Staten Island) | integer; used as numeric categorical |
| `ZIP CODE` | Sales roll | USPS ZIP | numeric code |
| `Census Tract 2020` | Sales roll | 2020 census tract | numeric code |
| `Community Board` | Sales roll | Community Board code | numeric code |
| `Council District` | Sales roll | Council District column from the sales roll | coexists with `CounDist` |
| `CounDist` | crosswalk | Final council district used by every district-level analysis | float; the CounDist of record |
| `RESIDENTIAL UNITS` | Sales roll/PLUTO | # residential units | numeric since Round 8 |
| `COMMERCIAL UNITS` | Sales roll/PLUTO | # commercial units | numeric since Round 8 |
| `TOTAL UNITS` | Sales roll/PLUTO | total units | ≈ res + com (99.6% equality) |
| `unitsres` | PLUTO | Snapshot residential units | PLUTO-side sibling of `RESIDENTIAL UNITS` |
| `unitstotal` | PLUTO | Snapshot total units | |
| `GROSS SQUARE FEET` | Sales roll | Building gross square footage | numeric since Round 8 |
| `LAND SQUARE FEET` | Sales roll | Lot square footage | numeric since Round 8 |
| `lotarea` | PLUTO | Lot area (sq ft) | |
| `bldgarea` | PLUTO | Building area (sq ft) | |
| `YEAR BUILT` | Sales roll | Year built (sales roll) | may disagree with PLUTO `yearbuilt` (diff audit in `verify_canonical_dataset_and_split.py`) |
| `yearbuilt` | PLUTO | Year built (PLUTO snapshot) | regression control |
| `numfloors` | PLUTO | Floors | regression control; bulk-missing rows are the "join-failure" group |
| `landuse` | PLUTO | Land use code | regression control |
| `residfar` | PLUTO | Residential floor-area ratio | scenario feature (liberal ×1.5 / conservative ×0.7) |
| `commfar` | PLUTO | Commercial floor-area ratio | regression control |
| `facilfar` | PLUTO | Facility FAR | |
| `TAX CLASS AT TIME OF SALE` | Sales roll | Tax class | numeric code |
| `election_year` | derived | Nearest council election cycle (2017/2021/2025) | joins the ideology features |

## Political-only feature (25th)

| Feature | Source | Meaning | Notes |
|---|---|---|---|
| `dem_share` | BOE returns (ED-level vote share, crosswalked to council districts) | Democratic vote share in the council district for the sale's election cycle | the paper's ideology feature; ONLY political-mode models see it; produced by `src/data_pipeline/build_ideology_scores.py` (`data/processed/ideology_by_council.parquet`) |

## Deliberately excluded (a sample; see `config/feature_blacklist.yaml`)

- **Targets / leakage:** `target_log_price`, `sale price`, `sale_price*`, `log_sale_price`.
- **PLUTO 2026 snapshot assessments** (`assesstot`, `assessland`, `exempttot`,
  `appbbl`, `taxmap`, `plutomapid`): encode future appreciation → leakage.
- **Temporal identifiers:** `SALE DATE`, `sale_date`, `sale_year/month/quarter/day/timestamp`.
- **Administrative dates:** `basempdate`, `dcasdate`, `edesigdate`, `landmkdate`,
  `masdate`, `polidate`, `rpaddate`, `zoningdate`, `appdate`.
- **Row keys / split-only IDs:** `bin`, `bbl`, `lot`, `block`, `taxlot`, `record_id`, `sale_id`.
- **`BBL_pluto`:** PLUTO-side duplicate BBL kept by pandas' merge suffixing — a
  parcel-identifier artifact, blacklisted in Round 7 (see
  `VERIFICATION_REPORT.md` Round 7).
- **Raw geo:** `latitude`, `longitude`, `xcoord`, `ycoord` (offset leaked into
  coordinates; Moran/spatial work uses official polygon centroids instead).
- **Raw election variables:** `district_ideology`, `weighted_dem`, `weighted_rep`,
  `dem_vote_share`, `rep_vote_share`, `turnout` (political mode uses only the
  production `dem_share`).
- **Derived target encodings:** `mean_price`, `borough_mean_price`,
  `council_district_mean_price`, `building class category_mean_price`.
- **100%-null:** `EASE-MENT`.

Failures are enforced by the editorial audit
(`src/verification/editorial_audit/verify_blacklist_fully.py`) on every full
run.
