# Data Refresh Impact Report — 2016–2025 Sales Refresh (2026-09-10/11)

This report catalogs **exactly what changed** when the canonical dataset was
refreshed in place from a January-2026 sales snapshot (2016–2024) to the full
2016–2025 sales data, and every downstream table, model, and figure was
regenerated. It is written to support the *separate, later* task of rewriting
`orbit_paper_CER.tex` around these new numbers — the manuscript itself was **not**
touched.

**Source of old numbers:** the pre-refresh result files
(`results/regression_table.csv`, `results/robustness_checks.json`,
`results/spatial_regression.json`, `results/morans_i_results.json`,
`results/FULL_PAPER_FACT_CHECK.md`, `experiments/results/*.json`,
`models/artifacts/lgbm_all_years_political_shap_summary.json`,
`results/scenario_comparison_clean.csv`) and the manuscript text, all read **before**
running the full 46-stage pipeline (`python main.py`) this session. Old values are
also recoverable from git history (`d0cdd12`).

---

## 1. What was done (Tasks 1–2)

1. **New sales pull.** Downloaded the current full
   `https://data.cityofnewyork.us/resource/w2pb-icbu` (SODA dataset `w2pb-icbu`)
   Citywide Annualized Calendar Sales CSV on **2026-09-10**:
   `845,607` rows, `sale_date` range `2016-01-01` → `2025-12-31`, `29` columns,
   `84,693` rows on/after `2025-01-01`. Saved to
   `data/raw/sales/NYC_Citywide_Annualized_Calendar_Sales_Update_20260910.csv`.
   The 2016–2024 remainder is exactly `760,914` rows — byte/record-identical to the
   old file's `760,914` rows; the refresh **only adds 2025** to historical data.
2. **Glob-based, self-enforcing raw-data loading** in
   `src/data_pipeline/rebuild_canonical.py` (sales and PLUTO paths). It now globs
   `NYC_Citywide_Annualized_Calendar_Sales_Update_*.csv` and
   `Primary_Land_Use_Tax_Lot_Output_(PLUTO)_*.csv`, prints the resolved filename
   prominently, and **refuses to run** (clear error) on zero or multiple matches.
   Verified live: a planted second sales file triggered the "multiple matches"
   refusal and named both files; temporarily removing the file triggered the
   "no match" refusal.
3. **`data/DATA_MANIFEST.json`** is now written on every canonical build (sales
   filename + raw row count + file mtime, PLUTO filename, output row count, run
   timestamp).
4. **Old sales file archived** (not deleted) to `data/raw/sales/_archive/`.
5. **QA gate** (Task 2a) — see §2 — then the new canonical was written **in place**
   to `data/canonical/modeling_dataset_canonical_v2.parquet` (same filename).
6. **Full pipeline** (`python main.py`) re-ran all **46 stages → 46 passed,
   0 failed** (`results/PIPELINE_RUN_REPORT_full_20260910.md`, generated
   2026-09-10T23:38:27). This re-trained all 7 models, re-registered every run,
   refreshed `champion.json`, and regenerated every table/figure/verification artifact.

---

## 2. QA gate result (build-correctness) — important nuance

The Task 2a gate compares the new build's `sale_year <= 2024` rows against the old
canonical. **Result: PASSES in substance**, with two export-side artifacts that make
a naive row-for-row `==` comparison fail. Full evidence is saved in
`results/qa_gate_v2_evidence.json`.

| Check | Result |
|---|---|
| `sale_year <= 2024` row count (old vs new) | **514,618 = 514,618** ✅ |
| Sales-record multiset `BBL\|sale_date\|sale_price` | **identical** ✅ (498,328 unique keys) |
| All 19 modeling columns (17 features + target + sale_year), multiset-aligned | **bit-identical** ✅ |
| Raw *non-modeling* string columns | differ in **formatting only** (see below) |

The naive comparison diverges for two reasons, **neither of which reflects a change
to the modeling data**:

1. **Row order changed** — Socrata's full export reordered rows between the Jan-2026
   and 2026-09-10 snapshots.
2. **String formatting changed** in Socrata's export: `SALE PRICE` lost the
   `$` and thousands-separator characters (`"$4,027,000"` → `"4027000"`), and the
   unit columns became float-formatted (`"0"` / empty → `"0.0"` / `nan`).

**Consequence that DOES matter (documented, not hidden):** the float reformatting
changed the inferred dtype of `RESIDENTIAL UNITS`, `COMMERCIAL UNITS`, and
`TOTAL UNITS` from `object` (string) to `float64`. The data pipeline's
`prepare_features()` keeps *numeric* non-blacklist columns, so these three columns —
previously excluded as strings — are **now admitted as 3 new model features**. This
changed the model feature counts **16 → 19 (base) and 17 → 20 (political)** and is
the root cause of every Table-2/Table-3 shift below. `SALE PRICE` also became
`int64` but remains excluded (it is a blacklisted leakage column).

This is a real, explainable side effect of the Socrata export change, **not** a
pipeline bug and **not** a data-corruption issue. It is surfaced here so the
manuscript rewrite can decide whether to (a) accept the 3 features, or (b) pin the
unit columns back to the pre-refresh treatment (e.g. blacklist them) to keep the
17-feature model.

---

## 3. Dataset shape

| Claim | Old | New | % change |
|---|---|---|---|
| Canonical rows × cols | 514,618 × 53 | 567,997 × 53 | +10.4% |
| `sale_year` span | 2016–2024 | 2016–2025 | +1 year |
| `election_year` rows: 2017 / 2021 / 2025 | 178,439 / 285,617 / 50,562 | 178,439 / 285,617 / **103,941** | 2025: +105.6% |

Per-year `sale_year` distributions are identical for 2016–2024 (60,756 / 60,866 /
56,817 / 55,626 / 43,788 / 70,829 / 65,899 / 49,475 / 50,562) and add
**53,379** rows for 2025.

## 4. Table 1 — OLS regression (Section 4.3)

`N` changes `470,535 → 521,220` (the +50,685 rows are the 2025 sales with complete
controls). Values from `results/regression_table.csv`.

| Claim | Old | New | % change |
|---|---|---|---|
| M1 bivariate `dem_share` β | 0.8890 | 0.9484 | +6.7% |
| M1 `dem_share` SE | 0.0070 | 0.0067 | −4.3% |
| M1 R² | 0.0336 | 0.0375 | +11.6% |
| M2 (+controls) `dem_share` β | 0.5741 | 0.6253 | +8.9% |
| M2 `dem_share` SE | 0.0071 | 0.0067 | −5.6% |
| M2 R² | 0.0932 | 0.1005 | +7.8% |
| M3 (+borough FE) `dem_share` β | 0.1843 | **0.3036** | **+64.7%** |
| M3 `dem_share` SE | 0.0093 | 0.0087 | −6.5% |
| M3 R² | 0.1478 | 0.1504 | +1.8% |

The headline M3 coefficient (the "80% shrinkage" result in the abstract) grows
substantially: `0.184 → 0.304`. Year-built in M3 stays insignificant (p 0.119 →
0.048 — marginally significant now).

## 5. Table 2 — model performance (Section 4.6.1)

All models were re-trained on the **unchanged** ML split (train 121,622 / test
56,817, all `sale_year` 2016–2018). Metrics change only because the feature set grew
from 16/17 to **19/20** (see §2). Values from `experiments/results/*.json`.

| Model | Old R² | New R² | Old MAE | New MAE | features old→new |
|---|---|---|---|---|---|
| LightGBM base | 0.5418 | 0.5649 | 0.42897 | 0.41099 | 16 → 19 |
| LightGBM political | 0.5413 | 0.5669 | 0.42934 | 0.41178 | 17 → 20 |
| Random Forest | 0.50702 | 0.48291 | 0.43681 | 0.42606 | 17 → 20 |
| Ridge | 0.11138 | **−4.97245** | 0.65580 | 0.74981 | 17 → 20 |
| ElasticNet | −0.00965 | −0.00965 | 0.69037 | 0.69037 | 17 → 20 |
| CatBoost | 0.50574 | 0.54866 | 0.45391 | 0.42646 | 17 → 20 |
| XGBoost | 0.52869 | 0.57707 | 0.43922 | 0.41149 | 17 → 20 |

Notes: (a) **Ridge collapses** to a strongly-negative R² — the 3 mostly-missing,
imputed unit features with poor scaling wreck the L2-only Ridge fit. (b)
**ElasticNet is unchanged to 14 dp** — its L1 penalty zeroes the 3 new features, so
predictions are identical. (c) Champion (`experiments/champion.json`) was re-selected
automatically and remains `lgbm_all_years_base` (new MAE 0.41099).

## 6. Table 3 — SHAP (Section 4.6.2)

**The stored political-model SHAP artifact
(`models/artifacts/lgbm_all_years_political_shap_summary.json`) is unchanged by this
run** (it was already a deliberately-left-stale 18-feature-era file, per
`results/FULL_PAPER_FACT_CHECK.md` §7). Its stored values (Community Board 0.27472,
… dem_share 0.03473, "rank 12 of 17") will still compare "OK" only because the file
itself was not touched, but it is now **doubly stale**: the current political model
has **20 features**, not 17. The 3 unit columns (`RESIDENTIAL UNITS`,
`COMMERCIAL UNITS`, `TOTAL UNITS`) are new SHAP features that must be added, and
`dem_share`'s rank re-derived, when Table 3 is regenerated during the manuscript
rewrite.

## 7. Table 4 — robustness (Section 4.4)

`N` changes `470,535 → 521,220`; `sale_year` list now ends in `2025`.
Values from `results/robustness_checks.json`. (`p` below tiny values are rounded to
one significant digit.)

| Specification | Old β | New β | Old p | New p |
|---|---|---|---|---|
| Model 3, classical SE | 0.1843 | 0.3036 | 1.0e-87 | 1.3e-265 |
| Model 3, SE clustered by CounDist (51) | 0.1843 | 0.3036 | **0.318** | **0.082** |
| Model 4 (+51 district FE), classical SE | 0.5682 | 0.8033 | 1.2e-165 | <1e-300 |
| Model 4, clustered SE | 0.5682 | 0.8033 | 8.4e-06 | 1.3e-14 |
| Model 5 (+year FE), classical SE | 0.0740 | 0.0820 | 4.3e-14 | 1.7e-17 |
| Model 5, clustered SE | 0.0740 | 0.0820 | 0.712 | 0.688 |
| Winsorized (1/99) | 0.1880 | 0.2993 | 1.6e-102 | 1.3e-292 |
| Cook's-D trimmed | 0.1304 | 0.2314 | 1.5e-75 | 6.7e-265 |

Sign-stable across all 8 checks in both runs (`sign_stable_across_all_checks: true`).
**Headline change:** under clustered-by-district SE the M3 p-value moves from `0.318`
(insignificant) to `0.082` (marginal), and the district-FE coefficient grows
`0.568 → 0.803` and stays highly significant under clustering.

## 8. Table 5 — spatial regression (Section 4.5)

`n = 51` districts throughout. Values from `results/spatial_regression.json`.

| Quantity | Old | New |
|---|---|---|
| Spatial lag ρ (`W_median_log_price`) | 0.7086 (p 7.5e-12) | 0.7022 (p 1.8e-11) |
| Spatial error λ | 0.7872 (p 1.0e-19) | 0.7804 (p 1.5e-18) |
| OLS `dem_share` p (β 0.2007 → 0.2429) | 0.385 | 0.311 |
| Spatial-lag `dem_share` p (β 0.2246 → 0.2529) | 0.170 | 0.138 |
| Spatial-error `dem_share` p (β 0.2288 → 0.2641) | 0.374 | 0.315 |
| OLS-residual Moran's I | 0.3894 (z 6.05) | 0.3901 (z 6.07) |

The qualitative story is unchanged: `dem_share` never reaches significance once the
strong spatial spillover is modeled directly, and ρ/λ remain large and highly
significant. The `dem_share` point estimates and p-values shift modestly.

## 9. Moran's I (Section 4.2)

`results/morans_i_results.json` (k=5 row-standardized KNN, 999 permutations, seed 42,
51 districts). `p_sim` stays 0.001 for both.

| Quantity | Old | New | % change |
|---|---|---|---|
| Price I | 0.5959 | 0.5901 | −1.0% |
| Price z | 7.4597 | 7.3850 | −1.0% |
| dem_share I | 0.5797 | 0.5703 | −1.6% |
| dem_share z | 7.9934 | 7.8608 | −1.7% |

(`dem_share` by district is unchanged — the ideology pipeline is untouched — but the
district-level price aggregate shifts because it now mixes in 2025 sales.)

## 10. Appendix A.1 — counterfactual scenarios

51×3 district-level percentage shifts. `results/scenario_comparison_clean.csv` is a
**stale 8/31 artifact not regenerated by the pipeline**; fresh values are in
`results/political_scenarios/{liberal,conservative,mixed}_policy_by_council.csv`.

| Scenario | Old mean | New mean | Old range | New range | sign change |
|---|---|---|---|---|---|
| Liberal | +7.42% (49 pos) | +6.71% (48 pos) | [−1.21, 32.44] | [−2.77, 22.99] | 3 districts |
| Conservative | −0.78% (35 neg) | −1.27% (41 neg) | [−7.29, 1.05] | [−5.30, 1.40] | 10 districts |
| Mixed | −3.39% (14 pos) | **−9.82% (1 pos)** | [−16.73, 13.32] | [−38.05, 0.65] | 15 districts |

The mixed-governance scenario moves materially more negative (only 1 district stays
positive vs 14 before) — driven by the changed (20-feature) political model.

---

## 11. Expected flag: verify scripts that now carry stale hardcoded comparison values

Several `src/verification/verify_*.py` scripts hardcode pre-refresh paper values as
their comparison target. After this refresh they correctly print a mismatch against
those stale constants — **this is the expected, correct behavior of a data refresh,
not a new bug**. (All 46 stages still reported PASS because these scripts exit 0 and
only *print* the discrepancy.) The scripts with stale hardcoded targets are:

| Script | Stale hardcoded targets |
|---|---|
| `verify_abstract_and_intro.py` | rows 514,618; bivariate β 0.889; M3 β 0.184; political MAE 0.429342 |
| `verify_canonical_dataset_and_split.py` | rows 514,618; 16/17 feature counts (now 19/20) |
| `verify_regression_ols.py` | Table 1 β 0.8890 / 0.5741 / 0.1843, SEs, R² 0.0336 / 0.0932 / 0.1478, t 19.859 |
| `verify_regression_missingness.py` | rows 514,618; shrinkage 0.889 → 0.574 → 0.184 |
| `verify_numfloors_missingness.py` | rows 514,618 (11,918 / 11,930 numfloors count itself is unchanged) |
| `verify_ml_performance.py` | base MAE/R² 0.428972 / 0.541772; political 0.429342 / 0.541345 |
| `verify_shap_importance.py` | Community Board SHAP 0.27472 (18-feature-era artifact) |
| `verify_spatial_regression.py` | ρ 0.7086, λ 0.7872, dem_share p 0.385 / 0.170 / 0.374, residual Moran's I 0.389 |
| `verify_appendix_a1_scenarios.py` | 153 scenario values + summary 7.42 / −0.78 / −3.39 |
| `verify_scenario_mechanics.py` | clip counts 109 / 9 / 816 (now 114 / 0 / 292) |

Scripts whose targets are **unchanged** (ideology pipeline untouched) keep passing:
`verify_cd1_dem_share.py` and `verify_ideology_construction.py` (0.715908),
`verify_spatial_pattern.py` (r 0.163, "49 of 51").

Note also: `src/analysis/compute_morans_i.py` embeds the paper Moran's-I targets
(`paper_targets`: 0.5959 / 7.46 / 0.5797 / 7.99) and now writes a
`discrepancy_flags` line (DEM z 7.86 vs 7.99), so the Moran's-I z-score mismatch is
recorded in `results/morans_i_results.json` itself.

## 12. Known limitation (flagged, NOT addressed) — pre-2023 council-district boundaries

`data/raw/election_districts/NYC_City_Council_Districts.shp` still describes itself
in its own metadata as "the redistricting as of the US Census 2010" — i.e. the
**pre-2023** council-district lines. NYC's current boundaries took effect for the
2023 elections. Consequences: (a) every CounDist assignment in the canonical dataset
(and thus every district-level result above, including Moran's I, the spatial
regression, and the robustness checks) is built on pre-2023 boundaries; (b) the
2025 sales are being tagged with pre-2023 lines even though the 2025 council seats
were elected from the post-2023 lines. Re-deriving council districts for 600,000+
properties against corrected, date-appropriate boundaries is a **separate, larger
task** and was deliberately **not** attempted here. This observation is confirmed
still accurate as of this run.

## 13. PLUTO currency note (flagged, not changed this round)

The canonical build kept the existing PLUTO file
`Primary_Land_Use_Tax_Lot_Output_(PLUTO)_20260105.csv`. A check found the NYC Open
Data PLUTO dataset (`64uk-42ks`) was updated **2026-08-24**, newer than the repo's
`20260105` snapshot, so a newer DCP PLUTO release exists. PLUTO was intentionally
held constant this round (the Task 2a QA gate requires the ≤2024 rows — including
PLUTO-derived building attributes — to stay identical, which only holds if PLUTO is
unchanged). Pulling/merging a newer PLUTO is a separate decision with its own QA gate.

## 14. New data sources for the causal-identification analysis

- **`data/processed/council_election_margins.parquet`** (built by
  `src/data_pipeline/build_council_election_margins.py`): council-district general-
  election winner/runner-up margins for **2013, 2017, 2021, 2025** (204 rows =
  51 districts × 4 cycles). Raw Recap files cached in `data/raw/council_elections/`.
  RCV years (2021, 2025) use the certified first-choice totals with fusion lines
  aggregated (documented in the script docstring). Spot-checked:
  **2021 D43 Brannan 26,360 vs Fox 25,158 (2.33%)** and **2021 D19 Paladino 25,580
  vs Avella 24,800 (1.55%)** — both narrow, as expected.
- **`data/processed/city_of_yes_roll_calls.parquet`** (built by
  `src/data_pipeline/build_city_of_yes_roll_calls.py`): City of Yes roll calls.
  **Housing Opportunity (2024-12-05, 31–20)** is fully present (51 districts) and
  cross-checks cleanly against the supplied reference table (20 No + 31 Yes = 51,
  no overlaps). **Carbon Neutrality (Dec 2023) and Economic Opportunity (June 2024)**
  member-level roll calls could **not** be retrieved this session — legistar's JSON
  API returns 403 without a token and its HTML search is JS-only (search-engine
  fallbacks were also unreachable) — so those two are deliberately **not fabricated**
  and are flagged for manual retrieval from legistar.council.nyc.gov. `member_name`
  is null (join key `council_district` is fully populated).

---

## 15. Files changed this round

**Raw data — created**
- `data/raw/sales/NYC_Citywide_Annualized_Calendar_Sales_Update_20260910.csv` (new 2026-09-10 sales pull: 845,607 rows, 2016–2025)
- `data/raw/council_elections/<year>/` (204 BOE Recap files: 2013 PDFs, 2017/2021/2025 CSVs)

**Raw data — archived (not deleted)**
- `data/raw/sales/_archive/NYC_Citywide_Annualized_Calendar_Sales_Update_20260105.csv`

**Canonical data — refreshed in place**
- `data/canonical/modeling_dataset_canonical_v2.parquet` (514,618 → 567,997 rows)
- `data/DATA_MANIFEST.json` (new)
- `data/splits/*.parquet` (regenerated; same row counts)

**New data sources**
- `data/processed/council_election_margins.parquet` + `src/data_pipeline/build_council_election_margins.py`
- `data/processed/city_of_yes_roll_calls.parquet` + `src/data_pipeline/build_city_of_yes_roll_calls.py`

**Code — modified**
- `src/data_pipeline/rebuild_canonical.py` (glob-based sales/PLUTO discovery + self-enforcing single-match rule + `DATA_MANIFEST.json` writer)
- `requirements.txt` (+ `pypdf==6.18.1`)

**Models / experiments — regenerated**
- `models/lgbm_all_years_{base,political}.txt`, `models/rf_*.joblib`, `models/ridge_*.joblib`, `models/elasticnet_*.joblib` (and CatBoost/XGBoost artifacts)
- `models/artifacts/*_features.json` (19/20 features), `*_metrics.json`, `*_shap_summary.json` (regenerated except the deliberately-stale political SHAP summary)
- `experiments/registry.json`, `experiments/champion.json` (champion re-selected: `lgbm_all_years_base`, MAE 0.41099)

**Results — regenerated (overwritten with new values)**
- `results/regression_table.csv`, `results/robustness_checks.json`, `results/spatial_regression.json`, `results/morans_i_results.json`, `results/regression_diagnostics.json`, `results/regression_model3_full_summary.txt`, `results/political_scenarios/*`, `results/logs/pipeline/*`, `results/PIPELINE_RUN_REPORT.md`, `results/PIPELINE_RUN_REPORT_full_20260910.md`
- `docs/figures/*`, `docs/research_summary.md` (regenerated)

**Results — deliberately NOT regenerated (stale, flagged)**
- `results/scenario_comparison_clean.csv` (8/31; fresh scenario data is in `results/political_scenarios/*_by_council.csv`)
- `models/artifacts/lgbm_all_years_political_shap_summary.json` (8/31; 18-feature-era artifact, per FULL_PAPER_FACT_CHECK §7)

**Documentation**
- `DATA_REFRESH_IMPACT_REPORT.md` (this file)
- `results/qa_gate_v2_evidence.json` (build-correctness evidence)

The manuscript `orbit_paper_CER.tex` (at `../CER Papers 5/CER Papers 5/main/`) was **not** modified.