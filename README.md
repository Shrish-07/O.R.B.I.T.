# O.R.B.I.T.

O.R.B.I.T. studies whether council-district voting patterns are associated with residential sale prices in New York City from 2016 through 2025. It compares three vote-share measures across fixed-effects, spatial, and quasi-experimental specifications. A separate prediction workflow tests whether ideology improves held-out price forecasts.

**Finding.** The analysis finds no robust ideology premium. With district fixed effects alone, the association changes sign across the three preferred vote-share measures. With district and year fixed effects or district-specific trends, all three estimates are statistically indistinguishable from zero at the 5% level using wild-cluster inference. The wave-mean placebo shows how citywide time trends can account for the district-only result. [Fixed-effects estimates](results/robustness_checks_v3.json) · [Time-trend placebo](results/time_trend_placebo_v3.json)

## Repository layout

```text
main.py                         Pipeline entry point
requirements.txt                Pinned Python dependencies
Dockerfile                      Docker image definition for the Streamlit app
.github/workflows/ci.yml         CI smoke checks
app/                            Streamlit interface
config/                         Pipeline configuration and feature blacklist
data/DATA_MANIFEST.json          Tracked input provenance
docs/
  PAPER_TO_CODE_MAP.md           Producer and verifier map
  VERIFICATION_REPORT.md         Historical verification record
  FEATURE_DICTIONARY.md          Model features and exclusions
  figures/                      Diagnostic plots
  paper_assets/                 Supplemental tables and figures
  manuscript_assets/           Generated locally, ignored by Git
experiments/                    Registry, champion, and model metrics
models/                         Training scripts and model artifacts
results/                        Estimates, audits, and verification status
scripts/                        Reporting and release utilities
src/
  data_pipeline/                Input preparation, crosswalks, and splits
  analysis/                     Regressions, spatial analysis, and modeling
  causal/                       Close-election IV and City of Yes DiD
  figures/                      Figure and table generators
  ops/                          Pipeline checks and experiment utilities
  verification/                 Independent recalculations and checks
tables/                         Generated robustness and spatial tables
legacy/                         Earlier standalone implementation and outputs
notebooks/                      Reference data-validation notebook
logs/actions.log                Experiment-selection audit log
```

## Data sources & provenance

The dataset directories and ignored model artifacts are absent from a fresh clone. The clone includes `data/DATA_MANIFEST.json`, the tracked LightGBM model files, and small tracked model artifacts. Obtain `data/raw/` separately and supply the processed and canonical Parquets and temporal splits. Supply prebuilt model artifacts for standalone stages that consume them. The prerequisite command checks directory presence. [Ignore rules](.gitignore) · [Data manifest](data/DATA_MANIFEST.json)

| Input | Snapshot or identifier recorded in the repository | Provenance |
|---|---|---|
| NYC annualized property sales | `NYC_Citywide_Annualized_Calendar_Sales_Update_20260910.csv`, 845,607 rows covering 2016–2025. The manifest records a file modification timestamp of `2026-09-11T01:13:10.669379+00:00`. | [Data manifest](data/DATA_MANIFEST.json) |
| NYC Department of City Planning PLUTO | `Primary_Land_Use_Tax_Lot_Output_(PLUTO)_20260105.csv`. | [Data manifest](data/DATA_MANIFEST.json) |
| NYC Board of Elections election-district returns | Mayoral elections in 2017, 2021, and 2025 and presidential elections in 2016, 2020, and 2024. Inputs use the `ed_results_<year>_<office>.csv` naming convention. | [Return normalization](src/data_pipeline/normalize_ed_results.py) · [v2 ideology builder](src/data_pipeline/build_ideology_scores_v2.py) |
| NYC Board of Elections Council election recaps | 2013, 2017, 2021, and 2025 general elections. The builder records the election summary URLs for those cycles. | [Council margins builder](src/data_pipeline/build_council_election_margins.py) |
| District boundaries | `NYC_City_Council_Districts.shp` and historical vintages `nyed_16a`, `nyed_18a`, `nyed_20a`, `nycc_13a`. The crosswalk audit traces boundary vintage differences. | [Historical crosswalk builder](src/data_pipeline/build_legacy_ed_crosswalk.py) · [Crosswalk audit](results/crosswalk_audit.json) |
| City of Yes for Housing Opportunity roll call | ULURP `N 240290 ZRY`, adopted 2024-12-05. | [Roll-call builder](src/data_pipeline/build_city_of_yes_roll_calls.py) |

Start with a populated replication copy. Stage 1 checks an existing canonical v2 dataset, and the production crosswalk stage validates the supplied `ed_to_council_crosswalk.parquet`. Several analyses also use the supplied legacy ideology audit Parquets, Council election margins, roll-call Parquet, and `results/council_district_borough_map.csv`. The complete stage order reads canonical v2, the original IV result, and the earlier OLS verification output before their respective writers run. [Stage registry](main.py) · [Canonical sanity check](src/ops/pipeline_sanity_check.py)

## Run the pipeline

Use Python 3.13, the version configured in CI. In PowerShell, run these commands from the root of a populated working copy. The dependency installation and activation were tested with Python 3.13.6. Keep local input and output paths relative to that copy. Run the pipeline in the copy so generated results stay separate from the files you intend to commit. [CI workflow](.github/workflows/ci.yml) · [Dependency pins](requirements.txt)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --check-data
python main.py --list
```

For a single stage, pass its ID from the stage listing. Its inputs must already exist.

```powershell
python main.py --stage sanity_check
```

For all 86 stages, run the command below. Set `MANUSCRIPT_TEX` first if you want manuscript cell verification as part of the run. Logs and the run report live under `results/`. The verified full run took 29.8 minutes.

```powershell
python main.py
```

## Analysis lineage and outputs

| Stages | Outputs and purpose |
|---|---|
| `build_ideology_scores_v2`, `rebuild_canonical` | Construct the alternative mayoral and presidential measures, including vintage-matched district mappings. Canonical v2 carries the modeling features and analysis flags. `data/DATA_MANIFEST.json` records the build. |
| `build_ideology_scores_v3`, `crosswalk_audit_v3`, `extend_canonical_v3` | Allocate split election districts by area weights or a dominant-district rule, audit the allocation, and extend canonical v2. See [`crosswalk_audit_v3.json`](results/crosswalk_audit_v3.json) and [`canonical_v3_provenance.json`](results/canonical_v3_provenance.json). |
| `robustness_checks_v2`, `robustness_checks_v3`, `robustness_pre2024_v3`, `time_trend_placebo_v3`, `hedonic_v3` | Write `robustness_checks_v2.json`, `robustness_checks_v3.json`, `robustness_pre2024_v3.json`, `time_trend_placebo_v3.json`, and `hedonic_v3.json` in [`results/`](results), respectively. |
| `compute_morans_i_v2`, `compute_morans_i_v3`, `spatial_regression_model_v2`, `spatial_regression_model_v3` | Write `morans_i_results_v2.json`, `morans_i_results_v3.json`, `spatial_regression_v2.json`, and `spatial_regression_v3.json` in [`results/`](results), respectively. |
| `iv_district_fe_regression_v2`, `city_of_yes_did`, `city_of_yes_did_v3` | Write `iv_district_fe_results_v2.json`, `city_of_yes_did_results.json` and `_v2.json`, and `city_of_yes_did_results_v3.json` under [`results/causal/`](results/causal). The original and rescue IV designs have separate outputs. |
| `generate_paper_assets`, `generate_manuscript_assets` | The first writes supplemental assets and a manifest under [`docs/paper_assets/`](docs/paper_assets). The second produces 17 table fragments and four figures under the ignored `docs/manuscript_assets/` directory. |

| Manuscript assets | Main inputs |
|---|---|
| Measure, sample funnel, and election-audit tables | Processed ideology Parquets, `sample_funnel.json`, `ideology_line_audit.json`, and the crosswalk audits |
| Baseline, main, placebo, and spatial tables | `round5_part_H_ols.json`, the robustness and placebo results, and v3 Moran and spatial JSONs |
| Robustness, restricted-sample, pre-2024, and hedonic appendix tables | v2 and v3 robustness JSONs, `robustness_pre2024_v3.json`, and `hedonic_v3.json` |
| DiD, IV, and local-linear IV tables | DiD, original and rescue IV, and v2 IV JSONs under `results/causal/` |
| Model and scenario appendix tables | Supplemental model performance, cross-model SHAP, and scenario summary fragments under `docs/paper_assets/` |
| Figures 1–4 | Processed ideology Parquets, `robustness_checks_v3.json`, v3 placebo and robustness JSONs, and the three DiD result versions, respectively |

The four figures embedded in the submitted manuscript came from a separate plotting script and differ in layout and pixel dimensions from the pipeline's corresponding fig1 through fig4 PNGs, so the files are not byte-identical.

The generators define the complete mapping from input files to individual fragments. [Table and figure source](src/figures/generate_manuscript_assets.py) · [Supplemental asset source](src/figures/generate_paper_assets.py) · [Historical verification record](docs/VERIFICATION_REPORT.md) · [Producer and verifier map](docs/PAPER_TO_CODE_MAP.md)

## Verify

The final stage aggregates verifier status files and rejects missing, stale, or mismatching reports. Once a full run has produced fresh statuses, rerun the gate or verify a local LaTeX manuscript directly. For the manuscript command, set `$manuscript` to the path of the `.tex` file on your machine. The table fragments must already exist.

```powershell
python src/ops/verification_summary.py
$env:MANUSCRIPT_TEX = (Resolve-Path -LiteralPath $manuscript).Path
python src/verification/verify_manuscript_tables.py
```

The repository hygiene verifier needs the ignored `archive/FULL_PAPER_FACT_CHECK.md` with its superseded header and the ignored `config/scrub_terms.txt`. It adds a scrub-result check when `results/anonymous_release_report.json` exists. The manuscript verifier records a skipped check when it cannot find a manuscript. [Hygiene verifier](src/verification/verify_repo_hygiene.py) · [Manuscript verifier](src/verification/verify_manuscript_tables.py) · [Aggregate gate](src/ops/verification_summary.py)

## Reproducibility notes

- A full run retrains all seven active model variants. Trainers write new UTC-based experiment IDs to the registry, and stage 21 appends a champion-selection record to `logs/actions.log`. The trainers can also call the selection routine. [Stage registry](main.py) · [Training scripts](models/training) · [Selection routine](src/ops/experiment_runner.py)
- In the baseline comparison, numeric research leaves in `results/**/*.json` reproduced within `2e-15`. The wider artifact audit measured differences up to `1.96e-12` in the property-level scenario Parquets, within the combined tolerance of `1e-9` absolute plus `1e-9` relative to the rerun value. The seven prediction-interval Parquets have varying bounds because NumPy draws residuals without a seed after a seeded pandas sample. Their point predictions matched exactly for six models and within `1.07e-14` for Random Forest. The manuscript asset generators use other inputs. [Random Forest result](results/rf_tuning_grid.json) · [Scenario result](results/round5_part_K_mechanics.json) · [Scenario code](src/analysis/political_scenarios.py) · [Interval code](src/analysis/confidence_intervals.py) · [Manuscript asset source](src/figures/generate_manuscript_assets.py)
- Seed 42 is used in the wild-cluster inference, v3 Moran permutation tests, Random Forest and CatBoost fits, and SHAP sampling. [Wild-cluster inference](src/analysis/wild_cluster_inference.py) · [Moran v3](src/analysis/compute_morans_i_v3.py) · [Random Forest](models/training/train_random_forest.py) · [CatBoost](models/training/train_catboost.py) · [SHAP sampling](src/ops/shap_tiers.py)
- An in-place run records the absolute repository path in `results/round5_part_M.json`. Run in a separate copy, or review the Git diff for absolute paths before staging regenerated results. [Path-writing verifier](src/verification/verify_borough_map_and_crosscutting.py)
- The committed verification summary records 841 checks with the manuscript verifier skipped. The checked rerun with `MANUSCRIPT_TEX` set and the optional anonymous-release report absent recorded **865 checks**, **27 of 27 verifiers**, and **zero mismatches**, including 26 manuscript checks. All nine PNGs from the two asset generators matched the baseline pixels. [Committed summary](results/verification_summary.json) · [Manuscript verifier](src/verification/verify_manuscript_tables.py) · [Hygiene verifier](src/verification/verify_repo_hygiene.py)

## Limitations

District vote shares are aggregate proxies for political preferences. Boundary changes and split election districts affect how those measures are assigned. The study's observational specifications do not establish a causal price premium. The close-election instrument is weak or unidentified in the principal designs, and the City of Yes DiD uses district vote preference as a proxy. Geographic and time coverage follow the input snapshots listed above. [Crosswalk audit](results/crosswalk_audit_v3.json) · [IV results](results/causal/iv_district_fe_results.json) · [DiD results](results/causal/city_of_yes_did_results_v3.json)

## License

MIT. [License text](LICENSE)
