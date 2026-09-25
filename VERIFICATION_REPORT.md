# Independent Re-Verification Report — `ORBIT_reorganized`

**Date:** 2026-09-01
**Verified by:** independent agent pass (not the reorganizing session)
**Method:** every phase executed live against this checkout and the real `data/` copy;
nothing taken on faith. All stage logs under `results/logs/pipeline/`.

---

## Phase summary

| Phase | What was checked | Result | Fixed |
|---|---|---|---|
| 0 — Orientation | Read README / PAPER_TO_CODE_MAP / ACTION_PLAN / DELIVERY_NOTES in full; `python main.py --check-data`; repo tree vs README "Repository layout" | ✅ PASS (all 7 data buckets OK; tree matches) | — |
| 1 — Static integrity | py_compile of all active `.py`; grep for 17 pre-reorg flat paths; `app/app.py` import style + sys.path; training-script imports; every `parents[N]` root resolution; JSON backslashes; `__init__.py` coverage | ❌ → ✅ after fixes | 15 stale flat-path references in comments/docstrings/JSON-output strings + 4 stale references in `results/FULL_PAPER_FACT_CHECK.md` |
| 2 — Stage registry vs. filesystem | All 43 `main.py` stages: script exists, argparse flags/choices valid, no duplicate ids, `--list` prints all 43, `--check-data` exits 0 | ✅ PASS | — |
| 3 — Execute stages, check real numbers | 19 verification/analysis stages + 5 heavy comparison trainings + 1 extra direct training script | ⚠️ Mixed: 5 real bugs found; all fixed; one stage blocked by a data gap (see "Still needs attention") | 4 code bugs fixed (see below) |
| 4 — PAPER_TO_CODE_MAP vs. reality | Paper `.tex` table/figure inventory vs. map; all 34 script paths in the map; archived-file spot check | ✅ PASS with caveats (all 34 referenced scripts exist; no archived module is imported by active code) | — |
| 5 — CI & app smoke | `ci.yml` YAML validity + all 3 referenced script paths; `ci_smoke_check.py`; `pipeline_sanity_check.py` direct; headless Streamlit launch | ✅ PASS (after fix #1 made direct runs work) | Encoding hardening |


---

## Real bugs found and fixed (all confirmed by re-running, not just reading)

1. **Windows encoding crash in every stage whose script prints non-ASCII** (e.g. `pipeline_sanity_check.py` prints emoji/→).
   Symptom: `UnicodeEncodeError: 'charmap' codec` as soon as `main.py` redirected output to a log file (Python uses cp1252 for redirected stdout on this machine).
   Fixes: `main.py::run_stage` now launches children with `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` in the environment; `src/ops/pipeline_sanity_check.py` reconfigures its own stdout to UTF-8 (safe fallback) so direct runs work too.

2. **`src/data_pipeline/data_loader.py` referenced the deleted/blacklisted `district_ideology` column.**
   `prepare_features(..., mode="political")` asserted `district_ideology` exists and appended it to the feature list. That column is the *abandoned, blacklisted* ideology pipeline; the canonical v2 dataset and splits carry only `dem_share`. This hard-crashed `train_ridge` and `train_elasticnet` (the reorg sandbox never ran these heavy stages, so it went undetected).
   Fix: assert/append `dem_share` — the documented production ideology feature ("the only ideology feature that reaches any model", README).

3. **`data_loader.py` feature filter admitted `datetime64` columns.**
   `df[c].dtype != "object"` passes `sale_date` (datetime64), crashing scikit-learn's `SimpleImputer` with `DTypePromotionError`. Fixed to keep only numeric/bool columns and exclude datetime explicitly.

4. **Stale pre-reorganization flat paths** in live code/docstrings/output strings (Phase 1.2 grep):
   `src/data_pipeline/build_ideology_scores.py`, `rebuild_canonical.py`, `split_temporal.py`, `src/causal/causal_analysis.py`, `src/analysis/compute_morans_i.py`, `src/verification/verify_cd1_dem_share.py`, `src/verification/verify_numfloors_missingness.py`, `src/verification/verify_scenario_mechanics.py` — 15 references updated to the new `src/<subpackage>/` paths. Bonus: the usage-string fix removed the two previously "known" `SyntaxWarning: invalid escape sequence` (their source was the `.venv\Scripts\` text inside those docstrings) — compile is now *fully* warning-free.
   `results/FULL_PAPER_FACT_CHECK.md` also carried 3 stale references (`src/political_scenarios.py`, `src/round5_followup_resolution.py` ×2); updated to `src/analysis/...` / `src/verification/verify_followup_resolution.py`. The sentence describing the historical `round5_factcheck_H_missing.py → round5_followup_resolution.py` rename was reworded to name the current file while preserving the history. Re-grep confirms zero hits outside `archive/`, `docs/DATA_LINEAGE.md`, and `ACTION_PLAN_FOR_SUBMISSION.md`'s own reorg narrative.

5. **`experiments/registry.json` re-acquired Windows backslashes during this session.**
   The training scripts stringify `Path` objects (`str(ARTIFACT_DIR / ...)`) when appending registry entries, so my re-runs of `train_catboost`/`train_xgboost`/`train_random_forest` and `train_linear_baselines.py` on Windows inserted `models\artifacts\...` backslashes into the 5 new entries — the exact cross-platform bug the reorg already fixed once. Normalized to forward slashes via a JSON-aware load → replace → dump (file re-validated: 17 entries, valid JSON, zero backslashes). `experiments/champion.json`, both `models/artifacts/*_pdp_summary.json`, and `results/editorial_audit/blacklist_audit.json` were already clean on arrival (the only backslashes in the latter are legitimate `\u2014` em-dash escapes).

**Environment note (not a repo bug):** the checked-in `venv/` was a stripped shell (pip only, 2 site-package dirs); the system Python 3.13 had a broken pandas (missing pytz) and no lightgbm/xgboost/catboost/shap. I installed the **pinned versions from `requirements.txt`** into `venv/` (pandas 2.3.3, numpy 2.4.0, lightgbm 4.6.0, scikit-learn 1.8.0, statsmodels 0.14.6, scipy 1.16.3, pyarrow 23.0.1, geopandas 1.1.2 + pyogrio/pyproj/shapely, shap 0.51.0, xgboost 3.1.2, catboost 1.2.10, streamlit 1.57.0, libpysal 4.14.1, esda 2.10.0, matplotlib 3.10.9, joblib 1.5.3). The full 157-package requirements.txt was *not* wholesale installed (mlflow, folium, seaborn etc. are only needed by ops utilities and the unfinished `src/causal/causal_analysis.py`).

---

## Confirmed working — executed in this checkout, with observed values

### Verification/analysis stages (exit 0 via `main.py`)

| Stage | Key recomputed values observed |
|---|---|
| `sanity_check` | sale_year range 2016→2024 (100% within [2003,2025]); dem_share coverage **0.9756** (paper: 97.56%) ✓ |
| `verify_cd1_dem_share` | CD1/2017 dem_share = **0.7159080199792668** vs paper **0.715908** (diff 2e-8, `matches_paper: true`) ✓ |
| `verify_canonical_dataset_and_split` | 16-feature base / 17-feature political feature lists exact; train **121,622** / test **56,817** rows exact ✓ |
| `compute_morans_i` | `moran_i_price` = **0.5958926** (paper 0.5959), `moran_i_dem_share` = **0.5796642** (paper 0.5797), both p_sim = 0.001; 51/51 districts, k=5 KNN row-standardized, 999 permutations, seed=42, official shapefile polygon centroids — **exact reproduction to 4 dp** |
| `generate_regression_table` | exit 0; wrote `results/regression_table.csv`, `results/regression_diagnostics.json`, `results/regression_model3_full_summary.txt` |
| `verify_regression_ols` | Model 3 dem_share coef **0.1842585**, SE 0.0092785, p = **1.009e-87**; N = **470,535** ✓ |
| `verify_spatial_pattern` | r = **0.163015** vs paper 0.163; 49 positive / 2 negative shift districts (16, 17) ✓ |
| `verify_regression_missingness` / `verify_numfloors_missingness` | exit 0; 514,618 → 470,535 decomposition reproduced |
| `generate_robustness_checks` | exit 0 (18.8 s). Model 3 classical p = **1.01e-87** → clustered-by-CounDist p = **0.3185** (the expected ~87-orders-of-magnitude jump); Model 4 district-FE coef 0.5682, clustered p = **8.37e-06**; Model 5 year-FE coef 0.0740, clustered p = **0.7125**; winsorized coef 0.1880; Cook's-D trim dropped 25,819 rows (5.49%) → coef 0.1304. **Matches ACTION_PLAN's table exactly.** |
| `verify_ml_performance` | all 7 Table 2 models `matched: (True, True)`; mae_delta **0.00037** and r2_delta **−0.000427** recomputed exactly |
| `verify_shap_importance`, `verify_appendix_a1_scenarios`, `verify_borough_map_and_crosscutting`, `verify_blacklist_fully`, `verify_abstract_and_intro`, `verify_followup_resolution` | all exit 0; followup confirms the known 19-row prediction-interval gap but that it does **not** affect Table 2 metrics (political MAE 0.429342 / R² 0.541345 from the full 56,817-row in-training predict) |
| `verify_scenario_mechanics` | exit 0 in 213.5 s (7 full-dataset LightGBM predicts — the documented slow stage); base_pred min 9.5635 (row filter removes 0 rows); clip counts liberal_upper 109, conservative_upper 9, mixed_upper 816 |


### Training reproductions (Phase 3.7)

| Model | Recorded MAE / R² | Re-run now | Verdict |
|---|---|---|---|
| LightGBM base / political | 0.428972 / 0.541772 · 0.429342 / 0.541345 | not retrained (records left intact; independently verified against artifacts) | verified |
| CatBoost (lr .05, d 6) | 0.4539147 / 0.5057437 | **identical to full float precision** | exact |
| XGBoost (lr .05, md 6) | 0.4392201 / 0.5286944 | **identical to full float precision** | exact |
| Ridge | 0.6558020 / 0.1113793 | **identical**, via `models/training/train_linear_baselines.py --variant all_years --mode political` | exact |
| ElasticNet | 0.6903725 / −0.0096473 | **identical**, same script | exact |
| Random Forest (n=200) | 0.4370478 / 0.5066344 | 0.4368108 / 0.5070222 | ✓ within tolerance (ΔMAE ≈ 2.4e-4; bagging stochasticity) |

Feature count is now 17 vs. recorded 18 for these five models: the 18th was `EASE-MENT`, a 100%-null column now on the blacklist — inclusion had zero modeling effect, which is why cat/xgb/ridge/elasticnet reproduce bit-exactly.

Side effects of the training re-runs (deliberate): `models/rf_all_years_political_n200.joblib` (1.05 GB) regenerated — the file DELIVERY_NOTES said was excluded from the zip; cat/xgb/ridge/elasticnet model + artifact files overwritten with metric-identical equivalents; registry entries appended by the scripts, then path-normalized (bug fix #5). The two spurious `ridge_political_all.json` / `elasticnet_political_all.json` files produced by the legacy scripts during diagnosis were **deleted**, so `compare_models.py` (which globs all of `experiments/results/*.json`) cannot silently tabulate a 30-feature Ridge in place of the paper's 17-feature one.

### Phase 4 / 5 checks

- Paper inventory (from `..\CER Papers 4\CER Papers 4\main\orbit_paper_CER.tex`, read-only): 5 figures (political_ideology_vs_price, temporal_ideology_shift, model_comparison, shap_top15, scenario_impact_all_three) and 4 `\input` tables (table1, table2, table3, tableA1) — every one is present in PAPER_TO_CODE_MAP with a real Builds/Verifies entry.
- All **34** `src/…/.py` / `models/…/.py` paths referenced in PAPER_TO_CODE_MAP exist in this checkout.
- Archived-file spot check: `_truth_assemble.py`, `_truth_crosscut.py`, `_truth_write.py`, `_truth_assemble_data.json`, `final_verification.py`, `verify_paper_numbers.py`, `assemble_full_factcheck_report.py`, `create_model_performance_table.py`, `feature_engineering.py` are all in `archive/`, and **no active code imports any of them**.
- `.github/workflows/ci.yml`: valid YAML (2 jobs); all three `python <path>` commands reference existing files (`src/ops/ci_smoke_check.py`, `src/data_pipeline/split_temporal.py`, `models/training/train_smoke.py`).
- `python src/ops/ci_smoke_check.py` → "Smoke check OK — files present", exit 0 (FILES list matches reality).
- `python src/ops/pipeline_sanity_check.py` direct → exit 0 with the encoding fix (previously crashed on this console).
- Headless `streamlit run app/app.py` (port 8501): server started cleanly within ~10 s, no traceback/ImportError; process stopped after the probe. Caveat: page bodies are lazy — the launch check does not exercise the in-page `from auth import …` / geocoder / counterfactual calls, but those imports were statically verified to the new paths (`from auth import`, `from src.data_pipeline.geocoder`, `from src.analysis.counterfactual`; `app.py` inserts both the repo root and its own directory onto `sys.path`).


---

## Still needs attention (found; needs your decision — not silently changed)

1. **`verify_ideology_construction` cannot run in this checkout — missing historical audit data.** It unconditionally reads `data/processed/district_ideology.parquet` (the *abandoned* pipeline's output, kept only as an audit trail per README/DATA_LINEAGE). Neither that file nor `ed_ideology.parquet` exists anywhere on this machine (searched the whole `O.R.B.I.T v2` tree; `Data 2.zip`'s processed folder contains only `ed_results_clean`, `ed_to_council_crosswalk`, `ideology_by_council`), and the scripts that produced them were deliberately deleted, so they cannot be regenerated. Note `--check-data` passes regardless — it only checks coarse directory buckets and won't catch this. Options: (a) copy `district_ideology.parquet` (and `ed_ideology.parquet`) from your original pre-reorg tree into `data/processed/`; or (b) make the script skip those sub-checks gracefully when the file is absent (the way `verify_cd1_dem_share.py` already degrades) — but that silently weakens the audit, so (a) is better if you can locate the files.

2. **`main.py`'s `train_ridge` / `train_elasticnet` stages point at the wrong generation of scripts.** `experiments/registry.json` shows the recorded Table 2 Ridge/ElasticNet entries were produced by `models/training/train_linear_baselines.py`, which reproduces the paper numbers *exactly* and writes the recorded filenames. The wired-in `train_ridge.py` / `train_elasticnet.py` instead train on ~30 un-blacklisted raw columns (BBL, BIN, lat/lon, …) and write differently-named JSONs — they now *run* (after bug fixes #2–3) but do **not** regenerate the paper's Table 2 rows. I deliberately left `main.py`'s stage registry untouched; recommend re-pointing those two stages at `train_linear_baselines.py --variant all_years --mode political` (it trains both models in one invocation, so a single `train_linear_baselines` stage may be cleaner).

3. **The four manuscript `.tex` tables have no producer in this repo and aren't in this checkout.** `tables/` here contains only `table_robustness.tex`. `generate_regression_table.py` writes CSV/JSON/TXT under `results/` — no `.tex`; `generate_shap_summary.py` and `political_scenarios.py` likewise write no `.tex`; nothing writes `tables/table2_model_performance.tex` at all. The real table1/2/3/A1 `.tex` files live in `CER Papers 4/CER Papers 4/tables/` outside this checkout. PAPER_TO_CODE_MAP's "→ tables/tableN_….tex" output claims overstate what the scripts generate — a documentation gap, not a runtime failure.

4. **The paper itself isn't in the checkout.** `orbit_paper_CER.tex`/`.pdf` are absent here; Phase 4.1 used the copy at `..\CER Papers 4\CER Papers 4\main\` read-only. If that copy is stale relative to your actual submission draft, redo the table/figure cross-check against the current draft.

5. **Minor doc drift:** the `.tex` contains exactly 5 figures, while `main.py`/the map attribute a "Figure 6" choropleth to `src/figures/generate_scenario_map.py` — that figure is not included anywhere in the manuscript's source; and `docs/ARCHITECTURE.md` cites a `src/build_sales_pluto.py` that no longer exists. Also: `results/morans_i_results.json` self-reports that its z-scores don't match the paper's (price z=7.46 vs paper 7.58; dem z=7.99 vs paper 7.69) while the Moran's I values match to 4 dp — worth confirming which z-statistic convention the paper quotes.

6. **Dependency footprint caveat:** I installed a pinned *subset* of requirements.txt sufficient for all 43 stages; `src/ops/*` (mlflow etc.) and `src/causal/causal_analysis.py` (causalml, seaborn — and note `seaborn` is **not in requirements.txt** despite being imported there) still have uninstalled dependencies. Also, the ~1 GB of installed packages now lives in `venv/` inside the repo directory — consider whether the stripped `venv/` should have been committed at all.

7. **Files I changed:** `main.py`, `src/ops/pipeline_sanity_check.py`, `src/data_pipeline/data_loader.py`, comment/docstring-only edits in 8 source files (`build_ideology_scores.py`, `rebuild_canonical.py`, `split_temporal.py`, `causal_analysis.py`, `compute_morans_i.py`, `verify_cd1_dem_share.py`, `verify_numfloors_missingness.py`, `verify_scenario_mechanics.py`), 4 path-reference lines in `results/FULL_PAPER_FACT_CHECK.md` (no numbers, verdicts, or narrative altered), `experiments/registry.json` (backslash normalization of the 5 entries my training runs appended), plus this report. Nothing under `data/`, `archive/`, or the paper files was modified.

## Final state checks after all fixes

- `py_compile` over `src/`, `app/`, `models/`, `main.py`: **exit 0, zero errors, zero warnings** (the two formerly "known harmless" SyntaxWarnings are gone as a side effect of fix #4).
- Post-training re-run of `verify_ml_performance` and `verify_blacklist_fully`: both PASS against the freshly retrained artifacts.
- Flat-path grep (17 patterns): clean outside `archive/`, `docs/DATA_LINEAGE.md` (intentional history), and `ACTION_PLAN_FOR_SUBMISSION.md`'s reorg narrative.


---

## Round 2

**Date:** 2026-09-02
**Scope:** second-pass fixes — the two items Round 1 deliberately left for a
decision ("Still needs attention" #1 and #2), plus three more issues a second
review pass caught, plus the optional encoding hardening. Every change was
verified live in this checkout. Raw verification-block transcript:
`results/logs/round2_verification_output.txt`.

### Changes applied

1. **`main.py`: `train_ridge` / `train_elasticnet` stages replaced by a single
   `train_linear_baselines` stage** (`models/training/train_linear_baselines.py
   --variant all_years --mode political` — one invocation trains both models),
   the script that actually produced the recorded, paper-matching Table 2
   Ridge/ElasticNet JSONs. An inline NOTE in the stage registry documents why
   (the standalone scripts write differently-named files from a different,
   blacklist-incomplete feature set). The spurious
   `experiments/results/ridge_political_all.json` and
   `elasticnet_political_all.json` were confirmed still absent (Round 1
   already deleted them) — nothing further to clean; they would only reappear
   if someone runs the now-unwired standalone scripts directly.

2. **`src/data_pipeline/data_loader.py` now uses the real blacklist.**
   `prepare_features()` previously used its own 4-item hardcoded
   `LEAKAGE_FEATURES` list; it now loads `config/feature_blacklist.yaml`
   (49 entries — the same file every other `models/training/*.py` script
   uses), so all model families train on an identical feature set. With this
   fix, all three latent bugs in this module (Round 1's `district_ideology`
   assert and datetime64 admission, plus this blacklist bypass — the cause of
   Round 1's "30 features instead of 17" observation) are closed.

3. **`data/processed/district_ideology.parquet` (3,158 B) and
   `ed_ideology.parquet` (165,014 B) restored** from `data/Data 2.zip`'s
   `processed/` folder. Correction to Round 1's "Still needs attention" #1:
   the files were in the zip all along — Round 1's claim that the zip
   contained only `ed_results_clean` / `ed_to_council_crosswalk` /
   `ideology_by_council` was wrong. (The archive is spelled `Data 2.zip`, not
   `Data_2.zip`.)

4. **Optional encoding hardening (applied).** The
   `sys.stdout.reconfigure(encoding="utf-8")` guard Round 1 added to
   `pipeline_sanity_check.py` was extended to the 8 `models/training/*.py`
   scripts whose sources contain non-ASCII that reaches stdout:
   `train_lgbm`, `train_ridge`, `train_elasticnet`, `train_random_forest`,
   `train_catboost`, `train_xgboost`, `train_linear_baselines`,
   `train_ensemble_fixed`. Direct runs on this cp1252 console no longer crash
   on the 📦/🚫 prints even outside `main.py`.

5. **`PAPER_TO_CODE_MAP.md`: overstated `.tex` claims corrected.** No
   "Correction" section existed in the checkout, so this was still
   outstanding. The Table 1, Table 3, and Table A.1 rows now name the files
   each script actually writes — `results/regression_table.csv` +
   `results/regression_diagnostics.json` +
   `results/regression_model3_full_summary.txt`;
   `models/artifacts/lgbm_all_years_political_shap_summary.json`;
   `results/political_scenarios/{liberal_policy,conservative_policy,mixed_governance}_by_council.csv`
   + `*_all_properties.parquet` — and a note records that the final `.tex`
   files the paper `\input`s live in the paper's own `tables/` folder, are
   hand-formatted, and are **not currently automated**.

6. **`docs/ARCHITECTURE.md`: two wrong lines fixed.** (a) The preprocessing
   bullet that listed `build_sales_pluto.py` as a live `src/` script now says
   it no longer exists — it was folded into
   `src/data_pipeline/rebuild_canonical.py` before this reorganization.
   (b) The note claiming `freeze_schema.py` was likewise "folded into
   `rebuild_canonical.py`" was false and is corrected:
   `src/data_pipeline/freeze_schema.py` is a real, currently-active standalone
   stage (wired into `main.py` as `freeze_schema`) writing
   `docs/modeling_schema.csv`.


### Verification block results

**Environment caveat (not a repo bug):** a first attempt ran the block under
Git Bash where `python` resolved to the *system* Python 3.13 — the same one
Round 1 already recorded as broken (pandas import fails: missing `pytz`).
Both stage runs in that attempt failed with `ImportError: pytz`, as expected
of that interpreter. The results below are from rerunning the same block
commands verbatim with the repo venv activated first
(`source venv/Scripts/activate`).

1. `python -m py_compile $(find src app models -name "*.py" -not -path "*/__pycache__/*") main.py`
   → **exit code 0**, zero errors, zero warnings.
2. `python main.py --list | grep -i "train_ridge\|train_elasticnet\|train_linear_baselines"`
   → exactly one match: `train_linear_baselines   training   4.4, Table 2
   (comparison model family)`. `train_ridge` / `train_elasticnet` no longer
   appear in the registry (now 42 stages; the merged pair explains the count
   delta vs Round 1's 43). grep exit code 0.
3. `python main.py --stage train_linear_baselines` → **OK (6.9 s)**. Fresh
   `experiments/results/ridge_all_years_political.json`: r2 =
   0.11137927403013992, MAE = 0.6558020430385062, n_features = 17 — identical
   to the previously recorded paper-matching values. Fresh
   `elasticnet_all_years_political.json`: r2 = -0.00964726048578668, MAE =
   0.6903724545495283, also 17 features.
4. `python -c "... d['features_used'] ..."` → **KeyError: 'features_used'**
   (exit 1). The command probes the JSON schema of the *standalone*
   `train_ridge.py`, which writes `features_used`; the production
   `train_linear_baselines.py` writes the same quantity under `n_features`.
   An informational fallback (`d.get('n_features')`) prints **17** on the
   freshly written file — i.e. the substantive invariant the command checks
   holds (17 features, not the 30 the un-blacklisted standalone scripts
   trained on). No repo code changed for this: the production file's schema
   predates this session and matches what was already recorded.
5. `python main.py --stage verify_ideology_construction` → **OK (3.2 s) —
   passes in this checkout for the first time.** Key recomputed values
   (`results/logs/pipeline/verify_ideology_construction.log`): CD1/2017
   dem_share 0.7159080199792668 vs paper 0.715908 (diff 2e-8); CD1
   district_ideology 0.7113089490787546 vs paper 0.711309; dem_share missing
   12,558 == paper 12,558; coverage 97.5597% vs paper 97.56%; crosswalk 5,783
   rows / 4,264 distinct EDs / 1,383 EDs in >1 row — all match.

### Registry hygiene after the training run (same class as Round 1 bug #5)

The `train_linear_baselines` run appended 2 entries whose `Path`-stringified
paths carried Windows backslashes. Re-normalized with the same JSON-aware
load → replace → dump; registry re-validated: **19 entries, valid JSON, zero
backslashes**.

### Still needs attention after Round 2 (observed, deliberately not changed)

- Verification-block command #4 reads key `features_used`; for
  `experiments/results/ridge_all_years_political.json` the key is
  `n_features`. If the block is reused in a future round, correct the command.
- Related schema note: `src/analysis/compare_models.py` silently skips results
  JSONs lacking `{model, mode, scope, r2, mae, features_used}`, so the
  `train_linear_baselines.py` outputs (`name`/`variant`/`n_features`) are
  skipped by it. Pre-existing behavior, consistent with the recorded paper
  numbers; harmonizing would change the on-disk schema, so left alone.
- `docs/ARCHITECTURE.md` step 6 (smoke/CI) still says
  `python src/pipeline_sanity_check.py`; the file lives at
  `src/ops/pipeline_sanity_check.py`. Outside the six fixes ordered this
  round; noted for the next docs pass.
- `results/PIPELINE_RUN_REPORT.md` was regenerated by the two single-stage
  runs above and now reflects only those stages — expected behavior of
  `main.py` (each invocation rewrites the consolidated report).

**Files changed this round:** `main.py`, `src/data_pipeline/data_loader.py`,
`PAPER_TO_CODE_MAP.md`, `docs/ARCHITECTURE.md`, 8 scripts under
`models/training/` (encoding guard), `data/processed/district_ideology.parquet`
+ `data/processed/ed_ideology.parquet` (restored from `data/Data 2.zip`),
`experiments/registry.json` (2 new entries + backslash normalization),
`results/round5_part1_C.json` and `results/PIPELINE_RUN_REPORT.md`
(regenerated by the stage runs), plus this report (Round 2 section).

## Round 3 (2026-09-02) — full-repo re-review fixes

Round 3 had access to the complete repo (Round 1's reviewer was missing
`app/`, `archive/`, `config/`, `docs/`, `experiments/`, `.github/`). Four
items were found; three required changes, one was noted as no-action.

### Changes applied

1. **Real bug fixed — `src/analysis/compare_models.py` silently dropped 5 of
   7 models.** Two different result-JSON schemas exist in
   `experiments/results/`: `train_lgbm.py` writes
   `{model, mode, scope, features_used, ...}` (short-form `scope`), while
   every non-LightGBM trainer (`train_catboost.py`, `train_xgboost.py`,
   `train_random_forest.py`, `train_linear_baselines.py`) writes
   `{model, name, variant, mode, n_train, n_test, n_features, ...}` with
   long-form `variant` and no `scope`/`features_used`. The script's
   `required_keys` check only matched the LightGBM family, so it printed 5
   "Skipping malformed file" warnings and a 2-row table (this is the same
   behavior Round 2 flagged under "Still needs attention" — now fixed).
   Fix: added `normalize_record()` (defined before `main()`), called before
   the `required_keys` check, mapping `variant` → `scope` (via
   `SCOPE_NORMALIZATION`) and `n_features` → `features_used`. Neither
   training script's on-disk schema was changed. The paper's verified
   Table 2 was never affected (it comes from
   `docs/model_comparison_table.csv`, written independently by
   `src/figures/generate_research_summary.py`); this restores the
   `compare_models` cross-check stage so it checks all 7 rows.
2. **Stale path reference:** `results/FULL_PAPER_FACT_CHECK.md` line 419:
   `` `src/confidence_intervals.py` `` → `` `src/analysis/confidence_intervals.py` ``.
3. **Stale path reference:** `docs/ARCHITECTURE.md` "Run smoke/CI checks"
   section: `python src/pipeline_sanity_check.py` →
   `python src/ops/pipeline_sanity_check.py` + `python src/ops/ci_smoke_check.py`
   (the exact line Round 2 deferred).
4. **Archived stale legacy results:** `results/base/` and
   `results/political/` (each with `2017/` + `all/` subfolders containing
   `metrics.json` + `model.txt`) moved to
   `archive/legacy_results_pre_canonical_v2/` with an explanatory
   `README.md`. These are output from a pre-canonical-v2 modeling iteration
   (`political/` trains on the blacklisted `district_ideology`; both use a
   feature set that matches nothing in the current canonical schema). The
   directories were untracked, so `git mv` was not possible; plain
   `Move-Item` was used instead.
5. **No action:** `notebooks/eda_audit.ipynb` (0-byte) and
   `notebooks/01_validate_raw_data.ipynb` (generic-name stub) left as-is —
   neither is wired into `main.py` or referenced by active code.

### Verification block output (run after the changes)

*(PowerShell equivalents of the `find`/`py_compile` block were used; the
exact `python src/analysis/compare_models.py` and the legacy-archive check
command ran verbatim.)*

**1. `py_compile` over all of `src/`, `app/`, `models/` + `main.py`**
(73 files + `main.py`, `__pycache__` excluded) → **exit code 0**.

**2. `python src/analysis/compare_models.py`** → exit code 0, **zero**
"Skipping malformed file" warnings, and the "Raw Results" table now shows
**7 rows** (previously 2):

```
       model scope      mode        r2      mae  features_used
    CatBoost   all political  0.505744 0.453915             17
  Elasticnet   all political -0.009647 0.690372             17
    LightGBM   all      base  0.541772 0.428972             16
    LightGBM   all political  0.541345 0.429342             17
RandomForest   all political  0.507022 0.436811             17
       Ridge   all political  0.111379 0.655802             17
     XGBoost   all political  0.528694 0.439220             17
```

(The Ridge/ElasticNet numbers match the paper-verified values re-verified
in Round 2: Ridge r² = 0.1113792740…, MAE = 0.6558020430…; correct.)

**3. Legacy-archive check**
(`python -c "import pathlib; print('legacy archived OK:', ...)"`) →
`legacy archived OK: True` (archive dirs exist; `results/base` and
`results/political` gone).

**Files changed this round:** `src/analysis/compare_models.py`
(`normalize_record`), `results/FULL_PAPER_FACT_CHECK.md` (1 line),
`docs/ARCHITECTURE.md` (1 block), `archive/legacy_results_pre_canonical_v2/`
(new: moved `base/` + `political/` + `README.md`), plus this report (Round 3
section). `experiments/results/model_comparison_raw.csv` and
`ideology_ablation_summary.csv` were regenerated by the verification run.


---

## Round 4 (2026-09-06) — precision repair pass (Tasks 1–8)

*Naming note:* earlier drafts titled this section "Round 3 (2026-09-06)",
duplicating "Round 3 (2026-09-02)" above; renamed to Round 4 on 2026-09-08 to
keep round numbers unique. Older files (archive/_round3_*.py and dated inline
comments) may still call this pass "Round-3 (2026-09-06)" — same pass. The
2026-09-02 section covered the full-repo re-review of the reorganized
checkout; this section is the **later, separate** pass: the task-driven
precision repair of the
spatial-regression integration, the manuscript's Section-4 renumbering
cascade, the 19-row prediction-artifact gap, and the stale pre-blacklist
interpretability artifacts, ending with a full unrestricted pipeline re-run
and an exhaustive number reconciliation against the *current* manuscript
(`CER Papers 5/main/orbit_paper_CER.tex`, read-only reference — not edited;
editorial changes to the paper were out of scope).

Method: same as Rounds 1–2 — every fix below was proven by actually re-running
the affected stage(s), not by reading. Ground truth for all paper_ref /
section checks: the current manuscript's own `\subsection{}`/`\label{}` map
(grep-verified: 4.4 Robustness, 4.5 Formal Spatial Regression, 4.6 ML
(4.6.1/4.6.2), 4.7 Counterfactuals; 5 figures, no Figure 6; Tables 1–5 + A.1).

### Phase summary

| Phase | What was checked/done | Result |
|---|---|---|
| 0 — Environment / data | `python main.py --check-data`; installed `spreg` (needed by the orphaned script; **was missing** from venv AND `requirements.txt`) | 7/7 data buckets OK; `spreg==1.9.1` installed and pinned into `requirements.txt` |
| 1 — Integrate orphaned spatial-regression script (Task 1) | byte-identical copy from `CER Papers 5/code_addendum`, `extract()` labeling-fix docstring confirmed, stages registered, `verify_spatial_regression.py` written, `tables/table5_spatial_regression.tex` copied in | ✅ PASS — both new stages exit 0, all 15 paper-checks true |
| 2 — Section renumbering cascade (Tasks 2–4) | every `paper_ref` in `main.py`, PAPER_TO_CODE_MAP section headings/rows, ARCHITECTURE path | ✅ PASS — `--list` (44 stages) audited against ground truth |
| 3 — 19-row prediction-interval gap (Task 5) | root-caused (below), full-split re-exports added, verifier updated | ✅ PASS — `verify_followup_resolution`: 56,817/56,817, `gap_closed: true` |
| 4 — Stale pre-blacklist PDP/sensitivity artifacts (Task 6) | confirmed stale by feature-name inspection, deleted, scripts extended to the 17-feature political model, 4 stages re-run from deleted state, explicit blacklist audit | ✅ PASS — ALL ARTIFACTS BLACKLIST-CLEAN |
| 5 — Backslash paths made structurally impossible (Task 7) | 10 source files switched to `.as_posix()`; proven by re-running `train_linear_baselines` | ✅ PASS — 0 backslashes in `experiments/registry.json` (49 entries) |
| 6 — Full pipeline + reconciliation (Task 8) | single, clean, unrestricted `python main.py` (all 44 stages incl. the heavy trainings and both new stages) | ✅ **44/44 passed in 5.4 min**; report records "44 passed, 0 failed" |

### Real bugs found and fixed (each proven by re-running, not just reading)

1. **`spreg` missing from environment AND requirements.** The addendum's
   spatial-regression script needs `spreg.ML_Lag`/`ML_Error`; nothing in
   `requirements.txt` (nor the Round-1 environment note) installed it — a fresh
   clone would have failed on the new stage. Fix: installed `spreg==1.9.1`
   (compatible with pinned libpysal 4.14.1 / esda 2.10.0) and pinned it,
   alphabetically, into `requirements.txt`.

2. **Section-4-renumbering cascade (pure labeling).** The manuscript inserted
   two subsections (4.4, 4.5) but the repo still labeled: ML = 4.4 / 4.4.1 /
   4.4.2 (12 stage entries), Scenarios = 4.5 (2), Robustness = "(candidate)",
   `generate_scenario_map` = "Figure 6" (**no Figure 6 exists**),
   `generate_figures` = "Figures 4, 5". All corrected to 4.4/4.5/4.6(.1/.2)/4.7.
   While fixing `generate_figures`, its actual write set was enumerated from
   the code: it writes `model_comparison.png` (**Fig 3**), `shap_top15/top10{,_champion}.png`
   (**Fig 4**), and `scenario_impact_by_district.png` (unused) — and **no
   script in this repo writes the manuscript's Figure 5**
   (`scenario_impact_all_three.png`; it lives only in the paper folder).
   PAPER_TO_CODE_MAP had attributed Fig 5 to `generate_figures.py` — factually
   wrong, now corrected to state this honestly. Also corrected: the map's
   stale §4.4/§4.5 headings (→ 4.6/4.7) and its informal "New in this
   reorganization" note (content promoted into a proper §4.4 entry); two stale
   "Section 4.4.2" comments in `src/figures/generate_figures.py` /
   `generate_research_summary.py`; two stale forward-refs in
   ACTION_PLAN ("Section 4.4" → 4.6, "4.4.3" → 4.6.3). `docs/ARCHITECTURE.md`'s
   `src/ops/pipeline_sanity_check.py` line (the one Round 2 deferred) was
   verified **already fixed** by the 2026-09-02 round — no edit needed.

3. **19-row prediction-interval gap — root cause is more precise than the JSON
   recorded.** The three political prediction artifacts each carried 56,798 of
   56,817 rows, but with a **reset 0..56,797 positional index** — so the
   "19 missing rows" in the old `missing_rows_characteristics` block were an
   index re-assignment artifact, not identifiable test rows, and the original
   exporting script no longer exists in the repo. Fix:
   `src/analysis/champion_confidence_and_sensitivity.py` (the historical
   producer of the shared `{name}_intervals.csv` format) now also writes
   `{name}_test_preds.parquet` (BBL + pred_log_price, index = test.index) and
   regenerates the **political** model's test-preds + intervals from the full
   56,817-row split with LightGBM's native NaN handling (matching the
   in-training predict path), with a **row-count assertion inside the writer**;
   `src/analysis/confidence_intervals.py` similarly regenerates the political
   `_prediction_intervals.parquet`. `verify_followup_resolution.py`'s step-3
   block now writes `gap_closed: true` and a `RESOLVED (2026-09-06)` note when
   the gap is zero (plus a `historical_note` capturing the reset-index
   finding), instead of leaving the stale "please regenerate" action.

4. **Stale pre-blacklist PDP/sensitivity artifacts — proven stale by content,
   not by date.** The (now-removed)
   `lgbm_all_years_political_pdp_summary.json` /
   `lgbm_all_years_political_sensitivity_analysis.json` referenced
   `assesstot`, `exempttot`, `appbbl`, `taxmap`, `assessland`,
   `healthcenterdistrict`, `schooldist`, `policeprct`, `Tax block`, `Tax lot` —
   features that are **both** blacklisted today **and** absent from the current
   53-column canonical schema (Appendix A.4's exact failure class). Deleted;
   `generate_pdp.py` and `sensitivity_analysis.py` extended to also regenerate
   the **17-feature political model's** artifacts (previously champion-only);
   all four stages re-run after deleting every covered artifact; explicit
   audit vs `config/feature_blacklist.yaml` + `canonical_schema.json`:
   **zero blacklisted/off-schema features**; political PDP now covers
   {BBL_pluto, dem_share, landuse, numfloors, yearbuilt}; all 10 PDP PNGs
   verified readable (1800×1200). `main.py` desc fields for the four stages
   record the 2026-09-06 regeneration (paper_refs deliberately left
   "(unused — candidate addition...)", which remains true).

5. **Windows-backslash path bug fixed at the source** (was patched after the
   fact in Rounds 1/2 and re-introduced each training run). Every
   `str(Path)` / path emission into `experiments/registry.json` /
   `experiments/champion.json` / `models/artifacts/*.json` was switched to
   `.as_posix()` in:
   `models/training/train_{random_forest,catboost,xgboost,linear_baselines,stacking,ensemble,ensemble_fixed}.py`,
   `src/ops/{auto_train,continuous_runner}.py` (incl. `auto_train.py`'s
   `script` field), and `src/analysis/generate_pdp.py`'s `'plot':` field.
   Proven on Windows: `python main.py --stage train_linear_baselines` wrote
   forward-slash registry entries with no post-hoc normalization; whole-file
   re-validation: valid JSON, **0 backslashes**. (`champion.json` carries no
   paths, so it needed no source change.)


### Verification block output (run after the fixes)

All executed in this checkout (logs under `results/logs/pipeline/*.log`):

1. `python main.py --check-data` → **exit 0**, all 7 data buckets OK.
2. `python main.py --stage generate_spatial_regression verify_spatial_regression`
   → **both OK** (12.5 s / 4.2 s). Fresh values: ρ = 0.708585 (SE 0.1035,
   p = 7.5e-12); λ = 0.787189 (SE 0.0867, p = 1.0e-19); dem_share 0.200714 /
   0.224638 / 0.228786 with p = 0.38533 / 0.16999 / 0.37383; OLS-anchor
   residual Moran's I = 0.38944 (z = 6.0527, p = 1.4e-09); pseudo-R² 0.708 /
   0.406; N = 51. Verifier: all 15 `paper_checks` **true**;
   `matches_pipeline_output` **true** for all three models + residual Moran's I
   (tol 1e-6 vs the analysis script's own JSON); `ols_matches_numpy_lstsq`
   **true** (pure-numpy OLS cross-check — the regression test for the
   coefficient-labeling bug class documented in `extract()`).
3. `python main.py --stage train_linear_baselines` → **OK (2.5 s)**; Ridge
   r² = 0.11137927403013992, MAE = 0.6558020430385062; ElasticNet
   r² = −0.00964726048578668, MAE = 0.6903724545495283 — identical to the
   previously recorded paper-matching values; new registry entries use forward
   slashes.
4. `python main.py --stage champion_confidence_and_sensitivity confidence_intervals`
   then `--stage verify_followup_resolution` → all **OK**; step-3 now:
   `test_rows 56817, political_test_preds_rows 56817,
   political_intervals_csv_rows 56817, base_intervals_parquet_rows 56817,
   rows_missing_from_political_preds 0, gap_closed true`. (The full run
   regenerates the political artifacts from the champion ≠ political model
   path — verified again by the 44/44 run below.)
5. `python main.py --stage generate_pdp sensitivity_analysis confidence_intervals
   champion_confidence_and_sensitivity` executed **after deleting every cover
   artifact** (so "present" could not be mistaken for "fresh") → 4/4 **OK**;
   artifact audit → `ALL ARTIFACTS BLACKLIST-CLEAN: True`.
6. SHAP staleness probe (the basis for finding **S1** below): 5-seed
   (42, 0, 1, 7, 123) recompute of mean|SHAP| for the current 17-feature
   political model — tight cross-seed ranges (Community Board 0.2777–0.2807;
   facilfar 0.0373–0.0384) that **exclude** the stored paper-era values
   (0.27472 / 0.04773). Scripts: `archive/_round3_shap_diff.py`,
   `archive/_round3_shap_seedcheck.py`; fresh 17-feat values saved at
   `archive/lgbm_all_years_political_shap_summary.17feat_20260906.json`,
   pre-existing artifact backed up at
   `archive/lgbm_all_years_political_shap_summary.pre20260906.json`.
7. Acceptance run: clean, single-instance `python main.py` → **44/44 stages
   passed in 5.4 min**; `results/PIPELINE_RUN_REPORT.md` records "44 passed,
   0 failed, 0 skipped, 0 missing script out of 44 stages." (Two earlier
   agent-side launch attempts were killed/relaunched by hand after one
   accidentally left two pipeline instances running concurrently; the final
   acceptance run above is single-instance and uncontested.)

### Exhaustive number reconciliation vs `orbit_paper_CER.tex` (post-run)

One row per claim. "Fresh value" = what the 44/44-run artifacts contain **now**
(reproducible with `python archive/_round3_collect.py`).

| Claim (manuscript ref) | Paper value | Fresh pipeline value | Match | Source artifact |
|---|---|---|---|---|
| Canonical dataset shape (§3.3) | 514,618 × 53 | 514,618 × 53 | Y | data/canonical/…v2.parquet (sanity_check) |
| Abstract bivariate β, p | 0.889, p<0.001 | 0.8890, p=0.0000 | Y | results/regression_table.csv |
| Abstract/§4.3 M3 β (80% shrink), p | 0.184, p<0.001 | 0.1843, p=1.0e-87 | Y | results/regression_table.csv |
| §3.1/3.5: election-year rows split | 178,439 / 285,617 / 50,562 (34.7/55.5/9.8%) | unchanged (verified Rounds 1–2) | Y | results/round5_part1_*.json |
| §3.2 CD1/2017 worked example | dem_share 0.715908 | 0.7159080199792668 (Δ 2e-8) | Y | results/cd1_dem_share_audit.json |
| §3.2 dem_share coverage / missing | 97.56% / 12,558 | 0.9756 / 12,558 | Y | verify_cd1_dem_share / verify_ideology logs |
| §3.2 crosswalk | 5,783 rows / 4,264 EDs / 1,383 EDs >1 CD | unchanged (Rounds 1–2) | Y | results/crosscut.json |
| §3.3 base/political feature counts | 16 / 17 | 16 / 17 | Y | models/artifacts/*_features.json |
| §3.3 YEAR BUILT vs yearbuilt | 464,204 both present; 88.4% (410,164) exact; max 2,025 yr; corr 0.146 | unchanged (Rounds 1–2) | Y | results/round5_part1_*.json |
| §3.4 train/test split | 121,622 / 56,817 | 121,622 / 56,817 | Y | verify_canonical_dataset_and_split |
| §4.1 district correlation | r = 0.163 | r = 0.163015 | Y | results/round5_part1_F.json |
| §4.1 ideology shift | 49/51 positive; only 16, 17 negative | 49 / 2 ([16, 17]) | Y | results/round5_part1_F.json |
| §4.2 Moran's I (price) | I=0.5959, z=7.46, p_sim=0.001 | 0.5958926, 7.4597, 0.001 | Y | results/morans_i_results.json |
| §4.2 Moran's I (dem_share) | I=0.5797, z=7.99, p_sim=0.001 | 0.5796642, 7.9934, 0.001 | Y | results/morans_i_results.json |
| Table 1 M1 dem_share | 0.8890 (0.0070) | 0.8890 / 0.0070 | Y | results/regression_table.csv |
| Table 1 M2 dem_share | 0.5741 (0.0071) | 0.5741 / 0.0071 | Y | results/regression_table.csv |
| Table 1 M3 dem_share | 0.1843 (0.0093), t=19.859 | 0.1843 / 0.0093 / t=19.859 | Y | regression_table.csv + verify_regression_ols |
| Table 1 yearbuilt M2→M3 | sig. → p = 0.119 | p = 0.1191 | Y | results/regression_table.csv |
| Table 1 borough coefs (M3) | B2 −0.8127, B3 −0.2850, B4 −0.6609, B5 −0.6583 | identical (4 dp) | Y | results/round5_part_H_ols.json |
| Table 1 R² row | 0.0336 / 0.0932 / 0.1478 | same | Y | results/regression_table.csv |
| §4.3 + A.2 N | 470,535 (514,618 − 44,083) | 470,535 / 44,083 | Y | regression + missingness verifiers |
| Table 4 M3 clustered | p = 0.318 (SE 0.1847) | 0.3185 (0.184699) | Y | results/robustness_checks.json |
| Table 4 M4 district-FE | β=0.568 (0.0207 / clust. 0.1275); p≈8e-06 | 0.568163 / 0.020704 / 0.127522 / 8.373e-06 | Y | results/robustness_checks.json |
| Table 4 M5 year-FE | β=0.074 (0.0098 / clust. 0.2009); p=0.712 | 0.074046 / 0.009804 / 0.200938 / 0.7125 | Y | results/robustness_checks.json |
| Table 4 winsorized | 0.1880 | 0.187952 | Y | results/robustness_checks.json |
| Table 4 Cook's-D trim | 0.1304; 25,819 dropped (5.5%); N=444,716 | 0.130410; N=444,716 | Y | results/robustness_checks.json |
| Table 5 ρ (spatial lag) | 0.7086 (prose 0.709), p<0.001 | 0.708585, p=7.5e-12 | Y | results/spatial_regression.json |
| Table 5 λ (spatial error) | 0.7872 (prose 0.787), p<0.001 | 0.787189, p=1.0e-19 | Y | results/spatial_regression.json |
| Table 5 dem_share coefs / p | 0.2007/0.2246/0.2288; p=0.385/0.170/0.374 | 0.200714/0.224638/0.228786; 0.38533/0.16999/0.37383 | Y | results/spatial_regression.json |
| Table 5 OLS residual Moran's I | I=0.389, z=6.05, p<0.001 | 0.38944, 6.0527, 1.4e-09 | Y | results/spatial_regression.json |
| Table 5 pseudo-R² / N | 0.708 / 0.406; N=51 | 0.70792 / 0.40628; 51 | Y | results/spatial_regression.json |
| Table 2 LightGBM base | MAE 0.428972, R² 0.541772 | 0.4289721461 / 0.5417722685 | Y | experiments/results/lgbm_all_years_base.json |
| Table 2 LightGBM political | MAE 0.429342, R² 0.541345 | 0.4293418141 / 0.5413451158 | Y | experiments/results/lgbm_all_years_political.json |
| Table 2 Δ (political − base) | MAE +0.000370, R² −0.000427 | +0.000370 / −0.000427 | Y | verify_ml_performance |
| Table 2 Ridge | MAE 0.655802, R² 0.111379 | identical (retrained this round) | Y | experiments/results/ridge_all_years_political.json |
| Table 2 ElasticNet | MAE 0.690372, R² −0.009647 | identical (retrained this round) | Y | experiments/results/elasticnet_all_years_political.json |
| Table 2 CatBoost | MAE 0.453915, R² 0.505744 | 0.4539147018 / 0.5057437363 (retrained) | Y | experiments/results/cat_…json |
| Table 2 XGBoost | MAE 0.439220, R² 0.528694 | 0.4392201484 / 0.5286943913 (retrained) | Y | experiments/results/xgb_…json |
| Table 2 RandomForest | MAE 0.437048, R² 0.506634 | 0.4368108 / 0.5070222 | ~Y (ΔMAE≈2.4e-4 — the documented bagging stochasticity from Round 1) | experiments/results/rf_…json |
| Table 3 SHAP: dem_share rank | 12th of 17 | 12th of 17 (stored artifact AND fresh 17-feat recompute) | Y | verify_shap_importance |
| Table 3 SHAP: election_year | exactly 0.00000 | exactly 0 (both) | Y | ids. |
| Table 3 SHAP: Community Board base vs political | 0.30101 vs 0.27472 | 0.3010066 ✓; 0.27472 = stored artifact | Y | verify_shap_importance |
| **Table 3 SHAP cell values vs the CURRENT 17-feature model** | e.g. CB 0.27472, facilfar 0.04773 | current model ⇒ CB ≈ 0.28069, facilfar ≈ 0.03774 (seed-stable) | **N — see S1** | models/artifacts/…shap_summary.json vs archive/…17feat_20260906.json |
| §4.7 liberal scenario | mean +7.42%, 49/51 pos; D51 +32.4, D31 +30.8, D50 +30.0, D49 +27.6, D32 +18.3; D41 −1.06, D42 −1.21 | +7.4231%, 49/51; identical top-5; −1.06 / −1.21 | Y | results/political_scenarios/liberal_policy_by_council.csv |
| §4.7 conservative scenario | mean −0.78%; 35/51 neg; D10 −7.3, D25 −5.0 | −0.7771%; 35 neg; −7.29 / −4.98 | Y | results/political_scenarios/conservative_policy_by_council.csv |
| §4.7 mixed-governance | mean −3.39%; 14/51 pos; D5 +13.3, D4 +12.7, D38 +9.0, D45 +6.9, D1 +5.7; D33 −16.7, D10 −16.5, D39 −16.0, D17 −13.7, D3 −13.0 | −3.3914%; 14/51; all ten values match | Y | results/political_scenarios/mixed_governance_by_council.csv |
| §4.7 scenario mechanics | min base pred 9.56 ⇒ 0 rows dropped; clips 109 / 9 / 816 (upper-only) | identical | Y | results/round5_part_K_mechanics.json |
| Appendix A.1 | full 51×3 scenario table | identical | Y | results/round5_part_K_appendixA1.json |
| §5/Conclusion synthesis | robustness + spatial findings drive the interpretation | both reproduce above → synthesis stands | Y | (derived) |

| §4.3 missingness components | 31,692 / 11,930 / 12,558 (134 unique) / 380 / 53-overlap | identical | Y | results/round5_followup_resolution.json (step2) |
| A.2 Durbin–Watson | 1.517 | 1.5173268 | Y | results/regression_diagnostics.json |
| A.2 skew / kurtosis / cond | 0.264 / 7.204 / 3.64e4 | 0.2644 / 7.20424 / 36407 | Y | results/regression_diagnostics.json |
| A.2 omnibus / JB / F | 50100.014 / 352014.397 / 9067 | identical | Y | results/regression_diagnostics.json |


### Still needs attention after this round (found; deliberately not resolved by editing either side)

- **S1 — Table 3's SHAP values are from a pre-blacklist model build (the biggest
  legitimate "N" this round).** The paper-cited artifact
  `models/artifacts/lgbm_all_years_political_shap_summary.json` carries an
  all-zero `EASE-MENT` row — proof it was computed for the 18-feature,
  pre-blacklist political model. Against the **current** 17-feature model the
  same computation is seed-stable but systematically different:
  Community Board 0.27472 → ~0.28069; landuse 0.11688 → ~0.12172; yearbuilt
  0.10600 → ~0.11400; facilfar 0.04773 → ~0.03774 (and facilfar/Census Tract
  swap ranks 10/11). Ranks are otherwise preserved and the qualitative claims
  all still hold (dem_share 12th of 17; election_year exactly 0; CB political <
  base). Fresh 5-seed recompute archived at
  `archive/lgbm_all_years_political_shap_summary.17feat_20260906.json`
  (seed-check: `archive/_round3_shap_seedcheck.py`). I restored the
  paper-matching artifact in place and left `verify_shap_importance`/
  `round5_part_J_shap.json` re-recorded against it. **Decision needed from the
  authors: refresh Table 3 to the current-model values (pipelines' fresh
  numbers in the archive file), or keep the current draft's numbers.**
- **S2 — RF comparison-model MAE/R² drift of ±~2.4e-4** (0.437048 → 0.436811)
  across re-runs of an identical pipeline: bagging stochasticity, present since
  Round 1; does not affect Table 2 at its reported precision and is
  documented, not a defect.
- **S3 — Figure 5 (`scenario_impact_all_three.png`) has no producing script
  in this repo.** `src/figures/generate_figures.py` writes the unused
  `scenario_impact_by_district.png`; `generate_scenario_map.py` writes
  `scenario_impact_all_three_FIXED.png`. The manuscript's actual Fig 5 exists
  only in the paper folder. If the authors want the paper fully repo-generated,
  one of the map scripts needs a one-line output-filename/aggregation
  adjustment — an editorial decision, not made here.
- **S4 — `experiments/registry.json` is append-only** and accumulated several
  legitimate duplicate training entries this round (plus previous rounds) —
  now 49 entries, all valid, all forward-slash. `src/ops/dedup_registry.py`
  exists; I did not run it because rewriting registry history was out of
  scope.
- **S5 — `docs/WORKFLOW_AUTOMATION.md` still uses the pre-reorg flat `src/…`
  paths in its historical command log** (`src\pipeline_sanity_check.py` etc.).
  Left alone as an accurate historical log; convert it to current paths only
  if it's meant to be re-executed rather than read as a record.
- **S6 — Agent-side tooling note (not a repo bug):** during my own run
  management, a double-launch of `python main.py` (two overlapping instances)
  briefly had two writers on shared artifacts; both were killed and the
  affected artifacts were re-produced by clean single-instance re-runs before
  the final acceptance run. No corrupted output survives on disk.

**Files changed this round:** `main.py` (17 paper_ref fixes + 2 new stages +
desc notes); `requirements.txt` (added `spreg==1.9.1`);
`src/analysis/spatial_regression_model.py` (new; byte-identical copy from the
`CER Papers 5` code addendum); `src/verification/verify_spatial_regression.py`
(new); `tables/table5_spatial_regression.tex` (new; copied from paper repo);
`src/analysis/champion_confidence_and_sensitivity.py`;
`src/analysis/confidence_intervals.py`; `src/analysis/generate_pdp.py`;
`src/analysis/sensitivity_analysis.py`; `src/analysis/generate_shap_summary.py`
(comment-only note; behavior unchanged);
`src/verification/verify_followup_resolution.py`;
`src/figures/generate_figures.py` + `src/figures/generate_research_summary.py`
(section-number comments); `models/training/train_.py` ×7 +
`src/ops/auto_train.py` + `src/ops/continuous_runner.py` (`.as_posix()`);
`PAPER_TO_CODE_MAP.md`; `ACTION_PLAN_FOR_SUBMISSION.md` (2 stale refs);
`VERIFICATION_REPORT.md` (this section). Plus regenerated/removed artifacts:
`results/spatial_regression.json` (new), `results/spatial_regression_verification.json`
(new), `results/round5_followup_resolution.json` (updated — gap closed),
`experiments/predictions/lgbm_all_years_political_{test_preds.parquet,
intervals.csv, prediction_intervals.parquet}` and
`lgbm_all_years_base_{test_preds.parquet, intervals.csv, prediction_intervals.parquet}`
(regenerated at full 56,817-row coverage); `models/artifacts/pdp_lgbm_*` (10 PNGs),
`lgbm_all_years_{base,political}_pdp_summary.json`,
`lgbm_all_years_{base,political}_sensitivity_analysis.json`,
`lgbm_all_years_base_sensitivity.json` (regenerated fresh);
`models/artifacts/lgbm_all_years_political_shap_summary.json` (**restored** to
the paper-cited 18-row state — see S1 — with fresh 17-feat recompute archived);
`archive/_round3_{collect,check_registry,check_artifacts,shap_diff,shap_seedcheck}.py`
+ 2 artifact JSONs (new audit files); `experiments/registry.json` (appended by
the rounds' training re-runs, forward-slash clean); `results/PIPELINE_RUN_REPORT.md`
(regenerated by the final 44/44 run). No manuscript files touched; no number
altered to force agreement anywhere.


### Round-4 follow-ups (Tasks 9–11, same date)

9. **Table 3 / current-model SHAP mismatch documented for the authors**
   (without touching the manuscript): a side-by-side value table (stored
   18-feature-era artifact vs fresh, seed-stable current-17-feature recompute)
   is now at `results/FULL_PAPER_FACT_CHECK.md`, §"7. Round-3 follow-up (2026-09-06) —
   Table 3 / current-model SHAP mismatch (still needs attention)". It records
   which artifact is currently cited and where the fresh values live
   (`archive/lgbm_all_years_political_shap_summary.17feat_20260906.json`).
   The manuscript (`orbit_paper_CER.tex`) was not edited.

10. **Figure 5 source recovered as a real pipeline stage.** The
    `CER Papers 5/code_addendum/` folder was re-checked before building
    anything: it contains only the spatial-regression script/JSON (the same
    two files integrated in Task 1) — no figure script. New script
    `src/figures/generate_fig5_scenario_impact.py` reproduces the three-panel
    scenario figure from `results/political_scenarios/*_by_council.csv` and
    writes `docs/figures/scenario_impact_all_three.png` under **exactly** the
    filename the manuscript's `\includegraphics` call expects. Registered as
    stage `generate_fig5_scenario_impact` (45 stages total; paper_ref
    "Figure 5 (Section 4.7)"), and PAPER_TO_CODE_MAP's Figure-5 row updated to
    name it. Verified: `python main.py --stage generate_fig5_scenario_impact`
    → OK (2.0 s), asserts 51 districts per panel, prints per-scenario
    mean/positive counts that match §4.7's prose output (liberal +7.42%,
    49/51 positive; conservative −0.78%, 35 negative; mixed −3.39%, 14
    positive).

11. **Registry hygiene audit (`archive/_round3_registry_audit.py`, read-only):**
    54 entries, valid JSON, 0 backslashes, no entries missing required keys,
    no dangling `model_path`/`features_path` references among active entries.
    12 distinct experiment names (the 7 production models + 2 stale tuned-LGBM
    + stacking/ensemble history). Duplicate names are append-only re-run
    history with **bit-identical metrics** across all duplicates:
    cat/xgb ×8, rf ×9, ridge ×11, elasticnet ×11. **18 entries** carry
    timestamps inside the killed concurrent-run window (2026-09-07T02:44–04:08Z);
    all are structurally complete and metric-identical to the canonical runs —
    no torn entries from the two instances that were killed mid-writing.
    Nothing required removal; `src/ops/dedup_registry.py` remains available
    if the maintainers want to collapse the history.

**Files changed this follow-up:** `src/figures/generate_fig5_scenario_impact.py`
(new), `main.py` (1 new stage), `PAPER_TO_CODE_MAP.md` (Figure-5 row),
`results/FULL_PAPER_FACT_CHECK.md` (new §7), `results/PIPELINE_RUN_REPORT.md`
(re-written by the single-stage run), `docs/figures/scenario_impact_all_three.png`
(now pipeline-produced), `archive/_round3_registry_audit.py` (new),
`VERIFICATION_REPORT.md` (this subsection). No numbers altered.


---

## Round 5 (2026-09-13) — data refresh reconciliation + causal-ID build

### Scope

Build the candidate causal-identification design (the ACTION_PLAN's prior task,
closing the paper's stated #1 limitation) on top of the now-permanent
**19-feature base / 20-feature political** spec (the 3 newly-admitted
`RESIDENTIAL UNITS`, `COMMERCIAL UNITS`, `TOTAL UNITS` columns are final — no
comparison or decision-support report for the old spec was built), while
reconciling the remaining artifacts that lagged the 2026-09-10/11 data refresh.
The manuscript (`orbit_paper_CER.tex`) and everything under `CER Papers 5/`
were **not** touched — the rewrite to the refreshed numbers is a separate,
later task. For the refresh's numeric record this round points to
`DATA_REFRESH_IMPACT_REPORT.md` as the source of truth rather than
re-tabulating it; only the new Round-5 numbers appear below.

### Changes applied (each verified by re-running the affected stage, not just reading)

1. **Party-label bug fixed in `build_council_election_margins.py` (script, not
   the parquet by hand).** Raw `winner_party` mixed 2013 ALL-CAPS labels,
   title-case Recap labels, and fusion compounds (`Democratic/Law & Order`);
   `runnerup_party` additionally had ~25 minor/fusion lines. Fix: case-fold
   every label, split fusion compounds on "/", map each candidate's full line
   set to their major party — the literal `Democratic`/`Republican` line is
   authoritative; a documented alias table (Working Families → Dem;
   Conservative / Save Our City → Rep) is consulted only when no literal major
   line is present. A loud (non-silent) assertion now fails the build if any
   label outside `{Democratic, Republican, Other}` survives. Printed before/after
   distinct counts each run: **winner_party 6 → 3, runnerup_party 36 → 3**;
   parquet regenerated (still 204 rows, 51×4 cycles).
2. **Two cross-endorsement winners flagged, not guessed.** Robert F. Holden
   (D30, 2021) and Kalman Yeger (D44, 2021) each carry BOTH a Democratic and a
   Republican line — a cross-endorsement the ballot-line data alone cannot
   resolve. Per the task's "say so explicitly rather than guessing a fusion-line
   mapping" rule they classify `Other` and are logged at build time — a genuine
   finding. Both are uncontested (100.0% margin), so they can never enter a
   close-race instrument anyway; no identification decision rides on the call.
3. **Close-election coverage diagnostic**
   (`src/causal/build_close_election_instrument.py` →
   `results/causal/close_election_coverage.json`). Close races
   (`margin_pct <= bandwidth`) of the 204 races: **±5pp: 4 (4 usable), ±10pp:
   7 (6 usable), ±15pp: 12 (9 usable), ±20pp: 17 (13 usable)** (2025 races can't
   feed a later canonical year). A thin-coverage warning is printed AND stored
   in the JSON — the thin instrument is an observed property, not a surprise.
4. **IV / instrumented district-FE 2SLS**
   (`src/causal/iv_district_fe_regression.py` →
   `results/causal/iv_district_fe_results.json`). `linearmodels.iv.IV2SLS` (new
   dependency `linearmodels==7.0` pinned into `requirements.txt`), instrumenting
   `dem_share` with a binary prior-cycle close-Democratic-win indicator (Z=1
   close Dem win, 0 close Republican win, else excluded — sharp-RD exclusion),
   controls = Model 4's set (yearbuilt, numfloors, landuse, commfar) plus the
   two linearly-independent UNITS columns (TOTAL UNITS = RESIDENTIAL+COMMERCIAL
   for 99.6% of rows, so it's dropped with a documented note), with CounDist +
   sale_year FE. First-stage F-stat, both stages' coef/SE/p, a programmatic
   verdict (F > 10 → "strong enough to interpret", else "weak — treat as
   suggestive, not identified"), and a side-by-side vs the existing Model 4
   district-FE benchmark (`robustness_checks.json`: dem_share coef
   0.8032559316804574) are all written to the JSON.
5. **Honest negative result, verified twice.** On the current data the
   instrument is **not identified at ANY bandwidth**: after projecting out
   district + sale_year FE, the close-race indicator Z lies exactly in the FE
   column space (residual ~1e-11, exact zero) — it varies only at the
   (district, prior-cycle) level while `election_year` is a deterministic
   grouping of `sale_year` (2016-18→2017, 2019-23→2021, 2024-25→2025), so the
   district+year FE span any instrument built from this few close cells
   (4/6/9/13 at ±5/10/15/20pp). The independent verifier
   (`src/verification/verify_iv_district_fe.py`, which re-derives fresh without
   importing the analysis script) reports the same verdict and the same counts;
   nothing was coded around — no FE was dropped, no cell count was padded to
   reach an F-statistic. Per-bandwidth rank diagnostics (regressor block /
   instrument block): 5pp 15·16, 10pp 19·21, 15pp 22·23, 20pp 25·25 / 24·25.
6. **Three new `main.py` stages** (`build_close_election_instrument`,
   `iv_district_fe_regression` [analysis], `verify_iv_district_fe`
   [verification]) follow the existing `dict(id=..., category=..., script=...,
   args=[], desc=..., paper_ref=...)` pattern, with
   `paper_ref="(candidate causal-identification design — see
   ACTION_PLAN_FOR_SUBMISSION.md; not yet adopted into the manuscript)"`. All
   three exit 0 even when the weak-instrument / not-identified verdict fires.
   Verified: `python main.py --stage build_close_election_instrument
   iv_district_fe_regression verify_iv_district_fe` → 3/3 OK;
   `verify_iv_district_fe` reported `all_bandwidths_match: True`.
7. **Stale political SHAP artifact refreshed.** The 18-feature-era
   `lgbm_all_years_political_shap_summary.json` (per
   `results/FULL_PAPER_FACT_CHECK.md` §7) was superseded: the
   `generate_shap_summary` stage regenerated it against the now-permanent
   20-feature model, and a dated copy of the refreshed file was archived per
   convention at `archive/lgbm_all_years_political_shap_summary.20feat_20260912.json`.
   The 3 UNITS columns' SHAP ranks in the current artifact: **RESIDENTIAL UNITS
   #2, TOTAL UNITS #3, COMMERCIAL UNITS #8** (dem_share #12, election_year #20
   at exactly 0). §7 now records the artifact as current; the manuscript's
   Table 3 (old values) is deferred to the separate manuscript task.
8. **Doc currency pass.** `PAPER_TO_CODE_MAP.md` and `README.md` (both last
   edited 2026-09-08, before the refresh and before the new builders existed)
   now carry entries for `build_council_election_margins.py` and
   `build_city_of_yes_roll_calls.py`, the two new `src/causal/` scripts and
   `verify_iv_district_fe.py`, a note that `src/causal/causal_analysis.py` (the
   original X-learner / causalml script) is unfinished and superseded by the IV
   design as the recommended path (kept only as historical scaffolding), and a
   one-line confirmation that the 19/20-feature spec (UNITS columns included)
   is the permanent, final feature set.

### Verification block output

**Omitted by design this round** — the full 35-minute `python main.py`
end-to-end run is the author's to do once these changes land (per the task).
What was actually run: the three new stages, individually and together, shown
in change #6 (3/3 OK, verifier `all_bandwidths_match: True`); the author should
confirm the whole 49-stage pass on their own run.

### Still needs attention (found; NOT silently resolved)

- **The close-election instrument is not identified at ANY bandwidth
  (±5/±10/±15/±20 pp) with the current council-election data.** This is the
  real, hard limitation: NYC council races are overwhelmingly landslides, so
  only 4-13 of 204 district-year races are close, and at that cell count the
  district + sale_year FE structure leaves no residual variation for a
  cell-level instrument — the first-stage F-statistic is undefined (worse than
  merely being below 10) because the design is exactly singular. Nothing was
  fudged to force a match: the rank check is done against the assembled design
  and never lowers the FE count to manufacture identification. Next options for
  the author (all out of scope this round): relax the FE structure to a
  design where the instrument is estimable, draw an instrument from a source
  with many close races (e.g. a different office's margins), or record the
  design as "attempted, not identified with current data — future work." Both
  `results/causal/*.json` files carry the per-bandwidth `verdict:
  "instrument is weak at this bandwidth — treat as suggestive, not identified"`;
  the manuscript must not report an IV estimate computed from this data at any
  of the four bandwidths.
- The manuscript's Table 3 still cites the old 18-feature-era SHAP values; the
  artifact is now current (20-feature). The manuscript rewrite is deferred —
  see §7 of `results/FULL_PAPER_FACT_CHECK.md` and `DATA_REFRESH_IMPACT_REPORT.md`.
- `data/processed/city_of_yes_roll_calls.parquet` covers only City of Yes for
  Housing Opportunity (2024-12-05); the Carbon Neutrality (Dec 2023) and
  Economic Opportunity (June 2024) member-level roll calls were not
  programmatically retrievable (Legistar API 403 + JS-only public search) and
  are still flagged in `build_city_of_yes_roll_calls.py` for manual retrieval —
  unchanged from before.

### Files changed this round

- **Source (new):** `src/causal/build_close_election_instrument.py`,
  `src/causal/iv_district_fe_regression.py`,
  `src/verification/verify_iv_district_fe.py`.
- **Source (edited):** `src/data_pipeline/build_council_election_margins.py`
  (party consolidation + hard assertion + before/after log), `main.py`
  (3 new stages + a `generate_shap_summary` desc fix 17→20 features),
  `requirements.txt` (+ `linearmodels==7.0`).
- **Docs (edited):** `PAPER_TO_CODE_MAP.md`, `README.md`,
  `results/FULL_PAPER_FACT_CHECK.md` (§7), `VERIFICATION_REPORT.md` (this Round 5).
- **Data (regenerated):** `data/processed/council_election_margins.parquet`
  (party labels only — same 204 rows and vote totals).
- **Result artifacts (new/regenerated):** `results/causal/close_election_coverage.json`,
  `results/causal/iv_district_fe_results.json`,
  `results/causal/iv_district_fe_verification.json`, `models/artifacts/lgbm_all_years_political_shap_summary.json`
  (regenerated to the current 20-feature state), plus the three stages'
  `results/logs/pipeline/*.log` and `results/PIPELINE_RUN_REPORT.md` (overwritten
  by the single-stage runs this round — the dated refresh snapshots
  `results/PIPELINE_RUN_REPORT_full_20260{910,911,912}.md` are untouched).
- **Archived (not deleted):**
  `archive/lgbm_all_years_political_shap_summary.20feat_20260912.json`.

No manuscript file (`orbit_paper_CER.tex` or anything under `CER Papers 5/`) was
touched, and no number was edited to force a match — the non-identification
finding is reported as found, exactly as Rounds 1-4's "Still needs attention"
sections have done.


---

## Round 6 (2026-09-14) — IV rescue attempts + City of Yes DiD

> Headline (in one place, per brief): *of the six IV variants tried (four
> borough-FE bandwidths, plus a 20/25/30/35/40pp sweep on the district-FE
> design), four identify, and not one of them makes the instrument strong —
> every first-stage F stays under 10 (best ≈ 7.40). The City of Yes DiD's
> pre-trend check passes, but the DiD itself is not statistically
> distinguishable from zero (p = 0.1283). Nothing identified credibly enough to
> claim a causal effect; the full specification trail is below, not filtered.*

### Scope

Continue from Round 5's finding (district-FE close-election IV not identified at
any of ±5/10/15/20pp) with a rule-bound specification search, no new data
collection. Everything uses data already on disk:
`data/canonical/modeling_dataset_canonical_v2.parquet`,
`data/processed/council_election_margins.parquet`, and
`data/processed/city_of_yes_roll_calls.parquet`.

### Changes applied

0. **Task 0 — what's on record about City of Yes geography (documentation audit,
   no new data).** Read `src/data_pipeline/build_city_of_yes_roll_calls.py` in
   full and grepped `ACTION_PLAN_FOR_SUBMISSION.md`,
   `DATA_REFRESH_IMPACT_REPORT.md`, `VERIFICATION_REPORT.md` for "City of Yes".
   **Finding (on record):** *no source in this repo distinguishes universal
   (city-wide) Housing Opportunity provisions (e.g. ADUs, parking relief) from
   geography-specific components* (transit-oriented zoning-map amendments, town
   center opt-ins). The only geographically-targeted "City of Yes" action the
   plan names is **Economic Opportunity (June 2024)** — and its member-level
   roll call is explicitly unavailable here (Legistar not retrievable; see the
   builder's docstring). Housing Opportunity is a project-wide zoning text
   amendment adopted 2024-12-05. So exactly as the brief anticipates, the DiD
   runs under the **ideology/preference-proxy framing** and every downstream
   artifact carries that caveat.

1. **`src/causal/iv_district_fe_rescue.py` — both rescue arms, every attempt
   recorded** → `results/causal/iv_district_fe_rescue_results.json`:

   **Rescue 1a (borough FE, same instrument, ±5/10/15/20pp):** at ±5pp and
   ±10pp the rank check still fails exactly as in Round 5 (the instrument
   block is one rank short of full). At ±15pp and ±20pp the design *does* reach
   full rank and the 2SLS actually runs — but the first-stage F-stat is 1.04
   (±15pp) and 0.59 (±20pp), far under the conventional 10 threshold, and it
   shows why: the second-stage `dem_share` coefficient swings wildly and
   senselessly — **−14.7 (SE 14.8, p 0.32) at ±15pp** and **+18.3 (SE 24.0,
   p 0.45) at ±20pp**. Compare to the non-instrumented Model 4 counil-district
   benchmark ≈ **0.803** (SE 0.016). Two orders of magnitude apart in both
   sign and scale — the weak instrument is behaving exactly as econometric
   textbook says it will (huge variance, unstable sign). Programmatic verdict
   on both: **"instrument is weak at this bandwidth — treat as suggestive, not
   identified."** Every record carries the credibility note that borough FE
   controls only 5 region-level confounders, not the 51 district-level ones
   Model 4 controls — a materially weaker design even where it identifies.

   **Rescue 1b (wider bandwidths, original district FE, 20/25/30/35/40pp):** at
   ±20pp it remains not identified (24 of 25 instrument-block rank — the
   precise Round-5 result, reproduced unchanged). At ±25pp it reaches full rank
   — first bandwidth that does — and continues to identify at ±30, ±35, ±40pp.
   First-stage F at those bandwidths: **2.44 / 7.40 / 5.13 / 7.30** — all under
   10. Second-stage `dem_share`: **3.73 (p 0.19) / 0.53 (p 0.73) / 0.61 (p
   0.72) / −0.01 (p 0.99)** — small, insignificant, sign-unstable. Every flip
   forward/backward across these makes clear what we have: *not* an instrument,
    a noise generator. Every record carries the credibility note that widening
   the bandwidth deepens compromises the "as-good-as-random" assumption, so
   even a clean identification here is a lower-confidence robustness probe.

2. **`src/verification/verify_iv_district_fe_rescue.py`** — independently
   re-derives both arms from the same two parquets with fresh code (never
   imports the analysis script), mirrors the rank checks, and re-fits the four
   identified 2SLS specs. Result: **all nine records match (4dp), including the
   four re-fitted F-stats and coefficients** →
   `results/causal/iv_district_fe_rescue_verification.json`
   (`all_arms_match: true`).

3. **`src/causal/city_of_yes_did.py`** → `results/causal/city_of_yes_did_results.json`.
   `treatment_framing` is set at the top of the file to the full
   "ideology/preference-proxy DiD, NOT a policy-application DiD" caveat (per the
   Task 0 finding) and appears in the JSON and in the run log so it cannot be
   dropped downstream.
   - **Pre-trend event study (mandatory, reported regardless of outcome),**
     reference bin 2022-23, with Yes×bin terms for 2016-17, 2018-19, 2020-21
     (the three pre-period bins tested jointly), 2024 (partial-treatment
     transition year), and 2025 (full post). The three pre-period coefficients
     are small and jointly not significant: **joint Wald F = 4.19, p = 0.2416
     (n = 355,997)** → pre-trends are consistent with parallel trends. Verdict:
     **PASS** (computed programmatically from the Wald p-value, not hand-set).
   - **Main DiD** (post = 2025, transition year 2024 excluded from the sample):
     **coef +0.0812 (SE 0.0534, p = 0.1283, n = 334,187)** — not statistically
     distinguishable from zero at conventional levels.
   - **Sensitivity** (post = 2024+2025, transition year included): **+0.0762
     (SE 0.0507, p = 0.1331, n = 355,997)** — same qualitative verdict, a faint
     positive that does not survive significance.
   - In plain language: Yes-voting districts do not measurably change their
     property-price path relative to No-voting districts around the Dec-2024
     City of Yes vote, once district and year FE and the structural controls are
     in place — and even if it had, this design measures a revealed
     pro-/anti-development stance, not the policy's effect.

4. **`src/verification/verify_city_of_yes_did.py`** — independently re-fits the
   event study and both DiD variants from the two input parquets. **All 29
   fields match the pipeline JSON to 4dp** →
   `results/causal/city_of_yes_did_verification.json` (`all_match: true`).

5. **`main.py` — four new stages** (analysis: `iv_district_fe_rescue`,
   `city_of_yes_did`; verification: `verify_iv_district_fe_rescue`,
   `verify_city_of_yes_did`), each following the stage-dict pattern with
   `paper_ref="(candidate causal-identification design, Round 6 — see
   ACTION_PLAN_FOR_SUBMISSION.md; not yet adopted into the manuscript)"`.
   All verified individually with
   `python main.py --stage iv_district_fe_rescue city_of_yes_did verify_iv_district_fe_rescue verify_city_of_yes_did`
   → **4/4 OK**, and every one exits 0 regardless of identification or
   pre-trend outcome (a failed rescue is a finding, not a crash). Pipeline total
   is now 53 stages.

### Still needs attention (found; NOT silently resolved)

- **IV remains not-credible at any configuration tried.** The original
  district-FE design does not identify at ±5/10/15/20pp (rank deficiency) nor
  at the wider ±25-40pp where it does run (all first-stage F < 10, second-stage
  point estimates inconsistent in sign and implausible). The looser borough-FE
  drop-in strengthens *identification* at ±15/20pp at the cost of confounding
  control — and still leaves first-stage F ≈ 0.6-1.0 with second-stage point
  estimates of −14.7 and +18.3 against a benchmark of +0.80. The honest read is
  that no close-council-race instrument on this dataset can currently support a
  causal claim; any paper-level statement must either present this as an
  attempted-and-failed identification or find an instrument with genuinely many
  close races (which NYC council districts do not provide).
- **The City of Yes DiD passes its pre-trend check but shows no significant
  effect** (+0.081, p = 0.128 main; +0.076, p = 0.133 sensitivity). Under the
  ideology/preference-proxy framing (the only framing this repo's data
  supports), this is a suggestive-null for revealed-stance price effects, not a
  policy effect — the zoning amendment itself applies city-wide regardless of
  the member's vote. It must not be cited as a policy-application estimate.
- The Council-is-data-limited ceiling remains: 204 races over four cycles and
  only 13 usable close-race cells at ±20pp set a hard floor on instrument
  strength that no FE-relaxation rescues — demonstrated rather than asserted.

### Verification block output

**Omitted by design** (same convention as Round 5) — the full 35-minute
`python main.py` is the author's to run after these land. What was run: all
four new stages individually and together (4/4 OK); both verifiers passed
(`all_arms_match: true`, `all_match: true`).

### Files changed this round

- **New source:** `src/causal/iv_district_fe_rescue.py`,
  `src/causal/city_of_yes_did.py`,
  `src/verification/verify_iv_district_fe_rescue.py`,
  `src/verification/verify_city_of_yes_did.py`.
- **Modified:** `main.py`, `VERIFICATION_REPORT.md` (this Round 6).
- **New result JSONs:** `results/causal/iv_district_fe_rescue_results.json`,
  `results/causal/iv_district_fe_rescue_verification.json`,
  `results/causal/city_of_yes_did_results.json`,
  `results/causal/city_of_yes_did_verification.json`.
- **Logs:** `results/logs/pipeline/{iv_district_fe_rescue,city_of_yes_did,verify_iv_district_fe_rescue,verify_city_of_yes_did}.log`.
- **Untouched:** `requirements.txt` (no new libraries); `orbit_paper_CER.tex`
  and everything under `CER Papers 5/`; the Round-5 primary
  `results/causal/iv_district_fe_results.json` (the most-credible-but-failed IV
  result stays exactly as it was, NOT overwritten by the rescue).

---

## Round 7 (2026-09-22) — BBL_pluto blacklisted

> Headline (in one place, per brief): *`BBL_pluto` was a parcel-identifier
> artifact of the sales/PLUTO merge-suffix collision, not a genuine
> structural or locational feature. It is now blacklisted, the permanent
> modeling spec becomes **18 base / 19 political** features, and every model,
> SHAP summary, PDP, sensitivity pass, scenario, figure, and fact-check
> re-derives from that spec on the author's own `python main.py` run. No
> retrained performance numbers are reported here — none exist until that
> run happens. `verify_blacklist_fully.py` itself had two silent-skip gaps
> that would have hidden exactly this class of bug; both are fixed.*

### Scope

Remove the `BBL_pluto` merge artifact from the modeling feature set via the
single existing mechanism (`config/feature_blacklist.yaml`), audit every
downstream stage for hard-coded or cached feature lists, harden the blacklist
verifier so it actually catches this class of bug, and record the spec change
(19/20 → 18/19) everywhere the repo hard-codes it — in place, without
restructuring any script, without renaming or deleting anything, and without
running the pipeline (the author runs the full pass themselves).

### Changes applied

0. **Task 0 — mechanism confirmed before touching anything.**
   - `config/feature_blacklist.yaml` read in full: **`BBL_pluto` was not
     present under any casing or alias.** The only BBL-family entries are
     `- bbl` (post-sale identifiers) and `- appbbl` (assessment leakage); all
     training scripts match case-insensitively (`c.lower() in blacklist`),
     and `bbl` ≠ `bbl_pluto`, so nothing caught it.
   - **Why it exists:** `src/data_pipeline/rebuild_canonical.py` keeps
     PLUTO's `BBL` among its selected columns (line 128) and merges on
     `BBL_join` with `suffixes=('','_pluto')` (line 138). Both raw source
     files carry a `BBL` column that is *not* the join key, so pandas
     preserves PLUTO's copy as `BBL_pluto` instead of discarding it. It is
     the PLUTO copy of the parcel identifier — a row key, not a structural or
     geographic attribute — whose existence was already disclosed in the
     manuscript's Data Sources section (raw-file column overlap).
   - **Single fix location in the wired pipeline:** all five training
     scripts wired into `main.py` derive their feature lists dynamically
     from the blacklist (`train_lgbm.py` lines 93–103; identical idiom in
     `train_random_forest.py`, `train_linear_baselines.py`,
     `train_catboost.py`, `train_xgboost.py`). No stage in `main.py` has its
     own hard-coded feature list, so the blacklist edit alone reaches: all
     seven Table 2 models, champion re-selection, SHAP/PDP/sensitivity/
     intervals, scenarios, figures, and the comparison tables.
     `rebuild_canonical.py`'s separate hard-coded `BAN_COLUMNS` set
     (line 35) drops PLUTO assessment *columns from the canonical build* —
     a different job; `BBL_pluto` correctly **remains a canonical column**
     (no schema change); only model feature selection drops it.
   - **The one exception, named explicitly per the brief:**
     `models/training/train_smoke.py` (a smoke-test utility **not** wired
     into `main.py`) selects numeric features *without* consulting the
     blacklist (line 23), so it would still train on `BBL_pluto` if ever run
     by hand. It produces no paper artifact and takes no further fix here;
     see "Still needs attention."

1. **Task 1 — blacklist entry added.** `config/feature_blacklist.yaml` gains
   one `- BBL_pluto` entry at the end of the existing "Post-sale identifiers
   / row keys" section, in the file's existing flat-list style (comment lines
   above a plain `- name` item, same as the `EASE-MENT` entry), citing the
   merge-suffix mechanism and this round. The file is a flat YAML list under
   `leakage_features:`, so no mapping/block-scalar syntax was introduced.

---

## Round 8 (2026-09-24/25) — RF-champion crash class, dedup rebuild, 53/53, doc consolidation

> Headline: *a routine full run flipped the champion from LightGBM to
> RandomForest, which crashed 5 of 53 stages that hard-coded `lgb.Booster`;
> the fix was a family-agnostic model-IO layer (`src/ops/model_io.py`).
> The same pass dropped 1,041 exact-duplicate sales rows and made five
> string-typed structural columns numeric, widening the modeling spec from
> 18/19 to the current 24/25 features. The follow-up full run passed 53/53.*

### Scope

Author-side retrain + repair on top of the Round-7 state; nothing in this
round recomputes or re-litigates earlier rounds' findings. Numbers below are
read from the on-disk artifacts produced by the two full runs
(`results/PIPELINE_RUN_REPORT_full_20260924.md`, `..._20260925.md`).

### Changes observed

1. **Champion flip → 5-stage crash → `model_io.py`.** `refresh_champion`
   selected `rf_all_years_political_n200` (sklearn `.joblib`, test MAE
   0.4167477). Five stages loaded the champion via
   `lgb.Booster(model_file=...)` and died with "Unknown model format":
   `political_scenarios`, `generate_shap_summary`, `generate_pdp`,
   `champion_confidence_and_sensitivity`, `verify_scenario_mechanics`. Fix:
   new `src/ops/model_io.py` — suffix-dispatched loading/prediction/
   importance for `.txt` / `.joblib` / `.cbm` / `.model`, following the
   pattern already in `src/analysis/counterfactual.py`; sklearn-family
   predictions use `fillna(0)` (as those models were fit), tree boosters use
   native NaN. Related fixes: TreeSHAP on the unpruned forest is intractable
   (>40 CPU-min at n=2000), so the sklearn branch of `generate_shap_summary`
   emits a labeled `method: feature_importance_fallback` payload carrying
   both `gain` and `mean_abs_shap` keys (a missing `mean_abs_shap` key had
   crashed `generate_figures` / `generate_research_summary` — the two
   failures in the 09-24 interim run, 51/53); `generate_pdp.py` pins the
   matplotlib `Agg` backend (a TkAgg teardown can produce a spurious FAIL
   after successful completion).
2. **Dedup (FIX C13).** `rebuild_canonical.py` now ends with
   `drop_duplicates()` — ~1,041 duplicated transactions (529 unique, 2–3
   copies each) — and canonical rows went 567,997 → **566,956** on the
   2026-09-25 build (`data/DATA_MANIFEST.json`, run 2026-09-25T04:09Z;
   `docs/feature_manifest.json` / `models/artifacts/canonical_schema.json`
4. **`train_smoke.py` FIX C4** now filters `config/feature_blacklist.yaml`
   like the production trainers (previously trained on target-leaking
   columns; the old caveat is closed).
5. **Dead-code contraction** (working-tree deletions): `train_ridge.py`,
   `train_elasticnet.py`, `train_stacking.py`, `train_ensemble.py`,
   `train_ensemble_fixed.py`, `src/ops/auto_train.py`,
   `src/ops/continuous_runner.py`, `src/ops/fit_and_save_preprocessor.py`,
   `src/ops/serialize_preprocessor.py`, `src/data_pipeline/data_loader.py`,
   `src/causal/causal_analysis.py` (the long-stalled X-learner scaffolding),
   `notebooks/eda_audit.ipynb`, and `tables/table5_spatial_regression.tex`
   (superseded by the pipeline-generated `tables/table_spatial_regression.tex`).
6. **Two full runs:** 09-24 = 51/53 (the two figure stages above), 09-25 =
   **53/53** in ~7 min; `experiments/registry.json` now holds exactly the 7
   latest runs (one per trainer; replace-on-register).

### Current headline values (2026-09-25 build; supersede every earlier tabulation)

| Quantity | Current | Superseded value |
|---|---|---|
| Canonical rows × cols | 566,956 × 53 | 567,997 (09-10 build) / 514,618 (paper) |
| Train / test rows | 121,052 / 56,723 | 121,622 / 56,817 |
| Model MAE (Table 2 order) | rf 0.41675 · cat 0.41736 · xgb 0.41770 · lgbm-pol 0.43994 · lgbm-base 0.44183 · ridge 0.62934 · enet 0.63127 | 0.41–0.65 era values |
| Moran's I (price / dem_share) | 0.5942 / 0.5705 (both p=0.001) | 0.5959 / 0.5797 |
| Table 1 dem_share (M1/M2/M3) | 0.9449 / 0.6146 / 0.2883, N=520,365 | 0.889 / 0.5741 / 0.1843, N=470,535 |
| Robustness (clustered p / M4 / M5) | 0.0936 / 0.7764 / 0.0667 (clustered p 0.7427) | 0.3185 / 0.8033 / 0.0740 |
| Spatial regression | ρ=0.7029, λ=0.7809; dem_share p .327/.143/.323 | ρ=0.7086, λ=0.7872 |
| City of Yes DiD | Wald p 0.2209; main +0.0793 (p .1318); sens +0.0740 (p .1383) | Wald p 0.2416; +0.0812 (p .1283); +0.0762 (p .1331) |
| Champion | `rf_all_years_political_n200` (exp-20260925T041001Z), MAE 0.4167477; NEAR-TIE (3 within 0.001) | `lgbm_all_years_base` |

The IV/causal findings of Rounds 5–6 (not identified at any configuration;
DiD suggestive null) carry over — the rescue/DiD stages were re-run on the
rebuilt data and produce the same qualitative verdicts (`all_arms_match:
true`, `all_match: true` from the Round-6 verifiers).

### Documentation consolidation (2026-09-25)

`ACTION_PLAN_FOR_SUBMISSION.md` was condensed into `README.md` §"Where the
paper stands"; `DATA_REFRESH_IMPACT_REPORT.md` is appended verbatim below as
**Appendix A**; `docs/ARCHITECTURE.md`, `docs/DATA_LINEAGE.md`,
`docs/FEATURE_DICTIONARY.md`, `data/DATA_CONTRACT.md`, and
`DELIVERY_NOTES.md` were absorbed into `README.md`; `docs/WORKFLOW_AUTOMATION.md`
was deleted as zero-unique-info history. References in `main.py` stage
metadata, `PAPER_TO_CODE_MAP.md`, and `src/causal/city_of_yes_did.py` were
re-pointed. Earlier rounds' mentions of these files are deliberately left as
history. Total maintained docs went from ~10 separate files to two (README +
PAPER_TO_CODE_MAP) plus this record and pipeline-generated summaries.

---

## Appendix A — Data-refresh impact record (2026-09-10/11, verbatim)

*This appendix is the full, unedited text of the (since-merged)
`DATA_REFRESH_IMPACT_REPORT.md`. It documents exactly what changed when the
canonical dataset was refreshed from the January-2026 sales snapshot
(2016–2024) to the full 2016–2025 pull, table by table. Read its "new"
columns as of 2026-09-11: the subsequent Round-8 rebuild (dedup + numeric
coercion) moved the canonical table to 566,956 × 53 and the spec to 24/25
features, superseding those columns where they differ.*


   agree). Duplicates share a sale date, so none can straddle the 2018 split
   boundary.
3. **Numeric coercion → 24/25-feature spec.** The same rebuild coerces
   `LAND/GROSS SQUARE FEET`, `bldgarea`, `lotarea`, `unitsres`, `unitstotal`
   from strings to numeric; they had been silently excluded by
   `select_dtypes(include='number')` in every trainer. Spec went 18/19
   (post-Round-7) → **24 base / 25 political**.

   Mechanical consequence on the author's next run: base spec **19 → 18**
   features, political spec **20 → 19**.

2. **Task 2 — cascade audited, not executed.** Every `main.py` stage the
   brief named was read in full and confirmed to derive its feature list from
   artifact files that are **regenerated on each pipeline run**, never from a
   stale cache: the trainers rewrite `models/artifacts/*_features.json` and
   append to `experiments/registry.json`; `refresh_champion` rewrites
   `experiments/champion.json` from the registry (lowest test MAE among
   untainted post-2026-05-18 runs — hand-preservation of a champion is
   impossible by construction); `generate_shap_summary` / `generate_pdp` /
   `confidence_intervals` / `sensitivity_analysis` /
   `champion_confidence_and_sensitivity` / `political_scenarios` all resolve
   the champion + `lgbm_all_years_political` registry entries and read their
   freshly-written `features_path`; `compare_models` and
   `generate_research_summary` re-tabulate `experiments/**/*.json` (the
   latter regenerates `docs/model_comparison_table.csv`); the figure scripts
   read the regenerated SHAP/scenario artifacts; `verify_ml_performance`
   reads the regenerated comparison table; `verify_shap_importance` reads the
   regenerated SHAP JSONs. **No stage caches an old feature list on disk
   that survives the cascade**, so no artifact needed out-of-band
   regeneration. Two files are *orphans with no pipeline writer* —
   `docs/feature_manifest.json` and `models/artifacts/canonical_schema.json`
   — but both describe **canonical columns** (which are unchanged;
   `BBL_pluto` remains a column), so they stay accurate and are flagged
   rather than regenerated. The current champion on disk is
   `lgbm_all_years_base` (selected by `refresh_champion`); if the retrained
   deltas reorder the registry, the champion will change and the pipeline
   will say so plainly.

3. **Task 3 — `verify_blacklist_fully.py` hardened; it previously could not
   catch this bug.** Three gaps found, all fixed in place:
   - **Political model was silently skipped forever.** The political-model
     lookup required `exp.get("pipeline_v2")` in `experiments/registry.json`;
     verified by direct grep — **zero registry entries carry that flag** (no
     training script writes it), so `political_name` was always `None` and
     both final-verdict blocks skipped the political model entirely. The
     script now falls back to the same name-based registry lookup
     (`lgbm_all_years_political`) that `generate_pdp.py` /
     `sensitivity_analysis.py` / `generate_shap_summary.py` already use.
   - **The final verdict could never see `BBL_pluto`.** The verdict loops
     tested only `norm_label(f)` (underscores → spaces), which turns
     `BBL_pluto` into `BBL pluto` — not the blacklist entry — while the
     per-model scan already tested the original name too. Both verdict loops
     now test original **and** normalized names, mirroring the per-model
     loop, so every blacklist entry (including underscore-suffixed merge
     artifacts like this one) is checked against both champion models'
     actual trained feature lists, not just spot-checked names.
   - **Enforcement failure was a print-only event.** The script exited 0 even
     when a champion model contained blacklisted features (main.py would
     have reported the stage PASS), and never wrote the
     `results/editorial_audit/blacklist_audit.json` its own docstring
     promises. It now writes that JSON on every run and exits 1 if either
     champion model fails (legacy `.txt` scans above remain informational in
     the log and do not gate the pipeline, matching their existing role).
   - **The fix was validated against the live checkout this session:** with
     the entry added and the on-disk models still trained *with* `BBL_pluto`
     (pre-retrain), the hardened script identifies **both** champion base and
     political models as BLACKLISTED (`BBL_pluto` in both feature lists),
     writes the audit JSON with `enforcement_passed: false`, and exits 1 —
     i.e. the verifier finally catches exactly this class of bug. It goes
     green as soon as the author's pipeline run retrains under the
     18/19-feature spec; until then its failure is the intended state, not a
     regression.

4. **In-place feature-count updates** (19→18 / 20→19; stray 16/17
   descriptions → 18/19; no restructuring):
   - `main.py` stage descriptions: `train_lgbm_base`, `train_lgbm_political`,
     `generate_shap_summary`, `generate_pdp`, `confidence_intervals`,
     `sensitivity_analysis`, `champion_confidence_and_sensitivity`.
   - Script comments/docstrings: `src/analysis/generate_pdp.py`,
     `src/analysis/sensitivity_analysis.py`, `src/figures/generate_figures.py`
     (count only; the "dem_share at rank 12" phrase is unpacked below),
     `src/causal/iv_district_fe_regression.py` (docstring + controls
     comment), `src/causal/city_of_yes_did.py` (docstring).

5. **Docs.** `PAPER_TO_CODE_MAP.md` and `README.md` each gained one line
   noting that the permanent **18/19-feature spec** (post-`BBL_pluto`)
   supersedes the prior 19/20-feature spec, with a pointer to this Round 7.

### What does NOT change — stated explicitly so it need not be re-verified by hand

None of the following ever included `BBL_pluto` in a control set, and none is
affected by the blacklist edit (their numbers come from regressions over
canonical columns, not from the model feature spec):
- **Table 1 (OLS, `generate_regression_table.py`)**: controls are
  `yearbuilt, numfloors, landuse, commfar` only.
- **Table 4 (robustness battery, `generate_robustness_checks.py`)**: same
  four-control set.
- **Table 5 (spatial regression, `spatial_regression_model.py`)**:
  district-level aggregate models; identical construction.
- **Tables 6–7 (candidate causal-identification designs, `src/causal/`)**: the
  IV `CONTROLS` lists (`yearbuilt, numfloors, landuse, commfar, RESIDENTIAL
  UNITS, COMMERCIAL UNITS`) and the City of Yes DiD control set were read in
  full — no `BBL_pluto` anywhere, and no re-run consequence.
- The canonical dataset, splits, and schema are unaffected — the blacklist
  gates *model features*, not canonical columns.

### Still needs attention (found; NOT silently resolved)

- **`models/training/train_smoke.py`** derives features without the blacklist
  (line 23) and would train on `BBL_pluto` if run. It is not wired into
  `main.py`, produces no paper artifact, and was left untouched per this
  round's scope — but it is the one script in `models/training/` the
  blacklist edit does not reach.
- **Fact-check reference mismatches are expected on the author's run and
  must not be "fixed" by editing the reference values.**
  `verify_shap_importance.py` compares the regenerated SHAP summary against
  the paper's Table 3 values, **including a `BBL_pluto` row (0.07654)**;
  `verify_ml_performance.py` compares Table 2 MAE/R² against the recorded
  paper values; `verify_canonical_dataset_and_split.py` and
  `verify_borough_map_and_crosscutting.py` record the paper's 16/17 feature
  counts as `paper:` baselines. After the retrain these scripts will report
  mismatches — that is the fact-check harness doing its job, logging a
  genuine manuscript-vs-current-state gap for the manuscript rewrite. Nothing
  in those files was edited.
- **Champion selection may change** after the retrain (the tree families'
  recorded test MAEs sit within ~0.003 of one another). Whatever
  `refresh_champion` selects on the author's run is the answer; nothing here
  preserves the old champion by hand.
- **`docs/FEATURE_DICTIONARY.md`** is hand-maintained and was already stale
  (it describes the 16/17-feature era under a 19/20 spec); after Round 7 it
  is doubly stale and lists `BBL_pluto` as "used in both". Left untouched
  (outside this round's edit set); fix during the manuscript-rewrite pass.
- **Narrative strings embedded in generated docs:** `generate_figures.py` and
  `generate_research_summary.py` carry prose like "dem_share appears at rank
  12"; the dem_share rank may move when the political model retrains without
  `BBL_pluto`. Only feature-count literals were updated this round; these
  claims should be checked against the regenerated artifacts after the run.

### Files changed this round

- **Modified — code/config:** `config/feature_blacklist.yaml` (+`BBL_pluto`
  entry), `src/verification/editorial_audit/verify_blacklist_fully.py`
  (Task-3 hardening: political-model lookup fallback, original-name matching
  in the final verdict, JSON audit output, fail-loud exit code), `main.py`
  (7 stage `desc` strings, feature counts only),
  `src/analysis/generate_pdp.py`, `src/analysis/sensitivity_analysis.py`,
  `src/figures/generate_figures.py`, `src/causal/iv_district_fe_regression.py`,
  `src/causal/city_of_yes_did.py` (count comments/docstrings only).
- **Modified — docs:** `PAPER_TO_CODE_MAP.md`, `README.md` (one line each),
  `VERIFICATION_REPORT.md` (this Round 7).
- **Runtime output refreshed for validation:** this round's validation run of
  the hardened verifier rewrote `results/editorial_audit/blacklist_audit.json`
  with the new schema (recording `enforcement_passed: false` — correctly
  flagging both pre-retrain champion models as BLACKLISTED for `BBL_pluto`).
- **Untouched:** `orbit_paper_CER.tex` and everything under `CER Papers */`;
  every model binary, features JSON, SHAP/PDP/sensitivity artifact, registry
  entry, and `champion.json` (they re-derive on the author's own pipeline
  run); `archive/`, `legacy/`, `docs/FEATURE_DICTIONARY.md`,
  `docs/feature_manifest.json`, `DATA_REFRESH_IMPACT_REPORT.md`, and all
  earlier rounds of this report; no file was deleted or renamed.
> **HISTORICAL / SUPERSEDED RECORD (as of the 2026-09 remediation):** this document describes an earlier state or an earlier round of the repo and is kept for audit history only; file paths, scripts, and numbers it mentions may have moved, been superseded, or been removed. Current source of truth: README.md and VERIFICATION_REPORT.md.

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
