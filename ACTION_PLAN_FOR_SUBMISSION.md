# Action Plan: Strengthening the Paper and Repo for This Year's Submission

> **⚠️ SUPERSEDED (2026-09-08) — read VERIFICATION_REPORT.md first.** This
> document was written against the pre-reorganization repo. Its parts about
> *repo state* (what's broken, what to fix, file paths, priorities) have since
> been acted on and verified; the current, run-proven record of what was found
> and fixed is `VERIFICATION_REPORT.md`, **Round 2–4 sections**, which are the
> source of truth going forward. Part 1's competitive analysis of the accepted
> CER pool and the causal-identification recommendation remain valid as
> research strategy — that is the one part still current. Everything else here
> is kept for historical context only; do not treat it as a to-do list.

This document has three parts: (1) where your paper stands against last
year's accepted pool and why, (2) a prioritized, file-specific list of
changes to make, ranked by how much they'd move you toward "top of the
pool," and (3) exactly what changed in the repo during this reorganization
and how to use it. Part 2 is the part you asked for by name — it's written
so each item names the exact file, the exact command, and what comes out.

---

## Part 1 — Where you stand

I read all 9 unique accepted papers from `CER_Journal_2025-...zip`
(Abbas, Kang, Law, Lin, Polesky, Torres, Yuan, Yun & Crowley, Zhang — all
Columbia Economic Review) against `orbit_paper_CER.tex`/`.pdf`.

| | Your paper | Accepted pool (n=9) |
|---|---|---|
| Length | 6,210 words, 34 pages | 9,290–22,944 words (avg ~13,000) |
| Tables | 12 | typically 4–9 dedicated to results, but embedded in much longer text |
| Figures | 6 | comparable |
| Explicit "Robustness" language | 0 mentions | present in most; several have a full subsection |
| Causal identification strategy | **None** — paper explicitly states this as its top limitation | DiD (Torres; Yun & Crowley), panel fixed effects (Zhang; Law), IV/gravity-style instruments (Kang); even the more descriptive ones frame results causally with heavier caveats than yours |
| Independent numerical fact-check of every claim | Yes — `results/FULL_PAPER_FACT_CHECK.md`, ~15 scripts | Not observed in any accepted paper I read; this is a genuine, unusual strength of yours |

**The honest read:** your paper is unusually well-audited — nobody else in
that pool appears to have independently re-derived every number they report
— but it is also shorter and more purely associational than every accepted
paper I compared it to. "No causal identification strategy" is *already*
your own stated #1 limitation; the accepted pool confirms that's exactly the
axis reviewers are grading hardest on. Fixing that (or getting materially
closer to it) is worth more than any amount of additional polish on what's
already there. Everything in Part 2 is ordered with that in mind.

I also found, independent of the paper-comparison, a **real methodological
issue** in the existing regression (Priority 0 below) — worth fixing before
anything else because it's fast, it's correct, and a reviewer who knows to
check for it will find it if you don't.

---

## Part 2 — Prioritized changes

### Priority 0 (do this first — already run, results are in the repo)

**Issue:** `src/analysis/generate_regression_table.py` fits all three OLS
models in Table 1 with classical standard errors. Section 4.2 independently
documents strong positive spatial autocorrelation in both price and
`dem_share` (Moran's I = 0.5959 / 0.5797, both p = 0.001). Positive spatial
autocorrelation in the regressors/errors of a model that doesn't account for
it is a textbook case where classical SEs understate the true uncertainty —
i.e., Table 1's p-values are likely too small.

**What I did:** wrote `src/analysis/generate_robustness_checks.py` and ran it
against your real canonical dataset (470,535 rows, matching the paper's own
missingness-adjusted N). It does **not** modify Table 1 — it's a new,
separate output. Results are already in `results/robustness_checks.json` and
`tables/table_robustness.tex`.

**What it found**, on the real data:

| Specification | `dem_share` coef. | SE | p-value |
|---|---|---|---|
| Model 3 (paper), classical SE | 0.1843 | 0.0093 | 1.0e-87 |
| Model 3, SE clustered by council district (51 clusters) | 0.1843 | 0.1847 | **0.318** |
| Model 4: 51 council-district FE (replaces borough FE), classical SE | 0.5682 | 0.0207 | 1.2e-165 |
| Model 4, clustered SE | 0.5682 | 0.1275 | 8.4e-06 |
| Model 5: Model 3 + sale-year FE (not district FE), classical SE | 0.0740 | 0.0098 | 4.3e-14 |
| Model 5, clustered SE | 0.0740 | 0.2009 | 0.712 |

**This is a real tension, not a bug — you need to decide how to handle it.**
Once you correctly account for the fact that Moran's I says your
observations aren't independent within a district, the *simple* clustered
version of Table 1's own Model 3 loses significance entirely. But a model
with full council-district fixed effects (absorbing everything
time-invariant about each district and identifying `dem_share` off
within-district movement over time) finds a *larger* coefficient that *does*
survive district clustering. A model that instead controls for sale-year
FE (rather than district FE) shrinks the effect further and loses
significance once clustered. Three genuinely different pictures depending on
what you control for — that's exactly the kind of specification-sensitivity
a rigorous referee will probe, and it's much better for you to have already
run and be ready to discuss it than to have it surface for the first time in
review.

**What to actually do about it (your call, but concretely):**
1. Add `cov_type='cluster', cov_kwds={'groups': df['CounDist']}` to at least
   Model 3 in `src/analysis/generate_regression_table.py` and report the
   clustered SE in Table 1 itself, not just as a robustness footnote — this
   is the more defensible default given you've documented the spatial
   autocorrelation yourself.
2. Decide whether Model 4 (district FE) or something like it becomes your
   new preferred/reported specification instead of Model 3 — it's the one
   result here that's both large and survives the correction. If you go this
   route, you'll need to explain in prose *why* district FE is the right
   choice (it identifies the effect from within-district change in ideology
   over time, which is arguably a cleaner story than the paper's current
   cross-sectional one) and you should say so explicitly rather than let a
   reader wonder why you picked it.
3. Add a "Robustness" subsection to Section 4.3 (or a new Section 4.6) that
   reports `tables/table_robustness.tex` in full and walks through this
   three-way tension honestly. Reviewers respond much better to "here is a
   sensitivity we found and here's our read on it" than to a table that
   doesn't mention it at all.
4. Wire the new stage into your run: it's already registered in `main.py`
   as `generate_robustness_checks` (category `analysis`).

*I did not pick a side for you above and I'm not going to — which
specification you lead with is a judgment call about your research design
that should be yours and your mentor's, not mine. I ran the numbers so you
have them to make that call with.*

### Priority 1 — Causal identification (the single biggest lever, and the hard one)

This is what separates the accepted pool from your current draft most
clearly, and it's also the one item I can't hand you a working script for,
because it requires a real, specific, defensible treatment definition your
data doesn't currently encode — inventing one would mean me putting
numbers in your paper that aren't grounded in anything real, which I won't
do. Concretely, here's what's needed and where it plugs in:

- **`src/causal/causal_analysis.py`** (moved into this reorganization,
  unchanged in content) currently references `data/intermediate/processed_data.parquet`
  (doesn't exist), a `post_2023_reform` column (doesn't exist), and an
  `ideology_score` column (doesn't exist — the paper correctly uses
  `dem_share`). It is genuinely unfinished scaffolding, exactly as your
  paper's own Future Work section says.
- To make it real, you need a **specific, dated policy event** that changed
  zoning/land-use rules in some council districts and not others (or at
  different times), e.g. a specific rezoning action, "City of Yes for
  Housing Opportunity" (adopted December 2024) if its provisions varied by
  district, or a specific prior rezoning with a known effective date and
  known treated districts. This requires you to (a) identify the real event
  and its real effective date(s) and treated/untreated districts from a
  primary source (NYC Department of City Planning, NYC Council records), and
  (b) encode that as a new column in the canonical build
  (`src/data_pipeline/rebuild_canonical.py`), not fabricate it.
- Once that column exists, the DiD/event-study machinery itself is
  standard and I can help you build it (e.g., `statsmodels`
  `PanelOLS`-style two-way FE, or an event-study version with leads/lags
  around the effective date) — but the treatment definition has to come
  from you, or from primary-source research, first. This is the honest
  path to closing the single biggest gap versus Torres, Yun & Crowley, and
  Zhang.
- **If a clean natural experiment isn't available before your deadline**,
  the next-best, fully achievable-with-existing-data step is a formal
  spatial regression (spatial-lag or spatial-error model), which your paper
  already flags as the natural next step and which reuses infrastructure you
  already have (`libpysal`/`esda`, already installed and used by
  `src/analysis/compute_morans_i.py`). This doesn't get you to causal
  identification, but it directly answers the referee-facing question "your
  own Moran's I shows your errors are spatially correlated — why are you
  still reporting plain OLS?", and it's a much more tractable few-days task
  than sourcing a real natural experiment. If you want this, say so and I'll
  build `src/analysis/spatial_lag_model.py` against your real data the same
  way I built the robustness-checks script above.

### Priority 2 — Surface work you've already paid for but aren't using

These are genuinely already computed and sitting on disk, unused in the
manuscript. This is the cheapest possible improvement in the whole list.

| Add to paper | Existing artifact | Script that produced it |
|---|---|---|
| A PDP figure/panel in Section 4.6 showing how price responds to `numfloors`, `yearbuilt`, `commfar`, `landuse`, `BBL_pluto` | `models/artifacts/pdp_lgbm_all_years_base_*.png` | `src/analysis/generate_pdp.py` |
| Prediction-interval coverage in a "model uncertainty" paragraph | `experiments/predictions/*_intervals*.csv` | `src/analysis/confidence_intervals.py` |
| A feature-perturbation sensitivity table (how much does a 1-SD change in each feature move the prediction) | `models/artifacts/*_sensitivity_analysis.json` | `src/analysis/sensitivity_analysis.py` |

Concretely: add a subsection "4.6.3 Model Interpretation and Uncertainty"
right after the existing SHAP subsection (4.6.2), pull 2–3 PDP panels as a new
Figure, and report interval coverage in one paragraph. This is a half-day
of writing, not new computation — everything above already ran.

### Priority 3 — Length and framing

Not a code change, but concrete: the accepted papers average roughly double
your word count, and the difference isn't padding — it's more literature
review (situating the result against 3–5 more specific prior studies, not
just general theory), more discussion of mechanism (why would `dem_share`
predict price at all — sorting? amenity provision? something else? — most
accepted papers spend a full paragraph or more on this per finding), and
more explicit engagement with alternative explanations before the
Limitations section rather than only within it. Priorities 0–2 above will
naturally add 1,500–2,500 words if you write them up properly; that alone
gets you much closer to the pool's length without padding.

### Priority 4 — One data-integrity loose end to close, low effort

`src/verification/verify_followup_resolution.py` (re-run during this
reorganization, confirmed still reproducible) found that
`experiments/predictions/lgbm_all_years_political_test_preds.parquet` and
its `_intervals.csv` have 56,798 rows, 19 short of the full 56,817-row test
set — confirmed **not** to affect the Table 2 metrics you report (those come
from a separate in-training `model.predict()` call on the full test set),
but it does mean the two Priority-2 artifacts above (prediction intervals in
particular) are currently missing 19 rows relative to the official test set.
Fix: regenerate those two files from the full 56,817-row split before
writing the Priority-2 paragraph, per the `recommended_action` field already
written into `results/robustness_checks.json`'s sibling file
`results/round5_followup_resolution.json`.

---

## Part 3 — What changed in the repo, and how to run it now

### The one command you asked for

```bash
python main.py                 # runs the entire pipeline, in dependency order,
                                # end to end — data build, all model training,
                                # all analysis, all figures, the full independent
                                # fact-check suite — and writes a consolidated
                                # results/PIPELINE_RUN_REPORT.md at the end.

python main.py --list          # see every stage, in order, with the exact
                                # paper section/table it feeds
python main.py --check-data    # verify data/ is populated correctly before
                                # a long run
python main.py --fast          # skip the 5 slower comparison-model trainings
                                # (random forest/ridge/elasticnet/catboost/xgboost),
                                # keep the 2 LightGBM models the headline
                                # numbers depend on
python main.py --category verification   # just re-run the fact-check suite
python main.py --stage compute_morans_i  # run exactly one stage
```

A new clone needs: `pip install -r requirements.txt`, then populate `data/`
per `docs/DATA_LINEAGE.md` (raw sources are gitignored, as before), then
`python main.py`.

**A real runtime note, found while testing this**: in a CPU-constrained
environment, a single `lightgbm` `.predict()` call over the full 514,618-row
canonical table took ~93 seconds in my sandbox.
`src/verification/verify_scenario_mechanics.py` alone calls `.predict()`
seven times, so budget several minutes for that one stage specifically (and
correspondingly longer for a full unrestricted `python main.py` run) on a
typical laptop — this is expected, not a bug.

### Reorganization summary

`src/` was a flat folder of ~70 scripts, including 15 non-descriptively
named (`round5_factcheck_AB.py`, `_C.py`, `_DE.py`, ... `_LM.py`) fact-check
scripts, several genuinely dead/superseded scripts, and no single entry
point. It's now:

```
src/
├── data_pipeline/     # raw sources -> canonical dataset
├── analysis/          # Moran's I, regression, SHAP, scenarios, robustness
├── figures/           # everything that writes to docs/figures/
├── verification/      # the independent fact-check suite (renamed by paper section)
│   └── editorial_audit/
├── causal/            # causal_analysis.py (unfinished; see Priority 1)
└── ops/                # training automation, CI checks, preprocessing utilities
app/
├── app.py
└── auth.py             # moved out of src/, it's only ever used by the app
archive/                # superseded/dead scripts, kept for history, not in main.py
```

Full old-path &rarr; new-path correspondence, and which paper table/figure
each file feeds, is in **`PAPER_TO_CODE_MAP.md`** — read that first if
you're trying to find "what produced Table 2."

**Two real bugs were found and fixed while verifying the reorganization
against your real data** (not hypothetical — both were caught by actually
running the pipeline, not just reading code):
1. `experiments/registry.json` and `experiments/champion.json` (plus 3 other
   JSON artifacts) stored Windows-style backslash paths
   (`models\artifacts\...`), which silently breaks on Linux/Mac. Normalized
   to forward slashes repo-wide.
2. `verify_scenario_mechanics.py` (renamed from `round5_factcheck_K_mech.py`)
   had a hardcoded path to `political_scenarios.py`'s old flat location.
   Fixed to the new `src/analysis/` location.

Both are confirmed fixed by actually re-running the affected scripts against
your real data in this session — not just inspected.

Everything under `src/verification/` that I ran (11 of the 15 scripts,
covering ideology construction, the canonical dataset, the train/test split,
Moran's I, OLS, ML performance, SHAP, scenario mechanics, the Appendix A.1
table, the borough map, the blacklist audit, and the two follow-up
resolutions) reproduced the paper's published numbers exactly, including
Moran's I to 4 decimal places. I did not run the 5 heavier comparison-model
training stages (random forest/ridge/elasticnet/catboost/xgboost) — trust
`main.py`'s stage list and `models/training/*.py`'s existing CLI args for
those, which I sourced directly from `experiments/results/*.json`'s
already-recorded hyperparameters, not guessed.
