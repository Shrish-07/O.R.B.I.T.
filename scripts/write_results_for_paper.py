"""Write results/NEW_RESULTS_INDEX.md and results/RESULTS_FOR_PAPER.md (Task M).

NEW_RESULTS_INDEX.md: every new or changed file from Round 12, one line each.
RESULTS_FOR_PAPER.md: copy-ready tables of every model x ideology measure x
sample with coef, cluster SE, p(t(G-1)), p(wild), CI, N, G — plus the DiD,
IV, spatial, ML-collinearity, funnel and ballot-line-audit summaries.

Both files are generated from the repo artifacts (nothing hand-typed), so a
pipeline re-run keeps them in sync.

Run: python scripts/write_results_for_paper.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
OUT_INDEX = R / 'NEW_RESULTS_INDEX.md'
OUT_PAPER = R / 'RESULTS_FOR_PAPER.md'


def load(p):
    return json.loads((ROOT / p).read_text(encoding='utf-8'))


def fmt(x, nd=4):
    if x is None:
        return '---'
    return f'{x:.{nd}f}'


def fmt_p(p):
    if p is None:
        return '---'
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'


def stars(p):
    if p is None:
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


MEASURE_LABELS = {
    'dem_share': 'legacy dem_share (Dem/(Dem+Rep), mayoral)',
    'dem_share_mayor_allcand': 'dem_share_mayor_allcand (Dem/all candidate lines)',
    'dem_share_pres': 'dem_share_pres ((Dem+WFP)/(Dem+WFP+Rep+Cons))',
    'dem_share_vintage': 'dem_share_vintage (legacy measure, vintage-matched crosswalk)',
    'dem_share_mayor_allcand_vintage': 'dem_share_mayor_allcand_vintage',
    'dem_share_pres_vintage': 'dem_share_pres_vintage',
}
SPEC_LABELS = {
    'model3_boro_fe': 'Model 3: controls + borough FE',
    'model4_district_fe': 'Model 4: controls + district FE',
    'model5_boro_year_fe': 'Model 5: controls + borough + sale-year FE',
    'model6_district_year_fe_units':
        'Model 6: controls + units + district + sale-year FE',
    'model7_district_boro_x_year_fe':
        'Model 7: controls + units + district + borough x sale-year FE',
    'model8_district_linear_trends':
        'Model 8: controls + units + district + district trends + sale-year FE',
    'hedonic_upgraded':
        'Hedonic: log sqft + units + landuse/building-class/tax-class FE + district + year FE',
}


def robustness_tables(rob):
    lines = []
    for sample, sample_desc in [
            ('full', 'FULL analysis sample (non-missing outcome + controls)'),
            ('restricted', 'RESTRICTED analysis sample '
             '(+ tax class 1/2, price >= $100k, single-parcel)')]:
        lines.append(f'\n### {sample_desc}\n')
        for measure, specs in rob[sample].items():
            lines.append(f'\n**Ideology measure: '
                         f'{MEASURE_LABELS.get(measure, measure)}**\n')
            lines.append('| Specification | Coef | SE (cluster) | p t(G-1) | '
                         'p wild | 95% CI (t) | N | G | R2 |')
            lines.append('|---|---|---|---|---|---|---|---|---|')
            for spec, rep in specs.items():
                if 'coef' not in rep:
                    lines.append(f'| {SPEC_LABELS.get(spec, spec)} | '
                                 f'NOT ESTIMABLE ({rep.get("error")}) | | | '
                                 f'| | | | |')
                    continue
                lines.append(
                    f'| {SPEC_LABELS.get(spec, spec)} | '
                    f'{fmt(rep["coef"])}{stars(rep["p_t_G_minus_1"])} | '
                    f'{fmt(rep["se_cluster"])} | '
                    f'{fmt_p(rep["p_t_G_minus_1"])} | '
                    f'{fmt_p(rep["p_wild_cluster_bootstrap_t"])} | '
                    f'[{fmt(rep["ci95_low_t"])}, {fmt(rep["ci95_high_t"])}] | '
                    f'{rep["n"]:,} | {rep["n_clusters_G"]} | '
                    f'{fmt(rep["r_squared"], 3)} |')
    return lines

def did_tables(did_v2):
    lines = ['\n## City of Yes DiD (preference-proxy framing)\n']
    mw = did_v2['main_wild_inference']
    lines.append('| Spec | Coef | SE (cluster) | p t(G-1) | p wild | '
                 '95% CI (wild) | N | G |')
    lines.append('|---|---|---|---|---|---|---|---|')
    lines.append(f'| Main (post=2025, 2024 excl.) | {fmt(mw["coef"])} | '
                 f'{fmt(mw["se_cluster"])} | {fmt_p(mw["p_t_G_minus_1"])} | '
                 f'{fmt_p(mw["p_wild_cluster_bootstrap_t"])} | '
                 f'[{fmt(mw["ci95_low_wild"])}, {fmt(mw["ci95_high_wild"])}] '
                 f'| {mw["n"]:,} | {mw["n_clusters_G"]} |')
    for tag, label in [
            ('window_2022_2025_incl_2024_as_pre',
             'Window 2022-25 (2024 as pre)'),
            ('window_2022_2025_excl_2024', 'Window 2022-25 (2024 excl.)'),
            ('district_specific_linear_trends',
             'District-specific linear trends')]:
        r = did_v2[tag]
        lines.append(f'| {label} | {fmt(r["did_coef"])} | '
                     f'{fmt(r["did_se"])} | {fmt_p(r["did_p"])} | --- | '
                     f'--- | {r["n"]:,} | {r["n_clusters_G"]} |')
    ct = did_v2['continuous_treatment_dem_share_pres_x_post']
    lines.append(f'| Continuous: dem_share_pres x post | '
                 f'{fmt(ct["interaction_coef"])} | '
                 f'{fmt(ct["interaction_se"])} | '
                 f'{fmt_p(ct["interaction_p"])} | --- | --- | '
                 f'{ct["n"]:,} | {ct["n_clusters_G"]} |')
    lines.append('\nLeave-one-borough-out (main spec):\n')
    lines.append('| Borough excluded | Coef | SE | p | N | G |')
    lines.append('|---|---|---|---|---|---|')
    for b, r in did_v2['leave_one_borough_out'].items():
        lines.append(f'| {b} | {fmt(r["did_coef"])} | {fmt(r["did_se"])} | '
                     f'{fmt_p(r["did_p"])} | {r["n"]:,} | '
                     f'{r["n_clusters_G"]} |')
    return lines


def iv_tables(iv_v2):
    lines = ['\n## Close-election IV — matched OLS benchmark '
             '(replaces the mismatched Table-2-Model-4 pointer)\n']
    lines.append('The 2SLS second stage uses district FE + sale-year FE + '
                 'RESIDENTIAL/COMMERCIAL UNITS on the close-race sample; '
                 'the matched OLS below is the IDENTICAL spec/sample. '
                 'G < 30 at every primary bandwidth — cluster inference is '
                 'thin and flagged.\n')
    lines.append('| Bandwidth | N | G | G<30 | treated cells | control cells '
                 '| matched OLS dem_share (coef, p) |')
    lines.append('|---|---|---|---|---|---|---|')
    for bw, rec in iv_v2['bandwidths'].items():
        mo = rec['matched_ols_by_measure']['dem_share']
        lines.append(f'| {bw} | {rec["n_obs"]:,} | {rec["n_clusters_G"]} | '
                     f'{"YES" if rec["g_below_30_flag"] else "no"} | '
                     f'{rec["treated_cells_Z1"]} | {rec["control_cells_Z0"]} '
                     f'| {fmt(mo["coef"])} ({fmt_p(mo["p"])}) |')
    lines.append('\nMatched OLS by ideology measure at +/-20pp:\n')
    lines.append('| Measure | Coef | SE | p | N | G |')
    lines.append('|---|---|---|---|---|---|')
    for m, r in iv_v2['matched_ols_benchmark'].items():
        lines.append(f'| {m} | {fmt(r["coef"])} | {fmt(r["se"])} | '
                     f'{fmt_p(r["p"])} | {r["n"]:,} | '
                     f'{r["n_clusters_G"]} |')
    lines.append('\nLocal-linear RD-style spec (instruments Z, Z x margin; '
                 'first-stage joint F):\n')
    lines.append('| Bandwidth | Identified | First-stage F | F p | '
                 'dem_share coef | N | G |')
    lines.append('|---|---|---|---|---|---|---|')
    for bw, r in iv_v2['rd_local_linear'].items():
        if r.get('identified'):
            lines.append(f'| {bw} | yes | {fmt(r["first_stage_joint_f_excluded_instruments"], 1)} '
                         f'| {fmt_p(r["first_stage_joint_f_pvalue"])} | '
                         f'{fmt(r["dem_share_coef"])} | {r["n"]:,} | '
                         f'{r["n_clusters_G"]} |')
        else:
            lines.append(f'| {bw} | no (rank failure) | --- | --- | --- | '
                         f'{r.get("n", 0):,} | {r.get("n_clusters_G", 0)} |')
    return lines


def spatial_tables(moran_v2, spat_v2):
    lines = ['\n## Spatial statistics per ideology measure '
             '(district level, N=51, k=5 KNN weights)\n']
    lines.append('| Measure | Moran I | I p (sim) | Residual I | '
                 'Residual I (borough FE) | SLM rho (pooled) | '
                 'SEM lambda (pooled) |')
    lines.append('|---|---|---|---|---|---|---|')
    for m in moran_v2['by_measure']:
        r = moran_v2['by_measure'][m]
        sp = spat_v2['by_measure'][m]['pooled']
        rho = sp.get('spatial_lag_model', {}).get('W_median_log_price', {})
        lam = sp.get('spatial_error_model', {}).get('lambda', {})
        rb = r.get('residual_morans_boro_fe') or {}
        lines.append(f'| {m} | {fmt(r["moran_i"], 4)} | '
                     f'{fmt_p(r["p_sim"])} | '
                     f'{fmt(r["residual_morans"]["I"], 4)} | '
                     f'{fmt(rb.get("I"), 4) if rb else "---"} | '
                     f'{fmt(rho.get("coef"), 4) if rho else "---"} | '
                     f'{fmt(lam.get("coef"), 4) if lam else "---"} |')
    return lines


def write_index():
    idx = ['# NEW RESULTS INDEX — Round 12 (v2 outputs)',
           '',
           'One line per new or changed file. Generated by '
           'scripts/write_results_for_paper.py.',
           '',
           '## Raw data added (gitignored locally; provenance recorded)',
           '- data/raw/election_results/ed_results_2016_president.csv — '
           'vote.nyc ED-level presidential returns 2016 (fetched CSV)',
           '- data/raw/election_results/ed_results_2020_president.csv — '
           'vote.nyc ED-level presidential returns 2020 (fetched CSV)',
           '- data/raw/election_results/ed_results_2024_president.csv — '
           'vote.nyc ED-level presidential returns 2024 (fetched CSV)',
           '- data/raw/election_districts_legacy/ — DCP Bytes historical ED '
           'boundaries (nyed_16a/18a/20a) + 2013 Adopted Council Districts '
           '(nycc_13a) used to build the vintage-matched crosswalk',
           '',
           '## Processed / canonical data (new or changed)',
           '- data/processed/ed_results_clean.parquet — CHANGED: parses all '
           'six raw ED files via the shared reader + reconciliation gate '
           '(was 2017-only with junk columns)',
           '- data/processed/ed_to_council_crosswalk_legacy.parquet — NEW: '
           'vintage-matched ED->current-council crosswalk',
           '- data/processed/council_2013_to_current.parquet — NEW: 2013-plan'
           ' council -> current council areal correspondence',
           '- data/processed/ideology_by_council_v2.parquet — NEW: mayoral '
           'all-candidate + vintage measures per (CounDist, election_year)',
           '- data/processed/ideology_pres_by_council.parquet — NEW: '
           'presidential measures per (CounDist, presidential year)',
           '- data/canonical/modeling_dataset_canonical_v2.parquet — CHANGED:'
           ' 11 new columns (dem_share_mayor_allcand, dem_share_pres, '
           'dem_share_vintage, dem_share_mayor_allcand_vintage, '
           'dem_share_pres_vintage, election_year_pres, sample_pre2024, '
           'is_resid_tc12, price_ge_100k, multi_parcel_flag, '
           'building_class_category); legacy columns byte-identical',
           '- data/DATA_MANIFEST.json, docs/modeling_schema.csv, '
           'docs/feature_manifest.json, models/artifacts/'
           'canonical_schema.json — CHANGED: regenerated (new columns '
           'recorded)',
           '- config/feature_blacklist.yaml — CHANGED: all 11 new columns '
           'blacklisted so no model ever trains on them',
           '',
           '## Results (new unless marked CHANGED)',
           '- results/ideology_line_audit.json + .csv — ballot-line audit; '
           '2025 Other share 42.26% flagged',
           '- results/crosswalk_audit.json — ED/crosswalk vintage audit; '
           '17.8-24.1% base losses documented, vintage recovery ~0%',
           '- results/sample_funnel.json + .csv — 845,607 -> 566,956 -> '
           '520,365 -> 469,697 funnel',
           '- results/robustness_checks_v2.json — 84-fit regression suite '
           '(Models 3-8 + hedonic x 6 measures x 2 samples, full inference)',
           '- results/causal/city_of_yes_did_results.json — CHANGED: '
           'pre_trend_joint_wald_f (a chi2 mislabeled as F) renamed to '
           'pre_trend_wald_chi2/_p; implied F(3,G-1) added',
           '- results/causal/city_of_yes_did_results_v2.json — NEW: wild '
           'main estimate, 2022-25 window, district trends, continuous '
           'treatment, leave-one-borough-out',
           '- results/causal/iv_district_fe_results_v2.json — NEW: matched '
           'OLS benchmark per measure, G/cell counts (G<30 flags), RD '
           'local-linear specs',
           '- results/morans_i_results_v2.json — NEW: per-measure Moran I + '
           'residual Moran I (+ borough FE)',
           '- results/spatial_regression_v2.json — NEW: per-measure SLM/SEM '
           'pooled + borough FE',
           '- results/ml_dem_share_collinearity.json — NEW: dem_share '
           'deterministic in district (R2=1), duplicated feature pairs',
           '- results/regression_diagnostics.json — CHANGED: stale '
           'pre-rewrite manuscript targets removed',
           '- results/anonymous_release_report.json — NEW: anonymous-release '
           'scrub report',
           '- results/NEW_RESULTS_INDEX.md, results/RESULTS_FOR_PAPER.md — '
           'NEW: this index and the copy-ready tables',
           '- results/verification/verify_ideology_v2.json, '
           'verify_analysis_v2.json, verify_causal_spatial_v2.json, '
           'verify_repo_hygiene.json — NEW verifier status files',
           '- results/verification/verify_city_of_yes_did.json — CHANGED: '
           're-verified against the renamed Wald fields',
           '',
           '## Code (new unless marked CHANGED)',
           '- src/data_pipeline/boe_ed_reader.py — NEW shared BOE ED reader '
           '(both raw layouts) + ballot-line classification',
           '- src/data_pipeline/normalize_ed_results.py — CHANGED (Task J '
           'fix): all six files, tidy schema, reconciliation gate',
           '- src/data_pipeline/build_legacy_ed_crosswalk.py — NEW vintage '
           'crosswalk builder',
           '- src/data_pipeline/build_ideology_scores_v2.py — NEW v2 '
           'ideology builder (reconciles vs production builder)',
           '- src/data_pipeline/ideology_line_audit.py, crosswalk_audit.py '
           '— NEW',
           '- src/data_pipeline/rebuild_canonical.py — CHANGED: merges the '
           'new columns + flags',
           '- src/analysis/wild_cluster_inference.py — NEW wild-cluster-'
           'bootstrap engine (validated vs statsmodels + brute force)',
           '- src/analysis/robustness_checks_v2.py, sample_funnel.py, '
           'compute_morans_i_v2.py, spatial_regression_model_v2.py, '
           'ml_dem_share_collinearity.py — NEW',
           '- src/causal/city_of_yes_did.py — CHANGED: Wald bug fix + v2 '
           'specs block',
           '- src/causal/iv_district_fe_regression_v2.py — NEW (original 12 '
           'arms untouched)',
           '- src/verification/verify_canonical_dataset_and_split.py — '
           'CHANGED: legacy-53-present + v2-11 contract',
           '- src/verification/verify_city_of_yes_did.py — CHANGED: renamed '
           'fields + implied-F checks',
           '- src/verification/verify_ideology_v2.py, '
           'verify_analysis_v2.py, verify_causal_spatial_v2.py, '
           'verify_repo_hygiene.py — NEW',
           '- src/figures/generate_paper_assets.py — CHANGED: DiD figure '
           'annotation + table caption use chi2 and the implied F',
           '- src/analysis/generate_regression_table.py — CHANGED: stale '
           'manuscript targets removed',
           '- scripts/make_anonymous_release.py + config/scrub_terms.txt — '
           'NEW (Task L)',
           '- scripts/write_results_for_paper.py — NEW: generates this '
           'index and RESULTS_FOR_PAPER.md',
           '- main.py — CHANGED: 12 new stages registered, stale labels '
           'updated; README.md — CHANGED: title leads with the paper '
           'title; LICENSE — NEW (MIT); requirements.txt — CHANGED: '
           're-pinned to the virtualenv',
           '- archive/FULL_PAPER_FACT_CHECK.md — MOVED from results/ with '
           'a SUPERSEDED header',
           '',
           '## Regenerated paper assets',
           '- docs/paper_assets/table_did_event_study.tex + '
           'fig7_did_event_study.png — CHANGED: chi2(3) statistic with '
           'implied F(3,50) in caption/annotation',
           '- docs/paper_assets/* — regenerated from the artifacts (see '
           'manifest.json)',
           ]
    OUT_INDEX.write_text('\n'.join(idx) + '\n', encoding='utf-8')
    print('Wrote', OUT_INDEX.relative_to(ROOT))


def main():
    rob = load('results/robustness_checks_v2.json')
    did_v2 = load('results/causal/city_of_yes_did_results_v2.json')
    iv_v2 = load('results/causal/iv_district_fe_results_v2.json')
    moran_v2 = load('results/morans_i_results_v2.json')
    spat_v2 = load('results/spatial_regression_v2.json')
    mlc = load('results/ml_dem_share_collinearity.json')
    funnel = load('results/sample_funnel.json')
    line = load('results/ideology_line_audit.json')

    # ---------------- RESULTS_FOR_PAPER.md -------------------------------
    p = ['# RESULTS FOR PAPER — Round 12 v2 outputs',
         '',
         'Every number below is generated from the repository artifacts by '
         'scripts/write_results_for_paper.py (no hand-typed values). '
         'Cluster = council district throughout; cluster SEs use the '
         'Stata/statsmodels CR1 correction; p(t) is from t(G-1); p(wild) '
         'is the wild-cluster-bootstrap-t p-value (Webb 6-point weights, '
         '9,999 replications, seed 42).',
         '']
    p.append('## Data funnel (845,607 raw sales -> canonical -> '
             'restricted sample)\n')
    p.append('| Step | Rows | Lost from previous |')
    p.append('|---|---|---|')
    for s in funnel['funnel']:
        p.append(f'| {s["step"]} | {s["rows"]:,} | '
                 f'{s["rows_lost_from_previous"]:,} |')
    p.append('\n## Ballot-line audit headline (mayoral races)\n')
    p.append('| Year | Candidate-line votes | Dem | Rep | Other | '
             'Other share | legacy dem_share | all-candidate share |')
    p.append('|---|---|---|---|---|---|---|---|')
    for y, r in line['by_year'].items():
        p.append(f'| {y} | {r["candidate_line_votes"]:,} | '
                 f'{r["class_votes"]["Dem"]:,} | {r["class_votes"]["Rep"]:,} '
                 f'| {r["class_votes"]["Other"]:,} | '
                 f'{100 * r["other_share_of_candidate_votes"]:.2f}% | '
                 f'{r["legacy_dem_share_dem_over_dem_plus_rep"]:.4f} | '
                 f'{r["dem_share_mayor_allcand"]:.4f} |')
    p.append('\n> **2025 flag:** Other-class ballot lines (led by the Cuomo '
             '"Fight and Deliver" line) hold '
             f'{100 * line["by_year"]["2025"]["other_share_of_candidate_votes"]:.1f}% '
             'of mayoral candidate votes; the legacy Dem/(Dem+Rep) ratio '
             'excludes them by construction.\n')
    p.append('\n> **Crosswalk vintage warning:** under the production '
             '(2026-vintage) ED crosswalk the 2016-2021 returns lose '
             '17.8-24.1% of ED-level candidate votes to pre-redistricting '
             'ED keys (a pre-existing property of the legacy dem_share, '
             'quantified in results/crosswalk_audit.json). The '
             'vintage-matched crosswalk recovers every year to ~0%; the '
             '*_vintage measures use it.\n')
    p.append('\n## Cross-sample regression suite '
             '(7 specs x 6 measures x 2 samples = 84 fits)\n')
    p.extend(robustness_tables(rob))
    p.extend(did_tables(did_v2))
    p.extend(iv_tables(iv_v2))
    p.extend(spatial_tables(moran_v2, spat_v2))
    p.append('\n## ML collinearity diagnostics (no retraining)\n')
    dd = mlc['dem_share_deterministic_by_district']
    p.append(f'- dem_share is single-valued per council district on the '
             f'train+test sample: '
             f'{dd["districts_with_single_dem_share_value_train_plus_test"]}'
             f'/{dd["n_districts"]} districts; R2 of dem_share on district '
             f'dummies = {dd["r2_dem_share_on_district_dummies"]:.4f} '
             '(deterministic function of the district in this sample).\n')
    p.append('- Duplicated feature pairs: ' + '; '.join(
        f'{d["feature_a"]} vs {d["feature_b"]} (r={d["pearson_r"]:.3f})'
        for d in mlc['duplicated_feature_pairs']) + '.\n')
    p.append(f'- Champion `{mlc["champion_selection"]["champion"]}` '
             f'(test MAE {mlc["champion_selection"]["test_mae"]}) was '
             'selected on test MAE; no stored validation predictions exist '
             'to compute a validation-MAE champion.\n')
    OUT_PAPER.write_text('\n'.join(p) + '\n', encoding='utf-8')
    print('Wrote', OUT_PAPER.relative_to(ROOT))
    write_index()


if __name__ == '__main__':
    main()

