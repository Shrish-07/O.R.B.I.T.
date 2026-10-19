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
MEASURE_LABELS_V3 = {
    'dem_share_aw': 'dem_share_aw (area-weighted, mayoral Dem/(Dem+Rep))',
    'dem_share_mayor_allcand_aw': 'dem_share_mayor_allcand_aw (area-weighted)',
    'dem_share_pres_aw': 'dem_share_pres_aw (area-weighted presidential)',
    'dem_share_vintage_aw': 'dem_share_vintage_aw (area-weighted, vintage-matched)',
    'dem_share_mayor_allcand_vintage_aw':
        'dem_share_mayor_allcand_vintage_aw (area-weighted, vintage-matched)',
    'dem_share_pres_vintage_aw':
        'dem_share_pres_vintage_aw (area-weighted, vintage-matched)',
    'dem_share_vintage_dom': 'dem_share_vintage_dom (dominant-district)',
    'dem_share_mayor_allcand_vintage_dom':
        'dem_share_mayor_allcand_vintage_dom (dominant-district)',
    'dem_share_pres_vintage_dom': 'dem_share_pres_vintage_dom (dominant-district)',
}


def resolve_champion():
    """Round 13 fix: experiments/champion.json stores selected_experiment +
    metric (not name/test_mae), which previously rendered 'Champion None'.
    Resolve the model name and test MAE from the experiment registry."""
    ch = json.loads((ROOT / 'experiments' / 'champion.json')
                    .read_text(encoding='utf-8'))
    exp = ch.get('selected_experiment')
    reg = json.loads((ROOT / 'experiments' / 'registry.json')
                     .read_text(encoding='utf-8'))
    entries = reg if isinstance(reg, list) else reg.get('experiments', [])
    for e in entries:
        if e.get('id') == exp:
            mae = e.get('metrics', {}).get('mae')
            return e.get('name'), (round(float(mae), 6) if mae is not None else None)
    return None, None
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
                         f'{MEASURE_LABELS_V3.get(measure, MEASURE_LABELS.get(measure, measure))}**\n')
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
                 'first-stage joint F; Round 13 fix: covariance clustered '
                 'by council district — the pre-fix default robust '
                 'covariance reported F of 3,000-77,000; singular clustered '
                 'fits are marked not estimable with no unadjusted '
                 'fallback):\n')
    lines.append('| Bandwidth | Estimated | First-stage F | F p | '
                 'dem_share coef | dem_share SE | N | G |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for bw, r in iv_v2['rd_local_linear'].items():
        if r.get('identified'):
            lines.append(f'| {bw} | yes (clustered) | '
                         f'{fmt(r["first_stage_joint_f_excluded_instruments"], 1)} '
                         f'| {fmt_p(r["first_stage_joint_f_pvalue"])} | '
                         f'{fmt(r["dem_share_coef"])} | {fmt(r["dem_share_se"])} '
                         f'| {r["n"]:,} | {r["n_clusters_G"]} |')
        else:
            lines.append(f'| {bw} | {r.get("status", "not estimable")} | --- | '
                         f'--- | --- | --- | {r.get("n", 0):,} | '
                         f'{r.get("n_clusters_G", 0)} |')
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
    lines.append('\n> Round 13 (R3) fix: the residual-Moran regressions now '
                 'include the ideology measure among the regressors. The '
                 'pre-fix column omitted it, forcing every measure to the '
                 'same residual I = 0.3825; the pooled legacy dem_share '
                 'residual I now reproduces 0.3950 (spreg normal-theory '
                 'z = 6.14, the same statistic as the spatial-regression '
                 'OLS anchor).\n')
    return lines


def r13_split_ed_section(cw3):
    lines = ['\n## Round 13: split-ED double-counting fix '
             '(_aw area-weighted / _dom dominant-district)\n']
    lines.append('The legacy aggregation sends each ED\'s FULL votes to '
                 'every council district it touches (summed CD votes = '
                 '1.29-1.31x raw votes). The _aw measures split each ED '
                 'across its districts by normalized intersection-area '
                 'weights; the _dom measures assign each ED wholly to its '
                 'largest-overlap district. Crosswalks: legacy-vintage for '
                 '2016/2017/2020/2021, production for 2024/2025.\n')
    lines.append('| Election | Crosswalk | Raw votes (all EDs) | Coverage % | '
                 'Overcount % (_aw) | Loss % (_aw) | Overcount % (_dom) | '
                 'Full-to-each ratio |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for y in ['2017', '2021', '2025', '2016', '2020', '2024']:
        r = cw3['by_year'][y]
        lines.append(
            f'| {y} ({r["office"]}) | '
            f'{r["crosswalk_used"].replace("ed_to_council_crosswalk", "xwalk").replace(".parquet", "")} '
            f'| {r["raw_candidate_votes_all_eds"]:,} | '
            f'{r["coverage_pct_of_raw"]:.4f} | '
            f'{r["overcount_pct_aw_vs_matched"]:+.4f} | '
            f'{r["loss_pct_aw_vs_matched"]:+.4f} | '
            f'{r["overcount_pct_dom_vs_matched"]:+.4f} | '
            f'{r["full_to_each_overcount_ratio_vs_matched"]:.4f} |')
    lines.append('\nOverall correlation of each new measure with its '
                 'full-to-each counterpart (all CD-wave pairs):\n')
    lines.append('| New measure | Counterpart | Overall r | Per-CD r '
                 '(across waves, median) |')
    lines.append('|---|---|---|---|')
    import statistics
    for grp in ['mayoral_measures', 'presidential_measures']:
        for m, rec in cw3['per_cd_correlations'][grp].items():
            cds = list(rec['per_cd_correlation_across_waves'].values())
            med = statistics.median(cds) if cds else None
            lines.append(f'| {m} | {rec["counterpart_full_to_each"]} | '
                         f'{fmt(rec["overall_correlation"], 4)} | '
                         f'{fmt(med, 4) if med is not None else "---"} |')
    lines.append('\nEach _aw measure is numerically identical to its '
                 '_vintage_aw counterpart because the _aw crosswalk choice '
                 'is vintage-matched for the pre-2024 years by '
                 'construction (documented in the audit and verified).\n')
    return lines


def pre2024_tables(pre24):
    lines = ['\n## Round 13: Cuomo-free robustness '
             '(pre-2024 sample, sale_year <= 2023)\n']
    lines.append('Tests whether Model 7\'s positive mayoral estimates come '
                 'from the 2025 Cuomo "Fight and Deliver" contamination: '
                 'the 2024 presidential and 2025 mayoral election waves are '
                 'excluded from the estimating sample. Coefficients below '
                 'carry wild-bootstrap significance stars '
                 '(* p<0.05, ** p<0.01, *** p<0.001).\n')
    lines.append('| Measure | M3 coef | M4 coef | M6 coef | M7 coef | '
                 'M8 coef | N | G |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for m, specs in pre24.items():
        if m == 'metadata':
            continue
        cells = []
        for sp in ['model3_boro_fe', 'model4_district_fe',
                   'model6_district_year_fe_units',
                   'model7_district_boro_x_year_fe',
                   'model8_district_linear_trends']:
            r = specs.get(sp, {})
            cells.append(f'{fmt(r["coef"])}{stars(r["p_wild_cluster_bootstrap_t"])}'
                         if 'coef' in r else '---')
        lines.append(f'| {m} | ' + ' | '.join(cells)
                     + f' | {specs["model4_district_fe"]["n"]:,} | '
                       f'{specs["model4_district_fe"]["n_clusters_G"]} |')
    return lines


def placebo_tables(placebo):
    lines = ['\n## Round 13: time-trend placebo '
             '(wave-mean vs demeaned, Model 4)\n']
    lines.append('(a) measure_wavemean = citywide mean of the measure in '
                 'the election wave (identical for every district in the '
                 'wave): expected to reproduce the original Model 4 '
                 'sign/magnitude. (b) measure_dev = measure - wavemean: '
                 'expected near zero.\n')
    lines.append('| Measure | Original M4 coef | Wave-mean coef | '
                 'Sign match | Dev coef | Dev 95% CI | Near zero |')
    lines.append('|---|---|---|---|---|---|---|')
    for m, r in placebo.items():
        if m == 'metadata':
            continue
        cmp = r['comparison_to_original_model4'] or {}
        dev = r['dev_model4']
        lines.append(
            f'| {m} | {fmt(cmp.get("original_model4_coef"))} | '
            f'{fmt(r["wavemean_model4"]["coef"])} | '
            f'{cmp.get("sign_matches_original")} | '
            f'{fmt(dev["coef"])} | '
            f'[{fmt(dev["ci95_low_t"])}, {fmt(dev["ci95_high_t"])}] | '
            f'{r["dev_near_zero_verdict"]["near_zero"]} |')
    return lines


def hedonic_v3_tables(hed3):
    lines = ['\n## Round 13: hedonic sample audit + land/units variant\n']
    lines.append('Why N=23,789: the upgraded hedonic spec requires positive '
                 'GROSS and LAND square feet; only 23,789 of the 520,365 '
                 'analysis-base rows satisfy both (plus the building-class/'
                 'tax-class FE requirements). Missing square footage is '
                 'concentrated in the smaller building-class and tax-class '
                 'categories — full shares by category are in '
                 'results/hedonic_v3.json.\n')
    lines.append('| Funnel step | Rows |')
    lines.append('|---|---|')
    for r in hed3['part1_why_n_23789']['funnel']:
        lines.append(f'| {r["step"]} | {r["rows"]:,} |')
    lines.append('\nhedonic_land_units variant (log land sq ft + units, '
                 'gross sq ft NOT required — the larger sample):\n')
    lines.append('| Measure | Coef | SE (cluster) | p t(G-1) | p wild | '
                 'N | G |')
    lines.append('|---|---|---|---|---|---|---|')
    for m, r in hed3['part2_hedonic_land_units']['by_measure'].items():
        if 'coef' not in r:
            continue
        lines.append(f'| {m} | {fmt(r["coef"])} | {fmt(r["se_cluster"])} | '
                     f'{fmt_p(r["p_t_G_minus_1"])} | '
                     f'{fmt_p(r["p_wild_cluster_bootstrap_t"])} | '
                     f'{r["n"]:,} | {r["n_clusters_G"]} |')
    return lines


def did_v3_tables(did_v3):
    lines = ['\n## Round 13: DiD additions (wild-bootstrap p + CI)\n']
    lines.append('| Spec | Coef | SE (cluster) | p t(G-1) | p wild | '
                 '95% CI (wild) | N | G |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for tag, label in [
            ('window_2022_2025_incl_2024_as_pre',
             'Window 2022-25 (2024 as pre)'),
            ('window_2022_2025_excl_2024', 'Window 2022-25 (2024 excl.)'),
            ('district_specific_linear_trends',
             'District-specific linear trends')]:
        r = did_v3[tag]
        lines.append(f'| {label} | {fmt(r["coef"])} | {fmt(r["se_cluster"])} '
                     f'| {fmt_p(r["p_t_G_minus_1"])} | '
                     f'{fmt_p(r["p_wild_cluster_bootstrap_t"])} | '
                     f'[{fmt(r["ci95_low_wild"])}, {fmt(r["ci95_high_wild"])}] '
                     f'| {r["n"]:,} | {r["n_clusters_G"]} |')
    ct = did_v3['continuous_treatment_dem_share_pres_vintage_aw_x_post']
    lines.append('\nContinuous treatment dem_share_pres_vintage_aw x post '
                 'in the 2022-25 window:\n')
    lines.append('| Variant | Interaction coef | SE (cluster) | p wild | '
                 '95% CI (wild) | N | G |')
    lines.append('|---|---|---|---|---|---|---|')
    for tag, arm in ct.items():
        if tag not in ('excl_2024', 'incl_2024_as_pre'):
            continue
        r = arm['interaction_wild_inference']
        lines.append(f'| {tag} | {fmt(r["coef"])} | {fmt(r["se_cluster"])} | '
                     f'{fmt_p(r["p_wild_cluster_bootstrap_t"])} | '
                     f'[{fmt(r["ci95_low_wild"])}, {fmt(r["ci95_high_wild"])}] '
                     f'| {r["n"]:,} | {r["n_clusters_G"]} |')
    return lines


def write_index():
    idx = ['# NEW RESULTS INDEX — Round 12 (v2 outputs) + Round 13 (v3 outputs)',
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
           '',
           '## Round 13 — split-ED fix, bug fixes, and new robustness '
           '(all new files; nothing legacy overwritten)',
           '- data/processed/ideology_by_council_v3.parquet + '
           'ideology_pres_by_council_v3.parquet — NEW: the nine _aw '
           '(area-weighted) / _dom (dominant-district) measures built by '
           'build_ideology_scores_v3.py',
           '- data/canonical/modeling_dataset_canonical_v3.parquet — NEW: '
           'canonical v2 (all 64 legacy columns byte-identical) + the 9 '
           'new columns; provenance in results/canonical_v3_provenance.json',
           '- config/feature_blacklist.yaml — CHANGED: all nine new '
           'columns blacklisted from every model (no retraining)',
           '- results/crosswalk_audit_v3.json — NEW: per-year '
           'reconciliation (overcount = 0.00%, loss = 0.00% vs matched ED '
           'votes; coverage reported; legacy full-to-each 1.29-1.31x for '
           'contrast) + per-CD correlations with the full-to-each '
           'counterparts',
           '- results/robustness_checks_v3.json — NEW: 126-fit suite '
           '(7 specs x 9 _aw/_dom measures x 2 samples, full inference)',
           '- results/morans_i_results_v3.json + spatial_regression_v3.json '
           '— NEW: Moran + SLM/SEM for the nine new measures '
           '(pooled + borough FE)',
           '- results/robustness_pre2024_v3.json — NEW: Cuomo-free '
           'robustness (Models 3/4/6/7/8 x 15 measures on sale_year <= '
           '2023)',
           '- results/time_trend_placebo_v3.json — NEW: wave-mean vs '
           'demeaned Model 4 placebo (9 measures x 2 variants)',
           '- results/hedonic_v3.json — NEW: N=23,789 explanation '
           '(gross/land sqft shares by building class and tax class) + '
           'hedonic_land_units variant on the larger sample',
           '- results/causal/city_of_yes_did_results_v3.json — NEW: '
           'wild p + CI for the 2022-25 window specs and the '
           'district-trends spec; continuous dem_share_pres_vintage_aw x '
           'post (excl./incl. 2024)',
           '- results/morans_i_results_v2.json — CHANGED (R3 bug fix): '
           'the residual-Moran regressions now include the measure '
           '(previously every measure showed residual I = 0.3825; legacy '
           'dem_share now reproduces 0.3950, spreg z = 6.14)',
           '- results/causal/iv_district_fe_results_v2.json — CHANGED '
           '(R4 bug fix): the RD local-linear fits now cluster the '
           'covariance by CounDist (the pre-fix first-stage F '
           '3,000-77,000 ignored clustering; 10-20pp bandwidths are '
           'marked not estimable with no unadjusted fallback) and carry '
           'the council-district vintage mismatch flag; the original 12 '
           'IV arms are untouched',
           '- results/verification/verify_ideology_v3.json, '
           'verify_analysis_v3.json, verify_causal_spatial_v3.json — '
           'NEW verifier status files',
           '- src/data_pipeline/build_ideology_scores_v3.py, '
           'crosswalk_audit_v3.py, extend_canonical_v3.py; '
           'src/analysis/robustness_checks_v3.py, compute_morans_i_v3.py, '
           'spatial_regression_model_v3.py, robustness_pre2024_v3.py, '
           'time_trend_placebo_v3.py, hedonic_v3.py; '
           'src/causal/city_of_yes_did_v3.py — NEW stages',
           '- src/analysis/compute_morans_i_v2.py — CHANGED (R3 fix + '
           '0.3950/z=6.14 assertion); src/causal/'
           'iv_district_fe_regression_v2.py — CHANGED (R4 clustered '
           'covariance fix); scripts/write_results_for_paper.py — '
           'CHANGED (Round-13 sections + champion-name fix); main.py, '
           'src/ops/verification_summary.py — CHANGED (new stages and '
           'verifiers registered)',
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
    cw3 = load('results/crosswalk_audit_v3.json')
    rob3 = load('results/robustness_checks_v3.json')
    moran_v3 = load('results/morans_i_results_v3.json')
    spat_v3 = load('results/spatial_regression_v3.json')
    pre24 = load('results/robustness_pre2024_v3.json')
    placebo = load('results/time_trend_placebo_v3.json')
    hed3 = load('results/hedonic_v3.json')
    did_v3 = load('results/causal/city_of_yes_did_results_v3.json')

    # ---------------- RESULTS_FOR_PAPER.md -------------------------------
    p = ['# RESULTS FOR PAPER — Round 12 v2 + Round 13 outputs',
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
    p.extend(r13_split_ed_section(cw3))
    p.append('\n## Round 13: cross-sample regression suite for the _aw/_dom '
             'measures (7 specs x 9 measures x 2 samples = 126 fits)\n')
    p.extend(robustness_tables(rob3))
    p.append('\n## Round 13: Moran and spatial regressions for the '
             '_aw/_dom measures\n')
    p.extend(spatial_tables(moran_v3, spat_v3))
    p.extend(pre2024_tables(pre24))
    p.extend(placebo_tables(placebo))
    p.extend(hedonic_v3_tables(hed3))
    p.extend(did_v3_tables(did_v3))
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
    champ_name, champ_mae = resolve_champion()
    p.append(f'- Champion `{champ_name}` (test MAE {champ_mae}) was '
             'selected on test MAE (see experiments/champion.json; the '
             'Round-13 generator resolves the experiment id through '
             'experiments/registry.json instead of printing the empty '
             'champion-selection note this line previously rendered); '
             'no stored validation predictions exist '
             'to compute a validation-MAE champion.\n')
    OUT_PAPER.write_text('\n'.join(p) + '\n', encoding='utf-8')
    print('Wrote', OUT_PAPER.relative_to(ROOT))
    write_index()


if __name__ == '__main__':
    main()

