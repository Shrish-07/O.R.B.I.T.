"""Manuscript asset generator -- rebuilds every table fragment and all four
figures of the manuscript straight from repository artifacts.

Why this exists
---------------
The manuscript (main.tex) is hand-typed: 17 tabulars, 4 figures and a large
number of in-text numbers were verified against results/*.json exactly once.
This stage makes that check permanent and the figures reproducible:

  * one LaTeX tabular fragment per manuscript table, written to
    docs/manuscript_assets/tables/<label>.tex, using the SAME column layout,
    the SAME rounding and the SAME star rules as main.tex (stars from the
    wild-cluster-bootstrap p: <0.01 ***, <0.05 **, <0.10 *; 3 dp; round half
    away from zero on the stored float);
  * the four figures, written to docs/manuscript_assets/figures/ under the
    exact filenames the manuscript \\includegraphics's:
        fig1_timepaths.png   citywide time paths, FIRST-BUILT district means
        fig2_speccurve.png   coef by specification x measure (vintage_aw),
                             95 percent WILD bootstrap intervals
        fig3_placebo.png     district-FE estimate and its two components
                             (vintage_aw, wild intervals)
        fig4_did.png         City of Yes event study + estimates by design
                             (wild intervals)
    Every plotted point is printed to stdout so the figure can be compared
    against the source JSON cell by cell.

Inputs -- repository artifacts ONLY.  This stage writes ONLY
docs/manuscript_assets/ ; nothing under data/ or results/ is written.

Run: python src/figures/generate_manuscript_assets.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TABLES_DIR = ROOT / 'docs' / 'manuscript_assets' / 'tables'
FIGS_DIR = ROOT / 'docs' / 'manuscript_assets' / 'figures'

PREFERRED = ('dem_share_vintage_aw', 'dem_share_mayor_allcand_vintage_aw',
             'dem_share_pres_vintage_aw')
PREFERRED_LABELS = ('Mayoral two-party', 'Mayoral all-candidate',
                    'Presidential two-party')

# ---------------------------------------------------------------------------
# Number formatting -- the manuscript's exact conventions
# ---------------------------------------------------------------------------
def stars(p):
    """Wild-cluster-bootstrap star rule: <0.01 ***, <0.05 **, <0.10 *."""
    if p is None:
        return ''
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.10:
        return '*'
    return ''


def rha(x, nd=3):
    """Round half AWAY FROM ZERO on the stored float (Decimal, not binary)."""
    if x is None:
        return None
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def f(x, nd=3):
    if x is None:
        return '--'
    return f'{rha(x, nd):.{nd}f}'


def sgn(x, nd=3, math_all=False):
    """The manuscript's minus convention.  Text-mode cells put only the minus
    in math mode ($-$0.096); whole-cell math cells wrap everything ($-0.0004$).
    Ties round away from zero, so -0.0004 at 4 dp stays -0.0004 and -0.0000
    collapses to 0.000 (no stray minus)."""
    if x is None:
        return '--'
    v = rha(x, nd)
    s = f'{abs(v):.{nd}f}'
    if v < 0:
        return ('$-' + s + '$') if math_all else ('$-$' + s)
    return s


def cell_coefs(coef, p=None, nd=3):
    """Coefficient cell: value + wild-bootstrap stars."""
    return sgn(coef, nd) + stars(p)


def cell_se(se, nd=3):
    return '(' + f(se, nd) + ')'


def cell_ci(lo, hi, nd=3):
    return '[' + sgn(lo, nd) + ', ' + sgn(hi, nd) + ']'


def cell_int(n):
    return '--' if n is None else f'{int(n):,}'


def cell_pct(x, nd=1):
    """Share in [0, 1] -> '6.3\\%'; None -> '--'."""
    if x is None:
        return '--'
    return f'{rha(100.0 * float(x), nd):.{nd}f}' + '\\%'


class Fragment:
    """One tabular fragment: identical column layout and cell text to the
    manuscript tabular it mirrors.  `header` holds the lines printed before
    the first \\midrule (column heads / multicolumn bands), `lines` the body."""

    def __init__(self, label, name, colspec):
        self.label = label
        self.name = name
        self.colspec = colspec
        self.header = []
        self.lines = []

    def head(self, *cells):
        self.header.append(' & '.join(cells) + ' \\\\')

    def rule(self, which='\\midrule'):
        self.lines.append(which)

    def raw(self, line):
        """Append a literal LaTeX line (cmidrule, addlinespace, ...)."""
        self.lines.append(line)

    def row(self, *cells, space=None):
        line = ' & '.join(cells) + ' \\\\'
        if space is not None:
            line += f'[{space}]'
        self.lines.append(line)

    def write(self):
        out = TABLES_DIR / f'{self.name}.tex'
        body = ['% GENERATED by src/figures/generate_manuscript_assets.py '
                '-- do not edit by hand',
                f'% manuscript table: {self.label}',
                f'\\begin{{tabular}}{{{self.colspec}}}',
                '\\toprule']
        body += self.header + self.lines
        body += ['\\bottomrule', '\\end{tabular}']
        out.write_text('\n'.join(body) + '\n', encoding='utf-8')
        n = len(self.header) + len(self.lines)
        print(f'wrote {out.relative_to(ROOT)}  ({n} rows)')
        return out


def _json(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def _fit(block, key):
    """Coef / clustered SE / wild p / wild CI of one robustness-checks fit."""
    b = block[key]
    return {'coef': b['coef'], 'se': b['se_cluster'],
            'p': b.get('p_wild_cluster_bootstrap_t'),
            'lo': b.get('ci95_low_wild'), 'hi': b.get('ci95_high_wild'),
            'n': b.get('n'), 'G': b.get('n_clusters_G'),
            'r2': b.get('r_squared')}


SPEC_KEYS = ['model3_boro_fe', 'model4_district_fe', 'model5_boro_year_fe',
             'model6_district_year_fe_units', 'model7_district_boro_x_year_fe',
             'model8_district_linear_trends']
SPEC_LABELS = ['Borough FE', 'District FE', 'Borough + year FE',
               'District + year FE', 'District + borough$\\times$year FE',
               'District + year FE + district trends']

# --- artifact handles (all read-only) ---------------------------------------
ROB2 = _json('results/robustness_checks_v2.json')       # legacy + vintage
ROB3 = _json('results/robustness_checks_v3.json')       # _aw + _dom
PRE24 = _json('results/robustness_pre2024_v3.json')     # sales through 2023
PLACEBO = _json('results/time_trend_placebo_v3.json')
SPATIAL = _json('results/spatial_regression_v3.json')
MORAN2 = _json('results/morans_i_results_v2.json')
MORAN3 = _json('results/morans_i_results_v3.json')
HEDONIC = _json('results/hedonic_v3.json')
FUNNEL = _json('results/sample_funnel.json')
LINES = _json('results/ideology_line_audit.json')
XWALK = _json('results/crosswalk_audit.json')
XWALK3 = _json('results/crosswalk_audit_v3.json')
DID1 = _json('results/causal/city_of_yes_did_results.json')
DID2 = _json('results/causal/city_of_yes_did_results_v2.json')
DID3 = _json('results/causal/city_of_yes_did_results_v3.json')
IV1 = _json('results/causal/iv_district_fe_results.json')
IV2 = _json('results/causal/iv_district_fe_results_v2.json')
IVR = _json('results/causal/iv_district_fe_rescue_results.json')

with (ROOT / 'results' / 'regression_table.csv').open(encoding='utf-8') as fh:
    REGROWS = list(csv.DictReader(fh))
REG = {(r['Model'], r['Variable']): r for r in REGROWS}

H_OLS = _json('results/round5_part_H_ols.json')['section_H_ols']
ROB1 = _json('results/robustness_checks.json')

IDEO_V2 = pd.read_parquet(ROOT / 'data' / 'processed' / 'ideology_by_council_v2.parquet')
IDEO_PRES = pd.read_parquet(ROOT / 'data' / 'processed' / 'ideology_pres_by_council.parquet')
IDEO_V3 = pd.read_parquet(ROOT / 'data' / 'processed' / 'ideology_by_council_v3.parquet')
IDEO_PRES_V3 = pd.read_parquet(ROOT / 'data' / 'processed' / 'ideology_pres_by_council_v3.parquet')

MAYOR_WAVES = [2017, 2021, 2025]
PRES_WAVES = [2016, 2020, 2024]


def wave_mean(frame, col, yearcol, waves):
    """One value per district-wave (mean over that wave's sales), then the
    UNWEIGHTED mean across the 51 council districts."""
    out = {}
    for w in waves:
        sub = frame[frame[yearcol] == w]
        per = sub.groupby('CounDist')[col].mean().dropna()
        out[w] = (float(per.mean()), int(len(per)))
    return out


WM = {
    'Mayoral two-party': wave_mean(IDEO_V2, 'dem_share', 'election_year', MAYOR_WAVES),
    'Mayoral all-candidate': wave_mean(IDEO_V2, 'dem_share_mayor_allcand',
                                       'election_year', MAYOR_WAVES),
    'Presidential two-party': wave_mean(IDEO_PRES, 'dem_share_pres',
                                        'pres_election_year', PRES_WAVES),
}

# ---------------------------------------------------------------------------
# Table fragments
# ---------------------------------------------------------------------------
def t_tab_measures():
    """Table 2: ideology measures and their averages across districts."""
    fr = Fragment('tab:measures', 'tab_measures', 'llcccc')
    fr.head('Measure', 'Contest', '2016/17', '2020/21', '2024/25', 'Change')
    fr.rule()
    rows = [('Mayoral two-party', 'Mayor', WM['Mayoral two-party']),
            ('Mayoral all-candidate', 'Mayor', WM['Mayoral all-candidate']),
            ('Presidential two-party', 'President', WM['Presidential two-party'])]
    for label, contest, wm in rows:
        waves = list(wm.keys())
        vals = [wm[w][0] for w in waves]
        chg = vals[-1] - vals[0]
        fr.row(label, contest, f(vals[0]), f(vals[1]), f(vals[2]),
               sgn(chg) if chg < 0 else '$+' + f(chg) + '$')
    fr.rule()
    other = [LINES['by_year'][str(y)]['other_share_of_candidate_votes']
             for y in MAYOR_WAVES]
    fr.row('\\multicolumn{2}{l}{Mayoral votes on other lines (citywide)}',
           cell_pct(other[0]), cell_pct(other[1]), cell_pct(other[2]), '')
    return fr


def t_tab_base():
    """Table 3: baseline regressions of log price on the first-built mayoral
    two-party share (columns 1-3)."""
    fr = Fragment('tab:base', 'tab_base', 'lccc')
    fr.head('', '(1)', '(2)', '(3)')
    fr.rule()
    m1 = H_OLS['model1']['dem_share']
    m2 = H_OLS['model2']['dem_share']
    m3 = H_OLS['model3']['dem_share']
    fr.row('Democratic share', f(m1['coef']), f(m2['coef']), f(m3['coef']))
    fr.row('Classical standard error', cell_se(m1['se']), cell_se(m2['se']),
           cell_se(m3['se']))
    cl = ROB1['check1_model3_clustered_by_council_district']
    fr.row('District-clustered standard error', '', '', cell_se(cl['dem_share_se']))
    tstat = m3['coef'] / cl['dem_share_se']
    dfree = int(ROB1['check1_n_clusters']) - 1
    p_t = 2.0 * stats.t.sf(abs(tstat), dfree)
    p_wild = _fit(ROB2['full']['dem_share'], 'model3_boro_fe')['p']
    fr.row('$p$-value, clustered $t$ / wild bootstrap', '', '',
           f(p_t, 3) + ' / ' + f(p_wild, 3), space='3pt')
    fr.row('Controls', 'No', 'Yes', 'Yes')
    fr.row('Borough fixed effects', 'No', 'No', 'Yes')
    fr.row('$R^2$', f(H_OLS['model1']['R2']['recomputed']),
           f(H_OLS['model2']['R2']['recomputed']),
           f(H_OLS['model3']['R2']['recomputed']))
    n = int(H_OLS['N']['recomputed'])
    fr.row('Observations', cell_int(n), cell_int(n), cell_int(n))
    return fr


def t_tab_main():
    """Table 4: coefficient across six specifications x the three preferred
    (boundary-matched, area-weighted) measures."""
    fr = Fragment('tab:main', 'tab_main', 'lrccc')
    fr.head('', '', '\\multicolumn{3}{c}{Ideology measure}')
    fr.raw('\\cmidrule(lr){3-5}')
    fr.head('Specification', '$N$', 'Mayoral two-party',
            'Mayoral all-candidate', 'Presidential two-party')
    fr.rule()
    for label, key in zip(SPEC_LABELS, SPEC_KEYS):
        fits = [_fit(ROB3['full'][m], key) for m in PREFERRED]
        fr.row(label, cell_int(fits[0]['n']),
               *[cell_coefs(x['coef'], x['p']) for x in fits])
        fr.row('', '', *[cell_se(x['se']) for x in fits], space='3pt')
    return fr


def t_tab_placebo():
    """Table 5: decomposing the district-fixed-effects estimate."""
    fr = Fragment('tab:placebo', 'tab_placebo', 'lccc')
    fr.head('Measure', 'Actual district-FE estimate',
            'Citywide wave means only', 'District deviations')
    fr.rule()
    for label, m in zip(PREFERRED_LABELS, PREFERRED):
        actual = _fit(ROB3['full'][m], 'model4_district_fe')
        wm = PLACEBO[m]['wavemean_model4']
        dv = PLACEBO[m]['dev_model4']
        fr.row(label, cell_coefs(actual['coef'], actual['p']),
               cell_coefs(wm['coef'], wm['p_wild_cluster_bootstrap_t']),
               sgn(dv['coef']))
        fr.row('', '',
               cell_se(wm['se_cluster']), cell_se(dv['se_cluster']))
        fr.row('', '',
               cell_ci(wm['ci95_low_wild'], wm['ci95_high_wild']),
               cell_ci(dv['ci95_low_wild'], dv['ci95_high_wild']), space='3pt')
    return fr


def t_tab_spatial():
    """Table 6: district-level spatial regressions (N = 51)."""
    fr = Fragment('tab:spatial', 'tab_spatial', 'lcccccccc')
    fr.head('', '', '\\multicolumn{3}{c}{Pooled districts}',
            '\\multicolumn{3}{c}{With borough fixed effects}')
    fr.raw('\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}')
    fr.head('Measure', "Moran's $I$", 'OLS', 'Lag', 'Error', 'OLS', 'Lag', 'Error')
    fr.rule()
    for label, m in zip(PREFERRED_LABELS, PREFERRED):
        blk = SPATIAL['by_measure'][m]
        moran = MORAN3['by_measure'][m]['moran_i']
        cells = []
        for region in ('pooled', 'borough_fe'):
            b = blk[region]
            cells += [f(b['ols_anchor'][m]['coef']),
                      f(b['spatial_lag_model'][m]['coef']),
                      sgn(b['spatial_error_model'][m]['coef'])
                      + ('$^{*}$' if b['spatial_error_model'][m]['p'] < 0.10 else '')]
        fr.row(label, f(moran), *cells)
        ses = []
        for region in ('pooled', 'borough_fe'):
            b = blk[region]
            ses += [cell_se(b['ols_anchor'][m]['se']),
                    cell_se(b['spatial_lag_model'][m]['se']),
                    cell_se(b['spatial_error_model'][m]['se'])]
        fr.row('', '', *ses, space='3pt')
    return fr


def t_tab_did():
    """Table 7: City of Yes difference-in-differences estimates (4 dp for the
    estimate and standard error, 3 dp for p and the interval bounds)."""
    fr = Fragment('tab:did', 'tab_did', 'lccccr')
    fr.head('Specification', 'Estimate', 'Std.\\ error', 'Wild $p$',
            '95\\% interval', '$N$')
    fr.rule()
    fr.row('\\multicolumn{6}{l}{\\emph{Yes $\\times$ Post}}')
    mw = DID2['main_wild_inference']
    fr.row('All earlier years as pre-period', sgn(mw['coef'], 4, math_all=True),
           f(mw['se_cluster'], 4), f(mw['p_wild_cluster_bootstrap_t']),
           cell_ci(mw['ci95_low_wild'], mw['ci95_high_wild']), cell_int(mw['n']))
    sens = DID1['did_sensitivity_2024_included']
    fr.row('Post = 2024--25 (clustered $t$)', sgn(sens['did_coef'], 4, math_all=True),
           f(sens['did_se'], 4), f(sens['did_p']) + '$^{\\dagger}$', '--',
           cell_int(sens['n']))
    for label, key in (('Pre-period limited to 2022--24',
                        'window_2022_2025_incl_2024_as_pre'),
                       ('Pre-period limited to 2022--23',
                        'window_2022_2025_excl_2024'),
                       ('All earlier years, with district trends',
                        'district_specific_linear_trends')):
        b = DID3[key]
        fr.row(label, sgn(b['coef'], 4, math_all=True), f(b['se_cluster'], 4),
               f(b['p_wild_cluster_bootstrap_t']),
               cell_ci(b['ci95_low_wild'], b['ci95_high_wild']), cell_int(b['n']))
    fr.rule()
    fr.row('\\multicolumn{6}{l}{\\emph{Presidential Democratic share '
           '$\\times$ Post, window 2022--25}}')
    ct = DID3['continuous_treatment_dem_share_pres_vintage_aw_x_post']
    for label, key in (('2024 excluded', 'excl_2024'),
                       ('2024 as pre-period', 'incl_2024_as_pre')):
        b = ct[key]['interaction_wild_inference']
        fr.row(label, sgn(b['coef'], 4, math_all=True), f(b['se_cluster'], 4),
               f(b['p_wild_cluster_bootstrap_t']),
               cell_ci(b['ci95_low_wild'], b['ci95_high_wild']), cell_int(b['n']))
    return fr


def t_tab_summary():
    """Table 8: summary of evidence.  Numeric tokens are derived from the same
    artifacts as Tables 4 and 7; the two text columns are fixed layout."""
    fr = Fragment('tab:summary', 'tab_summary', 'p{4.0cm}p{5.1cm}p{5.2cm}')
    fr.head('Test', 'Result', 'Reading')
    fr.rule()
    plus = lambda x: ('$+' + f(x, 2) + '$') if x >= 0 else sgn(x, 2)
    row1 = [_fit(ROB3['full'][m], 'model3_boro_fe') for m in PREFERRED]
    row4 = [_fit(ROB3['full'][m], 'model4_district_fe') for m in PREFERRED]
    fr.row('Borough comparison',
           ', '.join(plus(x['coef']) for x in row1) + '; none significant',
           'Sign depends on the measure')
    fr.row('District fixed effects',
           ', '.join(plus(x['coef']) for x in row4) + '; all significant',
           "Sign follows each measure's citywide drift")
    fr.row('Placebo (wave means only)', 'Reproduces the sign of the estimate',
           'Estimate reflects time trends')
    fr.row('Year effects', 'Insignificant for all measures',
           'No association once time is controlled')
    fr.row('Borough-by-year effects',
           'Positive for mayoral, zero for presidential',
           'Not measure-independent; vanishes with district trends')
    fr.row('Spatial models',
           'Insignificant pooled; zero or slightly negative with borough effects',
           'Spatial clustering explains the raw association')
    win = DID3['window_2022_2025_incl_2024_as_pre']
    fr.row('City of Yes, matched pre-period',
           sgn(win['coef'], 4, math_all=True) + ' (interval '
           + sgn(win['ci95_low_wild'], 3) + ' to '
           + sgn(win['ci95_high_wild'], 3) + ')', 'Precise null at one year')
    fr.row('Close-election instrument', 'Unidentified or weak', 'Uninformative')
    return fr


def t_tabA_funnel():
    """Table A1: construction of the sales samples."""
    fr = Fragment('tabA:funnel', 'tabA_funnel', 'lrr')
    fr.head('Step', 'Sales remaining', 'Removed')
    fr.rule()
    by = {s['step']: s for s in FUNNEL['funnel']}
    fr.row('Raw sales records, 2016--2025',
           cell_int(by['raw_sales_rows']['rows']), '')
    fr.row('Sale price above \\$10{,}000', cell_int(by['price_gt_10k']['rows']),
           cell_int(by['price_gt_10k']['rows_lost_from_previous']))
    fr.row('Exact duplicates removed', cell_int(by['canonical_rows']['rows']),
           cell_int(by['canonical_rows']['rows_lost_from_previous']))
    fr.row('Non-missing controls (analysis sample)',
           cell_int(by['analysis_base_sample']['rows']),
           cell_int(by['analysis_base_sample']['rows_lost_from_previous']))
    unit_n = _fit(ROB2['full']['dem_share'], 'model6_district_year_fe_units')['n']
    fr.row('\\quad with unit counts (specifications with year effects)',
           cell_int(unit_n), '')
    fr.row('\\quad restricted sample: tax class 1 or 2, price $\\geq$ '
           '\\$100{,}000, single parcel',
           cell_int(by['restricted_analysis_sample']['rows']),
           cell_int(by['restricted_analysis_sample']['rows_lost_from_previous']))
    return fr


def t_tabA_audit():
    """Table A2: election returns -- ballot lines, boundary matching, district
    assignment."""
    fr = Fragment('tabA:audit', 'tabA_audit', 'llrcccc')
    fr.head('Election', 'Contest', 'Candidate-line votes', 'Other lines',
            'Unmatched ED keys', 'Full-assignment ratio')
    fr.rule()
    contests = {'2016': 'President', '2020': 'President', '2024': 'President'}
    for year in ('2016', '2017', '2020', '2021', '2024', '2025'):
        xy = XWALK['by_year'][year]
        a3 = XWALK3['by_year'][year]
        if year in ('2017', '2021', '2025'):
            votes = LINES['by_year'][year]['candidate_line_votes']
            other = cell_pct(LINES['by_year'][year]['other_share_of_candidate_votes'])
        else:
            votes = xy['total_candidate_votes_raw']
            other = '--'
        fr.row(year, contests.get(year, 'Mayor'), cell_int(votes), other,
               cell_pct(xy['missing_key_share']),
               f(a3['full_to_each_overcount_ratio_vs_matched'], 2))
    return fr


# 15 measure variants of Table A3: manuscript row label -> (source, measure)
A3_ROWS = [
    ('Mayoral two-party, as first built', 'v2', 'dem_share'),
    ('\\quad area-weighted', 'v3', 'dem_share_aw'),
    ('\\quad boundary-matched', 'v2', 'dem_share_vintage'),
    ('\\quad boundary-matched, area-weighted (preferred)', 'v3', 'dem_share_vintage_aw'),
    ('\\quad boundary-matched, dominant district', 'v3', 'dem_share_vintage_dom'),
    ('Mayoral all-candidate, as first built', 'v2', 'dem_share_mayor_allcand'),
    ('\\quad area-weighted', 'v3', 'dem_share_mayor_allcand_aw'),
    ('\\quad boundary-matched', 'v2', 'dem_share_mayor_allcand_vintage'),
    ('\\quad boundary-matched, area-weighted (preferred)', 'v3',
     'dem_share_mayor_allcand_vintage_aw'),
    ('\\quad boundary-matched, dominant district', 'v3',
     'dem_share_mayor_allcand_vintage_dom'),
    ('Presidential two-party, as first built', 'v2', 'dem_share_pres'),
    ('\\quad area-weighted', 'v3', 'dem_share_pres_aw'),
    ('\\quad boundary-matched', 'v2', 'dem_share_pres_vintage'),
    ('\\quad boundary-matched, area-weighted (preferred)', 'v3',
     'dem_share_pres_vintage_aw'),
    ('\\quad boundary-matched, dominant district', 'v3',
     'dem_share_pres_vintage_dom'),
]


def _a3_source(tag):
    return ROB2['full'] if tag == 'v2' else ROB3['full']


def t_tabA_all():
    """Table A3: all fifteen measure variants, full sample."""
    fr = Fragment('tabA:all', 'tabA_all', 'p{4.1cm}cccccc')
    fr.head('Measure', 'Borough FE', 'District FE', 'Borough + year',
            'District + year', 'District + boro$\\times$yr', 'District trends')
    fr.rule()
    for label, tag, mkey in A3_ROWS:
        src = _a3_source(tag)[mkey]
        fits = [_fit(src, k) for k in SPEC_KEYS]
        fr.row(label, *[cell_coefs(x['coef'], x['p']) for x in fits])
        fr.row('', *[cell_se(x['se']) for x in fits], space='2pt')
    return fr


A4_ROWS = [
    ('Mayoral two-party, as first built', 'v2', 'dem_share'),
    ('Mayoral two-party, preferred', 'v3', 'dem_share_aw'),
    ('Mayoral all-candidate, preferred', 'v3', 'dem_share_mayor_allcand_aw'),
    ('Presidential two-party, preferred', 'v3', 'dem_share_pres_aw'),
]


def t_tabA_restricted():
    """Table A4: restricted sample (tax class 1 or 2, price >= $100,000,
    single parcel)."""
    fr = Fragment('tabA:restricted', 'tabA_restricted', 'p{4.1cm}cccccc')
    fr.head('Measure', 'Borough FE', 'District FE', 'Borough + year',
            'District + year', 'District + boro$\\times$yr', 'District trends')
    fr.rule()
    for label, tag, mkey in A4_ROWS:
        src = (ROB2['restricted'] if tag == 'v2' else ROB3['restricted'])[mkey]
        fits = [_fit(src, k) for k in SPEC_KEYS]
        fr.row(label, *[cell_coefs(x['coef'], x['p']) for x in fits])
        fr.row('', *[cell_se(x['se']) for x in fits], space='2pt')
    return fr


PRE_KEYS = ['model3_boro_fe', 'model4_district_fe',
            'model6_district_year_fe_units', 'model7_district_boro_x_year_fe',
            'model8_district_linear_trends']
A5_ROWS = [
    ('Mayoral two-party, as first built', 'dem_share'),
    ('Mayoral two-party, preferred', 'dem_share_aw'),
    ('Mayoral all-candidate, preferred', 'dem_share_mayor_allcand_aw'),
    ('Presidential two-party, preferred', 'dem_share_pres_aw'),
]


def t_tabA_pre():
    """Table A5: sales through 2023."""
    fr = Fragment('tabA:pre', 'tabA_pre', 'p{3.6cm}ccccc')
    fr.head('Measure', 'Borough FE', 'District FE', 'District + year',
            'District + boro$\\times$yr', 'District trends')
    fr.rule()
    for label, mkey in A5_ROWS:
        fits = [_fit(PRE24[mkey], k) for k in PRE_KEYS]
        fr.row(label, *[cell_coefs(x['coef'], x['p']) for x in fits])
        fr.row('', *[cell_se(x['se']) for x in fits], space='2pt')
    return fr


def t_tabA_hedonic():
    """Table A6: hedonic specifications."""
    fr = Fragment('tabA:hedonic', 'tabA_hedonic', 'lcccc')
    land = HEDONIC['part2_hedonic_land_units']['by_measure']
    full_n = int(_fit(ROB3['full'][PREFERRED[0]], 'hedonic_upgraded')['n'])
    alt_n = int(land[PREFERRED[0]]['n'])
    fr.head('', f'\\multicolumn{{2}}{{c}}{{Full hedonic ($N={full_n:,}$)}}',
            f'\\multicolumn{{2}}{{c}}{{Land area and units only '
            f'($N={alt_n:,}$)}}')
    fr.raw('\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}')
    fr.head('Measure', 'Coef. (SE)', 'Wild $p$', 'Coef. (SE)', 'Wild $p$')
    fr.rule()
    for label, m in zip(PREFERRED_LABELS, PREFERRED):
        full = _fit(ROB3['full'][m], 'hedonic_upgraded')
        alt = land[m]
        fr.row(label,
               sgn(full['coef']) + ' ' + cell_se(full['se']),
               f(full['p'], 3),
               sgn(alt['coef']) + ' ' + cell_se(alt['se_cluster']),
               f(alt['p_wild_cluster_bootstrap_t'], 3))
    return fr


def t_tabA_iv():
    """Table A7: close-election instrument -- all twelve specifications."""
    fr = Fragment('tabA:iv', 'tabA_iv', 'llcccc')
    fr.head('Fixed effects', 'Bandwidth', 'Identified', 'First-stage $F$',
            'Coefficient', '$p$')
    fr.rule()
    arms = []
    for bw in (5, 10, 15, 20):
        arms.append(('District', bw, IV1[f'bw_{bw}pp']))
    for rec in IVR['rescue_1a_borough_fe_variant']:
        arms.append(('Borough', int(rec['bandwidth_pp']), rec))
    for rec in IVR['rescue_1b_wider_bandwidth_district_fe']:
        if int(rec['bandwidth_pp']) > 20:
            arms.append(('District', int(rec['bandwidth_pp']), rec))
    for fe, bw, rec in arms:
        ident = bool(rec['identified'])
        fs = rec.get('first_stage') or {}
        ss = rec.get('second_stage') or {}
        if ident:
            fr.row(fe, f'$\\pm{bw}$ pp', 'Yes', f(fs['f_stat'], 2),
                   sgn(ss['dem_share_coef'], 2, math_all=True),
                   f(ss['dem_share_p']))
        else:
            fr.row(fe, f'$\\pm{bw}$ pp', 'No', '--', '--', '--')
    return fr


def t_tabA_rd():
    """Table A8: local-linear close-election specification."""
    fr = Fragment('tabA:rd', 'tabA_rd', 'lccccrr')
    fr.head('Bandwidth', 'First-stage $F$', 'Coefficient', 'Std.\\ error',
            '$p$', 'Clusters', '$N$')
    fr.rule()
    for bw in (10, 15, 20, 25, 30, 35, 40):
        rec = IV2['rd_local_linear'][f'bw_{bw}pp']
        if not rec['identified']:
            fr.row(f'$\\pm{bw}$ pp', '\\multicolumn{3}{c}{not estimable}', '',
                   str(int(rec['n_clusters_G'])), cell_int(rec['n']))
        else:
            fr.row(f'$\\pm{bw}$ pp',
                   f(rec['first_stage_joint_f_excluded_instruments'], 2),
                   sgn(rec['dem_share_coef'], 3, math_all=True),
                   f(rec['dem_share_se']), f(rec['dem_share_p']),
                   str(int(rec['n_clusters_G'])), cell_int(rec['n']))
    return fr


def _paper_asset_rows(fname):
    """Parse the body rows of a generated docs/paper_assets tabular
    (cells already split on '&', trailing rule removed)."""
    txt = (ROOT / 'docs' / 'paper_assets' / fname).read_text(encoding='utf-8')
    body = txt.split('\\midrule', 1)[1].split('\\bottomrule', 1)[0]
    rows = []
    for raw in body.split('\\\\'):
        raw = raw.strip()
        if not raw or raw.startswith('\\end'):
            continue
        rows.append([c.strip() for c in raw.split('&')])
    return rows


def _untex(s):
    """Inverse of the generator's escaping, used to key parsed tables by the
    raw artifact name (docs/paper_assets writes rf\\_all\\_years\\_...)."""
    for a, b in (('\\_', '_'), ('\\&', '&'), ('\\%', '%'), ('\\#', '#'),
                 ('\\$', '$'), ('$', ''), ('{', ''), ('}', '')):
        s = s.replace(a, b)
    return s.strip()


ML_DISPLAY = {
    'rf_all_years_political_tuned': 'Random forest (tuned)',
    'cat_all_years_political_lr0.05_d6': 'CatBoost',
    'xgb_all_years_political_lr0.05_md6': 'XGBoost',
    'lgbm_all_years_political': 'LightGBM, with ideology',
    'lgbm_all_years_base': 'LightGBM, without ideology',
    'ridge_all_years_political': 'Ridge',
    'elasticnet_all_years_political': 'Elastic net',
}
ML_ORDER = list(ML_DISPLAY)


def t_tabA_ml():
    """Table A9: held-out performance and importance of ideology."""
    fr = Fragment('tabA:ml', 'tabA_ml', 'lcccc')
    fr.head('Model', 'MAE', '$R^2$', 'Rank of ideology (of 25)',
            'Mean $|$SHAP$|$')
    fr.rule()
    perf = {_untex(r[0]): r for r in _paper_asset_rows('table2_model_performance.tex')}
    shap = {_untex(r[0]): r for r in _paper_asset_rows('table_shap_cross_model.tex')}
    for key in ML_ORDER:
        name = ML_DISPLAY[key]
        mae = float(perf[key][3])
        r2 = float(perf[key][4])
        rank_raw = shap[key][2]
        val_raw = shap[key][3]
        if r'\_' in rank_raw or rank_raw in ('', '--'):
            rank, val = '--', '--'
        elif 'zeroed' in rank_raw:
            rank = 'excluded by penalty'
            val = f(float(val_raw), 3)
        else:
            rank = rank_raw.split(' of ')[0]
            val = f(float(val_raw), 3)
        fr.row(name, f(mae, 4), f(r2, 3), rank, val)
    return fr


CF_ORDER = ML_ORDER


def t_tabA_cf():
    """Table A10: simulated mean percentage change in predicted price."""
    fr = Fragment('tabA:cf', 'tabA_cf', 'lccc')
    fr.head('Model', 'Upzoning', 'Downzoning', 'Democratic share $+0.15$')
    fr.rule()
    scen = {_untex(r[0]): r for r in _paper_asset_rows('table_scenario_summary.tex')}
    signed2 = lambda x: ('$+' + f(x, 2) + '$') if x > 0 else (
        sgn(x, 2) if x < 0 else f(0.0, 2))
    for key in CF_ORDER:
        cells = [float(scen[key][1]), float(scen[key][3]), float(scen[key][5])]
        fr.row(ML_DISPLAY[key], *[signed2(c) for c in cells])
    return fr


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# House style: serif, white background (matches generate_paper_assets.py so the
# manuscript's four figures match the LaTeX tables).
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.edgecolor': '0.2',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': ':',
})
COLORS = ['#1f4e79', '#c00000', '#2e7d32']


def _save(fig, name):
    out = FIGS_DIR / name
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {out.relative_to(ROOT)}')
    return out


def fig1_timepaths():
    """Figure 1: citywide time paths of the three FIRST-BUILT measures."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for (label, wm), color in zip(WM.items(), COLORS):
        waves = list(wm.keys())
        vals = [wm[w][0] for w in waves]
        ns = [wm[w][1] for w in waves]
        ax.plot(range(len(waves)), vals, marker='o', color=color, label=label)
        for i, (w, v, n) in enumerate(zip(waves, vals, ns)):
            print(f'[fig1_timepaths] {label} wave={w} districts={n} '
                  f'mean={v:.6f} (3dp {f(v)})')
            ax.annotate(f(v), (i, v), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=8, color=color)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['2016/17', '2020/21', '2024/25'])
    ax.set_ylabel('Unweighted mean across the 51 council districts')
    ax.set_title('Citywide time paths of the three ideology measures '
                 '(first-built)')
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, 'fig1_timepaths.png')


def fig2_speccurve():
    """Figure 2: coefficient by specification x measure, preferred
    (vintage_aw) measures, 95 percent WILD bootstrap intervals.  The shaded
    band marks District FE without time controls."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axvspan(0.5, 1.5, color='0.88', zorder=0)
    for j, (label, m, color) in enumerate(zip(PREFERRED_LABELS, PREFERRED,
                                              COLORS)):
        xs, ys, los, his = [], [], [], []
        for i, key in enumerate(SPEC_KEYS):
            b = _fit(ROB3['full'][m], key)
            xs.append(i + (j - 1) * 0.22)
            ys.append(b['coef'])
            los.append(b['lo'])
            his.append(b['hi'])
            print(f'[fig2_speccurve] {label} spec={key} coef={b["coef"]:.6f} '
                  f'p_wild={b["p"]} stars="{stars(b["p"])}" '
                  f'wild_ci=[{b["lo"]:.6f}, {b["hi"]:.6f}] n={b["n"]}')
        yerr = [np.array(ys) - np.array(los), np.array(his) - np.array(ys)]
        ax.errorbar(xs, ys, yerr=yerr, fmt='o', ms=4, capsize=2.5,
                    color=color, label=label)
    ax.axhline(0, color='0.3', lw=1)
    ax.set_xticks(range(len(SPEC_KEYS)))
    ax.set_xticklabels(SPEC_LABELS, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Coefficient on the Democratic vote share')
    ax.set_title('Estimate by specification and measure, with 95 percent '
                 'wild-bootstrap intervals')
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, 'fig2_speccurve.png')


def fig3_placebo():
    """Figure 3: the district-FE estimate and its two components (preferred
    measures, wild intervals for both components)."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    labels = ['District FE\n(actual)', 'Citywide wave\nmeans only',
              'District\ndeviations']
    for j, (label, m) in enumerate(zip(PREFERRED_LABELS, PREFERRED)):
        actual = _fit(ROB3['full'][m], 'model4_district_fe')
        wm = PLACEBO[m]['wavemean_model4']
        dv = PLACEBO[m]['dev_model4']
        pts = [(actual['coef'], actual['coef'] - 1.96 * actual['se'],
                actual['coef'] + 1.96 * actual['se'], 'cluster-SE CI'),
               (wm['coef'], wm['ci95_low_wild'], wm['ci95_high_wild'],
                'wild CI'),
               (dv['coef'], dv['ci95_low_wild'], dv['ci95_high_wild'],
                'wild CI')]
        xs = [i + (j - 1) * 0.2 for i in range(3)]
        for i, (c, lo, hi, kind) in enumerate(pts):
            flat = labels[i].replace('\n', ' ')
            print(f'[fig3_placebo] {label} {flat} coef={c:.6f} ({kind}) '
                  f'lo={lo:.6f} hi={hi:.6f}')
        ax.errorbar(xs, [p[0] for p in pts],
                    yerr=[[p[0] - p[1] for p in pts],
                          [p[2] - p[0] for p in pts]],
                    fmt='o', ms=5, capsize=3, color=COLORS[j], label=label)
    ax.axhline(0, color='0.3', lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Coefficient on the Democratic vote share')
    ax.set_title('The district-fixed-effects estimate and its two components')
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, 'fig3_placebo.png')


def fig4_did():
    """Figure 4: (a) City of Yes event study; (b) estimates by design."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ev = DID1['event_study_pre_trends']
    ref = ev['reference_bin']
    order = ['2016-17', '2018-19', '2020-21', ref, '2024', '2025']
    pos = {n: i for i, n in enumerate(order)}
    plot = {ref: (0.0, 0.0, 0.0)}
    for term, rec in ev['period_terms'].items():
        nm = term.replace('yes_x_', '')
        c, s = rec['coef'], rec['se']
        plot[nm] = (c, c - 1.96 * s, c + 1.96 * s)
        print(f'[fig4_did a] {nm} coef={c:.6f} se={s:.6f} '
              f'ci95=[{c - 1.96 * s:.6f}, {c + 1.96 * s:.6f}] p={rec["p"]}')
    print(f'[fig4_did a] {ref} coef=0.000000 (reference bin, plotted at 0)')
    xs = [pos[nm] for nm in order]
    err = [[plot[nm][0] - plot[nm][1] for nm in order],
           [plot[nm][2] - plot[nm][0] for nm in order]]
    axa.errorbar(xs, [plot[nm][0] for nm in order], yerr=err, fmt='o', ms=5,
                 capsize=3, color='#1f4e79')
    axa.axhline(0, color='0.3', lw=1)
    axa.axvline(pos[ref] + 0.5, color='0.6', ls='--', lw=1)
    axa.set_xticks(list(range(len(order))))
    axa.set_xticklabels(order, rotation=30, ha='right', fontsize=8)
    axa.set_ylabel('Yes $-$ No, relative to ' + ref)
    axa.set_title('(a) Event study')

    mw = DID2['main_wild_inference']
    win = DID3['window_2022_2025_incl_2024_as_pre']
    excl = DID3['window_2022_2025_excl_2024']
    trend = DID3['district_specific_linear_trends']
    ct = DID3['continuous_treatment_dem_share_pres_vintage_aw_x_post']
    designs = [
        ('All earlier years as pre-period', mw['coef'], mw['ci95_low_wild'],
         mw['ci95_high_wild']),
        ('Pre-period 2022--24', win['coef'], win['ci95_low_wild'],
         win['ci95_high_wild']),
        ('Pre-period 2022--23', excl['coef'], excl['ci95_low_wild'],
         excl['ci95_high_wild']),
        ('District trends', trend['coef'], trend['ci95_low_wild'],
         trend['ci95_high_wild']),
        ('Presidential share, 2022--25 excl. 2024',
         ct['excl_2024']['interaction_wild_inference']['coef'],
         ct['excl_2024']['interaction_wild_inference']['ci95_low_wild'],
         ct['excl_2024']['interaction_wild_inference']['ci95_high_wild']),
    ]
    for nm, c, lo, hi in designs:
        print(f'[fig4_did b] {nm} coef={c:.6f} wild_ci=[{lo:.6f}, {hi:.6f}]')
    axb.errorbar([c for _, c, _, _ in designs], range(len(designs)),
                 xerr=[[c - lo for _, c, lo, _ in designs],
                       [hi - c for _, c, _, hi in designs]],
                 fmt='o', ms=5, capsize=3, color='#c00000')
    axb.axvline(0, color='0.3', lw=1)
    axb.set_yticks(range(len(designs)))
    axb.set_yticklabels([d[0] for d in designs], fontsize=8)
    axb.set_xlabel('Estimate')
    axb.set_title('(b) Estimates by design')
    fig.tight_layout()
    return _save(fig, 'fig4_did.png')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
TABLES = [t_tab_measures, t_tab_base, t_tab_main, t_tab_placebo, t_tab_spatial,
          t_tab_did, t_tab_summary, t_tabA_funnel, t_tabA_audit, t_tabA_all,
          t_tabA_restricted, t_tabA_pre, t_tabA_hedonic, t_tabA_iv, t_tabA_rd,
          t_tabA_ml, t_tabA_cf]
FIGURES = [fig1_timepaths, fig2_speccurve, fig3_placebo, fig4_did]


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 78)
    print('MANUSCRIPT ASSETS (tables + figures) from repository artifacts')
    print('=' * 78)
    print(f'\n--- {len(TABLES)} table fragments -> '
          f'{TABLES_DIR.relative_to(ROOT)}')
    for fn in TABLES:
        fn().write()
    print(f'\n--- {len(FIGURES)} figures -> {FIGS_DIR.relative_to(ROOT)}')
    for fn in FIGURES:
        fn()
    print('\nDone.')


if __name__ == '__main__':
    main()
