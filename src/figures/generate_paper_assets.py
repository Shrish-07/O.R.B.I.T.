#!/usr/bin/env python3
"""Generate the paper's tables and figures from repository artifacts (Round 11).

Round 11 fixes on top of Round 10:
  * headers/captions/notes are LaTeX and are emitted VERBATIM (esc() is
    applied to data/text cells ONLY) -- Round 10 passed headers through
    esc(), mangling $R^2$, \\beta and % into \\textasciicircum{},
    \\textbackslash{}beta and literal \\%;
  * Appendix A.1 shows borough NAMES derived from
    results/council_district_borough_map.csv (sanity-asserted against the
    documented district ranges) in a two-block longtable (1-26 | 27-51),
    2 decimals, never '-0.00';
  * the scenario summary is WIDE (one row per model; Liberal/Conservative/
    Mixed each with mean % and # districts > 0);
  * the cross-model SHAP table prints 'zeroed (tie among N zeros)' when
    dem_share's importance is exactly 0 -- a positional rank is meaningless
    for an all-zero tail;
  * the IV table/figure sort bandwidths numerically, deduplicate the +-20pp
    district-FE spec (present in both the primary search and the wider-
    bandwidth rescue arm; listed once, footnoted), and Figure 6 is a
    horizontal bar chart so arm labels no longer overlap;
  * Figure 3 uses readable display names, marks the champion (resolved via
    experiments/champion.json -> registry name, never hard-coded) and the
    two LightGBM specs (champion_ranking.PAPER_MODEL_NAMES);
  * after writing, every generated .tex is wrapped in a tiny standalone
    document (booktabs, amsmath, float, longtable) and compiled with pdflatex
    when it is on PATH (SKIPPED with a clear message otherwise); any compile
    error fails the stage loudly.

Replaces the hand-assembled path for Tables 1-5, the SHAP tables, the
scenario tables (incl. Appendix A.1), and Figures 3-7 with a single
generator that reads ONLY repo artifacts -- no number is hard-coded:

  experiments/registry.json (via src/ops/champion_ranking, NAME-resolved)
      -> Table 2 (all eligible models, MAE/R^2) + Figure 3
  models/artifacts/*_shap_summary.json
      -> Table 3 (full lgbm_all_years_political SHAP table, dem_share row
         highlighted) + Figure 4; plus the cross-model provenance table
         (method, dem_share rank of N, dem_share importance, top feature,
         count of exactly-zero-importance features; 'feature_importance_
         fallback' is rendered as 'Gain (not SHAP)' -- it is NOT SHAP)
  results/political_scenarios/by_model/<model>/<scenario>_by_council.csv
      (name-keyed copies written by political_scenarios.py; rank->name map
      kept in scenario_run_metadata.json)
      -> per-scenario 7-model summary table, Appendix A.1 (51x3) and the
         3-panel Figure 5 for lgbm_all_years_political
  results/causal/iv_district_fe_results.json + iv_district_fe_rescue_results.json
      -> IV specification-search table + Figure 6
  results/causal/city_of_yes_did_results.json
      -> DiD / event-study table + Figure 7
  results/regression_table.csv, results/robustness_checks.json,
  results/spatial_regression.json
      -> Tables 1, 4, 5

Model resolution is ALWAYS by experiment name (via champion_ranking), never
by rank number -- rank assignments shift whenever the eligible set changes.

Output: docs/paper_assets/ (LaTeX .tex tables + PNG figures + manifest.json).
Figure house style: serif font, white background (matches the manuscript's
compiled look; the exploratory docs/figures/ dark theme is separate).

Run: python src/figures/generate_paper_assets.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ops.champion_ranking import PAPER_MODEL_NAMES  # noqa: E402
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

OUT_DIR = ROOT / 'docs' / 'paper_assets'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The paper's headline SHAP / scenario model, by NAME.
PAPER_MODEL = 'lgbm_all_years_political'

MANIFEST = {'generated_by': 'src/figures/generate_paper_assets.py',
            'outputs': []}

# House style: serif, white background, consistent with tables/*.tex look.
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.edgecolor': '0.2',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
})


def esc(s):
    """Minimal LaTeX escaping for table cell text."""
    return (str(s).replace('\\', '\\textbackslash{}').replace('&', r'\&')
                  .replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')
                  .replace('^', r'\textasciicircum{}'))


def fmt(v, digits=4):
    """Format numbers for tables; None/NaN -> '--'."""
    if v is None:
        return '--'
    try:
        if isinstance(v, str) or np.isnan(float(v)):
            return '--'
        return f'{float(v):.{digits}f}'
    except (TypeError, ValueError):
        return '--'


def fmt2(v):
    """2-decimal formatting with the '-0.00' guard (never emit '-0.00')."""
    s = fmt(v, 2)
    return '0.00' if s == '-0.00' else s


def write_tex(name, caption, label, header, rows, note=None, col_fmt=None):
    """Write a booktabs LaTeX table; returns the output Path (also added to
    MANIFEST)."""
    n_cols = len(header)
    col_fmt = col_fmt or ('l' + 'c' * (n_cols - 1))
    lines = [
        '% GENERATED by src/figures/generate_paper_assets.py -- do not edit by hand',
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{' + col_fmt + '}',
        r'\toprule',
        # Headers are LaTeX (math, \%, \#) and MUST be emitted verbatim --
        # only caller data/text cells get esc(). (Round 11)
        ' & '.join(str(h) for h in header) + r' \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(' & '.join(str(c) for c in row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    if note:
        lines += [r'\vspace{2pt}', r'\begin{minipage}{0.95\textwidth}',
                  r'\footnotesize ' + note, r'\end{minipage}']
    lines.append(r'\end{table}')
    path = OUT_DIR / name
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST['outputs'].append({'file': f'docs/paper_assets/{name}', 'kind': 'tex'})
    print('wrote', path.relative_to(ROOT))
    return path


def save_fig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    MANIFEST['outputs'].append({'file': f'docs/paper_assets/{name}', 'kind': 'png'})
    print('wrote', path.relative_to(ROOT))
    return path


def _display_name(e):
    """Readable figure label for a ranked entry. Resolution stays by name/id;
    this is a COSMETIC mapping only (raw name as fallback). (Round 11)"""
    name = str(e.get('name'))
    fam = str(e.get('family') or '')
    mode = str(e.get('mode') or '')
    if name == 'rf_all_years_political_tuned':
        return 'Tuned Random Forest'
    if fam == 'linear':
        if 'ridge' in name:
            return 'Ridge'
        if 'elasticnet' in name:
            return 'ElasticNet'
    fam_disp = {'lightgbm': 'LightGBM', 'catboost': 'CatBoost',
                'xgboost': 'XGBoost', 'sklearn': 'Random Forest'}.get(fam)
    if fam_disp is not None:
        return fam_disp + (f' ({mode})' if mode else '')
    return name


# ---------------------------------------------------------------------------
# Table 2 + Figure 3: every eligible model, resolved by NAME
# ---------------------------------------------------------------------------
def table2_and_fig3(models):
    rows = []
    for e in models:
        rows.append([esc(e['name']), esc(e.get('family', '')),
                     esc(e.get('mode', '')),
                     fmt(e.get('test_mae'), 5), fmt(e.get('test_r2'), 4)])
    write_tex(
        'table2_model_performance.tex',
        'Predictive performance on the held-out temporal test split '
        '(all eligible registry models).',
        'tab:model_performance',
        ['Model', 'Family', 'Mode', 'MAE (log price)', r'$R^2$'], rows,
        note=('Generated from \\texttt{experiments/registry.json} via '
              '\\texttt{src/ops/champion\\_ranking.py} (untainted, non-stale, '
              'post-2026-05-18 entries only). The Random Forest row is the '
              'complexity-constrained, validation-tuned forest of '
              '\\texttt{models/training/train\\_random\\_forest.py}: selection '
              'used a time-ordered validation slice of the training years only '
              'under a pre-specified $\\leq 3000$ mean-leaves-per-tree cap; the '
              'test set was scored exactly once. Full audit: '
              '\\texttt{results/rf\\_tuning\\_grid.json}.'))

    # ---- Figure 3: readable display names; champion + both LightGBM paper
    # specs marked. Champion resolved via experiments/champion.json ->
    # registry NAME (never hard-coded); LightGBM pair via the exported
    # champion_ranking.PAPER_MODEL_NAMES. (Round 11)
    from matplotlib.patches import Patch
    champ_id = json.loads((ROOT / 'experiments' / 'champion.json')
                          .read_text(encoding='utf-8')
                          ).get('selected_experiment')
    champ_name = next((e['name'] for e in models
                       if e.get('experiment_id') == champ_id), None)
    if champ_name is None:
        print('NOTE: champion.json selected_experiment not among eligible '
              'models; Figure 3 champion highlight skipped.')
    entries = models[::-1]
    labels, maes, colors = [], [], []
    for e in entries:
        lab = _display_name(e)
        if e['name'] == champ_name:
            lab += ' [CHAMPION]'
        labels.append(lab)
        maes.append(e.get('test_mae'))
        colors.append('#c0392b' if e['name'] == champ_name
                      else '#e69f00' if e['name'] in PAPER_MODEL_NAMES
                      else '#2c7fb8')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(labels, maes, color=colors)
    for i, v in enumerate(maes):
        ax.text(v, i, f' {v:.4f}', va='center', fontsize=8)
    ax.set_xlabel('Test MAE (log price)')
    ax.set_title('Model comparison — test MAE (eligible registry models)')
    ax.set_xlim(0, max(v for v in maes if v is not None) * 1.22)
    ax.legend(handles=[
        Patch(color='#c0392b', label='Champion (champion.json, by name)'),
        Patch(color='#e69f00', label='LightGBM paper specs'),
        Patch(color='#2c7fb8', label='Other eligible families')],
        fontsize=8, loc='lower right')
    save_fig(fig, 'fig3_model_mae.png')


# ---------------------------------------------------------------------------
# SHAP artifacts: Table 3 + Figure 4 (paper model) + cross-model provenance
# ---------------------------------------------------------------------------
def _load_shap_summary(name):
    p = ROOT / 'models' / 'artifacts' / f'{name}_shap_summary.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def table3_fig4_and_crossmodel(models):
    shap = _load_shap_summary(PAPER_MODEL)
    rows = []
    dem_rank = '--'
    for i, r in enumerate(shap['summary'], 1):
        feat, val = esc(r['feature']), fmt(r['mean_abs_shap'], 5)
        if r['feature'] == 'dem_share':
            feat, val = r'\textbf{dem\_share}', r'\textbf{%s}' % val
            dem_rank = i
        rows.append([i, feat, val])
    write_tex(
        'table3_shap_political.tex',
        f'SHAP feature importance, \\texttt{{{esc(PAPER_MODEL)}}} '
        f'(mean $|$SHAP$|$; \\textbf{{dem\\_share}} is rank {dem_rank} of '
        f'{len(rows)}).',
        'tab:shap_political',
        ['Rank', 'Feature', r'Mean $|$SHAP$|$'], rows,
        note=('Exact TreeSHAP computed by \\texttt{src/ops/shap\\_tiers.py}, '
              f"tier {esc(shap.get('shap_fallback_tier', '--'))}: "
              f"{esc(shap.get('shap_fallback_note', ''))}"))

    top = shap['summary'][:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [r['feature'] for r in top]
    vals = [r['mean_abs_shap'] for r in top]
    colors = ['#c0392b' if x == 'dem_share' else '#2c7fb8' for x in labels]
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title(f'Top 15 features by mean |SHAP| — {PAPER_MODEL}')
    save_fig(fig, 'fig4_shap_political.png')

    rows = []
    for e in models:
        name = e['name']
        s = _load_shap_summary(name)
        if s is None:
            rows.append([esc(name), 'MISSING', '--', '--', '--', '--'])
            continue
        summary = s.get('summary', [])
        method = ('Gain (not SHAP)'
                  if s.get('method') == 'feature_importance_fallback'
                  else f"SHAP (tier {s.get('shap_fallback_tier', '?')})")
        feats = [r['feature'] for r in summary]
        vals = {r['feature']: r.get('mean_abs_shap') for r in summary}
        n_zero = int(sum(1 for v in vals.values() if v == 0))
        if 'dem_share' in vals and vals['dem_share'] is not None:
            dem_val = vals['dem_share']
            if dem_val == 0:
                # An all-zero tail: every exactly-zero-importance feature is
                # TIED for last place, so printing a positional rank like
                # '25 of 25' is meaningless noise. Say what it is. (Round 11)
                dem_rank = f'zeroed (tie among {n_zero} zeros)'
            else:
                dem_rank = f"{feats.index('dem_share') + 1} of {len(feats)}"
        else:
            dem_val, dem_rank = None, '--'
        rows.append([esc(name), esc(method), dem_rank, fmt(dem_val, 5),
                     esc(feats[0] if feats else '--'), n_zero])
    write_tex(
        'table_shap_cross_model.tex',
        "Cross-model feature-importance provenance. Rows marked 'Gain (not "
        "SHAP)' are impurity/gain importances, NOT SHAP values; SHAP rows "
        'show the tier that produced them.',
        'tab:shap_cross_model',
        ['Model', 'Method', r'\texttt{dem\_share} rank',
         r'\texttt{dem\_share} value', 'Top feature', r'\# zero-importance'],
        rows,
        note=('Generated from \\texttt{models/artifacts/*\\_shap\\_summary.json}; '
              'models whose feature list lacks \\texttt{dem\\_share} (base-mode '
              'spec) show *--*. The tier/method labels are self-reported by '
              'the producing payload.'))


# ---------------------------------------------------------------------------
# Scenario outputs (name-keyed by_model copies -- never rank-suffixed reads)
# ---------------------------------------------------------------------------
SCEN_DIR = ROOT / 'results' / 'political_scenarios'
SCEN_LIST = ['liberal_policy', 'conservative_policy', 'mixed_governance']
SCEN_TITLES = {'liberal_policy': 'Liberal Policy (R3->R7 zoning, FAR x1.5)',
               'conservative_policy': 'Conservative Policy (R7->R3, FAR x0.7)',
               'mixed_governance': 'Mixed Governance (dem_share + 0.15)'}


def _load_scenario_by_council(model_name, scen):
    """Read the NAME-keyed by_council CSV for (model, scenario) written by
    src/analysis/political_scenarios.py. Falls back (loudly) to the rank-1
    legacy filename only when the name-keyed layout is absent. Returns a
    (CounDist, mean_pct_change) DataFrame or None if neither exists."""
    p = SCEN_DIR / 'by_model' / model_name / f'{scen}_by_council.csv'
    if not p.exists() and model_name == PAPER_MODEL:
        alt = SCEN_DIR / f'{scen}_by_council.csv'
        if alt.exists():
            print(f'NOTE: by_model copy missing for {model_name}/{scen}; '
                  f'falling back to legacy {alt.name} (run political_scenarios '
                  'first to refresh)')
            p = alt
    if not p.exists():
        return None
    d = pd.read_csv(p)
    ccol = next((c for c in d.columns if 'coun' in c.lower()), d.columns[0])
    return (d[[ccol, 'mean_pct_change']]
            .rename(columns={ccol: 'CounDist'})
            .assign(CounDist=lambda x: x['CounDist'].astype(int))
            .sort_values('CounDist')
            .reset_index(drop=True))


def _borough_map():
    """Council district -> borough NAME, derived from
    results/council_district_borough_map.csv (prefers its borough_name
    column). The documented district ranges are an assertion target only --
    the mapping values come from the CSV, never from this code. (Round 11)"""
    p = ROOT / 'results' / 'council_district_borough_map.csv'
    if not p.exists():
        return None
    d = pd.read_csv(p)
    dcol = next((c for c in d.columns if 'coun' in c.lower()
                 or 'dist' in c.lower()), None)
    bcols = [c for c in d.columns if 'boro' in c.lower()]
    bcol = next((c for c in bcols if 'name' in c.lower()),
                bcols[0] if bcols else None)
    if dcol is None or bcol is None:
        return None
    m = (d[[dcol, bcol]].rename(columns={dcol: 'CounDist', bcol: 'Borough'})
         .assign(CounDist=lambda x: x['CounDist'].astype(int))
         .drop_duplicates('CounDist'))
    # Sanity assertion (values themselves are NOT hard-coded above):
    # Manhattan 1-10, Bronx 11-18, Queens 19-32, Brooklyn 33-48, SI 49-51.
    expected = ({i: 'Manhattan' for i in range(1, 11)}
                | {i: 'Bronx' for i in range(11, 19)}
                | {i: 'Queens' for i in range(19, 33)}
                | {i: 'Brooklyn' for i in range(33, 49)}
                | {i: 'Staten Island' for i in range(49, 52)})
    bad = [(int(k), str(v), expected.get(int(k)))
           for k, v in m.set_index('CounDist')['Borough'].items()
           if expected.get(int(k)) is not None
           and str(v).replace(' ', '').lower()
           != expected[int(k)].replace(' ', '').lower()]
    if bad:
        raise SystemExit(
            'council_district_borough_map.csv disagrees with the documented '
            f'district->borough ranges: {bad[:5]}')
    return m


def scenario_assets(models):
    # ---------------- WIDE summary table: one row per model ----------------
    # (Round 11) Columns: Liberal / Conservative / Mixed, each with the mean
    # district-level % change and the number of districts > 0; short headers
    # sized for a 6.5-inch text width.
    rows = []
    for e in models:
        name = e['name']
        cells = [esc(name)]
        for scen in SCEN_LIST:
            d = _load_scenario_by_council(name, scen)
            if d is None:
                cells += ['--', '--']
            else:
                cells += [fmt2(d['mean_pct_change'].mean()),
                          str(int((d['mean_pct_change'] > 0).sum()))]
        rows.append(cells)
    write_tex(
        'table_scenario_summary.tex',
        'Counterfactual scenario summary across all eligible models: '
        'mean district-level percent price change and the number of council '
        'districts with a positive change, per scenario.',
        'tab:scenario_summary',
        ['Model', r'Lib. mean (\%)', r'Lib. \#$>0$',
         r'Cons. mean (\%)', r'Cons. \#$>0$',
         r'Mixed mean (\%)', r'Mixed \#$>0$'], rows,
        note=('Generated from the name-keyed outputs '
              '\\texttt{results/political\\_scenarios/by\\_model/<model>/'
              '<scenario>\\_by\\_council.csv} written by '
              '\\texttt{src/analysis/political\\_scenarios.py}.'))

    # ---------------- Appendix A.1 (51 districts x 3 scenarios) ------------
    frames = []
    for scen in SCEN_LIST:
        d = _load_scenario_by_council(PAPER_MODEL, scen)
        frames.append(d.rename(columns={'mean_pct_change': scen})
                      .set_index('CounDist'))
    merged = pd.concat(frames, axis=1).reset_index()
    bmap = _borough_map()
    if bmap is not None:
        merged = merged.merge(bmap, on='CounDist', how='left')
    else:
        merged['Borough'] = ''
    merged = merged.sort_values('CounDist').reset_index(drop=True)

    def _a1_cells(r):
        return [str(int(r['CounDist'])), esc(r['Borough']),
                fmt2(r['liberal_policy']), fmt2(r['conservative_policy']),
                fmt2(r['mixed_governance'])]

    # Two-block longtable: districts 1-26 (left) | 27-51 (right), 10 columns.
    half = (len(merged) + 1) // 2
    a1_head = (r'District & Borough & Liberal (\%) & Conservative (\%) & '
               r'Mixed gov.\ (\%)')
    lines = [
        '% GENERATED by src/figures/generate_paper_assets.py -- do not edit by hand',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{longtable}{llccc@{\hspace{1.2em}}llccc}',
        ('\\caption{Appendix A.1. District-level mean percent price change '
         'under each counterfactual scenario, computed with '
         f'\\texttt{{{esc(PAPER_MODEL)}}} ({len(merged)} council districts; '
         'two blocks: districts 1--26 \\textbar{} 27--51).}'
         '\\label{tab:appendix_a1} \\\\'),
        r'\toprule',
        a1_head + ' & ' + a1_head + r' \\',
        r'\midrule',
        r'\endfirsthead',
        (r'\multicolumn{10}{l}{\footnotesize\emph{Appendix A.1 (continued)}}'
         r' \\'),
        r'\toprule',
        a1_head + ' & ' + a1_head + r' \\',
        r'\midrule',
        r'\endhead',
        r'\bottomrule',
        r'\endlastfoot',
    ]
    for i in range(half):
        left = _a1_cells(merged.iloc[i])
        j = i + half
        right = _a1_cells(merged.iloc[j]) if j < len(merged) else [''] * 5
        lines.append(' & '.join(left) + ' & ' + ' & '.join(right) + r' \\')
    lines += [
        r'\end{longtable}',
        (r'{\footnotesize District means of property-level percent changes; '
         r'rows outside the training support of the mixed-governance '
         r'\texttt{dem\_share} shock are extrapolation artifacts (see '
         r'\texttt{mixed\_governance\_support\_diag*.json}).}'),
    ]
    path = OUT_DIR / 'appendix_a1_scenarios.tex'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST['outputs'].append(
        {'file': 'docs/paper_assets/appendix_a1_scenarios.tex', 'kind': 'tex'})
    print('wrote', path.relative_to(ROOT))

    # ---------------- Figure 5: 3-panel bar chart -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, scen in zip(axes, SCEN_LIST):
        d = merged.sort_values(scen)
        vals = d[scen]
        colors = ['#c0392b' if v < 0 else '#2c7fb8' for v in vals]
        ax.barh(d['CounDist'].astype(int).astype(str), vals, color=colors,
                alpha=0.9)
        ax.axvline(0, color='0.3', linewidth=0.8)
        ax.set_title(SCEN_TITLES[scen], fontsize=11)
        ax.set_xlabel('Mean % price change')
        ax.tick_params(axis='y', labelsize=6)
    fig.suptitle(f'Political scenario impact by NYC council district — '
                 f'{PAPER_MODEL}', fontsize=14)
    save_fig(fig, 'fig5_scenario_impact.png')


# ---------------------------------------------------------------------------
# Causal identification: IV specification search (table + Figure 6)
# ---------------------------------------------------------------------------
CAUSAL_DIR = ROOT / 'results' / 'causal'


def _fmt_p(p):
    if p is None:
        return '--'
    p = float(p)
    return ('%.3e' % p) if p < 0.001 else ('%.4f' % p)


def iv_assets():
    iv = json.loads((CAUSAL_DIR / 'iv_district_fe_results.json')
                    .read_text(encoding='utf-8'))
    rescue = json.loads((CAUSAL_DIR / 'iv_district_fe_rescue_results.json')
                        .read_text(encoding='utf-8'))

    arms = []
    # NUMERIC sort by bandwidth: the raw key order is lexicographic, which
    # would put 5pp AFTER 20pp ("bw_5pp" > "bw_20pp"). (Round 11)
    bw_keys = sorted((k for k in iv if k.startswith('bw_')),
                     key=lambda k: int(k.split('_')[1].rstrip('pp')))
    for key in bw_keys:
        d = dict(iv[key])
        if d.get('bandwidth_pp') is None:
            d['bandwidth_pp'] = int(key.split('_')[1].rstrip('pp'))
        arms.append(('Main search (CounDist FE)', d))
    deduped_20pp = False
    for arm_key, arm_label in (
            ('rescue_1a_borough_fe_variant', 'Rescue 1a (Borough FE)'),
            ('rescue_1b_wider_bandwidth_district_fe',
             'Rescue 1b (CounDist FE, wider bw)')):
        for d in sorted(rescue.get(arm_key, []),
                        key=lambda x: x.get('bandwidth_pp') or 0):
            if (arm_key == 'rescue_1b_wider_bandwidth_district_fe'
                    and d.get('bandwidth_pp') == 20):
                # The +-20pp district-FE spec duplicates the primary bw_20pp
                # arm (identical design); list it once, footnote the repeat.
                deduped_20pp = True
                continue
            arms.append((arm_label, dict(d)))

    rows = []
    for label, d in arms:
        f1 = d.get('first_stage') or {}
        s2 = d.get('second_stage') or {}
        coef, se = s2.get('dem_share_coef'), s2.get('dem_share_se')
        rows.append([
            esc(label), d.get('bandwidth_pp', '--'), d.get('n_obs', '--'),
            d.get('n_close_cells', d.get('n_instrument_cells', '--')),
            d.get('n_z_cells', d.get('n_instrument_z_cells', '--')),
            d.get('n_districts', '--'),
            fmt(f1.get('f_stat'), 3),
            (fmt(coef, 4) + ' (' + fmt(se, 4) + ')'
             if coef is not None else '--'),
            'yes' if d.get('identified') else r'\textbf{no}',
        ])
    write_tex(
        'table_iv_spec_search.tex',
        'Close-election IV specification search: every recorded bandwidth / '
        'fixed-effect arm with its sample, first-stage $F$ (weak-instrument '
        f"threshold $F > {fmt(iv.get('weak_f_threshold'), 1)}$), the 2SLS "
        'dem\\_share estimate when identified, and the identification verdict.',
        'tab:iv_spec_search',
        ['Design arm', r'Bandwidth (pp)', r'$N$', 'Close cells',
         'Z=1 cells', 'Districts', r'1st-stage $F$',
         r'2SLS $\beta_{\mathrm{dem\_share}}$ (SE)', 'Identified?'], rows,
        note=('From \\texttt{results/causal/iv\\_district\\_fe\\_results.json} '
              'and \\texttt{results/causal/iv\\_district\\_fe\\_rescue\\_results.json}; '
              'every attempted arm is shown, identified or not.'
              + (' The $\\pm$20\\,pp district-FE specification appears in '
                 'both the primary search and the wider-bandwidth rescue arm; '
                 'it is listed once (the primary arm).'
                 if deduped_20pp else '')),
        col_fmt='lcccccccc')

    # ---- Figure 6: HORIZONTAL bars -- arm labels live on the y-axis so they
    # cannot overlap (Round 11; the Round-10 vertical bars with two-line 7pt
    # x-tick labels collided). ----
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.55 * len(arms) + 1.8)))
    labels, fstats, identified = [], [], []
    for label, d in arms:
        f1 = d.get('first_stage') or {}
        labels.append(f"{label} ({d.get('bandwidth_pp')}pp)")
        fstats.append(f1.get('f_stat'))
        identified.append(bool(d.get('identified')))
    ys = np.arange(len(labels))
    xv = [v if v is not None else 0.0 for v in fstats]
    colors = ['#2c7fb8' if ok else '#bdbdbd' for ok in identified]
    ax.barh(ys, xv, color=colors)
    thr = iv.get('weak_f_threshold')
    xmax = max([v for v in fstats if v is not None]
               + ([float(thr)] if thr is not None else [0.0]) + [1.0])
    if thr is not None:
        ax.axvline(float(thr), color='#c0392b', linestyle='--', linewidth=1,
                   label=f'weak-instrument threshold F={float(thr):.0f}')
        ax.legend(loc='lower right', fontsize=8)
    for i, v in enumerate(fstats):
        ax.text(xv[i] + xmax * 0.01, i,
                ('%.2f' % v) if v is not None else 'unidentified',
                va='center', fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.15)
    ax.set_xlabel('First-stage F-statistic')
    ax.set_ylabel('Design arm (bandwidth)')
    ax.set_title('Close-election IV specification search — first-stage strength')
    save_fig(fig, 'fig6_iv_spec_search.png')


# ---------------------------------------------------------------------------
# City of Yes DiD: table + Figure 7
# ---------------------------------------------------------------------------
def did_assets():
    did = json.loads((CAUSAL_DIR / 'city_of_yes_did_results.json')
                     .read_text(encoding='utf-8'))
    ev = did['event_study_pre_trends']
    ref = ev.get('reference_bin')
    pt = ev.get('period_terms', {})
    raw_bins = [k[len('yes_x_'):] for k in pt]
    bins = [b for b in ['2016-17', '2018-19', '2020-21', '2022-23', '2024',
                        '2025'] if b in raw_bins or b == ref]

    rows = []
    for b in bins:
        key = f'yes_x_{b}'
        if b == ref:
            rows.append([esc(b) + ' (reference)', '0', '--', '--'])
        else:
            rows.append([esc(b), fmt(pt[key]['coef'], 4),
                         fmt(pt[key]['se'], 4), _fmt_p(pt[key]['p'])])
    rows.append(['---', '---', '---', '---'])
    main = did['did_main']
    sens = did['did_sensitivity_2024_included']
    rows.append(['DiD main (post=2025; 2024 excluded)', fmt(main['did_coef'], 4),
                 fmt(main['did_se'], 4), _fmt_p(main['did_p'])])
    rows.append(['DiD sensitivity (post={2024, 2025})', fmt(sens['did_coef'], 4),
                 fmt(sens['did_se'], 4), _fmt_p(sens['did_p'])])
    write_tex(
        'table_did_event_study.tex',
        'City of Yes Housing Opportunity difference-in-differences: '
        'event-study period terms (reference: '
        f"{esc(ref)}), joint pre-trend Wald test "
        f"($F={fmt(ev.get('pre_trend_joint_wald_f'), 2)}$, "
        f"$p={_fmt_p(ev.get('pre_trend_joint_wald_p'))}$), and the main / "
        'sensitivity DiD estimates. District FE + sale-year FE + controls; '
        'cluster-robust SE by council district.',
        'tab:did',
        ['Period / estimate', 'Coef (Yes-district effect)', 'SE', r'$p$'], rows,
        note=(f"Event-study $N={int(ev.get('n', 0)):,}$; DiD main "
              f"$N={int(main['n']):,}$, sensitivity $N={int(sens['n']):,}$. "
              'From \\texttt{results/causal/city\\_of\\_yes\\_did\\_results.json}.'),
        col_fmt='lccc')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs, ys, los, his, lbls = [], [], [], [], []
    for i, b in enumerate(bins):
        xs.append(i)
        lbls.append(b)
        if b == ref:
            ys.append(0.0); los.append(0.0); his.append(0.0)
        else:
            c = pt[f'yes_x_{b}']['coef']
            s = pt[f'yes_x_{b}']['se']
            ys.append(c); los.append(c - 1.96 * s); his.append(c + 1.96 * s)
    ax.errorbar(xs, ys,
                yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                fmt='o-', color='#2c7fb8', ecolor='#7fc0db', capsize=3)
    ax.axhline(0, color='0.3', linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(lbls)
    ax.set_ylabel('Yes-district period coefficient (log price)')
    ax.set_title('City of Yes event study — Yes-district price effects')
    ax.text(0.01, 0.02,
            f"pre-trend joint Wald F={ev.get('pre_trend_joint_wald_f'):.2f}, "
            f"p={_fmt_p(ev.get('pre_trend_joint_wald_p'))}; "
            f"main DiD = {main['did_coef']:.4f} (p={_fmt_p(main['did_p'])})",
            transform=ax.transAxes, fontsize=8)
    save_fig(fig, 'fig7_did_event_study.png')


# ---------------------------------------------------------------------------
# Tables 1, 4, 5 (OLS regression, robustness, spatial regression)
# ---------------------------------------------------------------------------
def tables_145():
    # -------- Table 1: OLS regression sequence --------
    reg = pd.read_csv(ROOT / 'results' / 'regression_table.csv')
    rows = [[esc(r['Model']), esc(r['Variable']), fmt(r['Coef'], 4),
             fmt(r['SE'], 4), _fmt_p(r['p']), fmt(r['R2'], 4), int(r['N'])]
            for _, r in reg.iterrows()]
    write_tex(
        'table1_ols_regression.tex',
        'Baseline OLS of log sale price on \\texttt{dem\\_share}: bivariate, '
        'with structural controls, and adding borough fixed effects.',
        'tab:ols',
        ['Model', 'Variable', 'Coef', 'SE', r'$p$', r'$R^2$', r'$N$'], rows,
        note=('From \\texttt{results/regression\\_table.csv} (freshly written '
              'by \\texttt{src/analysis/generate\\_regression\\_table.py}).'))

    # -------- Table 4: robustness checks --------
    rb = json.loads((ROOT / 'results' / 'robustness_checks.json')
                    .read_text(encoding='utf-8'))
    rob_keys = [
        'model3_classical_se',
        'check1_model3_clustered_by_council_district',
        'check2_model4_council_district_fe_classical_se',
        'check2_model4_council_district_fe_clustered_se',
        'check3_model5_borough_and_year_fe',
        'check3_model5_borough_and_year_fe_clustered_se',
        'check4a_winsorized_1_99pct',
        'check4b_model3_cooks_trimmed',
    ]
    rows = []
    for k in rob_keys:
        s = rb.get(k)
        if not isinstance(s, dict) or 'dem_share_coef' not in s:
            rows.append([esc(k), 'MISSING', '--', '--', '--', '--'])
            continue
        rows.append([esc(s.get('label', k)), fmt(s.get('dem_share_coef'), 4),
                     fmt(s.get('dem_share_se'), 4), _fmt_p(s.get('dem_share_p')),
                     fmt(s.get('r_squared'), 4), int(s.get('n'))])
    n_dropped = rb.get('check4b_n_dropped')
    write_tex(
        'table4_robustness.tex',
        'Robustness of the \\texttt{dem\\_share} coefficient across alternative '
        'specifications (clustered SE by council district; district and '
        "sale-year fixed effects; winsorized and Cook's-D-trimmed samples).",
        'tab:robustness',
        ['Specification', r'$\beta_{\mathrm{dem\_share}}$', 'SE', r'$p$',
         r'$R^2$', r'$N$'], rows,
        note=('From \\texttt{results/robustness\\_checks.json} '
              '(\\texttt{src/analysis/generate\\_robustness\\_checks.py}).'
              + (f" Cook's-D trim dropped {int(n_dropped):,} rows."
                 if n_dropped is not None else '')))

    # -------- Table 5: district-level spatial regression --------
    sp = json.loads((ROOT / 'results' / 'spatial_regression.json')
                    .read_text(encoding='utf-8'))
    specs = [('ols_anchor', 'OLS'), ('spatial_lag_model', 'Spatial lag'),
             ('spatial_error_model', 'Spatial error')]
    var_order = ['dem_share', 'mean_numfloors', 'mean_yearbuilt', 'mean_commfar']

    def cell(spec_key, var):
        c = sp.get(spec_key, {}).get('coefficients', {}).get(var)
        if c is None:
            return '--'
        return fmt(c['coef'], 4) + ' (' + fmt(c['se'], 4) + ')'

    rows = []
    for var in var_order:
        rows.append([esc(var)] + [cell(k, var) for k, _ in specs])
    rows.append([r'$\rho$ (spatial lag)', '--', cell('spatial_lag_model',
                                                      'W_median_log_price'),
                 '--'])
    rows.append([r'$\lambda$ (spatial error)', '--', '--',
                 cell('spatial_error_model', 'lambda')])
    rows.append(['pseudo-' + r'$R^2$',
                 fmt(sp['ols_anchor'].get('pr2'), 3),
                 fmt(sp['spatial_lag_model'].get('pr2'), 3),
                 fmt(sp['spatial_error_model'].get('pr2'), 3)])
    mi = sp.get('ols_residual_morans_i', {})
    write_tex(
        'table5_spatial_regression.tex',
        'District-level ('
        f"$N={sp.get('n_districts', 51)}$) spatial regression: median log "
        'price on mean \\texttt{dem\\_share} with structural controls.',
        'tab:spatial',
        ['Coefficient (SE in parentheses)', 'OLS', 'Spatial Lag (ML)',
         'Spatial Error (ML)'], rows,
        note=(f"OLS residual Moran's I = {fmt(mi.get('I'), 3)} "
              f"(z = {fmt(mi.get('z'), 2)})."
              ' From \\texttt{results/spatial\\_regression.json} (written by '
              '\\texttt{src/analysis/spatial\\_regression\\_model.py}).'),
        col_fmt='lccc')


# ---------------------------------------------------------------------------
# Standalone-compile smoke check (Round 11): every generated .tex must compile
# with ONLY booktabs, amsmath, float, longtable loaded.
# ---------------------------------------------------------------------------
_STANDALONE_PREAMBLE = (
    '\\documentclass{article}\n'
    '\\usepackage{booktabs}\n'
    '\\usepackage{amsmath}\n'
    '\\usepackage{float}\n'
    '\\usepackage{longtable}\n'
    '\\begin{document}\n')
_STANDALONE_FOOT = '\\end{document}\n'


def check_tables_compile():
    """Wrap each generated .tex in a tiny standalone document and run pdflatex
    on it. SKIPS with a clear message when pdflatex is not on PATH; any
    compile error fails the stage loudly. (Round 11)"""
    import shutil
    import subprocess
    import tempfile
    tex_files = [OUT_DIR / o['file'].split('/', 1)[-1]
                 for o in MANIFEST['outputs'] if o.get('kind') == 'tex']
    pdflatex = shutil.which('pdflatex')
    if pdflatex is None:
        print('NOTE: pdflatex NOT found on PATH -- standalone-compile check '
              f'SKIPPED for {len(tex_files)} .tex files. Install a TeX '
              'distribution (MiKTeX / TeX Live) to enable this check; the '
              'generated tables are still written.')
        return
    failures = []
    with tempfile.TemporaryDirectory(prefix='paper_assets_texcheck_') as tmp:
        tmp = Path(tmp)
        for tf in tex_files:
            body = tf.read_text(encoding='utf-8')
            (tmp / 'check.tex').write_text(
                _STANDALONE_PREAMBLE + body + _STANDALONE_FOOT,
                encoding='utf-8')
            proc = subprocess.run(
                [pdflatex, '-interaction=nonstopmode', '-halt-on-error',
                 'check.tex'],
                cwd=tmp, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                log_tail = '\n'.join((proc.stdout or '').splitlines()[-25:])
                failures.append((tf.name, log_tail))
    if failures:
        for name, tail in failures:
            print(f'!!! pdflatex FAILED for {name}')
            print(tail)
        raise SystemExit(
            'STANDALONE-COMPILE CHECK FAILED for: '
            + ', '.join(n for n, _ in failures))
    print(f'Standalone-compile check: all {len(tex_files)} .tex files compile '
          '(booktabs + amsmath + float + longtable only).')


# ---------------------------------------------------------------------------
def main():
    from src.ops.champion_ranking import get_all_models
    models = get_all_models(write_audit=False)
    print('Eligible models (name-resolved):',
          ', '.join(f"{e['name']} (rank {e['rank']})" for e in models))

    table2_and_fig3(models)
    table3_fig4_and_crossmodel(models)
    scenario_assets(models)
    iv_assets()
    did_assets()
    tables_145()

    MANIFEST['models'] = [{'name': e['name'],
                           'experiment_id': e['experiment_id'],
                           'family': e.get('family'),
                           'test_mae': e.get('test_mae')} for e in models]
    mpath = OUT_DIR / 'manifest.json'
    mpath.write_text(json.dumps(MANIFEST, indent=2), encoding='utf-8')
    print('wrote', mpath.relative_to(ROOT))
    check_tables_compile()  # Round 11: fail loudly if any .tex is broken
    print('Paper assets complete:', len(MANIFEST['outputs']), 'files in',
          OUT_DIR.relative_to(ROOT))


if __name__ == '__main__':
    main()





