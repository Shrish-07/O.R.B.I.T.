#!/usr/bin/env python3
"""verify_manuscript_tables.py -- makes the (once-off, hand-made) check of the
hand-typed manuscript permanent.

The manuscript source (main.tex) is NOT in this repository: it is gitignored
because the repository is public while the paper is under double-anonymous
review. This verifier therefore reads the manuscript from

    $MANUSCRIPT_TEX            (explicit path, wins if set)
    manuscript/main.tex        (default location, used when it exists locally)

and reports a clear SKIPPED status when neither is available, so CI and fresh
clones do not fail.

When the manuscript IS present:
  * every tabular fragment produced by src/figures/generate_manuscript_assets.py
    (docs/manuscript_assets/tables/*.tex) is matched to the manuscript table
    carrying the same \\label and compared CELL BY CELL -- numeric cells at half
    a unit of the printed decimal, text cells after normalisation;
  * a list of key in-text numbers (abstract/intro estimates, sample sizes,
    520,365 / 355,480 / 469,697, the 31 Yes / 20 No roll call, the 6.3/4.4/42.3
    mayoral other-line shares, the 17.8/23.7/24.1/23.8/0.5 crosswalk losses,
    Moran's I and the model MAEs) is checked for presence against the values
    recomputed here from the repository artifacts.
Every result is recorded through src/verification/_shared.record_status(); any
mismatch (a differing cell, a missing table, a missing prose number) fails.

Run: python src/verification/verify_manuscript_tables.py
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ROOT, record_status, check_val  # noqa: E402

TABLES_DIR = ROOT / 'docs' / 'manuscript_assets' / 'tables'
CHECKS = []


def _json(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


# --------------------------------------------------------------------------
# Normalisation / parsing
# --------------------------------------------------------------------------
_DROP = re.compile(r'\\(?:hline|midrule|toprule|bottomrule|cmidrule(?:\(lr\))?'
                   r'\{[^}]*\}|addlinespace|smallskip|medskip|bigskip|quad|,|;|!)')


def norm(cell):
    """Loose equality key for a LaTeX cell: strip rules/spacing/braces of
    presentation, collapse whitespace, normalise the minus and the percent."""
    s = cell
    s = _DROP.sub(' ', s)
    s = s.replace('\\%', '%').replace('\\_', '_').replace('\\&', '&')
    s = s.replace('$-$', '-').replace('$+', '+')
    s = re.sub(r'\$\^\{(\*+)\}\$?', r'\1', s)  # $^{*}$ / $^{*} -> * (kept)
    s = re.sub(r'\^\{(\*+)\}', r'\1', s)        # ^{***} -> *** (kept)
    s = s.replace('$', '').replace('{', '').replace('}', '')
    s = s.replace('--', '-').replace('~', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def as_number(cell):
    """Float value of a cell if it is numeric (optionally signed, optionally
    with stars / percent), else None."""
    s = norm(cell)
    s = s.replace('%', '').replace('*', '').strip()
    s = s.replace(',', '')
    if s in ('', '-', '+'):
        return None
    m = re.fullmatch(r'[+-]?(?:\d+\.?\d*|\.\d+)', s)
    return float(s) if m else None


def decimals(cell):
    s = norm(cell).replace('%', '').replace('*', '').strip()
    return len(s.split('.')[1]) if '.' in s else 0


_MC_RE = re.compile(r'\\multicolumn\{(\d+)\}\{(?:[^{}]|\{[^{}]*\})*\}\{')
_TABULAR_OPEN = r'\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}'


def _expand_cell(cell):
    """Expand \\multicolumn{n}{spec}{text} into n cells: `text` in the first,
    empty strings for the remaining n-1.  Other cells pass through."""
    out, i = [], 0
    for m in _MC_RE.finditer(cell):
        if cell[i:m.start()].strip():
            out.append(cell[i:m.start()].strip())
        n = int(m.group(1))
        j = m.end() - 1            # index of the '{' opening the text argument
        depth, k = 0, j
        while k < len(cell):
            if cell[k] == '{':
                depth += 1
            elif cell[k] == '}':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        out.append(cell[j + 1:k].strip())
        out.extend([''] * (n - 1))
        i = k + 1
    if cell[i:].strip():
        out.append(cell[i:].strip())
    return out if out else [cell.strip()]


def _drop_spacer_columns(rows):
    """Remove columns that are empty in EVERY row (pure spacer columns)."""
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    keep = [c for c in range(width)
            if any(c < len(r) and r[c].strip() for r in rows)]
    return [[r[c] for c in keep if c < len(r)] for r in rows]


def split_rows(body):
    """Data rows of a tabular body -> list of lists of cell strings.

    Rule-only rows (\\toprule / \\midrule / \\bottomrule / \\cmidrule ...) are
    skipped, so row counts compare DATA rows only; \\multicolumn{n}{spec}{text}
    expands to n cells; all-empty spacer columns are dropped."""
    rows = []
    for raw in body.split('\\\\'):
        line = raw.strip()
        if not line:
            continue
        line = re.sub(_TABULAR_OPEN, ' ', line)          # glued \begin{tabular}
        line = re.sub(r'^\[[^\]]*\]\s*', '', line)       # leading [3pt] of prev row
        line = re.sub(r'\\\[[^\]]*\]\s*$', '', line).strip()  # trailing [3pt]
        line = _DROP.sub(' ', line).strip()              # rule commands
        if not line or line.startswith('\\end'):
            continue                                     # rule-only row
        cells = []
        for part in line.split('&'):
            cells.extend(_expand_cell(part.strip()))
        rows.append(cells)
    return _drop_spacer_columns(rows)


TABLE_ENV = re.compile(r'\\begin\{(table\*?|longtable\*?)\}(.*?)'
                       r'\\end\{\1\}', re.S)
LABEL_RE = re.compile(r'\\label\{([^}]*)\}')
CAPTION_RE = re.compile(r'\\caption(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}')
TABULAR_RE = re.compile(_TABULAR_OPEN + r'(.*?)\\end\{tabular\}', re.S)


def parse_manuscript(text):
    """label -> {'caption', 'rows'} for every table-like environment."""
    out = {}
    for m in TABLE_ENV.finditer(text):
        block = m.group(2)
        lab = LABEL_RE.search(block)
        cap = CAPTION_RE.search(block)
        tbody = TABULAR_RE.search(block)
        if tbody is None:
            continue
        rows = split_rows(tbody.group(1))
        key = lab.group(1) if lab else None
        out[key] = {'caption': norm(cap.group(1)) if cap else '',
                    'rows': rows}
    return out
# --------------------------------------------------------------------------
# Table-by-table comparison
# --------------------------------------------------------------------------
def fragment_label(path):
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('% manuscript table:'):
            return line.split(':', 1)[1].strip()
    return None


def fragment_rows(path):
    text = path.read_text(encoding='utf-8')
    body = TABULAR_RE.search(text)
    return split_rows(body.group(1))


def cell_equal(frag_cell, ms_cell):
    # significance stars are part of the cell text: they must match exactly
    if norm(frag_cell).count('*') != norm(ms_cell).count('*'):
        return False
    fa, ma = as_number(frag_cell), as_number(ms_cell)
    if fa is not None and ma is not None:
        tol = 0.5 * 10 ** (-min(decimals(frag_cell), decimals(ms_cell)))
        return abs(fa - ma) <= tol
    return norm(frag_cell) == norm(ms_cell)


def compare_fragments(ms_tables):
    frags = sorted(TABLES_DIR.glob('*.tex'))
    if not frags:
        CHECKS.append(check_val('fragments_present', True, False))
        print('  !! no fragments in', TABLES_DIR.relative_to(ROOT))
        return
    for path in frags:
        label = fragment_label(path)
        name = path.stem
        if label is None or label not in ms_tables:
            print(f'  {name}: label {label!r} NOT found in the manuscript')
            CHECKS.append(check_val(f'{name}:table_found_in_manuscript',
                                    True, False))
            continue
        frows, mrows = fragment_rows(path), ms_tables[label]['rows']
        CHECKS.append(check_val(f'{name}:row_count', len(frows), len(mrows)))
        n_bad = 0
        for ri in range(min(len(frows), len(mrows))):
            for ci in range(min(len(frows[ri]), len(mrows[ri]))):
                if not cell_equal(frows[ri][ci], mrows[ri][ci]):
                    n_bad += 1
                    CHECKS.append(check_val(
                        f'{name}:r{ri + 1}c{ci + 1}',
                        norm(frows[ri][ci]), norm(mrows[ri][ci])))
        print(f'  {name} [{label}]: {len(frows)}x'
              f'{max(len(r) for r in frows)} cells vs {len(mrows)} manuscript '
              f'rows -> {n_bad} differing cell(s)')


# --------------------------------------------------------------------------
# Key in-text numbers
# --------------------------------------------------------------------------
def flat_text(text):
    """Manuscript prose with presentation stripped but digits/commas kept."""
    s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', text)
    s = s.replace('{', ' ').replace('}', ' ')
    s = s.replace('$', '').replace('~', ' ')
    return re.sub(r'\s+', ' ', s)


def prose_checks(text):
    flat = flat_text(text)
    rob3 = _json('results/robustness_checks_v3.json')
    steps = {s['step']: s for s in _json('results/sample_funnel.json')['funnel']}
    import pandas as pd  # local import: only needed when the manuscript exists
    roll = pd.read_parquet(ROOT / 'data' / 'processed' /
                           'city_of_yes_roll_calls.parquet')
    n_yes = int((roll['member_vote'] == 'Yes').sum())
    n_no = int((roll['member_vote'] == 'No').sum())
    line_audit = _json('results/ideology_line_audit.json')
    xwalk = _json('results/crosswalk_audit.json')
    mor3 = _json('results/morans_i_results_v3.json')
    rank = _json('results/rank1_rank2_selection.json')['ranked_eligible_models']

    items = [
        ('prose:n_520365', ['520,365'], 'full analysis sample'),
        ('prose:n_355480', [f"{rob3['full']['dem_share_vintage_aw']
                              ['model6_district_year_fe_units']['n']:,}"],
         'district+year FE (unit counts required) sample'),
        ('prose:n_469697', [f"{steps['restricted_analysis_sample']['rows']:,}"],
         'restricted analysis sample'),
        ('prose:mayoral_other_lines_6.3_4.4_42.3',
         [f"{100 * line_audit['by_year'][y]['other_share_of_candidate_votes']:.1f}"
          for y in ('2017', '2021', '2025')],
         'mayoral other-line share of candidate votes'),
        ('prose:crosswalk_losses_17.8_23.7_24.1_23.8_0.5',
         [f"{xwalk['by_year'][y]['missing_key_share_pct']:.1f}"
          for y in ('2016', '2017', '2020', '2021', '2024')
          if y in xwalk['by_year']],
         'ED candidate votes lost to pre-redistricting keys'),
        ('prose:morans_i_price_0.594',
         [f"{mor3['median_log_price']['I']:.3f}"], "Moran's I, log price"),
        ('prose:morans_i_dem_share_aw_0.412',
         [f"{mor3['by_measure']['dem_share_vintage_aw']['moran_i']:.3f}"],
         "Moran's I, dem_share_vintage_aw"),
        ('prose:mae_rank1_to_rank7',
         [f"{r['test_mae']:.4f}" for r in rank], 'held-out MAE per ranked model'),
    ]
    for name, needles, desc in items:
        found = [n for n in needles if n in flat]
        CHECKS.append(check_val(name, True, len(found) == len(needles)))
        tail = '' if len(found) == len(needles) else (
            '  MISSING ' + str(sorted(set(needles) - set(found))))
        print(f'  {name}: {len(found)}/{len(needles)} found ({desc}){tail}')

    # Roll call: any one of the accepted phrasings counts.
    roll_alts = [
        [f'{n_yes} Yes', f'{n_no} No'],
        [f'{n_yes} to {n_no}'],
        [f'{n_yes} members; {n_no} voted No'],
    ]
    hit = next((alt for alt in roll_alts if all(n in flat for n in alt)), None)
    CHECKS.append(check_val('prose:rollcall_31_yes_20_no', True,
                            hit is not None))
    print(f'  prose:rollcall_31_yes_20_no: matched {hit!r} '
          f'(City of Yes roll call; accepted: {roll_alts})')


def find_manuscript():
    env = os.environ.get('MANUSCRIPT_TEX')
    if env:
        p = Path(env)
        return p, (p if p.exists() else None)
    default = ROOT / 'manuscript' / 'main.tex'
    return default, (default if default.exists() else None)


def main():
    tried, path = find_manuscript()
    if path is None:
        note = (f'SKIPPED: manuscript LaTeX not available (MANUSCRIPT_TEX='
                f'{os.environ.get("MANUSCRIPT_TEX")!r}, default tried: {tried}). '
                'The manuscript source is gitignored -- the repository is public '
                'and the paper is under double-anonymous review -- so it is not '
                'present on a fresh clone or in CI. The generated '
                'fragments in docs/manuscript_assets/tables/ are the record of '
                'what the manuscript was checked against; re-run this stage with '
                'MANUSCRIPT_TEX set to re-verify the manuscript itself.')
        print('=' * 78)
        print('MANUSCRIPT TABLE VERIFICATION')
        print('=' * 78)
        print(note)
        CHECKS.append(check_val('manuscript_available', 'SKIPPED', 'SKIPPED'))
        record_status('verify_manuscript_tables', CHECKS, note=note)
        return 0

    text = path.read_text(encoding='utf-8', errors='ignore')
    print('=' * 78)
    print(f'MANUSCRIPT TABLE VERIFICATION  ({path})')
    print('=' * 78)
    ms_tables = parse_manuscript(text)
    print(f'parsed {len(ms_tables)} labelled table(s) from the manuscript')
    print('--- generated fragment vs manuscript, cell by cell')
    compare_fragments(ms_tables)
    print('--- key in-text numbers')
    prose_checks(text)
    record_status('verify_manuscript_tables', CHECKS,
                  note=f'compared against {path}')
    return 1 if any(not c['match'] for c in CHECKS) else 0


if __name__ == '__main__':
    sys.exit(main())


