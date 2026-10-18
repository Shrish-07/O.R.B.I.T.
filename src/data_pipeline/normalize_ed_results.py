"""Normalize every raw NYC BOE ED-level CSV into one tidy parquet (Task J fix).

Round 12 rewrite. The previous version used a plain pd.read_csv, which
(a) produced junk columns on the 2021/2025 mayoral (and 2020/2024
presidential) files whose header string is prepended to EVERY row, and
therefore (b) silently ended up holding only the 2017 mayoral rows. Nothing
consumed that broken output, but it is now correct by construction:

  * parsing re-uses the line-oriented reader shared with
    src/data_pipeline/build_ideology_scores_v2.py
    (src/data_pipeline/boe_ed_reader.py), which handles BOTH raw layouts
    (clean 2016/2017 files and per-row-header 2020/2021/2024/2025 files)
    exactly the way the production ideology builder does;
  * every file matching ed_results_<year>_<office>.csv is parsed (mayoral
    2017/2021/2025 AND presidential 2016/2020/2024) with a `year` and
    `office` column;
  * each row carries the ballot-line classification (Dem / Rep / Other /
    NonCandidate) used by the ideology-line audit and the v2 ideology
    builder;
  * a reconciliation gate compares this stage's per-(AD,ED) Dem/Rep totals
    for the mayoral years against build_ideology_scores.parse_year_file()
    (the production reader) and refuses to write on any mismatch.

Output: data/processed/ed_results_clean.parquet

Run: python src/data_pipeline/normalize_ed_results.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boe_ed_reader import read_boe_dataframe, iter_boe_rows, STANDARD_HEADER  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / 'data' / 'raw' / 'election_results'
OUT = ROOT / 'data' / 'processed' / 'ed_results_clean.parquet'
LEGACY_BUILDER = ROOT / 'src' / 'data_pipeline' / 'build_ideology_scores.py'

def main():
    files = sorted(RAW_DIR.glob('ed_results_*.csv'))
    if not files:
        raise SystemExit(f'No ed_results_*.csv files found in {RAW_DIR}')
    frames = []
    for path in files:
        m = re.match(r'ed_results_(\d{4})_([a-z]+)\.csv$', path.name)
        if not m:
            print(f'  skipping unrecognized file: {path.name}')
            continue
        year, office = int(m.group(1)), m.group(2)
        df = read_boe_dataframe(path, year)
        extra = pd.DataFrame([
            {h: rec[h] for h in STANDARD_HEADER
             if h not in ('AD', 'ED', 'Tally', 'Unit Name', 'County',
                          'EDAD Status')}
            for rec in iter_boe_rows(path)])
        df = pd.concat([df.reset_index(drop=True),
                        extra.reset_index(drop=True)], axis=1)
        df['office'] = office
        frames.append(df)
        print(f'  parsed {path.name}: {len(df):,} rows')
    if not frames:
        raise SystemExit('No files matched ed_results_<year>_<office>.csv')
    final = pd.concat(frames, ignore_index=True)

    # --- reconciliation with the production ideology builder -------------
    import importlib.util
    spec = importlib.util.spec_from_file_location('legacy_builder', LEGACY_BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mismatches = 0
    for year in [2017, 2021, 2025]:
        legacy = mod.parse_year_file(year, RAW_DIR / f'ed_results_{year}_mayor.csv')
        mine = (final[(final['year'] == year) & (final['office'] == 'mayor')
                      & (final['class'].isin(['Dem', 'Rep']))]
                .groupby(['AD', 'ED', 'class'])['Tally'].sum().unstack(fill_value=0))
        for (ad, ed), r in mine.iterrows():
            lv = legacy.get((int(ad), int(ed)), {'dem': 0, 'rep': 0})
            if (abs(r.get('Dem', 0) - lv['dem']) > 0.5
                    or abs(r.get('Rep', 0) - lv['rep']) > 0.5):
                mismatches += 1
        print(f'  reconcile {year} mayoral: {len(mine)} ED rows checked '
              'against production builder')
    if mismatches:
        raise SystemExit(
            f'RECONCILIATION FAILED: {mismatches} (AD,ED) Dem/Rep totals '
            'differ from build_ideology_scores.parse_year_file; refusing '
            'to write ed_results_clean.parquet.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT, index=False)
    print('ed_results_clean.parquet saved')
    print('Rows:', len(final))
    print('Years:', sorted(final['year'].unique()))
    print('Offices:', sorted(final['office'].unique()))


if __name__ == '__main__':
    main()

