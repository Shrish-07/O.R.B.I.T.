"""Build ideology_by_council.parquet from raw ED CSVs (line-by-line).

Usage: .venv\Scripts\python.exe src/build_ideology_scores.py

This script reads raw ED CSVs for years [2017,2021,2025] line-by-line,
skips any repeated-header lines where the first field equals 'AD',
extracts AD, ED, Unit Name, and Tally from valid lines, aggregates
dem_total and rep_total per (AD,ED) per year, computes ElectDist_num = AD*1000+ED,
joins to the precomputed ED->Council crosswalk and aggregates to council-level
dem_share per election_year, and writes
`data/processed/ideology_by_council.parquet`.

This file intentionally uses a simple, line-oriented parser and strict
skipping of rows whose first field equals 'AD' (embedded headers), as
requested.
"""
import csv
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw' / 'election_results'
CROSSWALK = ROOT / 'data' / 'processed' / 'ed_to_council_crosswalk.parquet'
OUT_DIR = ROOT / 'data' / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / 'ideology_by_council.parquet'

YEARS = [2017, 2021, 2025]


def _int_from_token(tok):
    if tok is None:
        return None
    s = str(tok)
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_year_file(year, path):
    """Parse the CSV line-by-line and return dict keyed by (AD,ED)->(dem,rep,total).

    Behavior:
    - Skip any line where the first field (after stripping) equals 'AD'.
    - If a header row is present (first encountered row containing 'AD' and 'ED'),
      record it to find indices for 'AD','ED','Unit Name','Tally' for aligned rows.
    - For valid data rows, attempt to extract AD and ED from the first two
      fields (fallback to other positions if not parseable). If AD/ED not
      parseable as integers, skip the row.
    - Unit Name and Tally are extracted using header indices when available,
      otherwise using heuristics (last columns for tally, near-end for Unit Name).
    """
    counts = defaultdict(lambda: {'dem': 0, 'rep': 0, 'total': 0})
    path = Path(path)
    if not path.exists():
        print(f"Missing file for {year}: {path}")
        return counts

    header_idxs = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            first = row[0].strip().strip('"')
            # detect header template (first row that includes AD and ED tokens)
            if not header_idxs and any(str(c).strip().upper() == 'AD' for c in row) and any(str(c).strip().upper() == 'ED' for c in row):
                hdr = [str(c).strip().strip('"') for c in row]
                # find indices if present
                for key in ['AD', 'ED', 'Unit Name', 'Tally']:
                    for i, h in enumerate(hdr):
                        if h.strip().upper() == key.upper():
                            header_idxs[key] = i
                            break
                # continue after capturing header
                continue

            # If row starts with AD, it may be a repeated header row OR a per-row header
            # where the actual numeric values are later in the row. Handle both cases.
            if str(first).upper() == 'AD':
                # find first numeric token index in the row
                first_num_idx = None
                for i, tok in enumerate(row):
                    if tok is None:
                        continue
                    t = str(tok).strip()
                    if re.match(r"^\d+$", t):
                        first_num_idx = i
                        break
                if first_num_idx is None:
                    # pure header-only row -> skip
                    continue
                # row contains header tokens then values (per-row header). Map header->value.
                header_part = [str(x).strip().strip('"') for x in row[:first_num_idx]]
                values_part = row[first_num_idx:first_num_idx + len(header_part)]
                if len(values_part) < len(header_part):
                    values_part = values_part + [''] * (len(header_part) - len(values_part))
                mapping = dict(zip(header_part, values_part))
                # extract using mapping
                ad_raw = mapping.get('AD', '')
                ed_raw = mapping.get('ED', '')
                unit = mapping.get('Unit Name', '') or mapping.get('Unit', '')
                tally_raw = mapping.get('Tally', '')
                # parse numbers
                ad = _int_from_token(ad_raw)
                ed = _int_from_token(ed_raw)
                try:
                    tally_val = int(re.sub(r"[^0-9]", "", str(tally_raw))) if str(tally_raw).strip() != '' else 0
                except Exception:
                    # fallback: find last numeric
                    tally_val = 0
                    for cand in reversed(values_part):
                        s = re.sub(r"[^0-9]", "", str(cand))
                        if s != '':
                            try:
                                tally_val = int(s)
                                break
                            except Exception:
                                continue
                if ad is None or ed is None:
                    # try scanning values_part for first two numbers
                    nums = []
                    for tok in values_part:
                        m = re.search(r"(\d+)", str(tok))
                        if m:
                            nums.append(int(m.group(1)))
                        if len(nums) >= 2:
                            break
                    if len(nums) >= 2:
                        ad, ed = nums[0], nums[1]
                if ad is None or ed is None:
                    continue
                lower = str(unit).strip().lower()
                key = (int(ad), int(ed))
                if '(democratic)' in lower or 'democratic' in lower or 'working families' in lower:
                    counts[key]['dem'] += tally_val
                if '(republican)' in lower or 'republican' in lower or 'conservative' in lower:
                    counts[key]['rep'] += tally_val
                counts[key]['total'] += tally_val
                continue

            # For normal rows that don't start with AD:
            ad = _int_from_token(row[0]) if len(row) >= 1 else None
            ed = _int_from_token(row[1]) if len(row) >= 2 else None

            # if header indices are known and appear to align, use them for unit/tally
            unit = None
            tally = None
            if header_idxs:
                if 'Unit Name' in header_idxs and header_idxs['Unit Name'] < len(row):
                    unit = row[header_idxs['Unit Name']]
                if 'Tally' in header_idxs and header_idxs['Tally'] < len(row):
                    tally = row[header_idxs['Tally']]

            # fallback heuristics
            if unit is None:
                candidates = [row[i] for i in range(max(0, len(row)-6), len(row)) if i < len(row)]
                unit = ''
                for cand in candidates[::-1]:
                    if cand and not re.match(r"^\s*\d+[\.,]*\d*\s*$", cand):
                        unit = cand
                        break

            if tally is None:
                tally = ''
                for cand in reversed(row):
                    s = re.sub(r"[^0-9]", "", str(cand))
                    if s != '':
                        tally = s
                        break

            if ad is None or ed is None:
                nums = []
                for tok in row:
                    m = re.search(r"(\d+)", str(tok))
                    if m:
                        nums.append(int(m.group(1)))
                    if len(nums) >= 2:
                        break
                if len(nums) >= 2:
                    ad, ed = nums[0], nums[1]

            if ad is None or ed is None:
                continue

            try:
                tally_val = int(re.sub(r"[^0-9]", "", str(tally))) if str(tally).strip() != '' else 0
            except Exception:
                try:
                    tally_val = int(re.sub(r"[^0-9]", "", str(row[-1])))
                except Exception:
                    continue

            unit_norm = str(unit).strip()
            lower = unit_norm.lower()
            key = (int(ad), int(ed))
            if '(democratic)' in lower or 'democratic' in lower or 'working families' in lower:
                counts[key]['dem'] += tally_val
            if '(republican)' in lower or 'republican' in lower or 'conservative' in lower:
                counts[key]['rep'] += tally_val
            counts[key]['total'] += tally_val

    return counts


def build_ideology_by_council():
    # Parse all years into ED-level aggregates
    ed_agg = []  # list of dicts: year, AD, ED, dem, rep, total
    for year in YEARS:
        p = RAW_DIR / f'ed_results_{year}_mayor.csv'
        print(f'Processing election results for {year}: {p}')
        year_counts = parse_year_file(year, p)
        for (ad, ed), vals in year_counts.items():
            ed_agg.append({'year': year, 'AD': int(ad), 'ED': int(ed), 'dem': float(vals['dem']), 'rep': float(vals['rep']), 'total_votes': float(vals['total'])})

    if not ed_agg:
        raise RuntimeError('No ED-level rows parsed from raw files')

    ed_df = pd.DataFrame(ed_agg)
    # compute ElectDist_num
    ed_df['ElectDist'] = ed_df['AD'].astype(int) * 1000 + ed_df['ED'].astype(int)

    # load crosswalk
    if not CROSSWALK.exists():
        raise FileNotFoundError(f'Missing ED->Council crosswalk: {CROSSWALK}')
    cw = pd.read_parquet(CROSSWALK)

    # robust: try to find an ElectDist-like column
    elect_col = None
    for c in cw.columns:
        if 'elect' in c.lower() or 'ed' in c.lower() and 'code' in c.lower():
            elect_col = c
            break
    if elect_col is None and 'ElectDist' in cw.columns:
        elect_col = 'ElectDist'

    if elect_col is not None:
        cw['ElectDist_num'] = pd.to_numeric(cw[elect_col], errors='coerce')
    else:
        # try AD/ED in crosswalk
        if 'AD' in cw.columns and 'ED' in cw.columns:
            cw['ElectDist_num'] = cw['AD'].astype(int) * 1000 + cw['ED'].astype(int)
        else:
            # fallback: try index
            try:
                cw['ElectDist_num'] = pd.to_numeric(cw.index.astype(str), errors='coerce')
            except Exception:
                cw['ElectDist_num'] = pd.Series(dtype='float')

    # find council column
    council_col = None
    for c in cw.columns:
        if 'coun' in c.lower() or 'council' in c.lower():
            council_col = c
            break
    if council_col is None:
        # try common names
        for cand in ['CounDist', 'council_district', 'council district']:
            if cand in cw.columns:
                council_col = cand
                break

    if council_col is None:
        raise KeyError('Could not find council district column in crosswalk')

    # ensure ElectDist_num numeric on both sides
    ed_df['ElectDist_num'] = pd.to_numeric(ed_df['ElectDist'], errors='coerce')
    cw['ElectDist_num'] = pd.to_numeric(cw['ElectDist_num'], errors='coerce')

    merged = ed_df.merge(cw[[council_col, 'ElectDist_num']], how='left', left_on='ElectDist_num', right_on='ElectDist_num')
    merged = merged.rename(columns={council_col: 'CounDist', 'year': 'election_year'})

    # drop rows without a council mapping
    merged = merged.dropna(subset=['CounDist'])
    # coerce council to integer where possible
    try:
        merged['CounDist'] = pd.to_numeric(merged['CounDist'], errors='coerce').astype('Int64')
    except Exception:
        pass

    # aggregate to council + year
    agg = merged.groupby(['CounDist', 'election_year']).agg({'dem': 'sum', 'rep': 'sum', 'total_votes': 'sum'}).reset_index()
    agg['dem_share'] = agg.apply(lambda r: (r['dem'] / (r['dem'] + r['rep'])) if (r['dem'] + r['rep']) > 0 else float('nan'), axis=1)

    # final format
    out = agg[['CounDist', 'election_year', 'dem_share']].copy()
    out.to_parquet(OUT_PATH, index=False)
    print(f'Saved ideology_by_council to {OUT_PATH}')


if __name__ == '__main__':
    print('Building ED-level ideology summaries by aggregating raw ED CSVs...')
    build_ideology_by_council()
