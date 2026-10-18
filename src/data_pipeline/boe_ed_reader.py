"""Shared line-oriented reader for NYC BOE ED-level election-result CSVs.

The Board of Elections posts its "Statement and Return Report by Election
District" files in two layouts:

  * CLEAN layout (2016 and 2017 raw files): a single header line, then
    11-field data rows ("AD","ED","County","EDAD Status","Event",
    "Party/Independent Body","Office/Position Title","District Key",
    "VoteFor","Unit Name","Tally").
  * PER-ROW-HEADER layout (2021/2025 mayoral and 2020/2024 presidential
    files): the 11 header names are PREPENDED TO EVERY ROW, so each physical
    row is 22 fields (11 header tokens followed by 11 value tokens), and the
    very first physical row is the header followed by one data record.

This module implements ONE reader that handles both layouts the same way the
production builder (src/data_pipeline/build_ideology_scores.py) does: find
the first pure-integer token in the row (the AD value); everything before it
is the header template, and the 11 tokens starting at it are the data
fields, taken positionally after validating the header template. A pure
header row (no integer token at all) is skipped.

It also provides the ballot-line classification shared by the ideology-line
audit, the v2 ideology builder, the crosswalk audit and the fixed
normalize_ed_results stage:

  * non-candidate counter rows (Public Counter, Manually Counted Emergency,
    Absentee / Military, Affidavit, Federal, Special Presidential,
    Scattered, Scattering, Blank, Void, Total Ballots) are excluded from
    every candidate-vote denominator;
  * candidate rows carry a ballot line in a trailing parenthetical, e.g.
    "Bill de Blasio (Democratic)". Class assignments:
      Dem  = Democratic + Working Families
      Rep  = Republican + Conservative
      Other = every other candidate line.
"""
import csv
import re
from pathlib import Path

import pandas as pd

STANDARD_HEADER = [
    'AD', 'ED', 'County', 'EDAD Status', 'Event', 'Party/Independent Body',
    'Office/Position Title', 'District Key', 'VoteFor', 'Unit Name', 'Tally',
]

# Counter/administrative rows that are NOT candidate ballot lines. These are
# excluded from every candidate-vote total. "Special Presidential" is a
# special-presidential-ballot counter row (it appears in the 2016/2020
# presidential files), not a candidate line.
NON_CANDIDATE_UNITS = {
    'public counter',
    'manually counted emergency',
    'absentee / military',
    'affidavit',
    'federal',
    'special presidential',
    'scattered',
    'scattering',
    'blank',
    'void',
    'total ballots',
}

DEM_LINES = {'democratic', 'working families'}
REP_LINES = {'republican', 'conservative'}


def _is_pure_int(tok):
    return tok is not None and re.match(r'^\d+$', str(tok).strip()) is not None


def _header_matches(template, observed):
    """Case-insensitive match of the observed per-row header tokens against
    the STANDARD_HEADER template, up to the length of the observed part."""
    obs = [str(t).strip().strip('"').lower() for t in observed]
    tmpl = [h.lower() for h in template[:len(obs)]]
    return obs == tmpl

def iter_boe_rows(path):
    """Yield one dict per data row (both layouts) with the 11 standard fields.

    Fields are returned as stripped strings; 'AD', 'ED', 'Tally' are also
    exposed as ints via the keys 'AD_int', 'ED_int', 'Tally_int' (None when
    the token is not a clean integer).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'BOE ED-level file not found: {path}')
    with open(path, 'r', newline='', encoding='utf-8-sig', errors='replace') as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            first_num = None
            for i, tok in enumerate(row):
                if _is_pure_int(tok):
                    first_num = i
                    break
            if first_num is None:
                # pure header template row (clean layout) or blank line
                continue
            header_part = row[:first_num]
            if header_part:
                if not _header_matches(STANDARD_HEADER, header_part):
                    # Unknown header layout: fall back to name-based mapping
                    # (same spirit as build_ideology_scores.py) instead of
                    # guessing positional fields.
                    names = [str(t).strip().strip('"') for t in header_part]
                    mapping = dict(zip(names, row[first_num:]))
                    vals = [mapping.get(h, '') for h in STANDARD_HEADER]
                else:
                    vals = row[first_num:first_num + len(STANDARD_HEADER)]
            else:
                vals = row[:len(STANDARD_HEADER)]
            if len(vals) < len(STANDARD_HEADER):
                # truncated row (should not happen in BOE exports); skip
                continue
            rec = {h: str(v).strip() for h, v in zip(STANDARD_HEADER, vals)}
            rec['AD_int'] = int(rec['AD']) if _is_pure_int(rec['AD']) else None
            rec['ED_int'] = int(rec['ED']) if _is_pure_int(rec['ED']) else None
            rec['Tally_int'] = int(rec['Tally']) if _is_pure_int(rec['Tally']) else None
            yield rec


def ballot_line(unit_name):
    """Return the trailing parenthetical ballot-line label, or None."""
    if not unit_name:
        return None
    m = re.search(r'\(([^()]*)\)\s*$', str(unit_name).strip())
    return m.group(1).strip() if m else None


def line_class(unit_name):
    """Classify a Unit Name as 'Dem', 'Rep', 'Other', or 'NonCandidate'."""
    u = str(unit_name).strip()
    if u.lower() in NON_CANDIDATE_UNITS:
        return 'NonCandidate'
    line = ballot_line(u)
    if line is None:
        # Candidate rows in these files always carry a parenthetical line;
        # anything else here is treated as a non-candidate counter row.
        return 'NonCandidate'
    low = line.lower()
    if low in DEM_LINES:
        return 'Dem'
    if low in REP_LINES:
        return 'Rep'
    return 'Other'


def read_boe_dataframe(path, year):
    """Read one BOE ED file into a typed DataFrame with a year column."""
    records = []
    for rec in iter_boe_rows(path):
        if rec['AD_int'] is None or rec['ED_int'] is None:
            continue
        records.append({
            'year': int(year),
            'AD': rec['AD_int'],
            'ED': rec['ED_int'],
            'County': rec['County'],
            'EDAD Status': rec['EDAD Status'],
            'Unit Name': rec['Unit Name'],
            'Tally': rec['Tally_int'] if rec['Tally_int'] is not None else 0,
            'line': ballot_line(rec['Unit Name']),
            'class': line_class(rec['Unit Name']),
        })
    if not records:
        raise RuntimeError(f'No data rows parsed from {path}')
    df = pd.DataFrame(records)
    df['ElectDist'] = df['AD'] * 1000 + df['ED']
    return df

