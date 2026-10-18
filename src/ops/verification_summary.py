#!/usr/bin/env python3
"""Verification summary stage (Round 11).

Aggregates the machine-countable per-verifier status files written by
src/verification/_shared.record_status() into results/verification_summary.json
and prints a compact table. Exits non-zero if ANY verifier reports a mismatch
beyond its tolerance -- or failed to report at all (a missing status file is a
coverage hole, not an implicit pass). This is the single gate that makes
verifier output machine-countable instead of print-only.

A verifier whose status file is older than the current pipeline run start
(when recorded by main.py in results/logs/pipeline/.run_start_utc) is flagged
'STALE' and also fails the summary: a status written by a previous run is not
evidence for this run.

Run: python src/ops/verification_summary.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS_DIR = ROOT / 'results' / 'verification'
OUT = ROOT / 'results' / 'verification_summary.json'
RUN_START_MARKER = ROOT / 'results' / 'logs' / 'pipeline' / '.run_start_utc'

# Every verification stage in main.py MUST appear here (and vice versa): a
# verifier that forgets record_status gets listed as NO REPORT and fails.
EXPECTED = [
    'verify_canonical_dataset_and_split',
    'verify_abstract_and_intro',
    'verify_ideology_construction',
    'verify_spatial_pattern',
    'verify_cd1_dem_share',
    'verify_regression_ols',
    'verify_regression_missingness',
    'verify_numfloors_missingness',
    'verify_ml_performance',
    'verify_shap_importance',
    'verify_appendix_a1_scenarios',
    'verify_borough_map_and_crosscutting',
    'verify_spatial_regression',
    'verify_iv_district_fe',
    'verify_iv_district_fe_rescue',
    'verify_city_of_yes_did',
    'verify_scenario_mechanics',
    'verify_followup_resolution',
    'verify_blacklist_fully',
    'verify_ideology_v2',
    'verify_analysis_v2',
    'verify_causal_spatial_v2',
    'verify_repo_hygiene',
]


def main():
    run_start = None
    if RUN_START_MARKER.exists():
        try:
            run_start = float(RUN_START_MARKER.read_text().strip())
        except ValueError:
            run_start = None

    rows = []
    total_checks = 0
    total_mismatches = 0
    n_problems = 0
    for name in EXPECTED:
        f = STATUS_DIR / f'{name}.json'
        if not f.exists():
            rows.append({'verifier': name, 'status': 'NO REPORT',
                         'n_checks': 0, 'n_mismatches': 0})
            n_problems += 1
            continue
        payload = json.loads(f.read_text(encoding='utf-8'))
        stale = (run_start is not None and f.stat().st_mtime < run_start - 1)
        mism = int(payload.get('n_mismatches', 0))
        rows.append({'verifier': name,
                     'status': 'STALE (older than this run)' if stale else
                               ('PASS' if mism == 0 else 'MISMATCH'),
                     'n_checks': int(payload.get('n_checks', 0)),
                     'n_mismatches': mism})
        total_checks += int(payload.get('n_checks', 0))
        total_mismatches += mism
        if mism > 0 or stale:
            n_problems += 1

    # Flag status files from verifiers no longer in the expected list (stale
    # leftovers from retired checks would otherwise read as a clean pass).
    extras = [f.stem for f in STATUS_DIR.glob('*.json')
              if f.stem not in EXPECTED]
    rows_out = sorted(rows, key=lambda r: (r['n_mismatches'] == 0
                                           and r['status'] == 'PASS',
                                           r['verifier']))
    print('=' * 78)
    print('VERIFICATION SUMMARY (machine-countable, Round 11)')
    print('=' * 78)
    print(f"{'verifier':<44} {'status':<26} {'checks':>6} {'mismatch':>8}")
    print('-' * 78)
    for r in sorted(rows, key=lambda r: r['verifier']):
        print(f"{r['verifier']:<44} {r['status']:<26} {r['n_checks']:>6} "
              f"{r['n_mismatches']:>8}")
    print('-' * 78)
    print(f'TOTAL: {total_checks} checks, {total_mismatches} mismatches, '
          f'{n_problems} problem verifier(s)')
    if extras:
        print('NOTE: stale status files from non-listed verifiers:', extras)

    payload = {'generated_utc': datetime.now(timezone.utc)
               .isoformat() + 'Z',
               'n_verifiers_expected': len(EXPECTED),
               'n_verifiers_reported': sum(1 for r in rows
                                           if r['status'] != 'NO REPORT'),
               'total_checks': total_checks,
               'total_mismatches': total_mismatches,
               'n_problem_verifiers': n_problems,
               'stale_status_files_ignored': extras,
               'verifiers': rows_out}
    OUT.write_text(json.dumps(payload, indent=2, default=str),
                   encoding='utf-8')
    print('WROTE', OUT.relative_to(ROOT))
    if total_mismatches > 0 or n_problems > 0:
        print(f'VERIFICATION SUMMARY: FAIL ({total_mismatches} mismatches, '
              f'{n_problems} problem verifiers)')
        sys.exit(1)
    print('VERIFICATION SUMMARY: PASS')


if __name__ == '__main__':
    main()
