"""Repo hygiene verifier (Round 12, Task K/L).

Checks the P2 hygiene items:
  * LICENSE exists and is MIT;
  * requirements.txt is pinned (every requirement line carries == and
    matches the installed version in this virtualenv for the packages
    that are importable);
  * the superseded Round-5 fact-check lives in archive/ with the
    SUPERSEDED header and no longer exists under results/;
  * results/regression_diagnostics.json no longer carries the stale
    pre-rewrite manuscript targets (paper_targets_appendix_a2 /
    kurtosis_matches_paper);
  * main.py no longer labels stages "not yet adopted into the manuscript";
  * the anonymous-release machinery exists (scripts/make_anonymous_release.py
    + config/scrub_terms.txt) and, if a release report exists, it reports
    zero remaining matches.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ROOT, record_status, check_val  # noqa: E402


def main():
    checks = []
    lic = ROOT / 'LICENSE'
    checks.append(check_val('license_exists', True, lic.exists()))
    checks.append(check_val('license_is_mit', True,
                            'MIT License' in lic.read_text(encoding='utf-8')
                            if lic.exists() else False))

    import importlib.metadata as md
    req_path = ROOT / 'requirements.txt'
    unpinned, mismatched = [], []
    if req_path.exists():
        for line in req_path.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if '==' not in s:
                unpinned.append(s)
                continue
            name, ver = s.split('==', 1)
            try:
                installed = md.version(name)
                if installed != ver:
                    mismatched.append(f'{name}: req={ver} venv={installed}')
            except md.PackageNotFoundError:
                pass  # e.g. Windows-only pywin32 variants
    checks.append(check_val('requirements_all_pinned', 0, len(unpinned),
                            tol=0))
    checks.append(check_val('requirements_match_venv', 0, len(mismatched),
                            tol=0))

    arch = ROOT / 'archive' / 'FULL_PAPER_FACT_CHECK.md'
    checks.append(check_val(
        'factcheck_archived_with_header', True,
        arch.exists() and arch.read_text(encoding='utf-8')
        .lstrip().startswith('> **SUPERSEDED')))
    checks.append(check_val(
        'factcheck_not_under_results', False,
        (ROOT / 'results' / 'FULL_PAPER_FACT_CHECK.md').exists()))

    diag = ROOT / 'results' / 'regression_diagnostics.json'
    dj = json.loads(diag.read_text()) if diag.exists() else {}
    checks.append(check_val('diagnostics_no_stale_targets', False,
                            'paper_targets_appendix_a2' in dj))
    checks.append(check_val('diagnostics_no_stale_kurtosis_flag', False,
                            'kurtosis_matches_paper' in dj))

    main_txt = (ROOT / 'main.py').read_text(encoding='utf-8')
    checks.append(check_val('no_stale_not_adopted_labels', 0,
                            main_txt.count('not yet adopted into the '
                                           'manuscript'), tol=0))

    checks.append(check_val(
        'anonymous_release_script_exists', True,
        (ROOT / 'scripts' / 'make_anonymous_release.py').exists()))
    checks.append(check_val(
        'scrub_terms_exist', True,
        (ROOT / 'config' / 'scrub_terms.txt').exists()))
    rep = ROOT / 'results' / 'anonymous_release_report.json'
    if rep.exists():
        r = json.loads(rep.read_text())
        checks.append(check_val('anonymous_release_no_remaining_matches', 0,
                                len(r.get('remaining_matches', [])), tol=0))

    record_status('verify_repo_hygiene', checks,
                  note='Task-K/L hygiene: LICENSE, pinned requirements, '
                       'archived superseded fact-check, cleaned '
                       'diagnostics/labels, anonymous-release machinery.')


if __name__ == '__main__':
    main()
