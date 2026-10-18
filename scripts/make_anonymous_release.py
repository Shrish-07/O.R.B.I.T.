"""Build an anonymized release copy of the repository (Task L).

Copies the repo WITHOUT .git (and without bulky/generated folders: venv,
__pycache__, raw/processed/canonical/split data, model binaries, logs,
MLflow artifacts, the release output itself and local archives) into
anonymous_release/, scrubs the identifying strings listed in
config/scrub_terms.txt from every text file in the copy (README, docs,
configs, results metadata, scripts), zips the result to
anonymous_release.zip, and writes results/anonymous_release_report.json
listing every remaining match (any hit there means scrub_terms.txt is
missing a term).

Git history is NOT rewritten — the original working tree is never
modified; only the anonymous_release/ copy is scrubbed.

Run: python scripts/make_anonymous_release.py
"""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'anonymous_release'
OUT_ZIP = ROOT / 'anonymous_release.zip'
REPORT = ROOT / 'results' / 'anonymous_release_report.json'
SCRUB_TERMS = ROOT / 'config' / 'scrub_terms.txt'

IGNORE_DIRS = {'.git', 'venv', '.venv', '__pycache__', 'catboost_info',
               'mlruns', '.mlruns', 'anonymous_release', '.idea', '.vscode',
               'data', 'logs', 'notebooks'}
IGNORE_EXT = {'.zip', '.7z', '.rar', '.pyc', '.pyo', '.joblib', '.cbm',
              '.model', '.parquet', '.png', '.jpg', '.ico', '.pdf',
              '.shapefile'}


def load_terms():
    terms = []
    for line in SCRUB_TERMS.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if s and not s.startswith('#'):
            terms.append(s)
    if not terms:
        raise SystemExit('config/scrub_terms.txt is empty — add the '
                         'identifying strings to scrub first.')
    return terms


def main():
    terms = load_terms()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    n_files = 0
    scrubbed_hits = []
    for src in ROOT.rglob('*'):
        if src.is_dir():
            continue
        rel = src.relative_to(ROOT)
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        if src.suffix.lower() in IGNORE_EXT:
            continue
        if src.name in {'mlflow.db', '.env'}:
            continue
        dst = OUT_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = src.read_text(encoding='utf-8')
        except (UnicodeDecodeError, ValueError):
            shutil.copy2(src, dst)
            n_files += 1
            continue
        low = text.lower()
        hits = [t for t in terms if t.lower() in low]
        if hits:
            for t in hits:
                scrubbed_hits.append({'file': rel.as_posix(), 'term': t,
                                      'occurrences': low.count(t.lower())})
                text = text.replace(t, '[REDACTED]')
            dst.write_text(text, encoding='utf-8')
        else:
            shutil.copy2(src, dst)
        n_files += 1

    # zip the scrubbed copy
    with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT_DIR.rglob('*')):
            if p.is_file():
                zf.write(p, p.relative_to(OUT_DIR))

    # report any REMAINING matches in the scrubbed copy (binary-ish files
    # copied verbatim could still contain a term)
    remaining = []
    for p in OUT_DIR.rglob('*'):
        if not p.is_file() or p.suffix.lower() in IGNORE_EXT:
            continue
        try:
            text = p.read_text(encoding='utf-8', errors='ignore').lower()
        except Exception:
            continue
        for t in terms:
            if t.lower() in text:
                remaining.append({'file': p.relative_to(OUT_DIR).as_posix(),
                                  'term': t})

    report = {
        'release_dir': str(OUT_DIR.relative_to(ROOT)),
        'zip': str(OUT_ZIP.relative_to(ROOT)),
        'n_files_copied': n_files,
        'scrub_terms': terms,
        'n_scrubbed_hits': len(scrubbed_hits),
        'scrubbed_hits': scrubbed_hits,
        'remaining_matches': remaining,
        'git_history': 'not touched — the working tree is never modified',
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'anonymous_release/: {n_files} files; '
          f'{len(scrubbed_hits)} scrubbed hits; '
          f'{len(remaining)} remaining matches '
          f'({"CLEAN" if not remaining else "*** REVIEW ***"})')
    print(f'zip: {OUT_ZIP.relative_to(ROOT)}')
    print(f'report: {REPORT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
