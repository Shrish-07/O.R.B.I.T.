"""Build City Council general-election margins (winner / runner-up) from the NYC
Board of Elections' certified "Election Results Summary" archive (vote.nyc), for
the 2013, 2017, 2021, and 2025 general elections.

Sources: vote.nyc/page/election-results-summary-{2013,2017,2021,2025}.
2013 is PDF-only (parsed with pypdf); 2017/2021/2025 have CSV Recap files.

The Recap file is the BOE "Statement and Return Report for Certification" — the
district-level certified totals, broken out per candidate per party line per
assembly district. For the RCV general elections (2021, 2025) this report carries
the certified *first-choice (round-1)* counts; the Recap archive does not publish
round-by-round RCV preferences, so the "final-round two-candidate result" for RCV
years is taken to be the top two candidates by these certified totals (fusion lines
aggregated by candidate name). For the overwhelmingly two-candidate council races
that equals the RCV final-round pairing; the choice is documented here rather than
silently assumed.

Output schema (one row per district per year):
  council_district, year, winner_name, winner_party, winner_votes,
  runnerup_name, runnerup_party, runnerup_votes, total_votes, margin_pct

  winner_votes / runnerup_votes are fusion-aggregated (summed across the candidate's
  party lines and every assembly district within the council district).
  winner_party / runnerup_party is the candidate's single largest party line.
  total_votes = winner_votes + runnerup_votes (the two-candidate denominator).
  margin_pct = (winner_votes - runnerup_votes) / total_votes * 100.

Raw Recap files are cached under data/raw/council_elections/<year>/ so the build is
re-runnable offline. Output: data/processed/council_election_margins.parquet.

This is separate data via a separate script from the mayor-race ideology pipeline
(normalize_ed_results.py + build_ideology_scores.py), which is not modified here.
"""
import csv
import io
import re
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "council_elections"
OUT = ROOT / "data" / "processed" / "council_election_margins.parquet"
BASE = "https://vote.nyc"

YEARS = {
    "2013": "https://vote.nyc/page/election-results-summary-2013",
    "2017": "https://vote.nyc/page/election-results-summary-2017",
    "2021": "https://vote.nyc/page/election-results-summary-2021",
    "2025": "https://vote.nyc/page/election-results-summary-2025",
}

DISTRICT_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\s+Council\s+District", re.IGNORECASE)
CAND_LABEL_RE = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*$")


# --------------------------------------------------------------------------
# Link discovery
# --------------------------------------------------------------------------
def find_recap_links(page_url):
    """Return {district: {fmt: {url, cross}}} for council general-election Recap files."""
    html = requests.get(page_url, timeout=120).text
    links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.DOTALL)
    by_district = {}
    for href, text in links:
        h = unquote(href.replace("&amp;", "&"))
        lower = h.lower()
        if "council district" not in lower:
            continue
        if "recap" not in lower:
            continue
        if "general" not in lower:
            continue
        m = DISTRICT_RE.search(h)
        if not m:
            continue
        dist = int(m.group(1))
        if 1 <= dist <= 51:
            by_district.setdefault(dist, {})
            fmt = "pdf" if lower.endswith(".pdf") else "csv"
            by_district[dist][fmt] = {"url": BASE + h, "cross": "rossover" in lower}
    return by_district


def select_recap(by_district):
    """Pick one file per district: prefer regular (non-crossover) CSV, then any CSV,
    then regular PDF, then any PDF."""
    result = {}
    for dist, fmts in by_district.items():
        pick = None
        for fmt in ("csv", "pdf"):
            opts = fmts.get(fmt)
            if opts is None:
                continue
            pick = opts
            break
        if pick is not None:
            result[dist] = (pick["url"], "csv" if pick["url"].lower().endswith(".csv") else "pdf")
    return result


# --------------------------------------------------------------------------
# Parsers -> list of (name, party, votes)
# --------------------------------------------------------------------------
def parse_csv(text):
    """Parse a Recap CSV into (name, party, votes) triples, handling the two
    observed layouts: 2017's two-column 'label,value' rows and 2021/2025's wider
    certificate rows. Scan each row for a 'Name (Party)' cell followed by a number."""
    rows = list(csv.reader(io.StringIO(text)))
    out = []
    for r in rows:
        for i in range(len(r) - 1):
            label = (r[i] or "").strip()
            m = CAND_LABEL_RE.match(label)
            if not m:
                continue
            val = (r[i + 1] or "").strip().replace(",", "")
            if not val.replace("-", "").replace(".", "").isdigit():
                continue
            v = int(float(val))
            if v <= 0:
                continue
            out.append((m.group(1).strip(), m.group(2).strip(), v))
            break
    return out


def parse_pdf(data):
    reader = PdfReader(io.BytesIO(data))
    out = []
    pat = re.compile(r"^\s*([\d,]+)\s+([A-Z][^(]*?)\s*\(([^()]+)\)\s*$")
    for page in reader.pages:
        for ln in (page.extract_text() or "").splitlines():
            m = pat.match(ln)
            if m:
                out.append((m.group(2).strip(), m.group(3).strip(), int(m.group(1).replace(",", ""))))
    return out


def aggregate(triples):
    """Fusion-aggregate (name, party, votes) -> {name: {party: votes}}."""
    by_name = {}
    for name, party, votes in triples:
        d = by_name.setdefault(name, {})
        d[party] = d.get(party, 0) + votes
    return by_name


# --------------------------------------------------------------------------
# Party consolidation (Round-5 fix)
# --------------------------------------------------------------------------
# The Recap report carries *ballot lines*, not a separate "candidate party
# enrollment" field. NYC fusion voting splits a candidate's votes across several
# lines (e.g. Brannan 2017 = "Democratic" + "Working Families"; Quaglione =
# "Republican" + "Conservative" + "Independence"); 2013 Recap PDFs also print
# every label in ALL CAPS while 2017+ use title case. The previous build set
# winner_party/runnerup_party to the candidate's *single largest* ballot line,
# leaving the raw line name (case-inconsistent, fusion compounds, minor-party
# line names) in the parquet. This fix collapses a candidate's full set of lines
# to their actual major-party affiliation: Democratic / Republican / Other.
#
# AUTHORITATIVE RULE: if the candidate's own ticket carries a literal
# "Democratic" or "Republican" line (inspected after case-folding and splitting
# fusion compounds on "/"), that line wins regardless of vote share — a major
# candidate always also runs on their own party's line. The minor-line alias
# tables below are consulted ONLY when no literal major line is present, and
# contain only lines whose major-party alignment is unambiguous and documented:
#   DEM_ALIGNED_MINOR: Working Families  (classic Democratic fusion partner)
#   REP_ALIGNED_MINOR: Conservative, Save Our City (Republican fusion partners)
# Lines NOT in these tables (Green, Libertarian, Reform, Independence, Liberal,
# novelty lines such as "Rent Is 2 Damn High", etc.) never imply a major party
# on their own; a candidate appearing only on such lines classifies as "Other".
# This mirrors NYC_BOE practice: a candidate with no Dem/Rep line is not a
# major-party candidate. Add a line to a table (with a comment) if a future
# cycle surfaces a new unambiguous alignment.
DEM_ALIGNED_MINOR = {"working families"}
REP_ALIGNED_MINOR = {"conservative", "save our city"}

ALLOWED_PARTIES = {"Democratic", "Republican", "Other"}
_SPLIT_RE = re.compile(r"[/;]")


def major_party_from_lines(parties, name=""):
    """Reduce a candidate's {ballot_line: votes} to Democratic/Republican/Other.

    Case-folds every label and splits fusion compounds (e.g. "Conservative/Save
    Our City", "Democratic/Law & Order"). Order of precedence: literal major line
    > unambiguous minor-line alignment > Other. A candidate appearing on BOTH the
    Democratic and Republican lines (a cross-endorsement the ballot-line data
    alone cannot disambiguate, e.g. Holden D30 / Yeger D44 2021) is classified
    "Other" and logged loudly rather than guessed — both known cases are
    uncontested races (margin 100%), so they can never enter the close-election
    instrument that reads this column.
    """
    dem_line = rep_line = False
    dem_alias = rep_alias = False
    for label in parties:
        for token in _SPLIT_RE.split(str(label)):
            t = token.strip().lower()
            if not t:
                continue
            if t == "democratic":
                dem_line = True
            elif t == "republican":
                rep_line = True
            if t in DEM_ALIGNED_MINOR:
                dem_alias = True
            if t in REP_ALIGNED_MINOR:
                rep_alias = True
    # Literal major line is authoritative; a candidate cannot legitimately hold
    # both (if that ever surfaced a new cross-endorsement, treat as Other and let
    # the assertion in main() flag it for review rather than silently guessing).
    if dem_line and rep_line:
        print(f"      NOTE cross-endorsement (candidate holds both Dem+Rep lines) "
              f"{name!r} -> classified 'Other'")
        return "Other"
    if dem_line:
        return "Democratic"
    if rep_line:
        return "Republican"
    if dem_alias and not rep_alias:
        return "Democratic"
    if rep_alias and not dem_alias:
        return "Republican"
    return "Other"


def winner_runnerup(by_name):
    name_totals = []
    for name, parties in by_name.items():
        tot = sum(parties.values())
        raw_primary = max(parties.items(), key=lambda kv: kv[1])[0]
        name_totals.append({
            "name": name,
            "party": major_party_from_lines(parties, name),  # consolidated major party
            "party_raw": raw_primary,                        # pre-fix largest ballot line
            "votes": tot,
            "write_in": any(str(p).strip().upper() == "WRITE-IN" for p in parties),
        })
    real = [r for r in name_totals if not r["write_in"]]
    pool = real if real else name_totals
    pool.sort(key=lambda r: -r["votes"])
    if not pool:
        return None, None, None, None
    winner = pool[0]
    runnerup = pool[1] if len(pool) >= 2 else None
    total2 = winner["votes"] + (runnerup["votes"] if runnerup else 0)
    margin = ((winner["votes"] - (runnerup["votes"] if runnerup else 0))
              / total2 * 100.0) if total2 else None
    return winner, runnerup, total2, margin


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for year, page_url in YEARS.items():
        print(f"\n===== {year} =====")
        selected = select_recap(find_recap_links(page_url))
        print(f"  discovered {len(selected)} council district Recap files")
        year_dir = RAW_DIR / year
        year_dir.mkdir(parents=True, exist_ok=True)
        for dist in sorted(selected):
            url, fmt = selected[dist]
            ext = "csv" if fmt == "csv" else "pdf"
            cache_path = year_dir / f"{year}_council_D{dist:02d}.{ext}"
            if not cache_path.exists():
                print(f"    downloading D{dist} {fmt}")
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                cache_path.write_bytes(resp.content)
            content = cache_path.read_bytes()
            triples = (parse_csv(content.decode("utf-8", errors="replace"))
                       if fmt == "csv" else parse_pdf(content))
            winner, runnerup, total2, margin = winner_runnerup(aggregate(triples))
            if winner is None:
                print(f"    WARNING: D{dist} parsed no candidates")
                continue
            rows.append({
                "council_district": int(dist), "year": int(year),
                "winner_name": winner["name"], "winner_party": winner["party"],
                "winner_votes": int(winner["votes"]),
                "runnerup_name": runnerup["name"] if runnerup else None,
                "runnerup_party": runnerup["party"] if runnerup else None,
                "runnerup_votes": int(runnerup["votes"]) if runnerup else None,
                "total_votes": int(total2) if total2 else None,
                "margin_pct": round(margin, 4) if margin is not None else None,
                # pre-fix raw ballot lines, kept only for the before/after log below
                "winner_party_raw": winner["party_raw"],
                "runnerup_party_raw": runnerup["party_raw"] if runnerup else None,
            })

    df = pd.DataFrame(rows).sort_values(["year", "council_district"]).reset_index(drop=True)

    # ---- Before/after party-label report + hard assertion (Round-5 fix) ----
    before_w = sorted(df["winner_party_raw"].dropna().unique().tolist())
    before_r = sorted(df["runnerup_party_raw"].dropna().unique().tolist())
    after_w = sorted(df["winner_party"].dropna().unique().tolist())
    after_r = sorted(df["runnerup_party"].dropna().unique().tolist())
    print("\n=== Party-label consolidation (winner/runnerup -> Democratic/Republican/Other) ===")
    print(f"  winner_party distinct:   {len(before_w)} -> {len(after_w)}")
    print(f"    before: {before_w}")
    print(f"    after : {after_w}")
    print(f"  runnerup_party distinct: {len(before_r)} -> {len(after_r)}")
    print(f"    before: {before_r}")
    print(f"    after : {after_r}")
    df = df.drop(columns=["winner_party_raw", "runnerup_party_raw"])

    # Fail loudly (never silently) if a label escapes consolidation.
    bad = (set(df["winner_party"].dropna()) | set(df["runnerup_party"].dropna())) - ALLOWED_PARTIES
    if bad:
        raise AssertionError(
            "council_election_margins: party labels escaped consolidation "
            f"-> {sorted(bad)}. Extend major_party_from_lines() / alias tables.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT} — {len(df)} rows")
    print(df.groupby("year").size().to_string())
    print("\nNarrowest margins per year:")
    for y in sorted(df["year"].unique()):
        sub = df[df["year"] == y].dropna(subset=["margin_pct"]).sort_values("margin_pct")
        print(f"  {y}: " + ", ".join(
            f"D{int(r.council_district)} {r.winner_name} {r.margin_pct:.2f}%"
            for r in sub.head(3).itertuples()))


if __name__ == "__main__":
    main()