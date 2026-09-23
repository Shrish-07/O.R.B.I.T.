"""Honest coverage diagnostic for the close-election instrument (Round 5, Task 1).

Before any instrument is built, this answers one question: how many of the 204
district-year NYC council races (51 districts x {2013, 2017, 2021, 2025}) were
actually close? "Close" is margin_pct <= a bandwidth in percentage points, where
margin_pct is the two-candidate (winner vs runner-up) margin in
data/processed/council_election_margins.parquet. Most NYC council races are
landslides, so the honest expectation is that only a handful of races qualify at
any reasonable bandwidth.

This script exists so the thinness of the instrument is *observed and recorded*
before src/causal/iv_district_fe_regression.py fits anything — not discovered
after the fact. It fits no models and edits no numbers.

Output: results/causal/close_election_coverage.json

Run:
    python src/causal/build_close_election_instrument.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MARGINS = ROOT / "data" / "processed" / "council_election_margins.parquet"
OUT = ROOT / "results" / "causal" / "close_election_coverage.json"

BANDWIDTHS_PP = [5, 10, 15, 20]
# A close-race count below this at the working bandwidth means the instrument is
# thin and its 2SLS estimates will have wide confidence intervals. Chosen per the
# task's "~15-20 races" guidance; kept explicit rather than hidden.
THIN_RACE_THRESHOLD = 20
# The working bandwidth iv_district_fe_regression.py treats as primary; the other
# three remain as robustness specs. 20 pp is chosen because it maximizes coverage.
WORKING_BANDWIDTH_PP = 20


def main():
    if not MARGINS.exists():
        raise SystemExit(f"{MARGINS} not found — run "
                         "src/data_pipeline/build_council_election_margins.py first")
    df = pd.read_parquet(MARGINS)
    n_total = int(len(df))

    out = {
        "source": str(MARGINS.relative_to(ROOT)).replace("\\", "/"),
        "n_district_year_races": n_total,
        "cycles": sorted(int(y) for y in df["year"].unique()),
        "bandwidths_pp": BANDWIDTHS_PP,
        "close_race_definition": "margin_pct <= bandwidth (margin_pct = "
                                 "(winner_votes - runnerup_votes)/(winner+runnerup)*100)",
        "thin_race_threshold": THIN_RACE_THRESHOLD,
        "working_bandwidth_pp": WORKING_BANDWIDTH_PP,
        "winner_party_domain": sorted(df["winner_party"].dropna().unique().tolist()),
        "by_bandwidth": {},
    }

    for bw in BANDWIDTHS_PP:
        close = df[df["margin_pct"] <= bw]
        # Races that can actually feed an instrumented estimate later: the close
        # race must be the *prior* cycle to a canonical election_year (2013->2017,
        # 2017->2021, 2021->2025). A 2025 close race has no downstream year in the
        # canonical dataset, so it contributes nothing to the instrument.
        usable = close[close["year"].isin([2013, 2017, 2021])]
        by_cycle = {int(y): int((close["year"] == y).sum())
                    for y in sorted(df["year"].unique())}
        usable_by_cycle = {int(y): int((usable["year"] == y).sum())
                           for y in sorted(df["year"].unique())}
        entry = {
            "total_close_races": int(len(close)),
            "by_cycle": by_cycle,
            "by_winner_party": {str(k): int(v) for k, v in
                                close["winner_party"].value_counts(dropna=False).items()},
            "by_cycle_and_winner_party": {
                str(y): {str(p): int(c) for p, c in sub["winner_party"].value_counts().items()}
                for y, sub in close.groupby("year")
            },
            "usable_for_instrument": {
                "note": "only 2013/2017/2021 races feed a later canonical "
                        "election_year; 2025 races have no downstream row",
                "total": int(len(usable)),
                "by_cycle": usable_by_cycle,
                "by_winner_party": {str(k): int(v) for k, v in
                                    usable["winner_party"].value_counts(dropna=False).items()},
            },
            "thin": int(len(close)) < THIN_RACE_THRESHOLD,
        }
        out["by_bandwidth"][str(bw)] = entry

    # Plain-language warning, printed AND stored. A thin instrument is a finding,
    # not a failure — this exits 0 either way.
    working = out["by_bandwidth"][str(WORKING_BANDWIDTH_PP)]
    if working["total_close_races"] < THIN_RACE_THRESHOLD:
        out["warning"] = (
            f"instrument coverage is thin at this bandwidth; at ±{WORKING_BANDWIDTH_PP}pp "
            f"only {working['total_close_races']} of {n_total} district-year races qualify "
            f"({working['usable_for_instrument']['total']} usable as a prior-cycle "
            "instrument); expect wide confidence intervals"
        )
    else:
        out["warning"] = None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"\n{n_total} district-year races; close-race counts by bandwidth "
          f"(total / usable-as-instrument):")
    for bw in BANDWIDTHS_PP:
        e = out["by_bandwidth"][str(bw)]
        print(f"  ±{bw:>2d}pp: {e['total_close_races']:>3d} total  /  "
              f"{e['usable_for_instrument']['total']:>3d} usable   "
              f"(by winner party: {e['by_winner_party']})")
    if out["warning"]:
        print(f"\nWARNING: {out['warning']}")
    else:
        print("\nCoverage looks adequate at the working bandwidth.")


if __name__ == "__main__":
    main()
