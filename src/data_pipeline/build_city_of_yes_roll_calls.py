"""Build data/processed/city_of_yes_roll_calls.parquet from the City of Yes
roll-call votes.

STATUS AND LIMITATIONS (documented, not silently resolved)
----------------------------------------------------------
* City of Yes for Housing Opportunity — ULURP N 240290 ZRY, adopted 2024-12-05,
  31 Yes / 20 No. The member-level roll call below is the already-verified
  reference table supplied for this task (districts are the join key used by the
  canonical dataset). It was cross-checked for internal consistency (20 No + 31
  Yes = 51 districts, no district appears twice) and no discrepancy was found.
* City of Yes for Carbon Neutrality (adopted December 2023) and City of Yes for
  Economic Opportunity (adopted June 2024): the exact member-level roll calls
  could NOT be programmatically retrieved in this session. legistar.council.nyc.gov's
  JSON API (webapi.legistar.com/v1/nyc) returns HTTP 403 without a token, and the
  public HTML search is a JavaScript/POST interface that is not scrapeable via
  plain GET in this environment (search-engine fallbacks were also unavailable).
  Those two votes therefore need the roll calls to be pulled by hand from
  legistar.council.nyc.gov's public HTML site (or with a Legistar API token) and
  appended to this table. They are deliberately NOT fabricated here.

Output schema:
  council_district (int), vote_name (str), vote_date (str ISO),
  member_vote ("Yes"/"No"), member_name (str or None).
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "city_of_yes_roll_calls.parquet"

# City of Yes for Housing Opportunity (ULURP N 240290 ZRY), adopted 2024-12-05,
# 31-20. Reference table supplied for this task (already verified).
HOUSING_OPPORTUNITY = {
    "vote_name": "City of Yes for Housing Opportunity",
    "vote_date": "2024-12-05",
    "no_districts": [1, 11, 13, 19, 20, 23, 24, 29, 30, 32, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51],
    "yes_districts": [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 21, 22, 25, 26, 27, 28, 31, 33, 34, 35, 36, 37, 38, 39, 40, 47],
}


def build():
    rows = []
    no = set(HOUSING_OPPORTUNITY["no_districts"])
    yes = set(HOUSING_OPPORTUNITY["yes_districts"])
    assert len(no) == 20 and len(yes) == 31, "reference tally mismatch"
    assert no.isdisjoint(yes), "district listed in both yes and no"
    assert no | yes == set(range(1, 52)), "districts do not cover 1..51"
    for d in range(1, 52):
        rows.append({
            "council_district": d,
            "vote_name": HOUSING_OPPORTUNITY["vote_name"],
            "vote_date": HOUSING_OPPORTUNITY["vote_date"],
            "member_vote": "No" if d in no else "Yes",
            "member_name": None,
        })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} — {len(df)} rows")
    print(df["member_vote"].value_counts().to_string())
    print("\nNOTE: Carbon Neutrality (Dec 2023) and Economic Opportunity (June 2024)")
    print("roll calls are pending manual Legistar retrieval — see module docstring.")


if __name__ == "__main__":
    build()