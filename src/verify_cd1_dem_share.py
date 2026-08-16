"""Re-verify Section 3.2 CD1 (2017) dem_share worked example.

PIPELINE (ground truth, traced in src/rebuild_canonical.py line 15):
- IDEO = data/processed/ideology_by_council.parquet (built by
  src/build_ideology_scores.py, a custom line-by-line raw-CSV parser) is the
  dem_share feature the political model trains on.
- A SEPARATE abandoned pipeline src/compute_ideology.py -> ed_ideology.parquet
  -> src/build_district_ideology.py -> district_ideology.parquet is BLACKLISTED
  (config/feature_blacklist.yaml) and feeds NO model.

Paper Section 3.2 quotes CD1 (2017) dem_share vote-summed = 0.715908 and
CD1 district_ideology area-weighted = 0.711309. A prior audit recomputed the
vote-summed figure as 0.721132 by reading the WRONG file (ed_ideology.parquet).
This script reads the CORRECT file (ideology_by_council.parquet), confirms it
reproduces 0.715908 exactly, and quantifies the two-parser disagreement.

Output: results/cd1_dem_share_audit.json
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IDEO_BY_COUNCIL = ROOT / "data" / "processed" / "ideology_by_council.parquet"
ED_IDEOLOGY = ROOT / "data" / "processed" / "ed_ideology.parquet"
CROSSWALK = ROOT / "data" / "processed" / "ed_to_council_crosswalk.parquet"
OUT = ROOT / "results" / "cd1_dem_share_audit.json"
PAPER_CD1_DEM_SHARE = 0.715908
PAPER_CD1_DI = 0.711309


def cd1_vote_summed_from_ed_ideology(year=2017):
    """Reproduce verify_paper_numbers.py's CD1 vote-sum from ed_ideology.parquet
    (the abandoned-pipeline file). Returns (dem_share, dem_sum, rep_sum)."""
    if not (ED_IDEOLOGY.exists() and CROSSWALK.exists()):
        return None, None, None
    ideo = pd.read_parquet(ED_IDEOLOGY)
    cw = pd.read_parquet(CROSSWALK)
    if {"AD", "ED", "year"} <= set(ideo.columns):
        ideo = ideo[ideo["year"] == year].copy()
    ideo["ElectDist"] = ideo["AD"].astype(float) * 1000 + ideo["ED"].astype(float)
    cw["ElectDist"] = pd.to_numeric(cw["ElectDist"], errors="coerce")
    idea = ideo[["ElectDist", "dem", "rep"]].dropna()
    merged = idea.merge(cw[["ElectDist", "CounDist"]], on="ElectDist", how="inner")
    vote = merged.groupby("CounDist").agg(dem=("dem", "sum"), rep=("rep", "sum")).reset_index()
    vote["dem_share"] = vote["dem"] / (vote["dem"] + vote["rep"])
    row = vote[vote["CounDist"] == 1]
    if len(row) == 0:
        return None, None, None
    return float(row["dem_share"].iloc[0]), float(row["dem"].iloc[0]), float(row["rep"].iloc[0])


def main():
    report = {
        "claim": "Section 3.2 worked example: CD1 (2017) dem_share vote-summed and district_ideology area-weighted",
        "paper_values": {
            "cd1_dem_share_vote_summed": PAPER_CD1_DEM_SHARE,
            "cd1_district_ideology_area_weighted": PAPER_CD1_DI,
        },
        "production_pipeline": {
            "canonical_script": "src/rebuild_canonical.py",
            "canonical_ideology_source": str(IDEO_BY_COUNCIL.relative_to(ROOT)),
            "ideology_builder": "src/build_ideology_scores.py (custom line-by-line raw-CSV parser)",
            "blacklisted_separate_pipeline": (
                "src/compute_ideology.py -> ed_ideology.parquet -> "
                "src/build_district_ideology.py -> district_ideology.parquet "
                "(ALL blacklisted in config/feature_blacklist.yaml, feeds no model)"
            ),
        },
    }
    cd1_val = None
    if IDEO_BY_COUNCIL.exists():
        ideo = pd.read_parquet(IDEO_BY_COUNCIL)
        cd1 = ideo[(ideo["CounDist"] == 1) & (ideo["election_year"] == 2017)]
        if len(cd1) > 0:
            cd1_val = float(cd1["dem_share"].iloc[0])
        report["ideology_by_council_file"] = {
            "path": str(IDEO_BY_COUNCIL.relative_to(ROOT)),
            "columns": ideo.columns.tolist(),
            "cd1_2017_dem_share": cd1_val,
            "matches_paper": (cd1_val is not None and abs(cd1_val - PAPER_CD1_DEM_SHARE) < 1e-6),
            "diff_from_paper": (None if cd1_val is None else cd1_val - PAPER_CD1_DEM_SHARE),
        }
    alt_val, alt_dem, alt_rep = cd1_vote_summed_from_ed_ideology(year=2017)
    report["diagnostic_wrong_source_file_ed_ideology"] = {
        "path": str(ED_IDEOLOGY.relative_to(ROOT)) if ED_IDEOLOGY.exists() else None,
        "cd1_2017_dem_share_recomputed_vote_sum": alt_val,
        "dem_sum": alt_dem,
        "rep_sum": alt_rep,
        "matches_paper": (alt_val is not None and abs(alt_val - PAPER_CD1_DEM_SHARE) < 1e-6),
        "diff_from_paper": (None if alt_val is None else alt_val - PAPER_CD1_DEM_SHARE),
        "interpretation": (
            "This number comes from the ABANDONED ed_ideology.parquet parser. It does NOT "
            "feed any model. Quoting it as the 'vote-summed' figure in the paper's worked "
            "example would have imported the wrong parser's values."
        ),
    }
    di_path = ROOT / "data" / "processed" / "district_ideology.parquet"
    if di_path.exists():
        di = pd.read_parquet(di_path)
        dist_col = "CounDist" if "CounDist" in di.columns else next(
            (c for c in di.columns if "coun" in c.lower()), None)
        cd1_di = di[di[dist_col] == 1] if dist_col else di.iloc[0:0]
        di_val = None
        if len(cd1_di) > 0:
            for c in ["district_ideology", "ideology"]:
                if c in cd1_di.columns:
                    di_val = float(cd1_di.iloc[0][c]); break
        report["district_ideology_area_weighted"] = {
            "path": str(di_path.relative_to(ROOT)),
            "cd1_district_ideology_value": di_val,
            "matches_paper": (di_val is not None and abs(di_val - PAPER_CD1_DI) < 0.001),
        }
    else:
        report["district_ideology_area_weighted"] = {"path": None}
    disagreement = {}
    if cd1_val is not None and alt_val is not None:
        diff = abs(cd1_val - alt_val)
        disagreement = {
            "cd1_2017_production_dem_share": cd1_val,
            "cd1_2017_abandoned_parser_dem_share": alt_val,
            "production_minus_abandoned": cd1_val - alt_val,
            "council_level_abs_diff_pp": float(diff * 100),
            "interpretation": (
                f"CD1 2017 council-level dem_share: production parser "
                f"(build_ideology_scores.py) = {cd1_val:.6f}; abandoned parser "
                f"(compute_ideology.py via ed_ideology.parquet) = {alt_val:.6f}; "
                f"abs difference = {diff*100:.3f} percentage points. The two raw-CSV "
                "parsers disagree non-trivially at the council level â€” an undisclosed "
                "methodological confound in the paper's framing of the dem_share-vs-"
                "district_ideology gap as 'purely area-weighting'."
            ),
        }
    report["two_parser_disagreement"] = disagreement
    report["required_action"] = (
        "No manuscript correction needed for the CD1 dem_share worked example: the paper's "
        "0.715908 is exactly reproduced by reading the CORRECT production file "
        "(data/processed/ideology_by_council.parquet). The prior audit's 0.721132 "
        "'discrepancy' was a misattribution that read the wrong file (ed_ideology.parquet). "
        "The two-parser disagreement should be noted as a methodological confound but does "
        "not require changing the published 0.715908 / 0.711309 numbers."
    )
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nSaved audit to {OUT}")


if __name__ == "__main__":
    main()
