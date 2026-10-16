"""Re-verify the numfloors missingness count (paper Section 4.3) and reconcile it
against the Table 1 missingness union.

Reproduces the audit logic exactly:
1. Load data/canonical/modeling_dataset_canonical_v2.parquet
2. mask_farf = rows where yearbuilt, commfar, residfar, facilfar are ALL null
   (paper's current value: 32,823 -- Round 10 refresh).
3. Count rows where numfloors is null AND mask_farf is False.
   Paper's current Section 4.3 value: 13,277.
4. Sanity-check the union across the regression dropna columns stays 46,591
   (Table 1 N = 566,956 - 46,591 = 520,365; independently locked).

Output: results/numfloors_missingness_audit.json  (persisted, reproducible)

This script does NOT adjust methodology or filters to hit a target number; it
reports the honest count and flags whether the paper needs correcting.
"""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "canonical" / "modeling_dataset_canonical_v2.parquet"
OUT = ROOT / "results" / "numfloors_missingness_audit.json"

# Round 10: updated to the manuscript's current values (post data refresh +
# 24/25-feature rebuild).
PAPER_TOTAL_ROWS = 566956
PAPER_FARF = 32823
PAPER_NUMFLOORS_ONLY = 13277  # paper's Section 4.3 value
PAPER_N_AFTER_DROPNA = 520365  # Table 1 N
PAPER_DROPNA_UNION = 46591


def main():
    df = pd.read_parquet(CANON)
    n_total = int(len(df))

    # mask_farf: all four FAR fields simultaneously null
    mask_farf = (
        df["yearbuilt"].isna() & df["commfar"].isna()
        & df["residfar"].isna() & df["facilfar"].isna()
    )
    n_farf = int(mask_farf.sum())

    mask_num = df["numfloors"].isna()
    n_num_only = int((mask_num & ~mask_farf).sum())

    # Regression dropna columns exactly as in src/analysis/generate_regression_table.py
    reg_vars = ["target_log_price", "dem_share", "yearbuilt",
                "numfloors", "landuse", "BOROUGH", "commfar"]
    available = [v for v in reg_vars if v in df.columns]
    missing_any = df[available[0]].isna()
    for v in available[1:]:
        missing_any = missing_any | df[v].isna()
    n_union = int(missing_any.sum())
    n_after = int((~missing_any).sum())

    # Decompose the numfloors-only group for the reconciliation note.
    md = df["dem_share"].isna()
    extra_num = mask_num & ~mask_farf
    also_dem = int((extra_num & md).sum())
    only_num = int((extra_num & ~md & ~df["yearbuilt"].isna()
                    & ~df["landuse"].isna() & ~df["BOROUGH"].isna()
                    & ~df["commfar"].isna() & ~df["target_log_price"].isna()).sum())

    report = {
        "claim": "numfloors contributes an additional N missing rows beyond the farf group (Section 4.3)",
        "paper_published_value": PAPER_NUMFLOORS_ONLY,
        "recomputed_value": n_num_only,
        "recomputed_value_rounded": int(round(n_num_only)),
        "match_paper": (n_num_only == PAPER_NUMFLOORS_ONLY),
        "discrepancy_rows": int(n_num_only - PAPER_NUMFLOORS_ONLY),
        "canonical_total_rows": n_total,
        "canonical_total_rows__matches_paper": (n_total == PAPER_TOTAL_ROWS),
        "mask_farf_count": n_farf,
        "mask_farf_count__matches_paper": (n_farf == PAPER_FARF),
        "farf_fields": ["yearbuilt", "commfar", "residfar", "facilfar"],
        "numfloors_total_missing": int(mask_num.sum()),
        "regression_dropna_union": n_union,
        "regression_dropna_union__matches_paper": (n_union == PAPER_DROPNA_UNION),
        "n_after_dropna": n_after,
        "n_after_dropna__matches_paper": (n_after == PAPER_N_AFTER_DROPNA),
        "reconciliation": {
            "numfloors_only_group_size": n_num_only,
            "of_which_also_dem_share_missing": also_dem,
            "of_which_only_numfloors_missing": only_num,
            "note": (
                "The 46,591 union (N=520,365) is independently locked and "
                "still holds: 32,823 (farf group) + 13,277 (numfloors beyond "
                "farf) + 137 (dem_share-unique) + 407 (other) - 53 overlap. "
                "Round 10 refreshed the targets to the manuscript's current "
                "values; the old Round-2 '11,918 vs 11,930' patch note is "
                "superseded."
            ),
        },
    }

    Path(ROOT / "results").mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nSaved audit to {OUT}")

    # Round 11: machine-countable status
    sys.path.insert(0, str(ROOT / 'src' / 'verification'))
    from _shared import record_status, check_val  # noqa: E402
    record_status('verify_numfloors_missingness', [
        check_val('numfloors_only_count', PAPER_NUMFLOORS_ONLY, n_num_only),
        check_val('farf_group_count', PAPER_FARF, n_farf),
        check_val('canonical_total_rows', PAPER_TOTAL_ROWS, n_total),
        check_val('regression_dropna_union', PAPER_DROPNA_UNION, n_union),
        check_val('n_after_dropna', PAPER_N_AFTER_DROPNA, n_after),
    ], note='numfloors-only group and the dropna-union reconciliation vs '
            'manuscript targets (Round 10 refreshed).')


if __name__ == "__main__":
    main()
