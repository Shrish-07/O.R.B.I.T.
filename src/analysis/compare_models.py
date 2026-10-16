import json
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("experiments/results")

SCOPE_NORMALIZATION = {
    "year2017": "2017",
    "all_years": "all",
}

MAX_ROWS_PRINT = 15   # hard cap to stop terminal floods
MAX_COL_WIDTH = 20   # prevent ultra-wide cells

pd.set_option("display.max_colwidth", MAX_COL_WIDTH)
pd.set_option("display.width", 120)

def normalize_record(data):
    """Reconcile the two result-JSON schemas this repo's training scripts
    actually write, so neither family is silently dropped.

    Family A (models/training/train_lgbm.py):
        {model, mode, scope, features_used, test_rows, train_rows, r2, mae}
        scope is already short-form ("all" / "2017").

    Family B (train_catboost.py, train_xgboost.py, train_random_forest.py,
    train_linear_baselines.py):
        {model, name, variant, mode, n_train, n_test, n_features, r2, mae}
        variant is long-form ("all_years" / "year2017"), and the feature
        count / row count keys are named differently.

    Before this fix, compare_models.py's required_keys check
    ({"model","mode","scope","r2","mae","features_used"}) only matched
    Family A, so every Family-B file (i.e. every model except LightGBM) was
    silently skipped — confirmed by running it: it printed only the 2
    LightGBM rows and 5 "Skipping malformed file" warnings. This function
    makes both families comparable without changing either training
    script's on-disk schema.
    """
    out = dict(data)

    if "scope" not in out and "variant" in out:
        out["scope"] = SCOPE_NORMALIZATION.get(out["variant"], out["variant"])

    if "features_used" not in out and "n_features" in out:
        out["features_used"] = out["n_features"]

    return out


def main():
    rows = []

    if not RESULTS_DIR.exists():
        raise FileNotFoundError("experiments/results directory does not exist")

    # sorted(): deterministic iteration order so the duplicate-key
    # drop_duplicates(keep='last') below deterministically resolves a retired
    # stale record (e.g. rf_all_years_political_n200) against its replacement
    # (rf_all_years_political_tuned -- alphabetically later) instead of
    # depending on filesystem ordering. Round 10.
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)

            data = normalize_record(data)

            required_keys = {"model", "mode", "scope", "r2", "mae", "features_used"}
            if not required_keys.issubset(data):
                print(f"⚠️  Skipping malformed file: {path.name}")
                continue

            # --- Normalize scope labels safely ---
            raw_scope = str(data["scope"])
            norm_scope = SCOPE_NORMALIZATION.get(raw_scope, raw_scope)

            row = {
                **data,
                "scope": norm_scope,
            }

            rows.append(row)

        except Exception as e:
            print(f"❌ Failed to read {path.name}: {e}")

    if not rows:
        raise RuntimeError("No valid result files found in experiments/results")

    df = pd.DataFrame(rows)

    # Normalize column types
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
    df["features_used"] = pd.to_numeric(df["features_used"], errors="coerce")

    # Drop rows that are now invalid
    before = len(df)
    df = df.dropna(subset=["model", "scope", "mode", "r2", "mae", "features_used"])
    if len(df) < before:
        print(f"⚠️  Dropped {before - len(df)} rows with invalid numeric values")

    # Detect duplicate experiment keys
    dupes = df.duplicated(subset=["model", "scope", "mode"], keep=False)
    if dupes.any():
        print("⚠️  Duplicate experiment keys detected:")
        print(df.loc[dupes, ["model", "scope", "mode"]])
        df = df.drop_duplicates(subset=["model", "scope", "mode"], keep="last")

    df = df.sort_values(["model", "scope", "mode"])

    print("\n📊 ORBIT v2 — Multi-Model Ideology Ablation (Raw Results)\n")

    preview_cols = ["model", "scope", "mode", "r2", "mae", "features_used"]
    preview_df = df[preview_cols].head(MAX_ROWS_PRINT)

    print(preview_df.to_string(index=False))

    if len(df) > MAX_ROWS_PRINT:
        print(f"\n… {len(df) - MAX_ROWS_PRINT} more rows not shown")

    raw_out_path = RESULTS_DIR / "model_comparison_raw.csv"
    df.to_csv(raw_out_path, index=False)

    # --- Ideology Ablation Table ---
    base_df = df[df["mode"] == "base"]
    pol_df = df[df["mode"] == "political"]

    ablation = pd.merge(
        base_df,
        pol_df,
        on=["model", "scope"],
        suffixes=("_base", "_political"),
        how="inner",
    )

    if ablation.empty:
        raise RuntimeError(
            "No matching base/political experiment pairs found. "
            "Check that both modes ran for each model + scope."
        )

    ablation["r2_delta"] = ablation["r2_political"] - ablation["r2_base"]
    ablation["mae_delta"] = ablation["mae_political"] - ablation["mae_base"]
    ablation["features_delta"] = (
        ablation["features_used_political"]
        - ablation["features_used_base"]
    )

    ablation = ablation[
        [
            "model",
            "scope",
            "r2_base",
            "r2_political",
            "r2_delta",
            "mae_base",
            "mae_political",
            "mae_delta",
            "features_used_base",
            "features_used_political",
            "features_delta",
        ]
    ].sort_values(["model", "scope"])

    print("\n📈 ORBIT v2 — Ideology Feature Ablation Summary\n")

    preview_ablation = ablation.head(MAX_ROWS_PRINT)
    print(preview_ablation.to_string(index=False))

    if len(ablation) > MAX_ROWS_PRINT:
        print(f"\n… {len(ablation) - MAX_ROWS_PRINT} more rows not shown")

    ablation_out_path = RESULTS_DIR / "ideology_ablation_summary.csv"
    ablation.to_csv(ablation_out_path, index=False)

    print(f"\n📁 Raw comparison saved to:       {raw_out_path}")
    print(f"📁 Ideology ablation saved to:    {ablation_out_path}")

    # --- Governance-Grade Summary ---
    print("\n🧠 ORBIT v2 — Governance Interpretation\n")

    for _, row in ablation.head(MAX_ROWS_PRINT).iterrows():
        model = row["model"]
        scope = row["scope"]
        r2_delta = row["r2_delta"]
        mae_delta = row["mae_delta"]

        verdict = "NEGLIGIBLE"
        if abs(r2_delta) >= 0.01:
            verdict = "MATERIAL"

        direction = "↑" if r2_delta > 0 else "↓"

        print(
            f"{model:14} | {scope:4} | "
            f"ΔR² = {r2_delta:+.4f} {direction} | "
            f"ΔMAE = {mae_delta:+.4f} | "
            f"Impact: {verdict}"
        )

    if len(ablation) > MAX_ROWS_PRINT:
        print(f"\n… {len(ablation) - MAX_ROWS_PRINT} more models not shown")


if __name__ == "__main__":
    main()
