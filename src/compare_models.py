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

def main():
    rows = []

    if not RESULTS_DIR.exists():
        raise FileNotFoundError("experiments/results directory does not exist")

    for path in RESULTS_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)

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
