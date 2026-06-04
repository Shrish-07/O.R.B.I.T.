import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path("experiments/results/ideology_ablation_summary.csv")
OUTPUT_PATH = Path("figures/ideology_ablation_r2.png")


def main():
    df = pd.read_csv(RESULTS_PATH)

    # Create display labels
    df["label"] = df["model"] + " (" + df["scope"].astype(str) + ")"

    # Sort for visual clarity
    df = df.sort_values(["model", "scope"])

    labels = df["label"]
    deltas = df["r2_delta"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, deltas)

    # Reference lines for "negligible impact"
    plt.axhline(0.01, linestyle="--", linewidth=1)
    plt.axhline(-0.01, linestyle="--", linewidth=1)
    plt.axhline(0, linewidth=1)

    plt.title("Ideology Feature Ablation — ΔR² (Political − Base)")
    plt.ylabel("ΔR²")
    plt.xticks(rotation=30, ha="right")

    # Annotate values
    for bar, value in zip(bars, deltas):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9
        )

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()

    print(f"📊 Saved figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
