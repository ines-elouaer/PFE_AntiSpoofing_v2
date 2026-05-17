from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.ablation_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cas proches du seuil
    near = df.copy()
    near["distance_to_threshold"] = (near["normal_score_spoof"] - 0.5).abs()
    near = near.sort_values("distance_to_threshold").head(20)

    x = range(len(near))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, near["normal_score_spoof"], marker="o", label="Normal")
    ax.plot(x, near["zero_behavior_score_spoof"], marker="o", label="Zero behavior")
    ax.plot(x, near["zero_rppg_score_spoof"], marker="o", label="Zero rPPG")
    ax.plot(x, near["zero_all_score_spoof"], marker="o", label="Zero all")
    ax.axhline(0.5, linestyle="--", label="Threshold 0.5")

    ax.set_title("Ablation impact on videos close to decision threshold")
    ax.set_xlabel("Videos sorted by distance to threshold")
    ax.set_ylabel("Spoof score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "ablation_scores_near_threshold.png", dpi=300)
    plt.close(fig)

    near.to_csv(out_dir / "ablation_near_threshold_cases.csv", index=False, encoding="utf-8")

    print("Saved:", out_dir / "ablation_scores_near_threshold.png")
    print("Saved:", out_dir / "ablation_near_threshold_cases.csv")


if __name__ == "__main__":
    main()
