from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc


def save_confusion_matrix(df, score_col, out_path, title):
    y = df["label"].astype(int).values
    pred = (df[score_col].astype(float).values >= 0.5).astype(int)

    cm = confusion_matrix(y, pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["REAL", "SPOOF"])

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["REAL", "SPOOF"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_roc(df, score_col, out_path, title):
    y = df["label"].astype(int).values
    scores = df[score_col].astype(float).values

    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_score_distribution(df, score_col, out_path, title):
    real = df[df["label"] == 0][score_col].astype(float)
    spoof = df[df["label"] == 1][score_col].astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(real, bins=20, alpha=0.6, label="REAL")
    ax.hist(spoof, bins=20, alpha=0.6, label="SPOOF")
    ax.axvline(0.5, linestyle="--", label="Threshold 0.5")

    ax.set_title(title)
    ax.set_xlabel("Spoof score")
    ax.set_ylabel("Number of videos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_branch_score_comparison(df, out_path, out_csv):
    cols = [
        "score_visual",
        "score_behavior",
        "score_rppg",
        "score_fusion_v4",
    ]

    rows = []

    for col in cols:
        rows.append({
            "score_type": col,
            "REAL_mean": df[df["label"] == 0][col].mean(),
            "SPOOF_mean": df[df["label"] == 1][col].mean(),
            "REAL_std": df[df["label"] == 0][col].std(),
            "SPOOF_std": df[df["label"] == 1][col].std(),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False, encoding="utf-8")

    x = np.arange(len(summary))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, summary["REAL_mean"], width, label="REAL")
    ax.bar(x + width / 2, summary["SPOOF_mean"], width, label="SPOOF")

    ax.set_title("Mean scores by branch")
    ax.set_ylabel("Mean spoof score")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["score_type"], rotation=20)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_policy_counts(df, out_path):
    if "decision_v4" not in df.columns:
        return

    counts = df["decision_v4"].value_counts().reset_index()
    counts.columns = ["decision", "count"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts["decision"], counts["count"])

    ax.set_title("V4 Banking Policy Decisions")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Number of videos")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.scores_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(
        df,
        "score_fusion_v4",
        out_dir / "fusion_v4_confusion_matrix.png",
        "V4 Constrained Fusion - Confusion Matrix",
    )

    save_roc(
        df,
        "score_fusion_v4",
        out_dir / "fusion_v4_roc_curve.png",
        "V4 Constrained Fusion - ROC Curve",
    )

    save_score_distribution(
        df,
        "score_fusion_v4",
        out_dir / "fusion_v4_score_distribution.png",
        "V4 Constrained Fusion - Score Distribution",
    )

    save_branch_score_comparison(
        df,
        out_dir / "branch_score_comparison.png",
        out_dir / "branch_score_comparison.csv",
    )

    save_policy_counts(
        df,
        out_dir / "policy_decision_counts.png",
    )

    print("Saved figures in:", out_dir)


if __name__ == "__main__":
    main()
