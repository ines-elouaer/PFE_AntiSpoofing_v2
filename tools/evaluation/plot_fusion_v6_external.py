from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc


def save_confusion_matrix(df, out_path):
    y = df["label"].astype(int).values
    pred = df["pred_fusion_v6"].astype(int).values

    cm = confusion_matrix(y, pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)

    ax.set_title("External V6 - Confusion Matrix")
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


def save_roc(df, out_path):
    y = df["label"].astype(int).values
    scores = df["score_fusion_v6"].astype(float).values

    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_title("External V6 - ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_score_distribution(df, out_path):
    real = df[df["label"] == 0]["score_fusion_v6"].astype(float)
    spoof = df[df["label"] == 1]["score_fusion_v6"].astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(real, bins=20, alpha=0.6, label="REAL")
    ax.hist(spoof, bins=20, alpha=0.6, label="SPOOF")
    ax.axvline(0.5, linestyle="--", label="Threshold 0.5")

    ax.set_title("External V6 - Score Distribution")
    ax.set_xlabel("Spoof score")
    ax.set_ylabel("Number of videos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_policy_counts(df, out_path):
    if "decision_v6" not in df.columns:
        return

    counts = df["decision_v6"].value_counts().reset_index()
    counts.columns = ["decision", "count"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts["decision"], counts["count"])
    ax.set_title("External V6 - Banking Policy")
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

    save_confusion_matrix(df, out_dir / "external_v6_confusion_matrix.png")
    save_roc(df, out_dir / "external_v6_roc_curve.png")
    save_score_distribution(df, out_dir / "external_v6_score_distribution.png")
    save_policy_counts(df, out_dir / "external_v6_policy_counts.png")

    print("Saved figures in:", out_dir)


if __name__ == "__main__":
    main()
