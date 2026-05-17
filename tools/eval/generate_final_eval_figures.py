from pathlib import Path
import argparse
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc


def save_confusion_matrix(df: pd.DataFrame, out_path: Path, title: str):
    y_true = df["label"].astype(int).values
    y_pred = df["pred_label"].astype(int).values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

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


def save_roc_curve(df: pd.DataFrame, out_path: Path, title: str):
    y_true = df["label"].astype(int).values
    scores = df["score_spoof"].astype(float).values

    fpr, tpr, _ = roc_curve(y_true, scores)
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


def save_score_distribution(df: pd.DataFrame, out_path: Path, title: str):
    real_scores = df[df["label"] == 0]["score_spoof"].astype(float)
    spoof_scores = df[df["label"] == 1]["score_spoof"].astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(real_scores, bins=20, alpha=0.6, label="REAL")
    ax.hist(spoof_scores, bins=20, alpha=0.6, label="SPOOF")

    ax.axvline(0.5, linestyle="--", label="Threshold = 0.5")

    ax.set_title(title)
    ax.set_xlabel("Spoof score")
    ax.set_ylabel("Number of videos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def compute_summary(df: pd.DataFrame):
    y_true = df["label"].astype(int)
    y_pred = df["pred_label"].astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    real_count = tn + fp
    spoof_count = fn + tp

    bpcer = fp / real_count if real_count else 0.0
    apcer = fn / spoof_count if spoof_count else 0.0
    acer = (apcer + bpcer) / 2.0
    acc = (tn + tp) / len(df) if len(df) else 0.0

    fpr, tpr, _ = roc_curve(y_true, df["score_spoof"].astype(float))
    roc_auc = auc(fpr, tpr)

    return {
        "total": int(len(df)),
        "real_count": int(real_count),
        "spoof_count": int(spoof_count),
        "REAL->REAL": tn,
        "REAL->SPOOF": fp,
        "SPOOF->REAL": fn,
        "SPOOF->SPOOF": tp,
        "accuracy": round(acc, 6),
        "APCER": round(apcer, 6),
        "BPCER": round(bpcer, 6),
        "ACER": round(acer, 6),
        "AUC": round(float(roc_auc), 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--title", default="PAD Evaluation")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)

    required = {"label", "pred_label", "score_spoof"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans predictions.csv : {missing}")

    save_confusion_matrix(
        df,
        out_dir / "confusion_matrix.png",
        f"{args.title} - Confusion Matrix",
    )

    save_roc_curve(
        df,
        out_dir / "roc_curve.png",
        f"{args.title} - ROC Curve",
    )

    save_score_distribution(
        df,
        out_dir / "score_distribution.png",
        f"{args.title} - Score Distribution",
    )

    summary = compute_summary(df)

    with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved figures in:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()