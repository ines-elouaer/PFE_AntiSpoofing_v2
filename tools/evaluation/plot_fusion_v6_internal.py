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

    ax.set_title("Internal V6 - Confusion Matrix")
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
        "score_behavior_pose",
        "score_fusion_v6",
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

    ax.set_title("Internal V6 - Mean Scores by Branch")
    ax.set_ylabel("Mean spoof score")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["score_type"], rotation=20)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_policy_counts(df, out_path, out_csv):
    if "decision_v6" not in df.columns:
        print("[WARN] decision_v6 column not found. Skipping policy figure.")
        return

    counts = df["decision_v6"].value_counts().reset_index()
    counts.columns = ["decision", "count"]

    counts.to_csv(out_csv, index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts["decision"], counts["count"])

    ax.set_title("Internal V6 - Banking Policy Decisions")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Number of videos")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_visual_vs_fusion_scores(df, out_path):
    sorted_df = df.copy()
    sorted_df = sorted_df.sort_values("score_visual").reset_index(drop=True)

    x = np.arange(len(sorted_df))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, sorted_df["score_visual"], marker="o", linewidth=1, markersize=3, label="V3 visual score")
    ax.plot(x, sorted_df["score_fusion_v6"], marker="o", linewidth=1, markersize=3, label="V6 fusion score")
    ax.axhline(0.5, linestyle="--", label="Threshold 0.5")

    ax.set_title("Internal V6 - Visual Score vs Fusion Score")
    ax.set_xlabel("Videos sorted by V3 visual score")
    ax.set_ylabel("Spoof score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_error_cases_csv(df, out_csv):
    out = df.copy()

    out["case"] = "UNKNOWN"

    out.loc[(out["label"] == 0) & (out["pred_fusion_v6"] == 0), "case"] = "REAL_TO_REAL"
    out.loc[(out["label"] == 0) & (out["pred_fusion_v6"] == 1), "case"] = "REAL_TO_SPOOF"
    out.loc[(out["label"] == 1) & (out["pred_fusion_v6"] == 0), "case"] = "SPOOF_TO_REAL"
    out.loc[(out["label"] == 1) & (out["pred_fusion_v6"] == 1), "case"] = "SPOOF_TO_SPOOF"

    cols = [
        "video_id",
        "label",
        "score_visual",
        "score_behavior_pose",
        "score_fusion_v6",
        "pred_fusion_v6",
        "decision_v6",
        "decision_reason",
        "case",
    ]

    cols = [c for c in cols if c in out.columns]
    out[cols].to_csv(out_csv, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.scores_csv)

    required = {
        "label",
        "score_visual",
        "score_behavior_pose",
        "score_fusion_v6",
        "pred_fusion_v6",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Columns={df.columns.tolist()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(
        df,
        out_dir / "internal_v6_confusion_matrix.png",
    )

    save_roc(
        df,
        "score_fusion_v6",
        out_dir / "internal_v6_roc_curve.png",
        "Internal V6 - ROC Curve",
    )

    save_score_distribution(
        df,
        "score_fusion_v6",
        out_dir / "internal_v6_score_distribution.png",
        "Internal V6 - Score Distribution",
    )

    save_branch_score_comparison(
        df,
        out_dir / "internal_v6_branch_score_comparison.png",
        out_dir / "internal_v6_branch_score_comparison.csv",
    )

    save_policy_counts(
        df,
        out_dir / "internal_v6_policy_counts.png",
        out_dir / "internal_v6_policy_counts.csv",
    )

    save_visual_vs_fusion_scores(
        df,
        out_dir / "internal_v6_visual_vs_fusion_scores.png",
    )

    save_error_cases_csv(
        df,
        out_dir / "internal_v6_cases_summary.csv",
    )

    print("Saved internal V6 figures in:", out_dir)


if __name__ == "__main__":
    main()
