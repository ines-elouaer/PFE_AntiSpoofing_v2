from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


FEATURES = [
    "ear_mean",
    "ear_std",
    "ear_min",
    "ear_max",
    "blink_count",
    "motion_mean",
    "motion_std",
    "motion_max",
    "skipped_rate",
    "rppg_dominant_freq",
    "rppg_hr_estimate",
    "rppg_snr",
    "rppg_signal_std",
    "rppg_skipped_rate",
    "rppg_valid",
]


def normalize_prediction_columns(pred: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit que predictions.csv contient bien :
    video_id, label, pred_label, score_spoof
    """
    pred = pred.copy()

    if "video_id" not in pred.columns:
        raise ValueError(f"Colonne video_id absente dans predictions.csv. Colonnes: {pred.columns.tolist()}")

    if "label" not in pred.columns:
        raise ValueError(f"Colonne label absente dans predictions.csv. Colonnes: {pred.columns.tolist()}")

    if "pred_label" not in pred.columns:
        raise ValueError(f"Colonne pred_label absente dans predictions.csv. Colonnes: {pred.columns.tolist()}")

    if "score_spoof" not in pred.columns:
        raise ValueError(f"Colonne score_spoof absente dans predictions.csv. Colonnes: {pred.columns.tolist()}")

    pred["video_id"] = pred["video_id"].astype(str)
    pred["label"] = pred["label"].astype(int)
    pred["pred_label"] = pred["pred_label"].astype(int)

    return pred


def normalize_behav_columns(behav: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie behav/rPPG pour éviter les conflits après merge.
    Si behav contient label, pred_label, score_spoof, on les supprime,
    car la référence doit être predictions.csv.
    """
    behav = behav.copy()

    if "video_id" not in behav.columns:
        raise ValueError(f"Colonne video_id absente dans behav_csv. Colonnes: {behav.columns.tolist()}")

    behav["video_id"] = behav["video_id"].astype(str)

    # Colonnes à supprimer pour éviter label_x / label_y
    conflict_cols = [
        "label",
        "pred_label",
        "pred_label_name",
        "label_name",
        "score_spoof",
    ]

    for col in conflict_cols:
        if col in behav.columns:
            behav = behav.drop(columns=[col])

    return behav


def decision_case(row):
    y = int(row["label"])
    p = int(row["pred_label"])

    if y == 0 and p == 0:
        return "REAL_TO_REAL"
    if y == 0 and p == 1:
        return "REAL_TO_SPOOF"
    if y == 1 and p == 0:
        return "SPOOF_TO_REAL"
    if y == 1 and p == 1:
        return "SPOOF_TO_SPOOF"

    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--behav_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    behav = pd.read_csv(args.behav_csv)

    pred = normalize_prediction_columns(pred)
    behav = normalize_behav_columns(behav)

    print("========== INPUT CHECK ==========")
    print("Predictions rows:", len(pred))
    print("Behavior rows   :", len(behav))
    print("Prediction videos:", pred["video_id"].nunique())
    print("Behavior videos  :", behav["video_id"].nunique())

    df = pred.merge(behav, on="video_id", how="left")

    if "label" not in df.columns:
        raise RuntimeError(f"Après merge, label est absent. Colonnes: {df.columns.tolist()}")

    if "pred_label" not in df.columns:
        raise RuntimeError(f"Après merge, pred_label est absent. Colonnes: {df.columns.tolist()}")

    df["case"] = df.apply(decision_case, axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available = [c for c in FEATURES if c in df.columns]
    missing_features = [c for c in FEATURES if c not in df.columns]

    print("\nAvailable behavior/rPPG features:", available)
    if missing_features:
        print("Missing features:", missing_features)

    # 1) Moyennes par cas
    mean_by_case = df.groupby("case")[available].mean().T
    mean_path = out_dir / "behavior_rppg_mean_by_case.csv"
    mean_by_case.to_csv(mean_path, encoding="utf-8")

    # 2) Écarts-types par cas
    std_by_case = df.groupby("case")[available].std().T
    std_path = out_dir / "behavior_rppg_std_by_case.csv"
    std_by_case.to_csv(std_path, encoding="utf-8")

    # 3) Counts par cas
    counts = df.groupby("case").size().reset_index(name="count")
    counts_path = out_dir / "case_counts.csv"
    counts.to_csv(counts_path, index=False, encoding="utf-8")

    # 4) Tableau complet fusionné
    merged_path = out_dir / "predictions_with_behavior_rppg_cases.csv"
    df.to_csv(merged_path, index=False, encoding="utf-8")

    print("\n========== CASE COUNTS ==========")
    print(counts.to_string(index=False))

    print("\n========== MEAN BY CASE ==========")
    print(mean_by_case.to_string())

    # 5) Figures simples pour features importantes
    important_features = [
        "blink_count",
        "ear_std",
        "motion_mean",
        "motion_std",
        "motion_max",
        "rppg_snr",
        "rppg_signal_std",
        "rppg_valid",
    ]

    for feat in important_features:
        if feat not in df.columns:
            continue

        plot_df = df.groupby("case")[feat].mean().reset_index()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(plot_df["case"], plot_df[feat])
        ax.set_title(f"Mean {feat} by decision case")
        ax.set_xlabel("Decision case")
        ax.set_ylabel(feat)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"{feat}_by_case.png", dpi=300)
        plt.close(fig)

    print("\nSaved:")
    print(mean_path)
    print(std_path)
    print(counts_path)
    print(merged_path)


if __name__ == "__main__":
    main()
