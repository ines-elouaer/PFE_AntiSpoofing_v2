from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


FEAT_COLS = [
    "ear_mean",
    "ear_std",
    "ear_min",
    "ear_max",
    "blink_count",
    "motion_mean",
    "motion_std",
    "motion_max",
    "skipped_rate",
]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str, project_root: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in FEAT_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df[FEAT_COLS] = (
        df[FEAT_COLS]
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .astype(float)
    )

    return df


def compute_casia_train_stats(casia_train_behav_csv: Path):
    train = pd.read_csv(casia_train_behav_csv)
    train = clean_features(train)

    x = train[FEAT_COLS].to_numpy(dtype=np.float64)

    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)

    std[std < 1e-12] = 1.0

    stats = {
        "source": str(casia_train_behav_csv),
        "feature_cols": FEAT_COLS,
        "mean": {col: float(mean[i]) for i, col in enumerate(FEAT_COLS)},
        "std": {col: float(std[i]) for i, col in enumerate(FEAT_COLS)},
    }

    return mean, std, stats


def normalize_file(input_csv: Path, output_csv: Path, mean, std):
    df = pd.read_csv(input_csv)
    df = clean_features(df)

    x = df[FEAT_COLS].to_numpy(dtype=np.float64)
    x_norm = (x - mean) / std

    for i, col in enumerate(FEAT_COLS):
        df[col] = x_norm[:, i]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    # Sauvegarde split files si colonne split existe
    if "split" in df.columns:
        split_dir = output_csv.parent / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)

        prefix = output_csv.stem.replace("_behav_norm", "")

        for split in ["train", "val", "test"]:
            split_df = df[df["split"] == split].copy()
            split_df.to_csv(
                split_dir / f"{prefix}_behav_{split}.csv",
                index=False,
                encoding="utf-8",
            )

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--casia_train_behav_csv",
        default="data/processed/casia/behav/train_behav.csv",
    )
    parser.add_argument(
        "--stats_out_json",
        default="data/processed/casia/behav/behav_scaler_9feat_stats.json",
    )

    args = parser.parse_args()

    project_root = get_project_root()

    input_csv = resolve_path(args.input_csv, project_root)
    output_csv = resolve_path(args.output_csv, project_root)
    casia_train_behav_csv = resolve_path(args.casia_train_behav_csv, project_root)
    stats_out_json = resolve_path(args.stats_out_json, project_root)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV introuvable: {input_csv}")

    if not casia_train_behav_csv.exists():
        raise FileNotFoundError(f"CASIA train behavior introuvable: {casia_train_behav_csv}")

    mean, std, stats = compute_casia_train_stats(casia_train_behav_csv)

    stats_out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    df_norm = normalize_file(
        input_csv=input_csv,
        output_csv=output_csv,
        mean=mean,
        std=std,
    )

    print("\n========== NORMALIZATION 9 FEATURES ==========")
    print(f"Input CSV      : {input_csv}")
    print(f"Output CSV     : {output_csv}")
    print(f"CASIA train    : {casia_train_behav_csv}")
    print(f"Stats JSON     : {stats_out_json}")
    print(f"Rows           : {len(df_norm)}")
    print(f"Videos         : {df_norm['video_id'].nunique() if 'video_id' in df_norm.columns else 'unknown'}")

    if "split" in df_norm.columns:
        print("\n========== SPLIT COUNTS ==========")
        print(df_norm["split"].value_counts())

    if "device_id" in df_norm.columns:
        print("\n========== DEVICE COUNTS ==========")
        print(df_norm["device_id"].value_counts())

    print("\n========== NORMALIZED FEATURE DESCRIBE ==========")
    print(df_norm[FEAT_COLS].describe().round(3).to_string())

    print("\n[OK] Normalisation terminée.")


if __name__ == "__main__":
    main()