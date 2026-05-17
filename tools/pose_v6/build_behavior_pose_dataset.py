from pathlib import Path
import argparse
import pandas as pd


DROP_RPPG_COLS = [
    "rppg_dominant_freq",
    "rppg_hr_estimate",
    "rppg_snr",
    "rppg_signal_std",
    "rppg_skipped_rate",
    "rppg_valid",
]


def build(base_behav_csv, pose_csv, out_csv):
    base = pd.read_csv(base_behav_csv)
    pose = pd.read_csv(pose_csv)

    base["video_id"] = base["video_id"].astype(str)
    pose["video_id"] = pose["video_id"].astype(str)

    # Supprimer rPPG du cœur comportemental
    for col in DROP_RPPG_COLS:
        if col in base.columns:
            base = base.drop(columns=[col])

    # Garder label depuis base
    if "label" in pose.columns:
        pose = pose.drop(columns=["label"])

    out = base.merge(pose, on="video_id", how="left")

    missing = out.isna().sum().sum()
    if missing > 0:
        print(f"[WARN] Missing values after merge: {missing}. Filling with 0.")
        out = out.fillna(0.0)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")

    print("Saved:", out_path)
    print("Rows:", len(out))
    print("Videos:", out["video_id"].nunique())
    print("Columns:", out.columns.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_behav_csv", required=True)
    parser.add_argument("--pose_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    build(args.base_behav_csv, args.pose_csv, args.out_csv)


if __name__ == "__main__":
    main()
