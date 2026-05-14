from pathlib import Path
import json
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def select_center_consecutive(group: pd.DataFrame, frames_per_video: int = 16) -> pd.DataFrame:
    group = group.sort_values("frame_idx").reset_index(drop=True)
    n = len(group)

    if n >= frames_per_video:
        start = max(0, n // 2 - frames_per_video // 2)
        selected = group.iloc[start:start + frames_per_video].copy()
    else:
        selected = group.copy()
        while len(selected) < frames_per_video:
            selected = pd.concat([selected, group.iloc[[-1]].copy()], ignore_index=True)

    selected = selected.reset_index(drop=True)
    selected["source_frame_index"] = selected["frame_idx"].astype(int)
    selected["frame_idx"] = list(range(len(selected)))
    selected["sampling_mode"] = "center_consecutive"
    return selected


def assign_splits_by_subject(subjects):
    subjects = sorted(set(subjects))
    n = len(subjects)

    if n < 3:
        raise RuntimeError("Il faut au moins 3 sujets pour train/val/test.")

    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15)))

    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1

    split_map = {}

    for s in subjects[:n_train]:
        split_map[s] = "train"

    for s in subjects[n_train:n_train + n_val]:
        split_map[s] = "val"

    for s in subjects[n_train + n_val:]:
        split_map[s] = "test"

    return split_map


def main():
    project_root = get_project_root()

    in_csv = project_root / "data" / "msu_mfsd_prepared" / "manifests" / "msu_real_frames_manifest.csv"
    out_dir = project_root / "data" / "msu_mfsd_prepared" / "manifests"
    splits_dir = out_dir / "splits"

    out_csv = out_dir / "msu_real_16_frames_manifest.csv"
    out_summary = out_dir / "msu_real_16_frames_summary.json"

    splits_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Manifest introuvable: {in_csv}")

    df = pd.read_csv(in_csv)

    required = {"path", "label", "label_name", "subject_id", "device_id", "video_id", "frame_idx"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Colonnes manquantes dans MSU manifest: {missing}")

    split_map = assign_splits_by_subject(df["subject_id"].unique())
    df["split"] = df["subject_id"].map(split_map)

    selected_groups = []

    for video_id, group in df.groupby("video_id"):
        selected = select_center_consecutive(group, frames_per_video=16)
        selected_groups.append(selected)

    out_df = pd.concat(selected_groups, ignore_index=True)

    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    for split in ["train", "val", "test"]:
        split_df = out_df[out_df["split"] == split].copy()
        split_df.to_csv(
            splits_dir / f"msu_real_16_frames_{split}.csv",
            index=False,
            encoding="utf-8",
        )

    video_df = out_df.drop_duplicates("video_id")

    summary = {
        "input_manifest": str(in_csv),
        "output_manifest": str(out_csv),
        "frames_per_video": 16,
        "sampling_mode": "center_consecutive",
        "frame_rows": int(len(out_df)),
        "videos": int(video_df["video_id"].nunique()),
        "subjects": int(video_df["subject_id"].nunique()),
        "video_counts_by_split": video_df["split"].value_counts().to_dict(),
        "video_counts_by_device": video_df["device_id"].value_counts().to_dict(),
        "video_counts_by_split_device": {
            f"{k[0]}::{k[1]}": int(v)
            for k, v in video_df.groupby(["split", "device_id"]).size().items()
        },
        "subjects_by_split": {
            split: sorted(video_df[video_df["split"] == split]["subject_id"].unique().tolist())
            for split in ["train", "val", "test"]
        },
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== MSU REAL 16 FRAMES SUMMARY ==========")
    print(f"Output manifest : {out_csv}")
    print(f"Summary         : {out_summary}")
    print(f"Videos          : {summary['videos']}")
    print(f"Subjects        : {summary['subjects']}")
    print(f"Frame rows      : {summary['frame_rows']}")

    print("\n========== VIDEO COUNTS BY SPLIT ==========")
    print(video_df["split"].value_counts())

    print("\n========== DEVICE COUNTS ==========")
    print(video_df["device_id"].value_counts())

    print("\n========== SPLIT x DEVICE ==========")
    print(video_df.groupby(["split", "device_id"]).size())

    print("\n[OK] MSU_REAL converti en 16 frames center_consecutive.")


if __name__ == "__main__":
    main()