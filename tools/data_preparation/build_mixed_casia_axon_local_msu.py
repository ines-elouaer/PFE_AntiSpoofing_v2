from pathlib import Path
import json
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


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = project_root()


def read_csv_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} introuvable: {path}")
    return pd.read_csv(path)


def ensure_cols(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    df = df.copy()
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    return df


def normalize_label_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label_name" not in df.columns:
        df["label_name"] = df["label"].map({0: "REAL", 1: "SPOOF"}).fillna("UNKNOWN")
    return df


def prefix_video_ids(df: pd.DataFrame, source_dataset: str) -> pd.DataFrame:
    """
    Évite les collisions de video_id entre CASIA, AXON, LOCAL_REAL, MSU.
    """
    df = df.copy()
    df["original_video_id"] = df["video_id"].astype(str)
    df["video_id"] = source_dataset + "__" + df["original_video_id"].astype(str)
    return df


def prepare_frames(df: pd.DataFrame, source_dataset: str, split: str) -> pd.DataFrame:
    df = df.copy()

    required = {"path", "label", "video_id", "frame_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Frames {source_dataset}/{split}: colonnes manquantes {missing}")

    df = normalize_label_name(df)
    df = prefix_video_ids(df, source_dataset)

    df = ensure_cols(df, {
        "subject_id": "unknown_subject",
        "device_id": "unknown_device",
        "condition": "unknown",
        "take": "unknown",
        "attack_type": "real_video",
        "domain": source_dataset,
        "split": split,
        "source_dataset": source_dataset,
    })

    df["source_dataset"] = source_dataset
    df["domain"] = source_dataset
    df["split"] = split

    keep_cols = [
        "path",
        "label",
        "label_name",
        "subject_id",
        "device_id",
        "video_id",
        "original_video_id",
        "frame_idx",
        "source_frame_index",
        "condition",
        "take",
        "attack_type",
        "domain",
        "source_dataset",
        "split",
    ]

    for c in keep_cols:
        if c not in df.columns:
            df[c] = "unknown"

    df["label"] = df["label"].astype(int)
    df["frame_idx"] = df["frame_idx"].astype(int)

    return df[keep_cols].copy()


def prepare_behav(df: pd.DataFrame, source_dataset: str, split: str) -> pd.DataFrame:
    df = df.copy()

    required = {"video_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Behav {source_dataset}/{split}: colonnes manquantes {missing}")

    df = normalize_label_name(df)
    df = prefix_video_ids(df, source_dataset)

    df = ensure_cols(df, {
        "subject_id": "unknown_subject",
        "device_id": "unknown_device",
        "condition": "unknown",
        "take": "unknown",
        "attack_type": "real_video",
        "domain": source_dataset,
        "split": split,
        "source_dataset": source_dataset,
    })

    for col in FEAT_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df[FEAT_COLS] = (
        df[FEAT_COLS]
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .astype(float)
    )

    df["source_dataset"] = source_dataset
    df["domain"] = source_dataset
    df["split"] = split
    df["label"] = df["label"].astype(int)

    keep_cols = [
        "video_id",
        "original_video_id",
        "label",
        "label_name",
        "subject_id",
        "device_id",
        "condition",
        "take",
        "attack_type",
        "domain",
        "source_dataset",
        "split",
    ] + FEAT_COLS

    for c in keep_cols:
        if c not in df.columns:
            df[c] = "unknown"

    return df[keep_cols].copy()


def find_axon_behav(split: str) -> Path:
    base = ROOT / "data" / "axon_prepared" / "manifests" / "splits"

    candidates = [
        base / f"axon_video_behav_{split}.csv",
        base / f"axon_video_behav_norm_{split}.csv",
        base / f"axon_video_{split}_behav.csv",
        base / f"axon_video_{split}_behav_norm.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Aucun fichier behavior Axon trouvé pour split={split}. "
        f"Candidats: {[str(p) for p in candidates]}"
    )


def load_casia(split: str):
    frames_path = ROOT / "data" / "processed" / "casia" / "splits_subject" / f"{split}.csv"
    behav_path = ROOT / "data" / "processed" / "casia" / "behav" / f"{split}_behav_norm.csv"

    frames = read_csv_required(frames_path, f"CASIA frames {split}")
    behav = read_csv_required(behav_path, f"CASIA behav {split}")

    frames = ensure_cols(frames, {
        "device_id": "casia_camera",
        "condition": "unknown",
        "take": "unknown",
        "attack_type": "casia_real_or_spoof",
    })

    behav = ensure_cols(behav, {
        "device_id": "casia_camera",
        "condition": "unknown",
        "take": "unknown",
        "attack_type": "casia_real_or_spoof",
    })

    return (
        prepare_frames(frames, "CASIA", split),
        prepare_behav(behav, "CASIA", split),
    )


def load_axon(split: str):
    frames_path = ROOT / "data" / "axon_prepared" / "manifests" / "splits" / f"axon_video_frames_{split}.csv"
    behav_path = find_axon_behav(split)

    frames = read_csv_required(frames_path, f"AXON frames {split}")
    behav = read_csv_required(behav_path, f"AXON behav {split}")

    frames = ensure_cols(frames, {
        "device_id": "axon_camera",
        "condition": "unknown",
        "take": "unknown",
    })

    behav = ensure_cols(behav, {
        "device_id": "axon_camera",
        "condition": "unknown",
        "take": "unknown",
    })

    return (
        prepare_frames(frames, "AXON", split),
        prepare_behav(behav, "AXON", split),
    )


def load_local_real(split: str):
    frames_path = ROOT / "data" / "local_real_prepared" / "manifests" / "local_real_frames_manifest_clean.csv"
    behav_path = ROOT / "data" / "local_real_prepared" / "manifests" / "local_real_behav_norm_clean.csv"

    frames_all = read_csv_required(frames_path, "LOCAL_REAL frames clean")
    behav_all = read_csv_required(behav_path, "LOCAL_REAL behav clean")

    frames = frames_all[frames_all["split"] == split].copy()
    behav = behav_all[behav_all["split"] == split].copy()

    return (
        prepare_frames(frames, "LOCAL_REAL", split),
        prepare_behav(behav, "LOCAL_REAL", split),
    )


def load_msu_real(split: str):
    frames_path = ROOT / "data" / "msu_mfsd_prepared" / "manifests" / "msu_real_16_frames_manifest_clean.csv"
    behav_path = ROOT / "data" / "msu_mfsd_prepared" / "manifests" / "msu_real_behav_norm_clean.csv"

    frames_all = read_csv_required(frames_path, "MSU_REAL frames clean")
    behav_all = read_csv_required(behav_path, "MSU_REAL behav clean")

    frames = frames_all[frames_all["split"] == split].copy()
    behav = behav_all[behav_all["split"] == split].copy()

    return (
        prepare_frames(frames, "MSU_MFSD", split),
        prepare_behav(behav, "MSU_MFSD", split),
    )


def validate_alignment(frames_df: pd.DataFrame, behav_df: pd.DataFrame, split: str):
    frame_videos = set(frames_df["video_id"].astype(str).unique())
    behav_videos = set(behav_df["video_id"].astype(str).unique())

    missing_behav = sorted(frame_videos - behav_videos)
    missing_frames = sorted(behav_videos - frame_videos)

    if missing_behav:
        raise RuntimeError(
            f"[{split}] Certaines vidéos frames n'ont pas de behavior: {missing_behav[:10]}"
        )

    if missing_frames:
        print(
            f"[WARN] [{split}] Certains behavior n'ont pas de frames "
            f"et seront ignorés: {missing_frames[:10]}"
        )
        behav_df = behav_df[behav_df["video_id"].isin(frame_videos)].copy()

    return behav_df


def build_split(split: str):
    frames_parts = []
    behav_parts = []

    for loader in [load_casia, load_axon, load_local_real, load_msu_real]:
        frames, behav = loader(split)
        frames_parts.append(frames)
        behav_parts.append(behav)

    frames_df = pd.concat(frames_parts, ignore_index=True)
    behav_df = pd.concat(behav_parts, ignore_index=True)

    behav_df = validate_alignment(frames_df, behav_df, split)

    frames_df = frames_df.sort_values(
        ["source_dataset", "video_id", "frame_idx"]
    ).reset_index(drop=True)

    behav_df = behav_df.sort_values(
        ["source_dataset", "video_id"]
    ).reset_index(drop=True)

    return frames_df, behav_df


def summarize_split(frames_df: pd.DataFrame, behav_df: pd.DataFrame):
    video_df = frames_df.drop_duplicates("video_id").copy()

    summary = {
        "num_frame_rows": int(len(frames_df)),
        "num_behavior_rows": int(len(behav_df)),
        "num_videos": int(video_df["video_id"].nunique()),
        "label_counts_video_level": video_df["label_name"].value_counts().to_dict(),
        "source_counts_video_level": video_df["source_dataset"].value_counts().to_dict(),
        "domain_label_counts": {
            f"{k[0]}::{k[1]}": int(v)
            for k, v in video_df.groupby(["source_dataset", "label_name"]).size().items()
        },
        "device_counts_video_level": video_df["device_id"].value_counts().to_dict(),
        "frames_per_video_min": int(frames_df.groupby("video_id")["frame_idx"].nunique().min()),
        "frames_per_video_max": int(frames_df.groupby("video_id")["frame_idx"].nunique().max()),
    }

    return summary


def main():
    out_dir = ROOT / "data" / "mixed_casia_axon_local_msu"
    out_dir.mkdir(parents=True, exist_ok=True)

    global_summary = {
        "output_dir": str(out_dir),
        "feature_cols": FEAT_COLS,
        "splits": {},
        "notes": [
            "video_id is prefixed by source_dataset to avoid collisions.",
            "LOCAL_REAL and MSU_MFSD are REAL-only complementary domains.",
            "CASIA and AXON preserve REAL and SPOOF samples.",
            "Use balanced sampler during fine-tuning; do not train naively on raw distribution.",
        ],
    }

    for split in ["train", "val", "test"]:
        print(f"\n========== BUILD MIXED SPLIT: {split.upper()} ==========")

        frames_df, behav_df = build_split(split)

        frames_out = out_dir / f"mixed_{split}_frames.csv"
        behav_out = out_dir / f"mixed_{split}_behav.csv"

        frames_df.to_csv(frames_out, index=False, encoding="utf-8")
        behav_df.to_csv(behav_out, index=False, encoding="utf-8")

        split_summary = summarize_split(frames_df, behav_df)
        global_summary["splits"][split] = split_summary

        print(f"Frames CSV : {frames_out}")
        print(f"Behav CSV  : {behav_out}")
        print(f"Frame rows : {split_summary['num_frame_rows']}")
        print(f"Videos     : {split_summary['num_videos']}")
        print("Labels     :", split_summary["label_counts_video_level"])
        print("Sources    :", split_summary["source_counts_video_level"])
        print("Frames/video min-max:",
              split_summary["frames_per_video_min"],
              split_summary["frames_per_video_max"])

    summary_out = out_dir / "mixed_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print("\n========== DONE ==========")
    print(f"Summary: {summary_out}")
    print("[OK] Dataset mixte CASIA + AXON + LOCAL_REAL + MSU_MFSD généré.")


if __name__ == "__main__":
    main()