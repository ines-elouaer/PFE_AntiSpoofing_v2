from pathlib import Path
import sys
import json
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_paths(project_root: Path):
    config_path = project_root / "configs" / "paths.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    axon_prepared_dir = Path(cfg["axon_prepared_dir"])

    if not axon_prepared_dir.is_absolute():
        axon_prepared_dir = (project_root / axon_prepared_dir).resolve()

    return axon_prepared_dir


def safe_stratified_split(video_df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split robuste par video_id.

    Priorité :
    1. Stratification par label + attack_type si possible.
    2. Si certaines classes sont trop petites, fallback par label.
    3. Si encore impossible, fallback manuel.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    df = video_df.copy()

    df["strata_attack"] = df["label_name"].astype(str) + "__" + df["attack_type"].astype(str)
    df["strata_label"] = df["label_name"].astype(str)

    # Split 1 : train / temp
    temp_ratio = val_ratio + test_ratio

    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=42,
            stratify=df["strata_attack"],
        )
        split_level_1 = "label_attack_type"
    except Exception:
        try:
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_ratio,
                random_state=42,
                stratify=df["strata_label"],
            )
            split_level_1 = "label_only"
        except Exception:
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_ratio,
                random_state=42,
                shuffle=True,
            )
            split_level_1 = "random_fallback"

    # Split 2 : val / test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            random_state=42,
            stratify=temp_df["strata_attack"],
        )
        split_level_2 = "label_attack_type"
    except Exception:
        try:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_ratio,
                random_state=42,
                stratify=temp_df["strata_label"],
            )
            split_level_2 = "label_only"
        except Exception:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_ratio,
                random_state=42,
                shuffle=True,
            )
            split_level_2 = "random_fallback"

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_video_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return split_video_df, {
        "split_level_1": split_level_1,
        "split_level_2": split_level_2,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }


def make_video_level_df(frames_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit le manifest frame-level en video-level.
    Une ligne = une vidéo.
    """

    video_df = (
        frames_df.sort_values(["video_id", "frame_idx"])
        .groupby("video_id")
        .first()
        .reset_index()
    )

    keep_cols = [
        "video_id",
        "label",
        "label_name",
        "subject_id",
        "attack_type",
        "level",
        "source_video_path",
        "source_manifest_row",
        "source_frame_count",
        "source_fps",
        "source_duration_sec",
    ]

    existing_cols = [c for c in keep_cols if c in video_df.columns]
    video_df = video_df[existing_cols].copy()

    return video_df


def attach_split_to_frames(frames_df: pd.DataFrame, split_video_df: pd.DataFrame) -> pd.DataFrame:
    split_map = split_video_df.set_index("video_id")["split"].to_dict()

    out = frames_df.copy()
    out["split"] = out["video_id"].map(split_map)

    if out["split"].isna().any():
        missing = out[out["split"].isna()]["video_id"].unique().tolist()
        raise RuntimeError(f"Certaines frames n'ont pas de split: {missing[:10]}")

    return out


def split_behavior_df(behav_df: pd.DataFrame, split_video_df: pd.DataFrame) -> pd.DataFrame:
    split_map = split_video_df.set_index("video_id")["split"].to_dict()

    out = behav_df.copy()
    out["split"] = out["video_id"].map(split_map)

    if out["split"].isna().any():
        missing = out[out["split"].isna()]["video_id"].unique().tolist()
        raise RuntimeError(f"Certaines features behavior n'ont pas de split: {missing[:10]}")

    return out


def count_summary(video_df: pd.DataFrame):
    summary = {}

    for split_name, g in video_df.groupby("split"):
        summary[split_name] = {
            "videos": int(len(g)),
            "label_counts": g["label_name"].value_counts().to_dict(),
            "attack_type_counts": g["attack_type"].value_counts().to_dict(),
            "level_counts": g["level"].value_counts().to_dict() if "level" in g.columns else {},
        }

    return summary


def check_no_overlap(split_video_df: pd.DataFrame):
    train_ids = set(split_video_df[split_video_df["split"] == "train"]["video_id"])
    val_ids = set(split_video_df[split_video_df["split"] == "val"]["video_id"])
    test_ids = set(split_video_df[split_video_df["split"] == "test"]["video_id"])

    return {
        "train_val_overlap": len(train_ids & val_ids),
        "train_test_overlap": len(train_ids & test_ids),
        "val_test_overlap": len(val_ids & test_ids),
    }


def main():
    project_root = PROJECT_ROOT
    axon_prepared_dir = load_paths(project_root)

    manifests_dir = axon_prepared_dir / "manifests"
    splits_dir = manifests_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    frames_csv = manifests_dir / "axon_video_frames_manifest.csv"
    behav_csv = manifests_dir / "axon_video_behav_norm.csv"

    if not frames_csv.exists():
        raise FileNotFoundError(f"Frames manifest introuvable: {frames_csv}")

    if not behav_csv.exists():
        raise FileNotFoundError(f"Behavior CSV introuvable: {behav_csv}")

    print("========== CONFIG ==========")
    print(f"Project root    : {project_root}")
    print(f"Frames CSV      : {frames_csv}")
    print(f"Behavior CSV    : {behav_csv}")
    print(f"Splits output   : {splits_dir}")
    print("Split ratios    : train=70%, val=15%, test=15%")

    frames_df = pd.read_csv(frames_csv)
    behav_df = pd.read_csv(behav_csv)

    print("\n========== INPUT ==========")
    print(f"Frame rows      : {len(frames_df)}")
    print(f"Unique videos   : {frames_df['video_id'].nunique()}")
    print(f"Behavior videos : {len(behav_df)}")

    video_df = make_video_level_df(frames_df)

    print("\n========== VIDEO-LEVEL LABELS ==========")
    print(video_df["label_name"].value_counts())

    print("\n========== VIDEO-LEVEL ATTACK TYPES ==========")
    print(video_df["attack_type"].value_counts())

    split_video_df, split_info = safe_stratified_split(
        video_df=video_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    frames_with_split = attach_split_to_frames(frames_df, split_video_df)
    behav_with_split = split_behavior_df(behav_df, split_video_df)

    # Sorties video-level
    split_video_path = splits_dir / "axon_video_split_video_level.csv"

    # Sorties frame-level
    frames_train_path = splits_dir / "axon_video_frames_train.csv"
    frames_val_path = splits_dir / "axon_video_frames_val.csv"
    frames_test_path = splits_dir / "axon_video_frames_test.csv"

    # Sorties behavior
    behav_train_path = splits_dir / "axon_video_behav_train.csv"
    behav_val_path = splits_dir / "axon_video_behav_val.csv"
    behav_test_path = splits_dir / "axon_video_behav_test.csv"

    # Summary
    summary_path = splits_dir / "axon_video_split_summary.json"

    split_video_df.to_csv(split_video_path, index=False, encoding="utf-8")

    frames_with_split[frames_with_split["split"] == "train"].to_csv(
        frames_train_path, index=False, encoding="utf-8"
    )
    frames_with_split[frames_with_split["split"] == "val"].to_csv(
        frames_val_path, index=False, encoding="utf-8"
    )
    frames_with_split[frames_with_split["split"] == "test"].to_csv(
        frames_test_path, index=False, encoding="utf-8"
    )

    behav_with_split[behav_with_split["split"] == "train"].to_csv(
        behav_train_path, index=False, encoding="utf-8"
    )
    behav_with_split[behav_with_split["split"] == "val"].to_csv(
        behav_val_path, index=False, encoding="utf-8"
    )
    behav_with_split[behav_with_split["split"] == "test"].to_csv(
        behav_test_path, index=False, encoding="utf-8"
    )

    overlap = check_no_overlap(split_video_df)
    counts = count_summary(split_video_df)

    summary = {
        "input_frames_csv": str(frames_csv),
        "input_behavior_csv": str(behav_csv),
        "split_info": split_info,
        "total_videos": int(len(split_video_df)),
        "total_frames": int(len(frames_df)),
        "overlap": overlap,
        "counts": counts,
        "outputs": {
            "video_level": str(split_video_path),
            "frames_train": str(frames_train_path),
            "frames_val": str(frames_val_path),
            "frames_test": str(frames_test_path),
            "behav_train": str(behav_train_path),
            "behav_val": str(behav_val_path),
            "behav_test": str(behav_test_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== SPLIT SUMMARY ==========")
    print(f"Split level 1 : {split_info['split_level_1']}")
    print(f"Split level 2 : {split_info['split_level_2']}")

    for split_name in ["train", "val", "test"]:
        g = split_video_df[split_video_df["split"] == split_name]

        print(f"\n--- {split_name.upper()} ---")
        print(f"Videos: {len(g)}")
        print("Labels:")
        print(g["label_name"].value_counts())
        print("Attack types:")
        print(g["attack_type"].value_counts())

    print("\n========== OVERLAP CHECK ==========")
    print(overlap)

    print("\n========== OUTPUTS ==========")
    print(f"Video-level split : {split_video_path}")
    print(f"Frames train      : {frames_train_path}")
    print(f"Frames val        : {frames_val_path}")
    print(f"Frames test       : {frames_test_path}")
    print(f"Behavior train    : {behav_train_path}")
    print(f"Behavior val      : {behav_val_path}")
    print(f"Behavior test     : {behav_test_path}")
    print(f"Summary           : {summary_path}")

    print("\n[OK] Splits Axon vidéo créés.")


if __name__ == "__main__":
    main()