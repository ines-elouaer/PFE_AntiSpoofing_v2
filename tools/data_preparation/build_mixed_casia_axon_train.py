from pathlib import Path
import sys
import json
import argparse
from typing import Optional, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


BEHAV_COLS = [
    "ear_mean",
    "ear_std",
    "ear_min",
    "ear_max",
    "blink_count",
    "perclos",
    "motion_mean",
    "motion_std",
    "motion_max",
    "lk_flow_mean",
    "lk_flow_std",
    "lk_flow_max",
    "mar_mean",
    "mar_std",
    "mar_max",
    "yaw_std",
    "pitch_std",
    "roll_std",
    "yaw_range",
    "pitch_range",
    "skipped_rate",
]


# ==========================================================
# PATH HELPERS
# ==========================================================

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


def first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_optional_path(value: str, project_root: Path) -> Optional[Path]:
    if not value:
        return None

    p = Path(value)

    if not p.is_absolute():
        p = (project_root / p).resolve()

    return p


def find_casia_frames_csv(project_root: Path, split: str) -> Optional[Path]:
    """
    Essaie plusieurs emplacements probables pour les CSV CASIA frame-level.
    CSV attendu : path,label,subject_id,video_id,frame_idx
    """

    candidates = [
        project_root / "data" / "processed" / "casia" / "splits_subject" / f"{split}.csv",
        project_root / "data" / "processed" / "casia" / "splits_subject" / f"casia_{split}.csv",
        project_root / "data" / "processed" / "casia" / "splits" / f"{split}.csv",
        project_root / "data" / "processed" / "casia" / "splits" / f"casia_{split}.csv",
        project_root / "data" / "processed" / "casia" / f"{split}.csv",
        project_root / "data" / "processed" / "casia" / f"casia_{split}.csv",
    ]

    return first_existing(candidates)


def find_casia_behav_csv(project_root: Path, split: str) -> Optional[Path]:
    """
    Essaie plusieurs emplacements probables pour les CSV behavior CASIA.
    CSV attendu : video_id,label + features comportementales.
    """

    candidates = [
        project_root / "data" / "processed" / "casia" / "behav" / f"behav_{split}.csv",
        project_root / "data" / "processed" / "casia" / "behav" / f"{split}_behav.csv",
        project_root / "data" / "processed" / "casia" / "behav" / f"casia_behav_{split}.csv",
        project_root / "data" / "processed" / "casia" / "behav" / f"casia_{split}_behav.csv",
        project_root / "data" / "processed" / "casia" / "behav" / f"{split}.csv",
    ]

    return first_existing(candidates)


# ==========================================================
# DATA HELPERS
# ==========================================================

def normalize_label_name(label: int) -> str:
    return "REAL" if int(label) == 0 else "SPOOF"


def ensure_frame_columns(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    df = df.copy()

    required = ["path", "label", "video_id", "frame_idx"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes obligatoires absentes dans frames {domain}: {missing}")

    if "subject_id" not in df.columns:
        df["subject_id"] = df["video_id"].astype(str)

    if "label_name" not in df.columns:
        df["label_name"] = df["label"].apply(normalize_label_name)

    if "attack_type" not in df.columns:
        df["attack_type"] = df["label"].apply(
            lambda x: "real_video" if int(x) == 0 else f"{domain.lower()}_spoof"
        )

    if "level" not in df.columns:
        df["level"] = domain

    df["domain"] = domain
    df["source_dataset"] = domain

    # Colonnes standard au début, puis les autres colonnes.
    front_cols = [
        "path",
        "label",
        "label_name",
        "subject_id",
        "video_id",
        "frame_idx",
        "attack_type",
        "level",
        "domain",
        "source_dataset",
    ]

    other_cols = [c for c in df.columns if c not in front_cols]
    return df[front_cols + other_cols]


def ensure_behav_columns(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    df = df.copy()

    required = ["video_id", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes obligatoires absentes dans behavior {domain}: {missing}")

    if "label_name" not in df.columns:
        df["label_name"] = df["label"].apply(normalize_label_name)

    if "subject_id" not in df.columns:
        df["subject_id"] = df["video_id"].astype(str)

    if "attack_type" not in df.columns:
        df["attack_type"] = df["label"].apply(
            lambda x: "real_video" if int(x) == 0 else f"{domain.lower()}_spoof"
        )

    if "level" not in df.columns:
        df["level"] = domain

    for col in BEHAV_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df["domain"] = domain
    df["source_dataset"] = domain

    front_cols = [
        "video_id",
        "label",
        "label_name",
        "attack_type",
        "level",
        "subject_id",
        "domain",
        "source_dataset",
    ]

    other_cols = [c for c in df.columns if c not in front_cols]
    return df[front_cols + other_cols]


def filter_behav_by_frames(behav_df: pd.DataFrame, frames_df: pd.DataFrame) -> pd.DataFrame:
    video_ids = set(frames_df["video_id"].astype(str).unique().tolist())

    out = behav_df.copy()
    out["video_id"] = out["video_id"].astype(str)
    out = out[out["video_id"].isin(video_ids)].copy()

    missing = video_ids - set(out["video_id"].unique().tolist())

    if missing:
        raise RuntimeError(
            f"Certaines vidéos frames n'ont pas de behavior features. Exemples: {list(missing)[:10]}"
        )

    return out


def align_and_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatène plusieurs DataFrames avec union des colonnes.
    Les colonnes absentes sont remplies par vide/0 selon pandas.
    """
    all_cols = []
    for df in dfs:
        for c in df.columns:
            if c not in all_cols:
                all_cols.append(c)

    aligned = []
    for df in dfs:
        x = df.copy()
        for c in all_cols:
            if c not in x.columns:
                x[c] = None
        aligned.append(x[all_cols])

    return pd.concat(aligned, ignore_index=True)


def video_level_counts(frames_df: pd.DataFrame):
    video_df = (
        frames_df.sort_values(["video_id", "frame_idx"])
        .groupby("video_id")
        .first()
        .reset_index()
    )

    out = {
        "videos": int(len(video_df)),
        "frames": int(len(frames_df)),
        "label_counts_videos": video_df["label_name"].value_counts().to_dict(),
        "domain_counts_videos": video_df["domain"].value_counts().to_dict(),
        "attack_type_counts_videos": video_df["attack_type"].value_counts().to_dict(),
    }

    return out


def check_unique_video_ids(frames_df: pd.DataFrame, name: str):
    duplicated = frames_df.groupby("video_id")["domain"].nunique()
    bad = duplicated[duplicated > 1]

    if len(bad) > 0:
        raise RuntimeError(
            f"Conflit video_id entre domaines dans {name}. Exemples: {bad.index[:10].tolist()}"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--casia_train_frames", type=str, default="")
    parser.add_argument("--casia_val_frames", type=str, default="")
    parser.add_argument("--casia_train_behav", type=str, default="")
    parser.add_argument("--casia_val_behav", type=str, default="")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/mixed_casia_axon",
    )

    args = parser.parse_args()

    project_root = PROJECT_ROOT
    axon_prepared_dir = load_paths(project_root)

    splits_dir = axon_prepared_dir / "manifests" / "splits"

    axon_train_frames_path = splits_dir / "axon_video_frames_train.csv"
    axon_val_frames_path = splits_dir / "axon_video_frames_val.csv"
    axon_train_behav_path = splits_dir / "axon_video_behav_train.csv"
    axon_val_behav_path = splits_dir / "axon_video_behav_val.csv"

    if not axon_train_frames_path.exists():
        raise FileNotFoundError(
            f"Axon train frames introuvable: {axon_train_frames_path}\n"
            "Lance d'abord: python tools\\axon\\build_axon_video_splits.py"
        )

    if not axon_val_frames_path.exists():
        raise FileNotFoundError(
            f"Axon val frames introuvable: {axon_val_frames_path}"
        )

    if not axon_train_behav_path.exists():
        raise FileNotFoundError(
            f"Axon train behavior introuvable: {axon_train_behav_path}"
        )

    if not axon_val_behav_path.exists():
        raise FileNotFoundError(
            f"Axon val behavior introuvable: {axon_val_behav_path}"
        )

    # CASIA paths
    casia_train_frames_path = resolve_optional_path(args.casia_train_frames, project_root)
    casia_val_frames_path = resolve_optional_path(args.casia_val_frames, project_root)
    casia_train_behav_path = resolve_optional_path(args.casia_train_behav, project_root)
    casia_val_behav_path = resolve_optional_path(args.casia_val_behav, project_root)

    if casia_train_frames_path is None:
        casia_train_frames_path = find_casia_frames_csv(project_root, "train")

    if casia_val_frames_path is None:
        casia_val_frames_path = find_casia_frames_csv(project_root, "val")

    if casia_train_behav_path is None:
        casia_train_behav_path = find_casia_behav_csv(project_root, "train")

    if casia_val_behav_path is None:
        casia_val_behav_path = find_casia_behav_csv(project_root, "val")

    missing_paths = {
        "casia_train_frames": casia_train_frames_path,
        "casia_val_frames": casia_val_frames_path,
        "casia_train_behav": casia_train_behav_path,
        "casia_val_behav": casia_val_behav_path,
    }

    for name, path in missing_paths.items():
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"{name} introuvable.\n"
                "Tu peux le fournir manuellement avec l'argument correspondant.\n"
                "Exemple:\n"
                "python tools\\axon\\build_mixed_casia_axon_train.py "
                "--casia_train_frames \"...\" --casia_val_frames \"...\" "
                "--casia_train_behav \"...\" --casia_val_behav \"...\"\n"
                f"Valeur détectée: {path}"
            )

    output_dir = resolve_optional_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    mixed_train_frames_path = output_dir / "mixed_train_frames.csv"
    mixed_val_frames_path = output_dir / "mixed_val_frames.csv"
    mixed_train_behav_path = output_dir / "mixed_train_behav.csv"
    mixed_val_behav_path = output_dir / "mixed_val_behav.csv"
    summary_path = output_dir / "mixed_summary.json"

    print("========== CONFIG ==========")
    print(f"Project root         : {project_root}")
    print(f"Output dir           : {output_dir}")

    print("\n========== CASIA INPUTS ==========")
    print(f"CASIA train frames   : {casia_train_frames_path}")
    print(f"CASIA val frames     : {casia_val_frames_path}")
    print(f"CASIA train behav    : {casia_train_behav_path}")
    print(f"CASIA val behav      : {casia_val_behav_path}")

    print("\n========== AXON INPUTS ==========")
    print(f"Axon train frames    : {axon_train_frames_path}")
    print(f"Axon val frames      : {axon_val_frames_path}")
    print(f"Axon train behav     : {axon_train_behav_path}")
    print(f"Axon val behav       : {axon_val_behav_path}")

    # Load
    casia_train_frames = pd.read_csv(casia_train_frames_path)
    casia_val_frames = pd.read_csv(casia_val_frames_path)
    casia_train_behav = pd.read_csv(casia_train_behav_path)
    casia_val_behav = pd.read_csv(casia_val_behav_path)

    axon_train_frames = pd.read_csv(axon_train_frames_path)
    axon_val_frames = pd.read_csv(axon_val_frames_path)
    axon_train_behav = pd.read_csv(axon_train_behav_path)
    axon_val_behav = pd.read_csv(axon_val_behav_path)

    # Normalize columns
    casia_train_frames = ensure_frame_columns(casia_train_frames, "CASIA")
    casia_val_frames = ensure_frame_columns(casia_val_frames, "CASIA")
    axon_train_frames = ensure_frame_columns(axon_train_frames, "AXON")
    axon_val_frames = ensure_frame_columns(axon_val_frames, "AXON")

    casia_train_behav = ensure_behav_columns(casia_train_behav, "CASIA")
    casia_val_behav = ensure_behav_columns(casia_val_behav, "CASIA")
    axon_train_behav = ensure_behav_columns(axon_train_behav, "AXON")
    axon_val_behav = ensure_behav_columns(axon_val_behav, "AXON")

    # Align behavior to frame video_ids
    casia_train_behav = filter_behav_by_frames(casia_train_behav, casia_train_frames)
    casia_val_behav = filter_behav_by_frames(casia_val_behav, casia_val_frames)
    axon_train_behav = filter_behav_by_frames(axon_train_behav, axon_train_frames)
    axon_val_behav = filter_behav_by_frames(axon_val_behav, axon_val_frames)

    # Combine
    mixed_train_frames = align_and_concat([casia_train_frames, axon_train_frames])
    mixed_val_frames = align_and_concat([casia_val_frames, axon_val_frames])

    mixed_train_behav = align_and_concat([casia_train_behav, axon_train_behav])
    mixed_val_behav = align_and_concat([casia_val_behav, axon_val_behav])

    check_unique_video_ids(mixed_train_frames, "mixed_train_frames")
    check_unique_video_ids(mixed_val_frames, "mixed_val_frames")

    # Save
    mixed_train_frames.to_csv(mixed_train_frames_path, index=False, encoding="utf-8")
    mixed_val_frames.to_csv(mixed_val_frames_path, index=False, encoding="utf-8")
    mixed_train_behav.to_csv(mixed_train_behav_path, index=False, encoding="utf-8")
    mixed_val_behav.to_csv(mixed_val_behav_path, index=False, encoding="utf-8")

    summary = {
        "inputs": {
            "casia_train_frames": str(casia_train_frames_path),
            "casia_val_frames": str(casia_val_frames_path),
            "casia_train_behav": str(casia_train_behav_path),
            "casia_val_behav": str(casia_val_behav_path),
            "axon_train_frames": str(axon_train_frames_path),
            "axon_val_frames": str(axon_val_frames_path),
            "axon_train_behav": str(axon_train_behav_path),
            "axon_val_behav": str(axon_val_behav_path),
        },
        "outputs": {
            "mixed_train_frames": str(mixed_train_frames_path),
            "mixed_val_frames": str(mixed_val_frames_path),
            "mixed_train_behav": str(mixed_train_behav_path),
            "mixed_val_behav": str(mixed_val_behav_path),
        },
        "train_summary": video_level_counts(mixed_train_frames),
        "val_summary": video_level_counts(mixed_val_frames),
        "behavior_rows": {
            "mixed_train_behav": int(len(mixed_train_behav)),
            "mixed_val_behav": int(len(mixed_val_behav)),
        },
        "notes": [
            "CASIA and Axon are combined without refitting behavior scaler.",
            "Axon behavior features are already normalized with CASIA train scaler.",
            "Final balancing must be done in DataLoader using stratified sampler.",
            "Do not train naively without sampler/class weights because Axon is highly imbalanced.",
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== MIXED TRAIN SUMMARY ==========")
    print(json.dumps(summary["train_summary"], indent=2, ensure_ascii=False))

    print("\n========== MIXED VAL SUMMARY ==========")
    print(json.dumps(summary["val_summary"], indent=2, ensure_ascii=False))

    print("\n========== OUTPUTS ==========")
    print(f"Mixed train frames : {mixed_train_frames_path}")
    print(f"Mixed val frames   : {mixed_val_frames_path}")
    print(f"Mixed train behav  : {mixed_train_behav_path}")
    print(f"Mixed val behav    : {mixed_val_behav_path}")
    print(f"Summary            : {summary_path}")

    print("\n[OK] Dataset mixte CASIA + Axon construit.")


if __name__ == "__main__":
    main()