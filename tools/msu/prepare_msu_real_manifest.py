from pathlib import Path
import json
import re
import pandas as pd
from collections import Counter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_paths(project_root: Path):
    config_path = project_root / "configs" / "paths.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    raw_dir = Path(cfg["msu_mfsd_raw_dir"])
    prepared_dir = Path(cfg["msu_mfsd_prepared_dir"])

    if not raw_dir.is_absolute():
        raw_dir = (project_root / raw_dir).resolve()

    if not prepared_dir.is_absolute():
        prepared_dir = (project_root / prepared_dir).resolve()

    return raw_dir, prepared_dir


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except Exception:
        # Si les frames MSU restent hors projet, on garde le chemin absolu.
        return str(path.resolve()).replace("\\", "/")


def parse_msu_video_dir(video_dir: Path):
    """
    Exemples attendus :
    real_client024_android_SD_scene01
    real_client002_laptop_SD_scene01

    On récupère :
    subject_id = client024
    device_id  = android / laptop / unknown
    video_id   = nom du dossier
    """
    name = video_dir.name.lower()
    video_id = video_dir.name

    m_client = re.search(r"client\d+", name)
    subject_id = m_client.group(0) if m_client else "unknown_subject"

    if "android" in name:
        device_id = "android"
    elif "laptop" in name:
        device_id = "laptop"
    elif "iphone" in name:
        device_id = "iphone"
    elif "ipad" in name:
        device_id = "tablet"
    else:
        device_id = "unknown_device"

    return {
        "video_id": video_id,
        "subject_id": subject_id,
        "device_id": device_id,
    }


def frame_index_from_name(path: Path):
    """
    frame_0000.png -> 0
    """
    m = re.search(r"(\d+)", path.stem)
    if not m:
        return None
    return int(m.group(1))


def infer_source_split(path: Path):
    parts = [p.lower() for p in path.parts]

    if "train" in parts:
        return "train"
    if "test" in parts:
        return "test"
    if "val" in parts or "dev" in parts or "devel" in parts:
        return "val"

    return "unknown"


def assign_split_from_msu_source(source_split: str):
    """
    On garde la logique Kaggle/MSU si présente.
    Mais comme MSU_REAL est complémentaire, on peut mapper :
    train -> train
    test  -> test
    unknown -> train
    """
    if source_split == "train":
        return "train"
    if source_split == "test":
        return "test"
    if source_split == "val":
        return "val"
    return "train"


def main():
    project_root = get_project_root()
    raw_dir, prepared_dir = load_paths(project_root)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Dossier MSU brut introuvable: {raw_dir}")

    manifests_dir = prepared_dir / "manifests"
    splits_dir = manifests_dir / "splits"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    out_manifest = manifests_dir / "msu_real_frames_manifest.csv"
    out_summary = manifests_dir / "msu_real_summary.json"

    print("========== CONFIG MSU_REAL ==========")
    print(f"Project root : {project_root}")
    print(f"Raw dir      : {raw_dir}")
    print(f"Prepared dir : {prepared_dir}")

    # On cherche uniquement les dossiers sous real/réel
    candidate_dirs = []

    for d in raw_dir.rglob("*"):
        if not d.is_dir():
            continue

        parts_lower = [p.lower() for p in d.parts]

        is_real_dir = (
            "real" in parts_lower
            or "réel" in parts_lower
            or "reel" in parts_lower
        )

        if not is_real_dir:
            continue

        frames = [
            p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

        if len(frames) >= 16:
            candidate_dirs.append(d)

    if not candidate_dirs:
        raise RuntimeError(
            "Aucun dossier vidéo MSU_REAL trouvé avec au moins 16 frames."
        )

    rows = []

    for video_dir in sorted(candidate_dirs):
        meta = parse_msu_video_dir(video_dir)
        source_split = infer_source_split(video_dir)
        split = assign_split_from_msu_source(source_split)

        frames = sorted([
            p for p in video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ], key=lambda p: frame_index_from_name(p) if frame_index_from_name(p) is not None else 10**9)

        for f in frames:
            idx = frame_index_from_name(f)
            if idx is None:
                continue

            rows.append({
                "path": relative_to_project(f, project_root),
                "label": 0,
                "label_name": "REAL",
                "subject_id": meta["subject_id"],
                "device_id": meta["device_id"],
                "video_id": meta["video_id"],
                "frame_idx": int(idx),
                "source_frame_index": int(idx),
                "condition": "unknown",
                "take": "unknown",
                "attack_type": "real_video",
                "domain": "MSU_MFSD",
                "source_split": source_split,
                "split": split,
                "source_video_dir": str(video_dir),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_manifest, index=False, encoding="utf-8")

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].copy()
        split_df.to_csv(
            splits_dir / f"msu_real_frames_{split}.csv",
            index=False,
            encoding="utf-8",
        )

    video_df = df.drop_duplicates("video_id")

    summary = {
        "raw_dir": str(raw_dir),
        "prepared_dir": str(prepared_dir),
        "manifest": str(out_manifest),
        "num_frame_rows": int(len(df)),
        "num_videos": int(video_df["video_id"].nunique()),
        "num_subjects": int(video_df["subject_id"].nunique()),
        "video_counts_by_split": video_df["split"].value_counts().to_dict(),
        "video_counts_by_device": video_df["device_id"].value_counts().to_dict(),
        "video_counts_by_source_split": video_df["source_split"].value_counts().to_dict(),
        "frame_counts_by_split": df["split"].value_counts().to_dict(),
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== SUMMARY MSU_REAL ==========")
    print(f"Manifest     : {out_manifest}")
    print(f"Summary      : {out_summary}")
    print(f"Videos       : {summary['num_videos']}")
    print(f"Subjects     : {summary['num_subjects']}")
    print(f"Frame rows   : {summary['num_frame_rows']}")

    print("\n========== VIDEO COUNTS BY SPLIT ==========")
    print(video_df["split"].value_counts())

    print("\n========== VIDEO COUNTS BY DEVICE ==========")
    print(video_df["device_id"].value_counts())

    print("\n[OK] Manifest MSU_REAL généré.")


if __name__ == "__main__":
    main()
    