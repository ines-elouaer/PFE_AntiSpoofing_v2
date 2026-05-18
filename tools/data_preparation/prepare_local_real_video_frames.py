from pathlib import Path
import json
import cv2
import pandas as pd
from collections import Counter, defaultdict


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_paths(project_root: Path):
    config_path = project_root / "configs" / "paths.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    raw_dir = Path(cfg["local_real_raw_dir"])
    prepared_dir = Path(cfg["local_real_prepared_dir"])

    if not raw_dir.is_absolute():
        raw_dir = (project_root / raw_dir).resolve()

    if not prepared_dir.is_absolute():
        prepared_dir = (project_root / prepared_dir).resolve()

    frames_per_video = int(cfg.get("local_real_frames_per_video", 16))

    return raw_dir, prepared_dir, frames_per_video


def parse_local_real_filename(path: Path):
    """
    Format attendu :
    pXX_device_condition_tXX.mp4

    Exemples :
    p01_iphone_normal_t01.mp4
    p02_pc_lowlight_t02.mp4
    p12_redmi_near_t03.mp4
    """
    stem = path.stem.lower()
    parts = stem.split("_")

    if len(parts) < 4:
        raise ValueError(
            f"Nom invalide: {path.name}. Format attendu: pXX_device_condition_tXX.mp4"
        )

    subject_id = parts[0]
    device_id = parts[1]
    take = parts[-1]
    condition = "_".join(parts[2:-1])

    if not subject_id.startswith("p"):
        raise ValueError(f"subject_id invalide dans {path.name}")

    if not take.startswith("t"):
        raise ValueError(f"take invalide dans {path.name}")

    video_id = stem

    return {
        "subject_id": subject_id,
        "device_id": device_id,
        "condition": condition,
        "take": take,
        "video_id": video_id,
    }


def relative_to_project(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")


def extract_consecutive_middle_frames(
    video_path: Path,
    output_dir: Path,
    frames_per_video: int = 16,
    jpeg_quality: int = 95,
):
    """
    Même principe que Axon :
    - extraction de T frames consécutives
    - depuis le milieu de la vidéo
    - si vidéo trop courte, répétition de la dernière frame disponible
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return [], {
            "opened": False,
            "reason": "video_not_opened",
            "frame_count": 0,
            "fps": 0.0,
            "duration_sec": 0.0,
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    duration_sec = frame_count / fps if fps > 0 else 0.0

    if frame_count <= 0:
        cap.release()
        return [], {
            "opened": True,
            "reason": "empty_video",
            "frame_count": frame_count,
            "fps": fps,
            "duration_sec": duration_sec,
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    if frame_count >= frames_per_video:
        start = max(0, frame_count // 2 - frames_per_video // 2)
        end = start + frames_per_video

        if end > frame_count:
            end = frame_count
            start = max(0, end - frames_per_video)

        indices = list(range(start, end))
    else:
        indices = list(range(frame_count))
        while len(indices) < frames_per_video:
            indices.append(indices[-1])

    saved = []

    for out_idx, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        out_path = output_dir / f"frame_{out_idx:03d}.jpg"

        ok = cv2.imwrite(
            str(out_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )

        if ok and out_path.exists():
            saved.append({
                "frame_idx": int(out_idx),
                "source_frame_index": int(frame_index),
                "path": out_path,
            })

    cap.release()

    info = {
        "opened": True,
        "reason": "ok" if len(saved) > 0 else "no_frame_saved",
        "frame_count": int(frame_count),
        "fps": float(fps),
        "duration_sec": float(duration_sec),
        "saved_frames": int(len(saved)),
        "sampling_mode": "center_consecutive",
    }

    return saved, info


def assign_splits_by_subject(subject_ids):
    """
    Split propre par personne :
    - aucune personne dans deux splits
    - train ≈ 70 %
    - val ≈ 15 %
    - test ≈ 15 %

    Pour peu de personnes :
    - garantit au moins 1 sujet val et 1 sujet test si possible.
    """
    subjects = sorted(set(subject_ids))
    n = len(subjects)

    if n < 3:
        raise RuntimeError(
            "Il faut au moins 3 personnes pour faire train/val/test proprement."
        )

    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15)))

    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])
    test_subjects = set(subjects[n_train + n_val:])

    split_by_subject = {}

    for s in train_subjects:
        split_by_subject[s] = "train"
    for s in val_subjects:
        split_by_subject[s] = "val"
    for s in test_subjects:
        split_by_subject[s] = "test"

    return split_by_subject


def main():
    project_root = get_project_root()
    raw_dir, prepared_dir, frames_per_video = load_paths(project_root)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Dossier LOCAL_REAL brut introuvable: {raw_dir}")

    frames_root = prepared_dir / "frames"
    manifests_dir = prepared_dir / "manifests"
    splits_dir = manifests_dir / "splits"

    frames_manifest_path = manifests_dir / "local_real_frames_manifest.csv"
    video_manifest_path = manifests_dir / "local_real_video_manifest.csv"
    summary_path = manifests_dir / "local_real_prepare_summary.json"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    video_files = [
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]

    if not video_files:
        raise RuntimeError(f"Aucune vidéo trouvée dans: {raw_dir}")

    print("========== CONFIG LOCAL_REAL ==========")
    print(f"Project root       : {project_root}")
    print(f"Raw dir            : {raw_dir}")
    print(f"Prepared dir       : {prepared_dir}")
    print(f"Frames root        : {frames_root}")
    print(f"Frames per video   : {frames_per_video}")
    print(f"Videos found       : {len(video_files)}")

    parsed_videos = []
    errors = []

    for video_path in sorted(video_files):
        try:
            meta = parse_local_real_filename(video_path)
            meta["video_path_abs"] = video_path
            parsed_videos.append(meta)
        except Exception as e:
            errors.append({
                "video_path": str(video_path),
                "reason": str(e),
            })

    if errors:
        print("\n[WARN] Certains fichiers ont un nom invalide:")
        for e in errors[:10]:
            print(e)

    if not parsed_videos:
        raise RuntimeError("Aucune vidéo valide après parsing des noms.")

    split_by_subject = assign_splits_by_subject(
        [v["subject_id"] for v in parsed_videos]
    )

    frame_rows = []
    video_rows = []
    warnings = []

    for idx, meta in enumerate(parsed_videos, start=1):
        video_path = meta["video_path_abs"]
        subject_id = meta["subject_id"]
        device_id = meta["device_id"]
        condition = meta["condition"]
        take = meta["take"]
        video_id = meta["video_id"]
        split = split_by_subject[subject_id]

        out_dir = frames_root / split / subject_id / device_id / video_id

        saved_frames, info = extract_consecutive_middle_frames(
            video_path=video_path,
            output_dir=out_dir,
            frames_per_video=frames_per_video,
        )

        if len(saved_frames) == 0:
            warnings.append({
                "video_path": str(video_path),
                "reason": info.get("reason", "unknown"),
            })
            continue

        video_rows.append({
            "video_path": str(video_path),
            "video_id": video_id,
            "label": 0,
            "label_name": "REAL",
            "subject_id": subject_id,
            "device_id": device_id,
            "condition": condition,
            "take": take,
            "attack_type": "real_video",
            "domain": "LOCAL_REAL",
            "split": split,
            "frame_count": int(info.get("frame_count", 0)),
            "fps": float(info.get("fps", 0.0)),
            "duration_sec": float(info.get("duration_sec", 0.0)),
            "saved_frames": int(len(saved_frames)),
            "sampling_mode": info.get("sampling_mode", "center_consecutive"),
        })

        for item in saved_frames:
            frame_rows.append({
                "path": relative_to_project(item["path"], project_root),
                "label": 0,
                "label_name": "REAL",
                "subject_id": subject_id,
                "device_id": device_id,
                "video_id": video_id,
                "frame_idx": int(item["frame_idx"]),
                "source_frame_index": int(item["source_frame_index"]),
                "condition": condition,
                "take": take,
                "attack_type": "real_video",
                "domain": "LOCAL_REAL",
                "split": split,
                "source_video_path": str(video_path),
            })

        if idx % 10 == 0:
            print(f"[INFO] Progression: {idx}/{len(parsed_videos)} vidéos traitées")

    frames_df = pd.DataFrame(frame_rows)
    videos_df = pd.DataFrame(video_rows)

    frames_df.to_csv(frames_manifest_path, index=False, encoding="utf-8")
    videos_df.to_csv(video_manifest_path, index=False, encoding="utf-8")

    # Sauvegarde splits frame-level
    for split in ["train", "val", "test"]:
        split_df = frames_df[frames_df["split"] == split].copy()
        split_df.to_csv(
            splits_dir / f"local_real_frames_{split}.csv",
            index=False,
            encoding="utf-8",
        )

    # Résumés par split, sujet, appareil
    summary = {
        "raw_dir": str(raw_dir),
        "prepared_dir": str(prepared_dir),
        "frames_manifest": str(frames_manifest_path),
        "video_manifest": str(video_manifest_path),
        "frames_per_video": frames_per_video,
        "videos_found": len(video_files),
        "valid_videos": int(len(videos_df)),
        "frame_rows": int(len(frames_df)),
        "warnings_count": len(warnings),
        "warnings_first_20": warnings[:20],
        "parse_errors_count": len(errors),
        "parse_errors_first_20": errors[:20],
        "split_subjects": {
            split: sorted([
                s for s, sp in split_by_subject.items()
                if sp == split
            ])
            for split in ["train", "val", "test"]
        },
        "video_counts_by_split": videos_df["split"].value_counts().to_dict()
        if len(videos_df) else {},
        "video_counts_by_device": videos_df["device_id"].value_counts().to_dict()
        if len(videos_df) else {},
        "video_counts_by_split_device": (
            videos_df.groupby(["split", "device_id"]).size().to_dict()
            if len(videos_df) else {}
        ),
        "video_counts_by_subject": videos_df["subject_id"].value_counts().to_dict()
        if len(videos_df) else {},
    }

    # JSON ne supporte pas les clés tuple
    summary["video_counts_by_split_device"] = {
        f"{k[0]}::{k[1]}": int(v)
        for k, v in summary["video_counts_by_split_device"].items()
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== SUMMARY LOCAL_REAL ==========")
    print(f"Video manifest  : {video_manifest_path}")
    print(f"Frames manifest : {frames_manifest_path}")
    print(f"Summary         : {summary_path}")
    print(f"Valid videos    : {len(videos_df)}")
    print(f"Frame rows      : {len(frames_df)}")

    print("\n========== SPLIT VIDEO COUNTS ==========")
    print(videos_df["split"].value_counts())

    print("\n========== DEVICE COUNTS ==========")
    print(videos_df["device_id"].value_counts())

    print("\n========== SPLIT x DEVICE ==========")
    print(videos_df.groupby(["split", "device_id"]).size())

    print("\n[OK] Préparation LOCAL_REAL terminée.")


if __name__ == "__main__":
    main()