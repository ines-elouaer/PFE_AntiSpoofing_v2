from pathlib import Path
import json
import hashlib
import cv2
import numpy as np
import pandas as pd


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_paths(project_root: Path):
    config_path = project_root / "configs" / "paths.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    axon_raw_dir = Path(cfg["axon_raw_dir"])
    axon_prepared_dir = Path(cfg["axon_prepared_dir"])

    if not axon_raw_dir.is_absolute():
        axon_raw_dir = (project_root / axon_raw_dir).resolve()

    if not axon_prepared_dir.is_absolute():
        axon_prepared_dir = (project_root / axon_prepared_dir).resolve()

    return axon_raw_dir, axon_prepared_dir


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def normalize_label(row) -> int:
    if "label" in row and not pd.isna(row["label"]):
        val = row["label"]

        if isinstance(val, str):
            val_upper = val.upper().strip()
            if val_upper in {"REAL", "BONAFIDE", "LIVE"}:
                return 0
            if val_upper in {"SPOOF", "ATTACK", "FAKE"}:
                return 1

        return int(val)

    if "label_name" in row and not pd.isna(row["label_name"]):
        label_name = str(row["label_name"]).upper().strip()
        if label_name in {"REAL", "BONAFIDE", "LIVE"}:
            return 0
        if label_name in {"SPOOF", "ATTACK", "FAKE"}:
            return 1

    raise ValueError("Impossible de déterminer le label pour une ligne Axon.")


def label_name_from_int(label: int) -> str:
    return "REAL" if int(label) == 0 else "SPOOF"


def find_video_path(row, project_root: Path, axon_raw_dir: Path, axon_prepared_dir: Path) -> Path:
    """
    Fonction robuste : elle essaie plusieurs colonnes possibles.
    """

    candidate_columns = [
        "absolute_path",
        "video_path",
        "prepared_path",
        "prepared_relative_path",
        "origin_raw_relative_path",
        "raw_relative_path",
        "path",
    ]

    values = []

    for col in candidate_columns:
        if col in row and not pd.isna(row[col]):
            values.append(str(row[col]))

    if not values:
        raise FileNotFoundError("Aucune colonne de chemin vidéo trouvée dans le manifest.")

    tried = []

    for value in values:
        p = Path(value)

        candidate_paths = []

        if p.is_absolute():
            candidate_paths.append(p)
        else:
            candidate_paths.extend([
                project_root / p,
                axon_raw_dir / p,
                axon_prepared_dir / p,
            ])

        for c in candidate_paths:
            c = c.resolve()
            tried.append(str(c))

            if c.exists() and c.suffix.lower() in VIDEO_EXTS:
                return c

    raise FileNotFoundError(
        "Vidéo introuvable. Chemins essayés:\n" + "\n".join(tried[:10])
    )


def safe_video_id(row, video_path: Path, row_idx: int) -> str:
    if "video_id" in row and not pd.isna(row["video_id"]):
        raw_id = str(row["video_id"])
    else:
        raw_id = f"{video_path.stem}_{row_idx}_{short_hash(str(video_path))}"

    raw_id = raw_id.replace("\\", "_").replace("/", "_")
    raw_id = raw_id.replace(" ", "_").replace(".", "_")

    return f"axon_{short_hash(raw_id + str(video_path), 14)}"

def extract_consecutive_frames(
    video_path: Path,
    output_dir: Path,
    frames_per_video: int = 16,
    jpeg_quality: int = 95,
):
    """
    Extrait T frames consécutives depuis le milieu de la vidéo.
    Cette stratégie est cohérente avec le meilleur modèle CASIA entraîné
    avec le mode consecutive.
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

    # Cas normal : prendre 16 frames consécutives autour du milieu.
    if frame_count >= frames_per_video:
        start = max(0, frame_count // 2 - frames_per_video // 2)
        end = start + frames_per_video

        if end > frame_count:
            end = frame_count
            start = max(0, end - frames_per_video)

        indices = list(range(start, end))

    # Cas rare : vidéo très courte, on répète la dernière frame disponible.
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

        if ok:
            saved.append({
                "frame_idx": out_idx,
                "source_frame_index": int(frame_index),
                "path": out_path,
            })

    cap.release()

    info = {
        "opened": True,
        "reason": "ok" if len(saved) > 0 else "no_frame_saved",
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": duration_sec,
        "saved_frames": len(saved),
        "sampling_mode": "consecutive_middle",
    }

    return saved, info
def main():
    project_root = get_project_root()
    axon_raw_dir, axon_prepared_dir = load_paths(project_root)

    manifests_dir = axon_prepared_dir / "manifests"
    video_manifest_path = manifests_dir / "axon_video_manifest.csv"

    if not video_manifest_path.exists():
        raise FileNotFoundError(f"Manifest vidéo introuvable: {video_manifest_path}")

    out_frames_root = axon_prepared_dir / "video_frames"
    out_manifest_path = manifests_dir / "axon_video_frames_manifest.csv"
    out_summary_path = manifests_dir / "axon_video_frames_summary.json"

    frames_per_video = 16

    print("========== CONFIG ==========")
    print(f"Project root      : {project_root}")
    print(f"Axon raw dir      : {axon_raw_dir}")
    print(f"Axon prepared dir : {axon_prepared_dir}")
    print(f"Video manifest    : {video_manifest_path}")
    print(f"Frames output     : {out_frames_root}")
    print(f"Frames per video  : {frames_per_video}")

    df = pd.read_csv(video_manifest_path)
    print(f"\n[INFO] Vidéos dans manifest: {len(df)}")

    rows = []
    warnings = []

    for i, row in df.iterrows():
        row_dict = row.to_dict()

        try:
            label = normalize_label(row_dict)
            label_name = label_name_from_int(label)
            class_dir = "real" if label == 0 else "spoof"

            video_path = find_video_path(
                row=row_dict,
                project_root=project_root,
                axon_raw_dir=axon_raw_dir,
                axon_prepared_dir=axon_prepared_dir,
            )

            video_id = safe_video_id(row_dict, video_path, i)
            attack_type = str(row_dict.get("attack_type", "unknown"))
            level = str(row_dict.get("level", "unknown"))
            subject_id = str(row_dict.get("subject_id", f"axon_subject_{short_hash(str(video_path), 8)}"))

            video_out_dir = out_frames_root / class_dir / video_id

            saved_frames, info = extract_consecutive_frames(
                video_path=video_path,
                output_dir=video_out_dir,
                frames_per_video=frames_per_video,
            )
            if len(saved_frames) == 0:
                warnings.append({
                    "row_idx": int(i),
                    "video_path": str(video_path),
                    "reason": info.get("reason"),
                })
                continue

            for item in saved_frames:
                rel_path = item["path"].resolve().relative_to(project_root)

                rows.append({
                    "path": str(rel_path).replace("\\", "/"),
                    "label": int(label),
                    "label_name": label_name,
                    "subject_id": subject_id,
                    "video_id": video_id,
                    "frame_idx": int(item["frame_idx"]),
                    "source_frame_index": int(item["source_frame_index"]),
                    "attack_type": attack_type,
                    "level": level,
                    "source_video_path": str(video_path),
                    "source_manifest_row": int(i),
                    "source_frame_count": int(info.get("frame_count", 0)),
                    "source_fps": float(info.get("fps", 0.0)),
                    "source_duration_sec": float(info.get("duration_sec", 0.0)),
                })

            if (i + 1) % 25 == 0:
                print(f"[INFO] Progression: {i + 1}/{len(df)} vidéos traitées")

        except Exception as e:
            warnings.append({
                "row_idx": int(i),
                "reason": str(e),
            })

    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_manifest_path, index=False, encoding="utf-8")

    summary = {
        "input_manifest": str(video_manifest_path),
        "frames_manifest": str(out_manifest_path),
        "frames_output_dir": str(out_frames_root),
        "frames_per_video": frames_per_video,
        "sampling_mode": "consecutive_middle",
        "input_videos": int(len(df)),
        "output_rows": int(len(out_df)),
        "unique_videos": int(out_df["video_id"].nunique()) if len(out_df) else 0,
        "label_counts_frames": out_df["label_name"].value_counts().to_dict() if len(out_df) else {},
        "label_counts_videos": out_df.drop_duplicates("video_id")["label_name"].value_counts().to_dict() if len(out_df) else {},
        "attack_type_counts_videos": out_df.drop_duplicates("video_id")["attack_type"].value_counts().to_dict() if len(out_df) else {},
        "warnings_count": len(warnings),
        "warnings_first_20": warnings[:20],
    }

    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== SUMMARY ==========")
    print(f"Frames manifest : {out_manifest_path}")
    print(f"Summary JSON    : {out_summary_path}")
    print(f"Input videos    : {summary['input_videos']}")
    print(f"Unique videos   : {summary['unique_videos']}")
    print(f"Output rows     : {summary['output_rows']}")
    print(f"Sampling mode   : {summary['sampling_mode']}")
    print(f"Warnings        : {summary['warnings_count']}")

    print("\n========== LABEL COUNTS VIDEOS ==========")
    for k, v in summary["label_counts_videos"].items():
        print(f"{k}: {v}")

    print("\n========== ATTACK COUNTS VIDEOS ==========")
    for k, v in summary["attack_type_counts_videos"].items():
        print(f"{k}: {v}")

    print("\n[OK] Préparation Axon vidéo terminée.")


if __name__ == "__main__":
    main()