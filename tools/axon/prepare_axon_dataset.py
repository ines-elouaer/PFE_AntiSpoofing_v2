from pathlib import Path
import json
import csv
import shutil
import cv2
import hashlib
from collections import Counter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def load_config():
    """
    Charge configs/paths.json.

    Comme ce fichier est placé dans tools/axon/,
    project_root = parents[2].
    """
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "paths.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    axon_raw_dir = Path(cfg["axon_raw_dir"])
    if not axon_raw_dir.is_absolute():
        axon_raw_dir = (project_root / axon_raw_dir).resolve()
    else:
        axon_raw_dir = axon_raw_dir.resolve()

    axon_prepared_dir = Path(cfg["axon_prepared_dir"])
    if not axon_prepared_dir.is_absolute():
        axon_prepared_dir = (project_root / axon_prepared_dir).resolve()
    else:
        axon_prepared_dir = axon_prepared_dir.resolve()

    frames_per_video = int(cfg.get("frames_per_video", 3))
    max_spoof_videos_for_static = int(cfg.get("max_spoof_videos_for_static", 300))

    return {
        "project_root": project_root,
        "config_path": config_path,
        "axon_raw_dir": axon_raw_dir,
        "axon_prepared_dir": axon_prepared_dir,
        "frames_per_video": frames_per_video,
        "max_spoof_videos_for_static": max_spoof_videos_for_static,
    }


def norm(path: Path) -> str:
    return str(path).lower().replace("\\", "/").replace("_", " ")


def infer_level(path: Path) -> str:
    parts = [p.lower() for p in path.parts]

    if "l1" in parts:
        return "L1"

    if "l2" in parts:
        return "L2"

    return "UNKNOWN"


def infer_label_attack_group(path: Path):
    """
    Retourne :
    - label : 0 REAL, 1 SPOOF, None si ignoré
    - label_name : REAL / SPOOF / UNKNOWN
    - attack_type : type d'attaque
    - media_group :
        static_real_image
        static_spoof_image
        video_real
        video_spoof
        ignore
    """

    text = norm(path)
    ext = path.suffix.lower()

    # ==========================================================
    # REAL
    # ==========================================================
    if "/1. real/" in text or "real - omit bias" in text:
        if ext in IMAGE_EXTS:
            return 0, "REAL", "real_selfie_or_image", "static_real_image"

        if ext in VIDEO_EXTS:
            return 0, "REAL", "real_video", "video_real"

    # ==========================================================
    # SPOOF
    # ==========================================================
    attack_type = None

    if "print + cut" in text:
        attack_type = "print_cut"

    elif "ylinder" in text:
        attack_type = "cylinder_attack"

    elif "/6. pc/" in text:
        if "image replay" in text:
            attack_type = "pc_image_replay"
        elif "video replay" in text:
            attack_type = "pc_video_replay"
        else:
            attack_type = "pc_replay"

    elif "/7. mobile/" in text:
        if "image replay" in text:
            attack_type = "mobile_image_replay"
        elif "video replay" in text:
            attack_type = "mobile_video_replay"
        else:
            attack_type = "mobile_replay"

    elif "on actor" in text:
        attack_type = "on_actor"

    elif "3d attack" in text:
        attack_type = "3d_attack"

    elif "silicone masks" in text:
        attack_type = "silicone_mask"

    elif "latex masks" in text:
        attack_type = "latex_mask"

    elif "wrapped 3d paper" in text:
        attack_type = "wrapped_3d_paper"

    elif "advanced paper attacks" in text:
        attack_type = "advanced_paper"

    if attack_type is not None:
        if ext in IMAGE_EXTS:
            return 1, "SPOOF", attack_type, "static_spoof_image"

        if ext in VIDEO_EXTS:
            return 1, "SPOOF", attack_type, "video_spoof"

    return None, "UNKNOWN", "unknown", "ignore"


def is_static_spoof_candidate(attack_type: str) -> bool:
    """
    Ces attaques sont utiles pour créer un dataset statique spoof
    à partir de frames vidéo.
    """
    return attack_type in {
        "print_cut",
        "pc_image_replay",
        "mobile_image_replay",
        "advanced_paper",
        "wrapped_3d_paper",
        "cylinder_attack",
    }


def relative_to_project(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")


def relative_to_axon(path: Path, axon_raw_dir: Path) -> str:
    return str(path.resolve().relative_to(axon_raw_dir.resolve())).replace("\\", "/")


def short_hash(path: Path, axon_raw_dir: Path) -> str:
    """
    Hash stable à partir du chemin relatif Axon.
    Évite les noms trop longs et garde la reproductibilité.
    """
    rel = str(path.resolve().relative_to(axon_raw_dir.resolve())).replace("\\", "/")
    return hashlib.md5(rel.encode("utf-8")).hexdigest()[:12]


def safe_stem(path: Path, max_len: int = 30) -> str:
    """
    Nettoie une petite partie du nom original.
    """
    stem = path.stem[:max_len]

    bad_chars = [" ", ":", "?", "+", "(", ")", ",", ";", "'", '"', "[", "]", "#", "&"]
    for ch in bad_chars:
        stem = stem.replace(ch, "_")

    return stem


def safe_filename(path: Path, axon_raw_dir: Path) -> str:
    """
    Nom court et stable basé sur :
    - level L1/L2
    - hash du chemin relatif
    - petit stem lisible

    Exemple :
    L1_a8f31c2d0912_10.jpg
    """
    rel = str(path.resolve().relative_to(axon_raw_dir.resolve())).replace("\\", "/")
    rel_low = rel.lower()

    if rel_low.startswith("l1/") or "/l1/" in f"/{rel_low}":
        level = "L1"
    elif rel_low.startswith("l2/") or "/l2/" in f"/{rel_low}":
        level = "L2"
    else:
        level = "UNK"

    h = short_hash(path, axon_raw_dir)
    stem = safe_stem(path)
    suffix = path.suffix.lower()

    return f"{level}_{h}_{stem}{suffix}"


def copy_image(src: Path, dst_dir: Path, axon_raw_dir: Path) -> Path:
    """
    Copie une image directe avec nom court.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    out_name = safe_filename(src, axon_raw_dir)
    dst = dst_dir / out_name

    if not dst.exists():
        shutil.copy2(src, dst)

    return dst


def extract_frames(video_path: Path, out_dir: Path, axon_raw_dir: Path, frames_per_video: int):
    """
    Extrait quelques frames depuis une vidéo d'attaque statique.

    Correction :
    - noms courts avec hash
    - vérification que cv2.imwrite a bien sauvegardé l'image
    - évite les chemins Windows trop longs
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[WARN] Impossible d'ouvrir la vidéo: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        positions = [0]
    else:
        base_positions = [0.25, 0.50, 0.75, 0.90, 0.10]
        positions = [int(total * p) for p in base_positions[:frames_per_video]]

    saved = []
    base = Path(safe_filename(video_path, axon_raw_dir)).stem

    for i, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        ret, frame = cap.read()
        if not ret:
            continue

        out_path = out_dir / f"{base}_frame_{i}.jpg"

        ok = cv2.imwrite(str(out_path), frame)

        if ok and out_path.exists():
            saved.append(out_path)
        else:
            print(f"[WARN] Frame non sauvegardée: {out_path}")

    cap.release()
    return saved


def save_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    cfg = load_config()

    project_root = cfg["project_root"]
    axon_raw_dir = cfg["axon_raw_dir"]
    axon_prepared_dir = cfg["axon_prepared_dir"]
    frames_per_video = cfg["frames_per_video"]
    max_spoof_videos_for_static = cfg["max_spoof_videos_for_static"]

    if not axon_raw_dir.exists():
        raise FileNotFoundError(f"Dossier AxonLabs introuvable: {axon_raw_dir}")

    if not (axon_raw_dir / "L1").exists():
        print(f"[WARN] L1 introuvable directement dans: {axon_raw_dir}")

    if not (axon_raw_dir / "L2").exists():
        print(f"[WARN] L2 introuvable directement dans: {axon_raw_dir}")

    static_real_dir = axon_prepared_dir / "static_images" / "real"
    static_spoof_dir = axon_prepared_dir / "static_images" / "spoof"
    manifest_dir = axon_prepared_dir / "manifests"

    static_manifest_path = manifest_dir / "axon_static_manifest.csv"
    video_manifest_path = manifest_dir / "axon_video_manifest.csv"
    summary_path = manifest_dir / "axon_prepare_summary.json"

    static_real_dir.mkdir(parents=True, exist_ok=True)
    static_spoof_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    files = [
        p for p in axon_raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS)
    ]

    print(f"[INFO] Fichiers image/vidéo trouvés: {len(files)}")

    static_rows = []
    video_rows = []

    ignored_count = 0
    spoof_video_used_for_static = 0
    open_video_warnings = 0

    for path in files:
        label, label_name, attack_type, media_group = infer_label_attack_group(path)
        level = infer_level(path)

        if media_group == "ignore":
            ignored_count += 1
            continue

        # ======================================================
        # 1) Manifest vidéo complet
        # ======================================================
        if path.suffix.lower() in VIDEO_EXTS:
            video_rows.append({
                "raw_relative_path": relative_to_axon(path, axon_raw_dir),
                "label": label,
                "label_name": label_name,
                "attack_type": attack_type,
                "level": level,
                "media_group": media_group,
                "source": "axon_raw",
            })

        # ======================================================
        # 2) Images réelles directes
        # ======================================================
        if media_group == "static_real_image":
            dst = copy_image(path, static_real_dir, axon_raw_dir)

            static_rows.append({
                "prepared_relative_path": relative_to_project(dst, project_root),
                "label": 0,
                "label_name": "REAL",
                "attack_type": attack_type,
                "level": level,
                "source": "axon_static_real_direct",
                "origin_raw_relative_path": relative_to_axon(path, axon_raw_dir),
            })

        # ======================================================
        # 3) Images spoof directes si elles existent
        # ======================================================
        elif media_group == "static_spoof_image":
            dst = copy_image(path, static_spoof_dir, axon_raw_dir)

            static_rows.append({
                "prepared_relative_path": relative_to_project(dst, project_root),
                "label": 1,
                "label_name": "SPOOF",
                "attack_type": attack_type,
                "level": level,
                "source": "axon_static_spoof_direct",
                "origin_raw_relative_path": relative_to_axon(path, axon_raw_dir),
            })

        # ======================================================
        # 4) Frames spoof extraites depuis vidéos d'attaque statique
        # ======================================================
        elif media_group == "video_spoof" and is_static_spoof_candidate(attack_type):
            if spoof_video_used_for_static >= max_spoof_videos_for_static:
                continue

            extracted = extract_frames(
                video_path=path,
                out_dir=static_spoof_dir,
                axon_raw_dir=axon_raw_dir,
                frames_per_video=frames_per_video,
            )

            if not extracted:
                open_video_warnings += 1

            spoof_video_used_for_static += 1

            for frame_path in extracted:
                static_rows.append({
                    "prepared_relative_path": relative_to_project(frame_path, project_root),
                    "label": 1,
                    "label_name": "SPOOF",
                    "attack_type": attack_type,
                    "level": level,
                    "source": "axon_frame_from_static_like_attack_video",
                    "origin_raw_relative_path": relative_to_axon(path, axon_raw_dir),
                })

    # ==========================================================
    # Sauvegarde CSV
    # ==========================================================
    save_csv(
        static_manifest_path,
        static_rows,
        [
            "prepared_relative_path",
            "label",
            "label_name",
            "attack_type",
            "level",
            "source",
            "origin_raw_relative_path",
        ],
    )

    save_csv(
        video_manifest_path,
        video_rows,
        [
            "raw_relative_path",
            "label",
            "label_name",
            "attack_type",
            "level",
            "media_group",
            "source",
        ],
    )

    # ==========================================================
    # Résumé
    # ==========================================================
    static_label_counts = Counter(r["label_name"] for r in static_rows)
    static_attack_counts = Counter(r["attack_type"] for r in static_rows)
    video_label_counts = Counter(r["label_name"] for r in video_rows)
    video_attack_counts = Counter(r["attack_type"] for r in video_rows)
    video_group_counts = Counter(r["media_group"] for r in video_rows)

    summary = {
        "axon_raw_dir": str(axon_raw_dir),
        "axon_prepared_dir": str(axon_prepared_dir),
        "static_manifest": str(static_manifest_path),
        "video_manifest": str(video_manifest_path),
        "total_input_files": len(files),
        "ignored_files": ignored_count,
        "static_rows": len(static_rows),
        "video_rows": len(video_rows),
        "static_label_counts": dict(static_label_counts),
        "static_attack_counts": dict(static_attack_counts),
        "video_label_counts": dict(video_label_counts),
        "video_attack_counts": dict(video_attack_counts),
        "video_group_counts": dict(video_group_counts),
        "spoof_videos_used_for_static": spoof_video_used_for_static,
        "video_open_warnings": open_video_warnings,
        "frames_per_video": frames_per_video,
        "max_spoof_videos_for_static": max_spoof_videos_for_static,
    }

    save_summary(summary_path, summary)

    print("\n========== SUMMARY ==========")
    print(f"Axon raw dir                 : {axon_raw_dir}")
    print(f"Axon prepared dir            : {axon_prepared_dir}")
    print(f"Static manifest              : {static_manifest_path}")
    print(f"Video manifest               : {video_manifest_path}")
    print(f"Summary JSON                 : {summary_path}")
    print(f"Input files                  : {len(files)}")
    print(f"Ignored files                : {ignored_count}")
    print(f"Static rows                  : {len(static_rows)}")
    print(f"Video rows                   : {len(video_rows)}")
    print(f"Static REAL                  : {static_label_counts.get('REAL', 0)}")
    print(f"Static SPOOF                 : {static_label_counts.get('SPOOF', 0)}")
    print(f"Video REAL                   : {video_label_counts.get('REAL', 0)}")
    print(f"Video SPOOF                  : {video_label_counts.get('SPOOF', 0)}")
    print(f"Spoof videos used for static : {spoof_video_used_for_static}")
    print(f"Video open warnings          : {open_video_warnings}")

    print("\n========== STATIC ATTACK TYPE COUNTS ==========")
    for k, v in sorted(static_attack_counts.items()):
        print(f"{k}: {v}")

    print("\n========== VIDEO ATTACK TYPE COUNTS ==========")
    for k, v in sorted(video_attack_counts.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()