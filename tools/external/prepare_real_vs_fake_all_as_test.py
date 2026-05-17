from pathlib import Path
import argparse
import os
import cv2
import pandas as pd


def get_external_root() -> Path:
    env = os.environ.get("PFE_EXTERNAL_DATA")
    if not env:
        raise RuntimeError(
            "Variable PFE_EXTERNAL_DATA introuvable. "
            "Exemple CMD: set PFE_EXTERNAL_DATA=E:\\PFE_EXTERNAL_DATA"
        )
    return Path(env)


def extract_frames_from_video(video_path: Path, out_dir: Path, max_frames: int = 64):
    """
    Extrait un clip CONTINU de max_frames frames au centre de la vidéo.

    Pourquoi ?
    Le modèle CNN+LSTM a été entraîné avec des frames consécutives.
    Il faut donc préserver la continuité temporelle pendant le test externe.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if total_frames <= 0:
        print(f"[WARN] Empty video: {video_path}")
        cap.release()
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    # CORRECTION IMPORTANTE :
    # Avant : frames réparties dans toute la vidéo.
    # Maintenant : clip continu au centre.
    if total_frames <= max_frames:
        selected_indices = list(range(total_frames))
    else:
        start = max(0, (total_frames - max_frames) // 2)
        selected_indices = list(range(start, start + max_frames))

    selected_set = set(selected_indices)
    saved = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx in selected_set:
            frame_name = f"frame_{idx:05d}.jpg"
            frame_path = out_dir / frame_name
            cv2.imwrite(str(frame_path), frame)

            saved.append({
                "frame_idx": idx,
                "path": str(frame_path),
                "video_fps": fps,
                "video_total_frames": total_frames,
            })

        idx += 1

    cap.release()
    return saved


def collect_videos(raw_root: Path):
    videos = []

    configs = [
        ("train", "real_video", 0, "REAL", "real_video"),
        ("train", "attack", 1, "SPOOF", "replay_attack"),
        ("test", "real_video", 0, "REAL", "real_video"),
        ("test", "attack", 1, "SPOOF", "replay_attack"),
    ]

    for split, folder, label, label_name, attack_type in configs:
        d = raw_root / split / folder

        if not d.exists():
            print(f"[WARN] Folder not found: {d}")
            continue

        for p in sorted(d.glob("*.mp4")):
            videos.append({
                "video_path": p,
                "split_original": split,
                "folder": folder,
                "label": label,
                "label_name": label_name,
                "attack_type": attack_type,
            })

    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="real_vs_fake_kaggle",
        help="Nom du dataset dans PFE_EXTERNAL_DATA."
    )
    parser.add_argument("--max_frames", type=int, default=64)
    args = parser.parse_args()

    external_root = get_external_root()
    dataset_root = external_root / args.dataset_name
    raw_root = dataset_root / "raw"
    processed_root = dataset_root / "processed"
    frames_root = processed_root / "frames"

    out_csv = processed_root / "external_all_frames.csv"

    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root introuvable: {raw_root}")

    videos = collect_videos(raw_root)

    print("========== PREPARE EXTERNAL REAL VS FAKE ==========")
    print("External root :", external_root)
    print("Raw root      :", raw_root)
    print("Processed root:", processed_root)
    print("Videos found  :", len(videos))
    print("Max frames    :", args.max_frames)
    print("Sampling      : center continuous clip")

    rows = []

    for i, item in enumerate(videos, start=1):
        video_path = item["video_path"]
        split_original = item["split_original"]
        folder = item["folder"]
        label = item["label"]
        attack_type = item["attack_type"]

        stem = video_path.stem

        video_id = f"REALVSFAKE__{split_original}_{folder}_{stem}"
        video_frame_dir = frames_root / video_id

        saved_frames = extract_frames_from_video(
            video_path=video_path,
            out_dir=video_frame_dir,
            max_frames=args.max_frames,
        )

        for fr in saved_frames:
            rows.append({
                "path": fr["path"],
                "label": label,
                "video_id": video_id,
                "frame_idx": fr["frame_idx"],
                "source_dataset": "REAL_VS_FAKE_KAGGLE",
                "device_id": "unknown",
                "subject_id": f"external_{split_original}_{stem}",
                "attack_type": attack_type,
                "original_video_id": str(video_path),
                "external_original_split": split_original,
                "video_fps": fr["video_fps"],
                "video_total_frames": fr["video_total_frames"],
            })

        print(
            f"[{i}/{len(videos)}] {video_id} | "
            f"label={label} | frames={len(saved_frames)}"
        )

    processed_root.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n========== SUMMARY ==========")
    print("Rows:", len(df))
    print("Videos:", df["video_id"].nunique())

    print("\nVideos by label:")
    print(df.groupby("label")["video_id"].nunique())

    print("\nVideos by original split:")
    print(df.groupby(["external_original_split", "label"])["video_id"].nunique())

    print("\nFPS summary:")
    print(df.groupby("video_id")["video_fps"].first().describe())

    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()