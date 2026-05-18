from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.behavior.mp_landmarks import FaceLandmarkerHelper, NOSE_TIP


T = 16

OUT_DIR = PROJECT_ROOT / "data" / "mixed_casia_axon_local_msu_motion_aware_test"
OUT_FRAMES_CSV = OUT_DIR / "mixed_test_frames_motion_aware.csv"

MIXED_TEST_FRAMES = PROJECT_ROOT / "data" / "mixed_casia_axon_local_msu" / "mixed_test_frames.csv"

LOCAL_CLEAN_FRAMES = PROJECT_ROOT / "data" / "local_real_prepared" / "manifests" / "local_real_frames_manifest_clean.csv"

MSU_FULL_FRAMES = PROJECT_ROOT / "data" / "msu_mfsd_prepared" / "manifests" / "msu_real_frames_manifest.csv"
MSU_CLEAN_FRAMES = PROJECT_ROOT / "data" / "msu_mfsd_prepared" / "manifests" / "msu_real_16_frames_manifest_clean.csv"

FACE_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"


def resolve_path(p):
    p = Path(str(p))
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def rel_to_project(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(p.resolve()).replace("\\", "/")


def center_indices(n, T=16):
    if n <= 0:
        return []

    if n >= T:
        start = max(0, n // 2 - T // 2)
        return list(range(start, start + T))

    idx = list(range(n))
    while len(idx) < T:
        idx.append(idx[-1])
    return idx


def compute_motion_scores_from_frames(frames, landmarker):
    scores = np.zeros(len(frames), dtype=np.float32)
    detected = np.zeros(len(frames), dtype=np.float32)

    prev_xy = None

    for i, frame in enumerate(frames):
        try:
            lm = landmarker.detect_landmarks(frame)
        except Exception:
            lm = None

        if lm is None:
            prev_xy = None
            continue

        h, w = frame.shape[:2]
        x = lm[NOSE_TIP].x * w
        y = lm[NOSE_TIP].y * h
        xy = np.array([x, y], dtype=np.float32)

        detected[i] = 1.0

        if prev_xy is not None:
            scores[i] = float(np.linalg.norm(xy - prev_xy))

        prev_xy = xy

    return scores, detected


def choose_best_window(scores, detected, T=16, min_detection_rate=0.50):
    n = len(scores)

    if n <= 0:
        return []

    if n < T:
        return center_indices(n, T)

    best_start = 0
    best_value = -1.0
    best_detection = 0.0

    for start in range(0, n - T + 1):
        end = start + T

        win_scores = scores[start:end]
        win_detected = detected[start:end]

        detection_rate = float(np.mean(win_detected))
        motion_score = float(np.mean(win_scores))

        value = motion_score * detection_rate

        if value > best_value:
            best_value = value
            best_start = start
            best_detection = detection_rate

    if best_detection < min_detection_rate:
        return center_indices(n, T)

    return list(range(best_start, best_start + T))


def read_video_all_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def build_local_real_motion_rows(landmarker):
    if not LOCAL_CLEAN_FRAMES.exists():
        raise FileNotFoundError(f"Missing: {LOCAL_CLEAN_FRAMES}")

    local_df = pd.read_csv(LOCAL_CLEAN_FRAMES)
    local_df = local_df[local_df["split"] == "test"].copy()

    rows = []
    out_root = OUT_DIR / "local_real_motion_frames"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[LOCAL_REAL] test videos: {local_df['video_id'].nunique()}")

    for video_id, g in local_df.groupby("video_id"):
        meta = g.sort_values("frame_idx").iloc[0].to_dict()

        if "source_video_path" not in g.columns:
            raise RuntimeError("source_video_path absent dans LOCAL_REAL clean manifest.")

        raw_video_path = Path(str(meta["source_video_path"]))

        if not raw_video_path.exists():
            raise FileNotFoundError(f"Vidéo source introuvable: {raw_video_path}")

        frames = read_video_all_frames(raw_video_path)
        scores, detected = compute_motion_scores_from_frames(frames, landmarker)
        selected_idx = choose_best_window(scores, detected, T=T)

        out_video_dir = out_root / str(video_id)
        out_video_dir.mkdir(parents=True, exist_ok=True)

        for new_idx, src_idx in enumerate(selected_idx):
            frame = frames[src_idx]
            out_path = out_video_dir / f"frame_{new_idx:03d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            rows.append({
                "path": rel_to_project(out_path),
                "label": 0,
                "label_name": "REAL",
                "subject_id": str(meta.get("subject_id", "unknown_subject")),
                "device_id": str(meta.get("device_id", "unknown_device")),
                "video_id": "LOCAL_REAL__" + str(video_id),
                "original_video_id": str(video_id),
                "frame_idx": int(new_idx),
                "source_frame_index": int(src_idx),
                "condition": str(meta.get("condition", "unknown")),
                "take": str(meta.get("take", "unknown")),
                "attack_type": "real_video",
                "domain": "LOCAL_REAL",
                "source_dataset": "LOCAL_REAL",
                "split": "test",
            })

    return pd.DataFrame(rows)


def build_msu_motion_rows(landmarker):
    if not MSU_FULL_FRAMES.exists():
        raise FileNotFoundError(f"Missing: {MSU_FULL_FRAMES}")

    if not MSU_CLEAN_FRAMES.exists():
        raise FileNotFoundError(f"Missing: {MSU_CLEAN_FRAMES}")

    msu_full = pd.read_csv(MSU_FULL_FRAMES)
    msu_clean = pd.read_csv(MSU_CLEAN_FRAMES)

    test_video_ids = set(msu_clean[msu_clean["split"] == "test"]["video_id"].astype(str).unique())

    msu_full = msu_full[msu_full["video_id"].astype(str).isin(test_video_ids)].copy()

    rows = []

    print(f"[MSU_MFSD] test videos: {len(test_video_ids)}")

    for video_id, g in msu_full.groupby("video_id"):
        g = g.sort_values("frame_idx").copy()
        meta = g.iloc[0].to_dict()

        frame_paths = [resolve_path(p) for p in g["path"].tolist()]
        frames = []

        valid_paths = []

        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
                valid_paths.append(p)

        if len(frames) == 0:
            print(f"[WARN] MSU video ignored, no frames readable: {video_id}")
            continue

        scores, detected = compute_motion_scores_from_frames(frames, landmarker)
        selected_idx = choose_best_window(scores, detected, T=T)

        for new_idx, local_src_idx in enumerate(selected_idx):
            src_path = valid_paths[local_src_idx]

            rows.append({
                "path": rel_to_project(src_path),
                "label": 0,
                "label_name": "REAL",
                "subject_id": str(meta.get("subject_id", "unknown_subject")),
                "device_id": str(meta.get("device_id", "unknown_device")),
                "video_id": "MSU_MFSD__" + str(video_id),
                "original_video_id": str(video_id),
                "frame_idx": int(new_idx),
                "source_frame_index": int(meta.get("frame_idx", local_src_idx)),
                "condition": str(meta.get("condition", "unknown")),
                "take": str(meta.get("take", "unknown")),
                "attack_type": "real_video",
                "domain": "MSU_MFSD",
                "source_dataset": "MSU_MFSD",
                "split": "test",
            })

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MIXED_TEST_FRAMES.exists():
        raise FileNotFoundError(f"Missing: {MIXED_TEST_FRAMES}")

    if not FACE_MODEL.exists():
        raise FileNotFoundError(f"Missing: {FACE_MODEL}")

    print("========== MOTION-AWARE TEST BUILD ==========")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Mixed test   : {MIXED_TEST_FRAMES}")
    print(f"Output CSV   : {OUT_FRAMES_CSV}")
    print(f"Face model   : {FACE_MODEL}")

    landmarker = FaceLandmarkerHelper(model_path=str(FACE_MODEL))

    mixed = pd.read_csv(MIXED_TEST_FRAMES)

    # CASIA et AXON restent inchangés pour garder le benchmark standard.
    fixed_part = mixed[mixed["source_dataset"].isin(["CASIA", "AXON"])].copy()

    print(f"[FIXED] CASIA/AXON videos: {fixed_part['video_id'].nunique()}")

    local_rows = build_local_real_motion_rows(landmarker)
    msu_rows = build_msu_motion_rows(landmarker)

    out_df = pd.concat(
        [fixed_part, local_rows, msu_rows],
        ignore_index=True,
    )

    out_df = out_df.sort_values(["source_dataset", "video_id", "frame_idx"]).reset_index(drop=True)

    out_df.to_csv(OUT_FRAMES_CSV, index=False, encoding="utf-8")

    video_df = out_df.drop_duplicates("video_id")

    print("\n========== SUMMARY ==========")
    print(f"Output frames CSV : {OUT_FRAMES_CSV}")
    print(f"Frame rows        : {len(out_df)}")
    print(f"Videos            : {video_df['video_id'].nunique()}")
    print("\nVideos by source:")
    print(video_df["source_dataset"].value_counts())
    print("\nLabels:")
    print(video_df["label_name"].value_counts())
    print("\nFrames per video min/max:")
    print(out_df.groupby("video_id")["frame_idx"].nunique().min())
    print(out_df.groupby("video_id")["frame_idx"].nunique().max())

    print("\n[OK] Motion-aware test frames generated.")


if __name__ == "__main__":
    main()