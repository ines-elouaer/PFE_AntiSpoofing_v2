from pathlib import Path
from collections import defaultdict
import argparse
import os
import cv2
import numpy as np
import pandas as pd


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = []
    for idx in eye_indices:
        if idx >= len(landmarks):
            return None
        pts.append((landmarks[idx].x * w, landmarks[idx].y * h))

    p1, p2, p3, p4, p5, p6 = pts
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal < 1e-6:
        return None

    return (vertical1 + vertical2) / (2.0 * horizontal)


def compute_motion(prev_gray, gray, size=(224, 224)):
    """
    Calcule le mouvement optique entre deux frames.
    Correction importante :
    les deux images doivent avoir exactement la même taille.
    """

    prev_gray = cv2.resize(prev_gray, size)
    gray = cv2.resize(gray, size)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=2,
        winsize=15,
        iterations=2,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag)), float(np.max(mag))

def load_frames(frames_csv):
    df = pd.read_csv(frames_csv)
    by_vid = defaultdict(list)

    for _, row in df.iterrows():
        by_vid[str(row["video_id"])].append(row.to_dict())

    for vid in by_vid:
        by_vid[vid] = sorted(by_vid[vid], key=lambda r: int(r["frame_idx"]))

    return by_vid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--model", default="models/face_landmarker.task")
    parser.add_argument("--blink_ear_threshold", type=float, default=0.21)
    args = parser.parse_args()

    from src.behavior.mp_landmarks import FaceLandmarkerHelper

    frames_csv = Path(args.frames_csv)
    out_csv = Path(args.out_csv)

    if not frames_csv.exists():
        raise FileNotFoundError(frames_csv)

    landmarker = FaceLandmarkerHelper(model_path=args.model)
    by_vid = load_frames(frames_csv)

    rows_out = []

    print("========== EXTERNAL BEHAVIOR EXTRACTION ==========")
    print("Frames CSV:", frames_csv)
    print("Videos:", len(by_vid))
    print("Out CSV:", out_csv)

    for i, (vid, rows) in enumerate(by_vid.items(), start=1):
        ears = []
        motion_means = []
        motion_maxs = []

        skipped = 0
        prev_gray = None

        label = int(rows[0]["label"])

        for r in rows:
            img_path = r["path"]
            frame = cv2.imread(img_path)

            if frame is None:
                skipped += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                try:
                    m_mean, m_max = compute_motion(prev_gray, gray)
                    motion_means.append(m_mean)
                    motion_maxs.append(m_max)
                except cv2.error as e:
                    skipped += 1
                    print(f"[WARN] Motion failed for {vid}: {e}")

            prev_gray = gray

            landmarks = landmarker.detect_landmarks(frame)

            if landmarks is None:
                skipped += 1
                continue

            h, w = frame.shape[:2]

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)

            if left_ear is None or right_ear is None:
                skipped += 1
                continue

            ears.append((left_ear + right_ear) / 2.0)

        n_frames = len(rows)
        skipped_rate = skipped / max(n_frames, 1)

        if len(ears) == 0:
            ear_mean = 0.0
            ear_std = 0.0
            ear_min = 0.0
            ear_max = 0.0
            blink_count = 0.0
        else:
            ears_arr = np.array(ears, dtype=np.float32)
            ear_mean = float(np.mean(ears_arr))
            ear_std = float(np.std(ears_arr))
            ear_min = float(np.min(ears_arr))
            ear_max = float(np.max(ears_arr))

            below = ears_arr < args.blink_ear_threshold
            blink_count = 0
            in_blink = False
            for b in below:
                if b and not in_blink:
                    blink_count += 1
                    in_blink = True
                elif not b:
                    in_blink = False

        if len(motion_means) == 0:
            motion_mean = 0.0
            motion_std = 0.0
            motion_max = 0.0
        else:
            motion_arr = np.array(motion_means, dtype=np.float32)
            motion_mean = float(np.mean(motion_arr))
            motion_std = float(np.std(motion_arr))
            motion_max = float(np.max(motion_maxs))

        rows_out.append({
            "video_id": vid,
            "label": label,
            "ear_mean": ear_mean,
            "ear_std": ear_std,
            "ear_min": ear_min,
            "ear_max": ear_max,
            "blink_count": float(blink_count),
            "motion_mean": motion_mean,
            "motion_std": motion_std,
            "motion_max": motion_max,
            "skipped_rate": skipped_rate,
        })

        if i % 10 == 0 or i == len(by_vid):
            print(
                f"[{i}/{len(by_vid)}] {vid} | "
                f"label={label} | EAR={ear_mean:.4f} | "
                f"motion={motion_mean:.4f} | skipped={skipped_rate:.2f}"
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_csv, index=False, encoding="utf-8")

    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()