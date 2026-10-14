from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("MediaPipe is required. Install with: python -m pip install mediapipe") from e


# Eye landmarks for EAR from MediaPipe FaceMesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

POSE_LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -63.6, -12.5],
    [-43.3, 32.7, -26.0],
    [43.3, 32.7, -26.0],
    [-28.9, -28.9, -24.1],
    [28.9, -28.9, -24.1],
], dtype=np.float64)


def resolve_path(p):
    p = Path(str(p))
    if p.exists():
        return p
    alt = Path.cwd() / p
    if alt.exists():
        return alt
    return p


def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def compute_ear(landmarks, ids, w, h):
    pts = []
    for idx in ids:
        lm = landmarks.landmark[idx]
        pts.append((lm.x * w, lm.y * h))

    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = dist(p2, p6)
    vertical_2 = dist(p3, p5)
    horizontal = dist(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def estimate_head_pose(face_landmarks, w, h):
    image_points = []

    for key in [
        "nose_tip",
        "chin",
        "left_eye_outer",
        "right_eye_outer",
        "left_mouth",
        "right_mouth",
    ]:
        lm = face_landmarks.landmark[POSE_LANDMARK_IDS[key]]
        image_points.append([lm.x * w, lm.y * h])

    image_points = np.array(image_points, dtype=np.float64)

    focal_length = w
    center = (w / 2.0, h / 2.0)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok:
        return 0.0, 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    return yaw, pitch, roll


def face_bbox_features(face_landmarks):
    xs = [lm.x for lm in face_landmarks.landmark]
    ys = [lm.y for lm in face_landmarks.landmark]

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    cx = float((x_min + x_max) / 2.0)
    cy = float((y_min + y_max) / 2.0)
    area = float(max(0.0, x_max - x_min) * max(0.0, y_max - y_min))

    return cx, cy, area


def compute_optical_flow(prev_gray, gray):
    if prev_gray is None:
        return 0.0, 0.0

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag)), float(np.max(mag))


def process_frames(frames_csv, out_csv, eye_closed_thr=0.20):
    df = pd.read_csv(frames_csv)

    required = {"video_id", "frame_idx", "path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Columns={df.columns.tolist()}")

    df["video_id"] = df["video_id"].astype(str)
    df = df.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

    rows = []

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        for video_id, g in df.groupby("video_id"):
            g = g.sort_values("frame_idx").reset_index(drop=True)

            prev_gray = None
            prev_cx = None
            prev_cy = None

            label = int(g["label"].iloc[0])

            for _, r in g.iterrows():
                img_path = resolve_path(r["path"])
                img = cv2.imread(str(img_path))

                frame_idx = int(r["frame_idx"])

                row = {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "label": label,
                    "ear": 0.0,
                    "eye_closed": 0.0,
                    "motion_mean": 0.0,
                    "motion_max": 0.0,
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0,
                    "face_area": 0.0,
                    "face_center_motion": 0.0,
                    "frame_valid": 0.0,
                }

                if img is None:
                    rows.append(row)
                    continue

                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_small = cv2.resize(gray, (128, 128))

                motion_mean, motion_max = compute_optical_flow(prev_gray, gray_small)
                prev_gray = gray_small

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                if not result.multi_face_landmarks:
                    row["motion_mean"] = motion_mean
                    row["motion_max"] = motion_max
                    rows.append(row)
                    continue

                landmarks = result.multi_face_landmarks[0]

                ear_left = compute_ear(landmarks, LEFT_EYE, w, h)
                ear_right = compute_ear(landmarks, RIGHT_EYE, w, h)
                ear = float((ear_left + ear_right) / 2.0)

                yaw, pitch, roll = estimate_head_pose(landmarks, w, h)
                cx, cy, face_area = face_bbox_features(landmarks)

                if prev_cx is None:
                    center_motion = 0.0
                else:
                    center_motion = float(np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2))

                prev_cx = cx
                prev_cy = cy

                row.update({
                    "ear": ear,
                    "eye_closed": float(ear < eye_closed_thr),
                    "motion_mean": motion_mean,
                    "motion_max": motion_max,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                    "face_area": face_area,
                    "face_center_motion": center_motion,
                    "frame_valid": 1.0,
                })

                rows.append(row)

            print(f"[OK] {video_id} | frames={len(g)}")

    out = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")

    print("\\nSaved:", out_path)
    print("Rows:", len(out))
    print("Videos:", out["video_id"].nunique())
    print(out.head().to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--eye_closed_thr", type=float, default=0.20)
    args = parser.parse_args()

    process_frames(
        frames_csv=args.frames_csv,
        out_csv=args.out_csv,
        eye_closed_thr=args.eye_closed_thr,
    )


if __name__ == "__main__":
    main()
