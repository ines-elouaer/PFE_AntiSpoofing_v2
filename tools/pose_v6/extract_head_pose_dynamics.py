from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        FACE_MESH_MODULE = mp.solutions.face_mesh
    else:
        from mediapipe.python.solutions import face_mesh as FACE_MESH_MODULE

except ImportError:
    raise ImportError("MediaPipe is required. Install it with: python -m pip install mediapipe")

LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],          # nose tip
    [0.0, -63.6, -12.5],      # chin
    [-43.3, 32.7, -26.0],     # left eye outer
    [43.3, 32.7, -26.0],      # right eye outer
    [-28.9, -28.9, -24.1],    # left mouth
    [28.9, -28.9, -24.1],     # right mouth
], dtype=np.float64)


def resolve_path(p):
    path = Path(str(p))
    if path.exists():
        return path

    project_root = Path.cwd()
    alt = project_root / path
    if alt.exists():
        return alt

    return path


def safe_autocorr(x):
    x = np.asarray(x, dtype=np.float64)

    if len(x) < 3:
        return 0.0

    a = x[:-1]
    b = x[1:]

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0

    c = np.corrcoef(a, b)[0, 1]

    if np.isnan(c):
        return 0.0

    return float(c)


def estimate_pose_from_landmarks(face_landmarks, width, height):
    image_points = []

    for key in [
        "nose_tip",
        "chin",
        "left_eye_outer",
        "right_eye_outer",
        "left_mouth",
        "right_mouth",
    ]:
        lm = face_landmarks.landmark[LANDMARK_IDS[key]]
        image_points.append([lm.x * width, lm.y * height])

    image_points = np.array(image_points, dtype=np.float64)

    focal_length = width
    center = (width / 2.0, height / 2.0)

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
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

    # OpenCV returns Euler-like angles in degrees.
    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    return yaw, pitch, roll


def compute_pose_features(yaw_seq, pitch_seq, roll_seq, valid_count, total_count):
    yaw_seq = np.asarray(yaw_seq, dtype=np.float64)
    pitch_seq = np.asarray(pitch_seq, dtype=np.float64)
    roll_seq = np.asarray(roll_seq, dtype=np.float64)

    if len(yaw_seq) == 0:
        return {
            "yaw_std": 0.0,
            "pitch_std": 0.0,
            "roll_std": 0.0,
            "yaw_range": 0.0,
            "pitch_range": 0.0,
            "roll_range": 0.0,
            "yaw_delta_mean": 0.0,
            "pitch_delta_mean": 0.0,
            "roll_delta_mean": 0.0,
            "pose_autocorr": 0.0,
            "pose_valid_rate": 0.0,
        }

    def mean_abs_delta(x):
        if len(x) < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(x))))

    pose_energy = yaw_seq + pitch_seq + roll_seq

    return {
        "yaw_std": float(np.std(yaw_seq)),
        "pitch_std": float(np.std(pitch_seq)),
        "roll_std": float(np.std(roll_seq)),
        "yaw_range": float(np.max(yaw_seq) - np.min(yaw_seq)),
        "pitch_range": float(np.max(pitch_seq) - np.min(pitch_seq)),
        "roll_range": float(np.max(roll_seq) - np.min(roll_seq)),
        "yaw_delta_mean": mean_abs_delta(yaw_seq),
        "pitch_delta_mean": mean_abs_delta(pitch_seq),
        "roll_delta_mean": mean_abs_delta(roll_seq),
        "pose_autocorr": safe_autocorr(pose_energy),
        "pose_valid_rate": float(valid_count / max(total_count, 1)),
    }


def extract_from_frames(frames_csv, out_csv, max_frames_per_video=64):
    df = pd.read_csv(frames_csv)

    required = {"video_id", "frame_idx", "path", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in frames_csv: {missing}. Columns={df.columns.tolist()}")

    df["video_id"] = df["video_id"].astype(str)
    df = df.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

    mp_face_mesh = FACE_MESH_MODULE

    rows = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        for video_id, g in df.groupby("video_id"):
            g = g.sort_values("frame_idx").reset_index(drop=True)

            if len(g) > max_frames_per_video:
                idx = np.linspace(0, len(g) - 1, max_frames_per_video).astype(int)
                g = g.iloc[idx].copy()

            yaw_seq = []
            pitch_seq = []
            roll_seq = []

            valid_count = 0
            total_count = len(g)
            label = int(g["label"].iloc[0])

            for _, r in g.iterrows():
                img_path = resolve_path(r["path"])
                img = cv2.imread(str(img_path))

                if img is None:
                    continue

                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                if not result.multi_face_landmarks:
                    continue

                face_landmarks = result.multi_face_landmarks[0]
                pose = estimate_pose_from_landmarks(face_landmarks, w, h)

                if pose is None:
                    continue

                yaw, pitch, roll = pose

                yaw_seq.append(yaw)
                pitch_seq.append(pitch)
                roll_seq.append(roll)
                valid_count += 1

            features = compute_pose_features(
                yaw_seq=yaw_seq,
                pitch_seq=pitch_seq,
                roll_seq=roll_seq,
                valid_count=valid_count,
                total_count=total_count,
            )

            row = {
                "video_id": video_id,
                "label": label,
                **features,
            }

            rows.append(row)

            print(
                f"[OK] {video_id} | valid={valid_count}/{total_count} | "
                f"yaw_std={features['yaw_std']:.4f} | pitch_std={features['pitch_std']:.4f}"
            )

    out = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")

    print("\\nSaved:", out_path)
    print("Videos:", out["video_id"].nunique())
    print(out.head().to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--max_frames_per_video", type=int, default=64)
    args = parser.parse_args()

    extract_from_frames(
        frames_csv=args.frames_csv,
        out_csv=args.out_csv,
        max_frames_per_video=args.max_frames_per_video,
    )


if __name__ == "__main__":
    main()
