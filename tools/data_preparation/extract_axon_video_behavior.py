from pathlib import Path
import argparse
import json
import math
import warnings

import cv2
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


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


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return default
        return float(x)
    except Exception:
        return default


def dist_2d(a, b) -> float:
    ax, ay = a
    bx, by = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def compute_ear(pts, idxs) -> float:
    """
    Eye Aspect Ratio approximatif avec indices FaceMesh.
    idxs = [p1, p2, p3, p4, p5, p6]
    """
    try:
        p1, p2, p3, p4, p5, p6 = [pts[i] for i in idxs]

        vertical_1 = dist_2d(p2, p6)
        vertical_2 = dist_2d(p3, p5)
        horizontal = dist_2d(p1, p4)

        if horizontal <= 1e-6:
            return 0.0

        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

    except Exception:
        return 0.0


def compute_mar(pts) -> float:
    """
    Mouth Aspect Ratio approximatif.
    """
    try:
        left = pts[61]
        right = pts[291]
        upper = pts[13]
        lower = pts[14]

        horizontal = dist_2d(left, right)
        vertical = dist_2d(upper, lower)

        if horizontal <= 1e-6:
            return 0.0

        return float(vertical / horizontal)

    except Exception:
        return 0.0


def estimate_head_pose(pts, image_w: int, image_h: int):
    """
    Estimation approximative yaw / pitch / roll avec solvePnP.

    Ce n'est pas parfait, mais suffisant pour extraire des stats compatibles
    avec la structure CASIA.
    """

    try:
        image_points = np.array(
            [
                pts[1],    # nose tip
                pts[152],  # chin
                pts[33],   # left eye outer
                pts[263],  # right eye outer
                pts[61],   # left mouth
                pts[291],  # right mouth
            ],
            dtype=np.float64,
        )

        model_points = np.array(
            [
                (0.0, 0.0, 0.0),          # nose tip
                (0.0, -63.6, -12.5),      # chin
                (-43.3, 32.7, -26.0),     # left eye
                (43.3, 32.7, -26.0),      # right eye
                (-28.9, -28.9, -24.1),    # left mouth
                (28.9, -28.9, -24.1),     # right mouth
            ],
            dtype=np.float64,
        )

        focal_length = image_w
        center = (image_w / 2.0, image_h / 2.0)

        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        return yaw, pitch, roll

    except Exception:
        return None


class AxonBehaviorExtractor:
    def __init__(
        self,
        project_root: Path,
        face_model_path: Path,
        blink_ear_threshold: float = 0.18,
    ):
        self.project_root = project_root
        self.face_model_path = face_model_path
        self.blink_ear_threshold = blink_ear_threshold

        if not self.face_model_path.exists():
            raise FileNotFoundError(f"FaceLandmarker introuvable: {self.face_model_path}")

        import mediapipe as mp

        self.mp = mp

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.face_model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

        print(f"[INFO] FaceLandmarker chargé: {self.face_model_path}")

    def extract_video_features(self, frame_paths):
        ear_values = []
        mar_values = []
        yaw_values = []
        pitch_values = []
        roll_values = []

        motion_values = []
        lk_flow_values = []

        detected_frames = 0
        total_frames = len(frame_paths)

        prev_gray = None

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        for frame_path in frame_paths:
            img = cv2.imread(str(frame_path))

            if img is None:
                continue

            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Motion simple : différence absolue frame t / t-1
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_values.append(float(np.mean(diff)))

                # Optical flow Lucas-Kanade
                try:
                    p0 = cv2.goodFeaturesToTrack(
                        prev_gray,
                        maxCorners=100,
                        qualityLevel=0.01,
                        minDistance=7,
                        blockSize=7,
                    )

                    if p0 is not None:
                        p1, st, _ = cv2.calcOpticalFlowPyrLK(
                            prev_gray,
                            gray,
                            p0,
                            None,
                        )

                        if p1 is not None and st is not None:
                            good_new = p1[st == 1]
                            good_old = p0[st == 1]

                            if len(good_new) > 0:
                                flow = np.linalg.norm(good_new - good_old, axis=1)
                                lk_flow_values.append(float(np.mean(flow)))
                except Exception:
                    pass

            prev_gray = gray

            # MediaPipe landmarks
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                mp_image = self.mp.Image(
                    image_format=self.mp.ImageFormat.SRGB,
                    data=rgb,
                )

                result = self.landmarker.detect(mp_image)

                if not result.face_landmarks:
                    continue

                detected_frames += 1

                landmarks = result.face_landmarks[0]

                pts = {}
                for idx, lm in enumerate(landmarks):
                    pts[idx] = (float(lm.x * w), float(lm.y * h))

                left_ear = compute_ear(pts, left_eye)
                right_ear = compute_ear(pts, right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear > 0:
                    ear_values.append(float(ear))

                mar = compute_mar(pts)
                if mar >= 0:
                    mar_values.append(float(mar))

                pose = estimate_head_pose(pts, w, h)
                if pose is not None:
                    yaw, pitch, roll = pose
                    yaw_values.append(float(yaw))
                    pitch_values.append(float(pitch))
                    roll_values.append(float(roll))

            except Exception:
                continue

        skipped_rate = 1.0
        if total_frames > 0:
            skipped_rate = 1.0 - (detected_frames / total_frames)

        features = self._aggregate_features(
            ear_values=ear_values,
            mar_values=mar_values,
            yaw_values=yaw_values,
            pitch_values=pitch_values,
            roll_values=roll_values,
            motion_values=motion_values,
            lk_flow_values=lk_flow_values,
            skipped_rate=skipped_rate,
        )

        return features

    def _aggregate_features(
        self,
        ear_values,
        mar_values,
        yaw_values,
        pitch_values,
        roll_values,
        motion_values,
        lk_flow_values,
        skipped_rate,
    ):
        ear = np.array(ear_values, dtype=np.float32)
        mar = np.array(mar_values, dtype=np.float32)
        yaw = np.array(yaw_values, dtype=np.float32)
        pitch = np.array(pitch_values, dtype=np.float32)
        roll = np.array(roll_values, dtype=np.float32)
        motion = np.array(motion_values, dtype=np.float32)
        lk_flow = np.array(lk_flow_values, dtype=np.float32)

        def mean(arr):
            return float(arr.mean()) if len(arr) else 0.0

        def std(arr):
            return float(arr.std()) if len(arr) else 0.0

        def minv(arr):
            return float(arr.min()) if len(arr) else 0.0

        def maxv(arr):
            return float(arr.max()) if len(arr) else 0.0

        blink_count = 0
        perclos = 0.0

        if len(ear):
            closed = ear < self.blink_ear_threshold
            blink_count = int(np.sum(closed))
            perclos = float(np.mean(closed))

        return {
            "ear_mean": mean(ear),
            "ear_std": std(ear),
            "ear_min": minv(ear),
            "ear_max": maxv(ear),
            "blink_count": blink_count,
            "perclos": perclos,

            "motion_mean": mean(motion),
            "motion_std": std(motion),
            "motion_max": maxv(motion),

            "lk_flow_mean": mean(lk_flow),
            "lk_flow_std": std(lk_flow),
            "lk_flow_max": maxv(lk_flow),

            "mar_mean": mean(mar),
            "mar_std": std(mar),
            "mar_max": maxv(mar),

            "yaw_std": std(yaw),
            "pitch_std": std(pitch),
            "roll_std": std(roll),
            "yaw_range": float(maxv(yaw) - minv(yaw)) if len(yaw) else 0.0,
            "pitch_range": float(maxv(pitch) - minv(pitch)) if len(pitch) else 0.0,

            "skipped_rate": safe_float(skipped_rate, 1.0),
        }


def find_scaler_auto(project_root: Path):
    patterns = [
        "**/*scaler*.pkl",
        "**/*scaler*.joblib",
        "**/*standard*.pkl",
        "**/*standard*.joblib",
    ]

    candidates = []

    for pattern in patterns:
        candidates.extend(project_root.glob(pattern))

    candidates = [
        p for p in candidates
        if "axon" not in str(p).lower()
    ]

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: len(str(p)))

    return candidates[0]


def load_scaler(scaler_path: Path):
    import joblib

    if scaler_path is None:
        return None

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler introuvable: {scaler_path}")

    return joblib.load(scaler_path)


def normalize_behavior(raw_df: pd.DataFrame, scaler, feature_cols):
    norm_df = raw_df.copy()

    for col in feature_cols:
        if col not in norm_df.columns:
            norm_df[col] = 0.0

    X = norm_df[feature_cols].astype(float).values
    X_norm = scaler.transform(X)

    for i, col in enumerate(feature_cols):
        norm_df[col] = X_norm[:, i]

    return norm_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scaler_path",
        type=str,
        default="",
        help="Chemin vers le StandardScaler fitté sur CASIA train (.pkl/.joblib).",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Sauvegarder seulement les features brutes sans normalisation.",
    )

    args = parser.parse_args()

    project_root = get_project_root()
    axon_prepared_dir = load_paths(project_root)

    manifests_dir = axon_prepared_dir / "manifests"
    frames_manifest_path = manifests_dir / "axon_video_frames_manifest.csv"

    raw_out_path = manifests_dir / "axon_video_behav_raw.csv"
    norm_out_path = manifests_dir / "axon_video_behav_norm.csv"
    summary_path = manifests_dir / "axon_video_behavior_summary.json"

    face_model_path = project_root / "models" / "face_landmarker.task"

    if not frames_manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest frames introuvable: {frames_manifest_path}\n"
            "Lance d'abord: python tools\\axon\\prepare_axon_video_frames.py"
        )

    df = pd.read_csv(frames_manifest_path)

    print("========== CONFIG ==========")
    print(f"Project root      : {project_root}")
    print(f"Frames manifest   : {frames_manifest_path}")
    print(f"Raw behavior CSV  : {raw_out_path}")
    print(f"Norm behavior CSV : {norm_out_path}")
    print(f"Face model        : {face_model_path}")

    extractor = AxonBehaviorExtractor(
        project_root=project_root,
        face_model_path=face_model_path,
    )

    rows = []

    grouped = df.sort_values(["video_id", "frame_idx"]).groupby("video_id")
    total_videos = len(grouped)

    print(f"\n[INFO] Vidéos à traiter: {total_videos}")

    for idx, (video_id, g) in enumerate(grouped, start=1):
        g = g.sort_values("frame_idx")

        frame_paths = [
            project_root / str(p)
            for p in g["path"].tolist()
        ]

        meta = g.iloc[0].to_dict()

        features = extractor.extract_video_features(frame_paths)

        row = {
            "video_id": video_id,
            "label": int(meta["label"]),
            "label_name": str(meta.get("label_name", "REAL" if int(meta["label"]) == 0 else "SPOOF")),
            "attack_type": str(meta.get("attack_type", "unknown")),
            "level": str(meta.get("level", "unknown")),
            "subject_id": str(meta.get("subject_id", "unknown")),
        }

        row.update(features)
        rows.append(row)

        if idx % 25 == 0:
            print(f"[INFO] Progression: {idx}/{total_videos} vidéos traitées")

    raw_df = pd.DataFrame(rows)

    for col in BEHAV_COLS:
        if col not in raw_df.columns:
            raw_df[col] = 0.0

    raw_df = raw_df[
        ["video_id", "label", "label_name", "attack_type", "level", "subject_id"] + BEHAV_COLS
    ]

    raw_df.to_csv(raw_out_path, index=False, encoding="utf-8")

    scaler_path = None
    scaler = None
    normalized = False
    scaler_feature_cols = BEHAV_COLS

    if not args.no_normalize:
        if args.scaler_path:
            scaler_path = Path(args.scaler_path)
            if not scaler_path.is_absolute():
                scaler_path = (project_root / scaler_path).resolve()
        else:
            scaler_path = find_scaler_auto(project_root)

        if scaler_path is not None:
            print(f"\n[INFO] Scaler trouvé/utilisé: {scaler_path}")
            scaler = load_scaler(scaler_path)

            if hasattr(scaler, "feature_names_in_"):
                scaler_feature_cols = list(scaler.feature_names_in_)

            missing_cols = [c for c in scaler_feature_cols if c not in raw_df.columns]
            if missing_cols:
                raise RuntimeError(
                    "Colonnes attendues par le scaler absentes dans Axon behavior: "
                    + str(missing_cols)
                )

            norm_df = normalize_behavior(
                raw_df=raw_df,
                scaler=scaler,
                feature_cols=scaler_feature_cols,
            )

            norm_df.to_csv(norm_out_path, index=False, encoding="utf-8")
            normalized = True

        else:
            print("\n[WARN] Aucun scaler CASIA trouvé.")
            print("[WARN] axon_video_behav_norm.csv ne sera pas généré.")
            print("[WARN] Utilise --scaler_path pour fournir le scaler CASIA train.")

    summary = {
        "frames_manifest": str(frames_manifest_path),
        "raw_behavior_csv": str(raw_out_path),
        "norm_behavior_csv": str(norm_out_path) if normalized else None,
        "scaler_path": str(scaler_path) if scaler_path is not None else None,
        "normalized": normalized,
        "num_videos": int(len(raw_df)),
        "label_counts": raw_df["label_name"].value_counts().to_dict(),
        "attack_type_counts": raw_df["attack_type"].value_counts().to_dict(),
        "feature_cols": BEHAV_COLS,
        "scaler_feature_cols": scaler_feature_cols,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== SUMMARY ==========")
    print(f"Raw behavior CSV  : {raw_out_path}")
    print(f"Norm behavior CSV : {norm_out_path if normalized else 'NON GENERE'}")
    print(f"Summary JSON      : {summary_path}")
    print(f"Videos            : {len(raw_df)}")
    print(f"Normalized        : {normalized}")

    print("\n========== LABEL COUNTS ==========")
    print(raw_df["label_name"].value_counts())

    print("\n========== ATTACK TYPE COUNTS ==========")
    print(raw_df["attack_type"].value_counts())

    print("\n[OK] Extraction behavior Axon terminée.")


if __name__ == "__main__":
    main()