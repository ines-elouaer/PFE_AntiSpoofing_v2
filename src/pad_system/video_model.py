from pathlib import Path
import math
import warnings
import json
import pickle

import cv2
import numpy as np
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD


warnings.filterwarnings("ignore")


class VideoPADModel:
    """
    Branche vidéo PAD finale.

    Version intégrée :
    - V3 multimodal : MobileNetV3 + LSTM + behavior/rPPG + gated fusion
    - Behavior-pose V6 : EAR + motion + Head Pose Dynamics
    - Fusion V6 : 0.85 * score_V3 + 0.15 * score_behavior_pose
    - Quality score vidéo pour la policy bancaire
    """

    MODEL_BEHAV_COLS = [
        "ear_mean",
        "ear_std",
        "ear_min",
        "ear_max",
        "blink_count",
        "motion_mean",
        "motion_std",
        "motion_max",
        "skipped_rate",
        "rppg_dominant_freq",
        "rppg_hr_estimate",
        "rppg_snr",
        "rppg_signal_std",
        "rppg_skipped_rate",
        "rppg_valid",
    ]

    OLD_BEHAV_COLS = [
        "ear_mean",
        "ear_std",
        "ear_min",
        "ear_max",
        "blink_count",
        "motion_mean",
        "motion_std",
        "motion_max",
        "skipped_rate",
    ]

    BEHAVIOR_POSE_V6_COLS = [
        "ear_mean",
        "ear_std",
        "ear_min",
        "ear_max",
        "blink_count",
        "motion_mean",
        "motion_std",
        "motion_max",
        "skipped_rate",
        "yaw_std",
        "pitch_std",
        "roll_std",
        "yaw_range",
        "pitch_range",
        "roll_range",
        "yaw_delta_mean",
        "pitch_delta_mean",
        "roll_delta_mean",
        "pose_autocorr",
        "pose_valid_rate",
    ]

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

    VIDEO_EXTS = [".avi", ".mp4", ".mov", ".mkv"]

    def __init__(
    self,
    checkpoint_path: str = r"E:\PFE_AntiSpoofing_v2\experiments\03_final_models\video_v6_behavior_pose\mixed_casia_axon_local_msu_gated_hard_balanced_rppg_v3\seed42\best_model.pth",
    test_csv: str = r"E:\PFE_AntiSpoofing_v2\data\mixed_casia_axon_local_msu\mixed_val_frames.csv",
    behav_test_csv: str = r"E:\PFE_AntiSpoofing_v2\data\mixed_casia_axon_local_msu_rppg\mixed_val_behav_rppg_norm.csv",
    behavior_stats_json: str = r"E:\PFE_AntiSpoofing_v2\data\mixed_casia_axon_local_msu_rppg\behav_rppg_norm_stats.json",
    scaler_path: str = "",
    behavior_pose_model_path: str = r"E:\PFE_AntiSpoofing_v2\experiments\03_final_models\banking_demo_models\final_models\behavior_pose_v6\behavior_pose_clf.pkl",
    img_size: int = 224,
    seq_len: int = 16,
    sample_mode: str = "center_consecutive",
):
        self.checkpoint_path = Path(checkpoint_path)
        self.test_csv = Path(test_csv)
        self.behav_test_csv = Path(behav_test_csv)
        self.behavior_stats_json = Path(behavior_stats_json) if behavior_stats_json else None
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.behavior_pose_model_path = (
            Path(behavior_pose_model_path) if behavior_pose_model_path else None
        )

        self.img_size = int(img_size)
        self.seq_len = int(seq_len)
        self.sample_mode = str(sample_mode)

        self.project_root = Path(__file__).resolve().parents[2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[VIDEO] Device: {self.device}")
        print(f"[VIDEO] Loading checkpoint: {self.checkpoint_path}")
        print(f"[VIDEO] Sampling mode: {self.sample_mode}")

        self.model, self.use_behav, self.behav_dim = self._load_model()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.frames_df = self._load_frames_csv()
        self.behav_df = self._load_behav_csv()
        self.behav_scaler = self._load_behavior_scaler()
        self.behav_stats = self._load_behavior_stats()
        self.behavior_pose_clf = self._load_behavior_pose_model()

    # ==========================================================
    # MODEL LOADING
    # ==========================================================

    def _extract_state_dict_and_config(self, ckpt):
        if isinstance(ckpt, dict):
            for key in ["model_state_dict", "model_state", "state_dict", "model", "net"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    print(f"[VIDEO] Checkpoint state chargé depuis la clé: {key}")
                    return ckpt[key], ckpt.get("config", {})

            print("[VIDEO][WARN] Aucune clé standard trouvée. Tentative checkpoint brut.")
            return ckpt, ckpt.get("config", {})

        raise RuntimeError(f"Format checkpoint non supporté: {type(ckpt)}")

    def _clean_state_dict_keys(self, state_dict):
        clean = {}

        for k, v in state_dict.items():
            if k.startswith("module."):
                clean[k[len("module."):]] = v
            else:
                clean[k] = v

        return clean

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint vidéo introuvable: {self.checkpoint_path}"
            )

        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)
        state_dict, cfg = self._extract_state_dict_and_config(ckpt)
        state_dict = self._clean_state_dict_keys(state_dict)

        use_behav = bool(cfg.get("use_behav", True))
        behav_dim = int(cfg.get("behav_dim", 15))
        behav_hidden = int(cfg.get("behav_hidden", 16))
        temporal_pool = cfg.get("temporal_pool", "median")
        use_gated_fusion = bool(cfg.get("use_gated_fusion", True))
        gate_hidden = int(cfg.get("gate_hidden", 128))

        model = CNN_LSTM_PAD(
            hidden=int(cfg.get("hidden", 256)),
            num_layers=int(cfg.get("num_layers", 1)),
            bidir=bool(cfg.get("bidir", False)),
            lstm_dropout=float(cfg.get("lstm_dropout", 0.2)),
            head_dropout=0.0,
            pretrained_backbone=False,
            temporal_pool=temporal_pool,
            use_behav=use_behav,
            behav_dim=behav_dim,
            behav_hidden=behav_hidden,
            use_gated_fusion=use_gated_fusion,
            gate_hidden=gate_hidden,
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print("\n========== VIDEO CHECKPOINT LOAD ==========")
        print(f"Checkpoint      : {self.checkpoint_path}")
        print(f"use_behav       : {use_behav}")
        print(f"behav_dim       : {behav_dim}")
        print(f"behav_hidden    : {behav_hidden}")
        print(f"temporal_pool   : {temporal_pool}")
        print(f"use_gated_fusion: {use_gated_fusion}")
        print(f"gate_hidden     : {gate_hidden}")
        print(f"Missing keys    : {len(missing)}")
        print(f"Unexpected keys : {len(unexpected)}")

        if len(missing) > 0:
            print("[VIDEO][WARN] Missing examples:", missing[:10])

        if len(unexpected) > 0:
            print("[VIDEO][WARN] Unexpected examples:", unexpected[:10])

        if len(missing) == 0 and len(unexpected) == 0:
            print("[VIDEO] Checkpoint chargé correctement.")
        else:
            print("[VIDEO][WARN] Checkpoint chargé avec différences.")

        model.to(self.device)
        return model, use_behav, behav_dim

    # ==========================================================
    # CSV / SCALER / STATS LOADING
    # ==========================================================

    def _load_frames_csv(self):
        if not self.test_csv.exists():
            print(f"[VIDEO][WARN] CSV test introuvable: {self.test_csv}")
            return None

        df = pd.read_csv(self.test_csv)

        required = {"video_id"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"Colonnes manquantes dans test_csv: {missing}")

        print(f"[VIDEO] test_csv chargé: {len(df)} lignes")
        return df

    def _load_behav_csv(self):
        if not self.behav_test_csv.exists():
            print(f"[VIDEO][WARN] Behavior CSV introuvable: {self.behav_test_csv}")
            return None

        df = pd.read_csv(self.behav_test_csv)
        print(f"[VIDEO] behav_test_csv chargé: {len(df)} lignes")
        return df

    def _load_behavior_scaler(self):
        if self.scaler_path is None:
            print("[VIDEO][WARN] Aucun scaler behavior fourni.")
            return None

        if not self.scaler_path.exists():
            print(f"[VIDEO][WARN] Scaler behavior introuvable: {self.scaler_path}")
            return None

        try:
            import joblib
            scaler = joblib.load(self.scaler_path)
            print(f"[VIDEO] Behavior scaler chargé: {self.scaler_path}")
            return scaler

        except Exception as e:
            print(f"[VIDEO][WARN] Impossible de charger le scaler behavior: {e}")
            return None

    def _load_behavior_stats(self):
        if self.behavior_stats_json is None:
            print("[VIDEO][WARN] Aucun fichier stats behavior+rPPG fourni.")
            return None

        if not self.behavior_stats_json.exists():
            print(f"[VIDEO][WARN] Stats behavior+rPPG introuvable: {self.behavior_stats_json}")
            return None

        try:
            with open(self.behavior_stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)

            print(f"[VIDEO] Stats behavior+rPPG chargées: {self.behavior_stats_json}")
            return stats

        except Exception as e:
            print(f"[VIDEO][WARN] Impossible de charger stats behavior+rPPG: {e}")
            return None

    def _load_behavior_pose_model(self):
        """
        Charge le classifieur behavior-pose V6.
        Supporte :
        - estimator sklearn direct avec predict_proba
        - Pipeline sklearn avec predict_proba
        - dict contenant scaler + clf/model
        """

        if self.behavior_pose_model_path is None:
            print("[VIDEO][WARN] Aucun modèle behavior-pose fourni.")
            return None

        if not self.behavior_pose_model_path.exists():
            print(f"[VIDEO][WARN] Modèle behavior-pose introuvable: {self.behavior_pose_model_path}")
            return None

        try:
            with open(self.behavior_pose_model_path, "rb") as f:
                obj = pickle.load(f)

            print(f"[VIDEO] Modèle behavior-pose V6 chargé: {self.behavior_pose_model_path}")
            return obj

        except Exception as e:
            print(f"[VIDEO][WARN] Impossible de charger behavior-pose V6: {e}")
            return None

    def _get_stat_mean_std(self, col: str):
        if self.behav_stats is None:
            return None, None

        if col in self.behav_stats and isinstance(self.behav_stats[col], dict):
            mean = self.behav_stats[col].get("mean", None)
            std = self.behav_stats[col].get("std", None)

            if mean is not None and std is not None:
                return float(mean), float(std)

        if "mean" in self.behav_stats and "std" in self.behav_stats:
            mean_map = self.behav_stats.get("mean", {})
            std_map = self.behav_stats.get("std", {})

            if col in mean_map and col in std_map:
                return float(mean_map[col]), float(std_map[col])

        return None, None

    # ==========================================================
    # FRAME PATH / SAMPLING
    # ==========================================================

    def _resolve_frame_path(self, row):
        for col in ["path", "frame_path", "image_path", "img_path"]:
            if col in row and isinstance(row[col], str):
                p = Path(row[col])

                if p.is_absolute() and p.exists():
                    return p

                candidate = self.project_root / p
                if candidate.exists():
                    return candidate

                candidate = self.project_root / "data" / p
                if candidate.exists():
                    return candidate

        raise FileNotFoundError(
            "Impossible de résoudre le chemin d'une frame. "
            "Vérifie les colonnes path/frame_path/image_path/img_path dans le CSV."
        )

    def _sample_rows(self, df_video):
        df_video = df_video.copy()

        if "frame_idx" in df_video.columns:
            df_video = df_video.sort_values("frame_idx")

        n = len(df_video)

        if n <= 0:
            raise RuntimeError("Aucune frame disponible pour cette vidéo.")

        if n >= self.seq_len:
            if self.sample_mode in ["consecutive", "center_consecutive", "consecutive_middle"]:
                start = max(0, (n - self.seq_len) // 2)
                sampled = df_video.iloc[start:start + self.seq_len]

            elif self.sample_mode == "uniform":
                idxs = [
                    int(round(i * (n - 1) / (self.seq_len - 1)))
                    for i in range(self.seq_len)
                ]
                sampled = df_video.iloc[idxs]

            else:
                raise ValueError(f"sample_mode invalide: {self.sample_mode}")

        else:
            rows = [df_video.iloc[i] for i in range(n)]

            while len(rows) < self.seq_len:
                rows.append(df_video.iloc[-1])

            sampled = pd.DataFrame(rows)

        return sampled

    # ==========================================================
    # SAFE STATS
    # ==========================================================

    @staticmethod
    def _safe_mean(values):
        return float(np.mean(values)) if len(values) > 0 else 0.0

    @staticmethod
    def _safe_std(values):
        return float(np.std(values)) if len(values) > 0 else 0.0

    @staticmethod
    def _safe_min(values):
        return float(np.min(values)) if len(values) > 0 else 0.0

    @staticmethod
    def _safe_max(values):
        return float(np.max(values)) if len(values) > 0 else 0.0

    @staticmethod
    def _safe_range(values):
        return float(np.max(values) - np.min(values)) if len(values) > 0 else 0.0

    @staticmethod
    def _safe_delta_mean(values):
        if len(values) < 2:
            return 0.0

        values = np.array(values, dtype=np.float32)
        return float(np.mean(np.abs(np.diff(values))))

    @staticmethod
    def _safe_autocorr(values):
        if len(values) < 3:
            return 0.0

        values = np.array(values, dtype=np.float32)

        if np.std(values[:-1]) < 1e-8 or np.std(values[1:]) < 1e-8:
            return 0.0

        corr = np.corrcoef(values[:-1], values[1:])[0, 1]

        if np.isnan(corr):
            return 0.0

        return float(corr)

    @staticmethod
    def _dist_2d(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _compute_ear(self, pts, eye_idxs):
        try:
            p1, p2, p3, p4, p5, p6 = [pts[i] for i in eye_idxs]

            vertical_1 = self._dist_2d(p2, p6)
            vertical_2 = self._dist_2d(p3, p5)
            horizontal = self._dist_2d(p1, p4)

            if horizontal <= 1e-6:
                return 0.0

            return float((vertical_1 + vertical_2) / (2.0 * horizontal))

        except Exception:
            return 0.0

    # ==========================================================
    # VIDEO READ / SAMPLING
    # ==========================================================

    def _read_video_rgb_gray_arrays(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

        frames_rgb = []
        frames_gray = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames_rgb.append(rgb)
            frames_gray.append(gray)

        cap.release()

        if len(frames_rgb) == 0:
            raise RuntimeError(f"Aucune frame lue depuis: {video_path}")

        return frames_rgb, frames_gray

    def _sample_raw_rgb_arrays(self, frames_rgb):
        frames_rgb = list(frames_rgb)

        if len(frames_rgb) >= self.seq_len:
            start = max(0, (len(frames_rgb) - self.seq_len) // 2)
            return frames_rgb[start:start + self.seq_len]

        while len(frames_rgb) < self.seq_len:
            frames_rgb.append(frames_rgb[-1])

        return frames_rgb

    def _sample_raw_gray_arrays(self, frames_gray):
        frames_gray = list(frames_gray)

        if len(frames_gray) >= self.seq_len:
            start = max(0, (len(frames_gray) - self.seq_len) // 2)
            return frames_gray[start:start + self.seq_len]

        while len(frames_gray) < self.seq_len:
            frames_gray.append(frames_gray[-1])

        return frames_gray

    def _read_raw_video_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

        frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            frames.append(img)

        cap.release()

        if not frames:
            raise RuntimeError(f"Aucune frame lue depuis: {video_path}")

        return frames

    def _sample_raw_frames(self, frames):
        frames = list(frames)

        if len(frames) >= self.seq_len:
            if self.sample_mode in ["consecutive", "center_consecutive", "consecutive_middle"]:
                start = max(0, (len(frames) - self.seq_len) // 2)
                frames = frames[start:start + self.seq_len]

            elif self.sample_mode == "uniform":
                idxs = [
                    int(round(i * (len(frames) - 1) / (self.seq_len - 1)))
                    for i in range(self.seq_len)
                ]
                frames = [frames[i] for i in idxs]

            else:
                raise ValueError(f"sample_mode invalide: {self.sample_mode}")

        else:
            while len(frames) < self.seq_len:
                frames.append(frames[-1])

        return frames

    # ==========================================================
    # BEHAVIOR V3 / rPPG
    # ==========================================================

    def _zero_behavior(self):
        return torch.zeros((1, self.behav_dim), dtype=torch.float32).to(self.device)

    def _get_behavior_vector(self, video_id):
        if self.behav_df is None:
            return self._zero_behavior()

        if "video_id" not in self.behav_df.columns:
            return self._zero_behavior()

        row = self.behav_df[self.behav_df["video_id"].astype(str) == str(video_id)]

        if row.empty:
            print(f"[VIDEO][WARN] Aucun behavior trouvé pour video_id={video_id}. Utilisation zéro.")
            return self._zero_behavior()

        row = row.iloc[0]
        values = []

        for col in self.MODEL_BEHAV_COLS:
            if col in self.behav_df.columns:
                values.append(float(row[col]))
            else:
                values.append(0.0)

        while len(values) < self.behav_dim:
            values.append(0.0)

        values = values[:self.behav_dim]

        return torch.tensor([values], dtype=torch.float32).to(self.device)

    def _extract_behavior_raw_unscaled(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"[VIDEO][WARN] Impossible d'ouvrir la vidéo pour behavior: {video_path}")
            return {col: 0.0 for col in self.OLD_BEHAV_COLS}

        frames_gray = []
        frames_rgb = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames_rgb.append(rgb)
            frames_gray.append(gray)

        cap.release()

        if len(frames_gray) == 0:
            print("[VIDEO][WARN] Aucune frame pour extraction behavior.")
            return {col: 0.0 for col in self.OLD_BEHAV_COLS}

        frames_rgb = self._sample_raw_rgb_arrays(frames_rgb)
        frames_gray = self._sample_raw_gray_arrays(frames_gray)

        motion_values = []

        for i in range(1, len(frames_gray)):
            diff = cv2.absdiff(frames_gray[i], frames_gray[i - 1])
            motion_values.append(float(np.mean(diff)))

        ear_values = []
        detected_frames = 0
        total_frames = len(frames_rgb)

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        face_model_path = self.project_root / "models" / "face_landmarker.task"

        if not face_model_path.exists():
            print(f"[VIDEO][WARN] face_landmarker.task introuvable: {face_model_path}")
            return {
                "ear_mean": 0.0,
                "ear_std": 0.0,
                "ear_min": 0.0,
                "ear_max": 0.0,
                "blink_count": 0.0,
                "motion_mean": self._safe_mean(motion_values),
                "motion_std": self._safe_std(motion_values),
                "motion_max": self._safe_max(motion_values),
                "skipped_rate": 1.0,
            }

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(face_model_path))

            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            with vision.FaceLandmarker.create_from_options(options) as landmarker:
                for rgb in frames_rgb:
                    h, w = rgb.shape[:2]
                    rgb = np.ascontiguousarray(rgb)

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb,
                    )

                    result = landmarker.detect(mp_image)

                    if not result.face_landmarks:
                        continue

                    detected_frames += 1
                    landmarks = result.face_landmarks[0]

                    pts = {}
                    for idx, lm in enumerate(landmarks):
                        pts[idx] = (float(lm.x * w), float(lm.y * h))

                    left_ear = self._compute_ear(pts, left_eye)
                    right_ear = self._compute_ear(pts, right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    if ear > 0:
                        ear_values.append(float(ear))

        except Exception as e:
            print(f"[VIDEO][WARN] MediaPipe behavior extraction échouée: {e}")
            detected_frames = 0
            ear_values = []

        skipped_rate = 1.0 - (detected_frames / total_frames) if total_frames > 0 else 1.0

        blink_count = 0
        if len(ear_values) > 0:
            closed = np.array(ear_values) < 0.18
            blink_count = int(np.sum(closed))

        return {
            "ear_mean": self._safe_mean(ear_values),
            "ear_std": self._safe_std(ear_values),
            "ear_min": self._safe_min(ear_values),
            "ear_max": self._safe_max(ear_values),
            "blink_count": float(blink_count),
            "motion_mean": self._safe_mean(motion_values),
            "motion_std": self._safe_std(motion_values),
            "motion_max": self._safe_max(motion_values),
            "skipped_rate": float(skipped_rate),
        }

    def _extract_rppg_from_raw_video(self, video_path: Path, max_frames: int = 64):
        from src.behavior.mp_landmarks import FaceLandmarkerHelper
        from src.behavior.extract_rppg import (
            get_cheek_roi,
            extract_rppg_features,
        )

        default_rppg = {
            "rppg_dominant_freq": 0.0,
            "rppg_hr_estimate": 0.0,
            "rppg_snr": 0.0,
            "rppg_signal_std": 0.0,
            "rppg_skipped_rate": 1.0,
            "rppg_valid": 0.0,
        }

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"[VIDEO][WARN] Impossible d'ouvrir la vidéo pour rPPG: {video_path}")
            return default_rppg

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if fps <= 1.0 or fps > 120.0:
            fps = 25.0

        if total_frames <= 0:
            cap.release()
            return default_rppg

        if total_frames <= max_frames:
            selected_indices = list(range(total_frames))
        else:
            start = max(0, (total_frames - max_frames) // 2)
            selected_indices = list(range(start, start + max_frames))

        selected_set = set(selected_indices)

        face_model_path = self.project_root / "models" / "face_landmarker.task"

        if not face_model_path.exists():
            print(f"[VIDEO][WARN] face_landmarker.task introuvable pour rPPG: {face_model_path}")
            cap.release()
            return default_rppg

        landmarker = FaceLandmarkerHelper(model_path=str(face_model_path))

        signal_r = []
        signal_g = []
        signal_b = []
        n_skipped = 0
        n_used = 0
        idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if idx not in selected_set:
                idx += 1
                continue

            n_used += 1

            landmarks = landmarker.detect_landmarks(frame)

            if landmarks is None:
                n_skipped += 1
                idx += 1
                continue

            h, w = frame.shape[:2]
            roi = get_cheek_roi(landmarks, w, h)

            if roi is None:
                n_skipped += 1
                idx += 1
                continue

            x1, y1, x2, y2 = roi
            patch = frame[y1:y2, x1:x2]

            if patch.size == 0:
                n_skipped += 1
                idx += 1
                continue

            signal_b.append(float(patch[:, :, 0].mean()))
            signal_g.append(float(patch[:, :, 1].mean()))
            signal_r.append(float(patch[:, :, 2].mean()))

            idx += 1

        cap.release()

        feats = extract_rppg_features(
            signal_r=np.array(signal_r, dtype=np.float32),
            signal_g=np.array(signal_g, dtype=np.float32),
            signal_b=np.array(signal_b, dtype=np.float32),
            fps=fps,
            n_frames=max(n_used, 1),
            n_skipped=n_skipped,
        )

        print("[VIDEO] rPPG raw video extrait.")
        print("[VIDEO] rPPG:", {k: round(float(v), 4) for k, v in feats.items()})
        print(f"[VIDEO] rPPG fps utilisé: {fps:.2f}")

        return feats

    def _normalize_raw_behavior(self, raw_features: dict):
        if self.behav_dim >= 15 and self.behav_stats is not None:
            values = []
            missing_stats = []

            for col in self.MODEL_BEHAV_COLS:
                value = float(raw_features.get(col, 0.0))
                mean, std = self._get_stat_mean_std(col)

                if mean is None or std is None:
                    missing_stats.append(col)
                    values.append(0.0)
                    continue

                if abs(std) < 1e-8:
                    std = 1.0

                value_norm = (value - mean) / std
                values.append(float(value_norm))

            if missing_stats:
                print("[VIDEO][WARN] Stats manquantes pour:", missing_stats)

            while len(values) < self.behav_dim:
                values.append(0.0)

            values = values[:self.behav_dim]

            print("[VIDEO] Normalisation V3 behavior+rPPG appliquée.")
            return values

        if self.behav_scaler is None:
            print("[VIDEO][WARN] Aucun scaler/stats. Features non normalisées.")
            values = [float(raw_features.get(c, 0.0)) for c in self.MODEL_BEHAV_COLS]

            while len(values) < self.behav_dim:
                values.append(0.0)

            return values[:self.behav_dim]

        try:
            arr = np.array(
                [[float(raw_features.get(c, 0.0)) for c in self.OLD_BEHAV_COLS]],
                dtype=np.float32,
            )

            arr_norm = self.behav_scaler.transform(arr)[0]
            values = [float(x) for x in arr_norm]

            while len(values) < self.behav_dim:
                values.append(0.0)

            return values[:self.behav_dim]

        except Exception as e:
            print(f"[VIDEO][WARN] Normalisation fallback échouée: {e}")
            values = [float(raw_features.get(c, 0.0)) for c in self.MODEL_BEHAV_COLS]

            while len(values) < self.behav_dim:
                values.append(0.0)

            return values[:self.behav_dim]

    def _extract_behavior_from_raw_video(self, video_path: Path):
        raw_features = self._extract_behavior_raw_unscaled(video_path)

        if self.behav_dim >= 15:
            rppg_features = self._extract_rppg_from_raw_video(video_path)
            raw_features.update(rppg_features)

        norm_values = self._normalize_raw_behavior(raw_features)

        while len(norm_values) < self.behav_dim:
            norm_values.append(0.0)

        norm_values = norm_values[:self.behav_dim]

        print("[VIDEO] Behavior+rPPG raw video extrait.")
        print("[VIDEO] Raw features:", {k: round(float(v), 4) for k, v in raw_features.items()})
        print("[VIDEO] Norm behavior+rPPG:", [round(float(v), 4) for v in norm_values])

        return torch.tensor([norm_values], dtype=torch.float32).to(self.device)

    # ==========================================================
    # BEHAVIOR-POSE V6
    # ==========================================================

    def _estimate_head_pose_from_landmarks(self, landmarks, w: int, h: int):
        try:
            image_points = []

            for key in [
                "nose_tip",
                "chin",
                "left_eye_outer",
                "right_eye_outer",
                "left_mouth",
                "right_mouth",
            ]:
                idx = self.POSE_LANDMARK_IDS[key]
                lm = landmarks[idx]
                image_points.append([float(lm.x * w), float(lm.y * h)])

            image_points = np.array(image_points, dtype=np.float64)

            focal_length = float(w)
            center = (w / 2.0, h / 2.0)

            camera_matrix = np.array([
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            ok, rotation_vec, _ = cv2.solvePnP(
                self.MODEL_POINTS_3D,
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

        except Exception:
            return 0.0, 0.0, 0.0

    def _extract_behavior_pose_v6_raw(self, video_path: Path):
        default = {col: 0.0 for col in self.BEHAVIOR_POSE_V6_COLS}

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"[VIDEO][WARN] Impossible d'ouvrir la vidéo pour behavior-pose: {video_path}")
            return default

        frames_rgb = []
        frames_gray = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames_rgb.append(rgb)
            frames_gray.append(gray)

        cap.release()

        if len(frames_rgb) == 0:
            return default

        frames_rgb = self._sample_raw_rgb_arrays(frames_rgb)
        frames_gray = self._sample_raw_gray_arrays(frames_gray)

        motion_values = []

        for i in range(1, len(frames_gray)):
            diff = cv2.absdiff(frames_gray[i], frames_gray[i - 1])
            motion_values.append(float(np.mean(diff)))

        ear_values = []
        yaw_values = []
        pitch_values = []
        roll_values = []

        detected_frames = 0
        total_frames = len(frames_rgb)

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        face_model_path = self.project_root / "models" / "face_landmarker.task"

        if not face_model_path.exists():
            print(f"[VIDEO][WARN] face_landmarker.task introuvable pour behavior-pose: {face_model_path}")

            default.update({
                "motion_mean": self._safe_mean(motion_values),
                "motion_std": self._safe_std(motion_values),
                "motion_max": self._safe_max(motion_values),
                "skipped_rate": 1.0,
                "pose_valid_rate": 0.0,
            })

            return default

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(face_model_path))

            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            with vision.FaceLandmarker.create_from_options(options) as landmarker:
                for rgb in frames_rgb:
                    h, w = rgb.shape[:2]
                    rgb = np.ascontiguousarray(rgb)

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb,
                    )

                    result = landmarker.detect(mp_image)

                    if not result.face_landmarks:
                        continue

                    detected_frames += 1
                    landmarks = result.face_landmarks[0]

                    pts = {}
                    for idx, lm in enumerate(landmarks):
                        pts[idx] = (float(lm.x * w), float(lm.y * h))

                    left_ear = self._compute_ear(pts, left_eye)
                    right_ear = self._compute_ear(pts, right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    if ear > 0:
                        ear_values.append(float(ear))

                    yaw, pitch, roll = self._estimate_head_pose_from_landmarks(
                        landmarks,
                        w,
                        h,
                    )

                    yaw_values.append(float(yaw))
                    pitch_values.append(float(pitch))
                    roll_values.append(float(roll))

        except Exception as e:
            print(f"[VIDEO][WARN] Behavior-pose V6 extraction échouée: {e}")
            detected_frames = 0
            ear_values = []
            yaw_values = []
            pitch_values = []
            roll_values = []

        skipped_rate = 1.0 - (detected_frames / total_frames) if total_frames > 0 else 1.0
        pose_valid_rate = detected_frames / total_frames if total_frames > 0 else 0.0

        blink_count = 0
        if len(ear_values) > 0:
            closed = np.array(ear_values) < 0.18
            blink_count = int(np.sum(closed))

        features = {
            "ear_mean": self._safe_mean(ear_values),
            "ear_std": self._safe_std(ear_values),
            "ear_min": self._safe_min(ear_values),
            "ear_max": self._safe_max(ear_values),
            "blink_count": float(blink_count),

            "motion_mean": self._safe_mean(motion_values),
            "motion_std": self._safe_std(motion_values),
            "motion_max": self._safe_max(motion_values),
            "skipped_rate": float(skipped_rate),

            "yaw_std": self._safe_std(yaw_values),
            "pitch_std": self._safe_std(pitch_values),
            "roll_std": self._safe_std(roll_values),
            "yaw_range": self._safe_range(yaw_values),
            "pitch_range": self._safe_range(pitch_values),
            "roll_range": self._safe_range(roll_values),
            "yaw_delta_mean": self._safe_delta_mean(yaw_values),
            "pitch_delta_mean": self._safe_delta_mean(pitch_values),
            "roll_delta_mean": self._safe_delta_mean(roll_values),
            "pose_autocorr": self._safe_autocorr(yaw_values),
            "pose_valid_rate": float(pose_valid_rate),
        }

        print("[VIDEO] Behavior-pose V6 raw features:")
        print({k: round(float(v), 4) for k, v in features.items()})

        return features
    def _is_behavior_pose_reliable(self, features: dict) -> tuple:
        """
        Vérifie si les features behavior-pose sont fiables en runtime webcam.

        Retourne :
        - reliable: bool
        - reason: str
        """

        pose_valid_rate = float(features.get("pose_valid_rate", 0.0))
        skipped_rate = float(features.get("skipped_rate", 1.0))

        yaw_range = abs(float(features.get("yaw_range", 0.0)))
        pitch_range = abs(float(features.get("pitch_range", 0.0)))
        roll_range = abs(float(features.get("roll_range", 0.0)))

        yaw_delta = abs(float(features.get("yaw_delta_mean", 0.0)))
        pitch_delta = abs(float(features.get("pitch_delta_mean", 0.0)))
        roll_delta = abs(float(features.get("roll_delta_mean", 0.0)))

        if pose_valid_rate < 0.70:
            return False, f"low_pose_valid_rate={pose_valid_rate:.3f}"

        if skipped_rate > 0.30:
            return False, f"high_skipped_rate={skipped_rate:.3f}"

        if yaw_range > 90 or pitch_range > 90 or roll_range > 90:
            return False, (
                f"unstable_pose_range="
                f"yaw:{yaw_range:.1f},pitch:{pitch_range:.1f},roll:{roll_range:.1f}"
            )

        if yaw_delta > 30 or pitch_delta > 30 or roll_delta > 30:
            return False, (
                f"unstable_pose_delta="
                f"yaw:{yaw_delta:.1f},pitch:{pitch_delta:.1f},roll:{roll_delta:.1f}"
            )

        return True, "behavior_pose_reliable"
    def _predict_behavior_pose_score(self, video_path: Path):
        """
        Calcule score_behavior_pose avec le modèle V6.
        Si les features sont instables, retourne None pour ne pas utiliser
        le score behavior-pose comme preuve d'attaque.
        """

        if self.behavior_pose_clf is None:
            return None, None, "behavior_pose_model_missing"

        try:
            features = self._extract_behavior_pose_v6_raw(video_path)

            reliable, reliability_reason = self._is_behavior_pose_reliable(features)

            if not reliable:
                print(f"[VIDEO][WARN] Behavior-pose non fiable: {reliability_reason}")
                return None, features, reliability_reason

            x = np.array(
                [[float(features.get(col, 0.0)) for col in self.BEHAVIOR_POSE_V6_COLS]],
                dtype=np.float32,
            )

            score = float(self.behavior_pose_clf.predict_proba(x)[0, 1])

            print(f"[VIDEO] score_behavior_pose V6: {score:.4f}")

            return score, features, reliability_reason

        except Exception as e:
            reason = f"behavior_pose_failed: {e}"
            print(f"[VIDEO][WARN] score_behavior_pose V6 échoué: {e}")
            return None, None, reason

    # ==========================================================
    # CSV VIDEO_ID INFERENCE
    # ==========================================================

    def _predict_from_csv_video_id(self, video_id):
        if self.frames_df is None:
            raise RuntimeError(
                "frames_df non chargé. Impossible d'utiliser le mode CSV video_id."
            )

        video_id = str(video_id).strip()

        df_video = self.frames_df[
            self.frames_df["video_id"].astype(str) == video_id
        ]

        if df_video.empty:
            available = self.frames_df["video_id"].drop_duplicates().head(10).tolist()

            raise FileNotFoundError(
                f"Aucune frame trouvée dans test_csv pour video_id={video_id}. "
                f"Exemples disponibles: {available}"
            )

        sampled = self._sample_rows(df_video)

        images = []

        for _, row in sampled.iterrows():
            frame_path = self._resolve_frame_path(row)
            img = Image.open(frame_path).convert("RGB")
            images.append(self.transform(img))

        seq = torch.stack(images, dim=0)
        seq = seq.unsqueeze(0).to(self.device)

        behav = self._get_behavior_vector(video_id)

        print(f"[VIDEO] CSV video_id={video_id} | frames utilisées={len(images)}")

        if self.use_behav:
            logits = self.model(seq, behav)
        else:
            logits = self.model(seq)

        proba = torch.softmax(logits, dim=1)
        return float(proba[0, 1].item())

    # ==========================================================
    # RAW VIDEO INFERENCE
    # ==========================================================

    def _predict_from_raw_video(self, video_path: Path):
        frames = self._read_raw_video_frames(video_path)
        frames = self._sample_raw_frames(frames)

        images = [self.transform(img) for img in frames]

        seq = torch.stack(images, dim=0)
        seq = seq.unsqueeze(0).to(self.device)

        behav = self._extract_behavior_from_raw_video(video_path)

        print(f"[VIDEO] Raw video={video_path.name} | frames utilisées={len(images)}")
        print("[VIDEO] Behavior+rPPG raw video: features extraites + normalisées.")

        if self.use_behav:
            logits = self.model(seq, behav)
        else:
            logits = self.model(seq)

        proba = torch.softmax(logits, dim=1)
        return float(proba[0, 1].item())

    # ==========================================================
    # VIDEO QUALITY + DETAILED PREDICTION
    # ==========================================================

    def compute_video_quality_score(self, video_path: str) -> float:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return 0.0

        brightness_values = []
        blur_values = []
        valid_frames = 0
        total_frames = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            total_frames += 1

            if total_frames % 3 != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            brightness = float(np.mean(gray) / 255.0)
            blur = float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()) / 10.0)

            brightness_values.append(brightness)
            blur_values.append(min(max(blur, 0.0), 1.0))
            valid_frames += 1

        cap.release()

        if valid_frames == 0:
            return 0.0

        brightness_mean = float(np.mean(brightness_values))
        brightness_std = float(np.std(brightness_values))
        blur_mean = float(np.mean(blur_values))
        frame_valid_rate = valid_frames / max(total_frames, 1)

        brightness_balance = 1.0 - abs(brightness_mean - 0.5) * 2.0
        brightness_balance = min(max(brightness_balance, 0.0), 1.0)

        lighting_stability = 1.0 - min(max(brightness_std, 0.0), 1.0)

        quality_score = (
            0.35 * blur_mean +
            0.30 * frame_valid_rate +
            0.20 * brightness_balance +
            0.15 * lighting_stability
        )

        return float(min(max(quality_score, 0.0), 1.0))

    @torch.no_grad()
    def predict_with_details(self, video_path_or_id: str) -> dict:
        """
        Prédiction finale détaillée V6.

        Sorties :
        - score_v3_multimodal : score du modèle vidéo V3
        - score_behavior_pose : score du modèle behavior-pose V6
        - score_final_v6      : fusion score-level
        - video_quality_score : score qualité vidéo
        - behavior_pose_status : indique si les features behavior-pose sont fiables ou instables
        """

        score_v3 = float(self.predict(video_path_or_id))

        p = Path(str(video_path_or_id))

        behavior_pose_features = None
        behavior_pose_status = "not_computed"

        if p.exists() and p.suffix.lower() in self.VIDEO_EXTS:
            video_quality_score = self.compute_video_quality_score(str(p))

            score_behavior_pose, behavior_pose_features, behavior_pose_status = (
                self._predict_behavior_pose_score(p)
            )
        else:
            video_quality_score = 0.70
            score_behavior_pose = None
            behavior_pose_status = "csv_or_video_id_mode"

        if score_behavior_pose is not None:
            score_final_v6 = (
                0.85 * float(score_v3)
                + 0.15 * float(score_behavior_pose)
            )
            fusion_status = "behavior_pose_loaded"
        else:
            score_final_v6 = float(score_v3)
            fusion_status = "fallback_score_v3_only"

        score_final_v6 = float(min(max(score_final_v6, 0.0), 1.0))

        return {
            "score_v3_multimodal": float(score_v3),

            "score_behavior_pose": (
                float(score_behavior_pose)
                if score_behavior_pose is not None
                else None
            ),

            "score_final_v6": float(score_final_v6),
            "video_quality_score": float(video_quality_score),

            "behavior_pose_status": behavior_pose_status,
            "behavior_pose_features": behavior_pose_features,

            "fusion": {
                "type": "score_level_fusion_v6",
                "formula": "score_final_v6 = 0.85 * score_v3_multimodal + 0.15 * score_behavior_pose",
                "w_v3_multimodal": 0.85,
                "w_behavior_pose": 0.15,
                "status": fusion_status,
            },
        }

    # ==========================================================
    # PUBLIC PREDICT
    # ==========================================================

    @torch.no_grad()
    def predict(self, video_path_or_id: str) -> float:
        print(f"[INFO] Analyse vidéo: {video_path_or_id}")

        p = Path(str(video_path_or_id))

        if p.exists() and p.suffix.lower() in self.VIDEO_EXTS:
            return self._predict_from_raw_video(p)

        if p.suffix.lower() in self.VIDEO_EXTS:
            video_id = p.stem
        else:
            video_id = str(video_path_or_id)

        video_id = video_id.strip()

        for ext in self.VIDEO_EXTS:
            video_id = video_id.replace(ext, "")

        return self._predict_from_csv_video_id(video_id)
