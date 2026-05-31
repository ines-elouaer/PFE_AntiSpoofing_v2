"""
video_model.py — VideoPADModel V6
==================================
Corrections appliquées :
1. FaceLandmarker singleton — créé une seule fois dans __init__, pas à chaque requête
2. Suppression double lecture vidéo — predict_with_details lit la vidéo UNE SEULE FOIS
3. Fix blink_count — seuil adaptatif relatif (baseline * 0.75) au lieu de 0.18 fixe
"""
import os 
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

    MODEL_BEHAV_COLS = [
        "ear_mean", "ear_std", "ear_min", "ear_max", "blink_count",
        "motion_mean", "motion_std", "motion_max", "skipped_rate",
        "rppg_dominant_freq", "rppg_hr_estimate", "rppg_snr",
        "rppg_signal_std", "rppg_skipped_rate", "rppg_valid",
    ]

    OLD_BEHAV_COLS = [
        "ear_mean", "ear_std", "ear_min", "ear_max", "blink_count",
        "motion_mean", "motion_std", "motion_max", "skipped_rate",
    ]

    BEHAVIOR_POSE_V6_COLS = [
        "ear_mean", "ear_std", "ear_min", "ear_max", "blink_count",
        "motion_mean", "motion_std", "motion_max", "skipped_rate",
        "yaw_std", "pitch_std", "roll_std",
        "yaw_range", "pitch_range", "roll_range",
        "yaw_delta_mean", "pitch_delta_mean", "roll_delta_mean",
        "pose_autocorr", "pose_valid_rate",
    ]

    POSE_LANDMARK_IDS = {
        "nose_tip": 1, "chin": 152,
        "left_eye_outer": 33, "right_eye_outer": 263,
        "left_mouth": 61, "right_mouth": 291,
    }

    MODEL_POINTS_3D = np.array([
        [0.0, 0.0, 0.0], [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0], [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1],
    ], dtype=np.float64)

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(
    self,
    checkpoint_path: str = "experiments/03_final_models/video_v6_behavior_pose/mixed_casia_axon_local_msu_gated_hard_balanced_rppg_v3/seed42/best_model.pth",
    test_csv: str = "data/mixed_casia_axon_local_msu/mixed_val_frames.csv",
    behav_test_csv: str = "data/mixed_casia_axon_local_msu_rppg/mixed_val_behav_rppg_norm.csv",
    behavior_stats_json: str = "data/mixed_casia_axon_local_msu_rppg/behav_rppg_norm_stats.json",
    scaler_path: str = "",
    behavior_pose_model_path: str = "experiments/03_final_models/banking_demo_models/final_models/behavior_pose_v6/behavior_pose_clf.pkl",
    img_size: int = 224,
    seq_len: int = 16,
    sample_mode: str = "center_consecutive",
):
        import os

        checkpoint_path = os.getenv("PAD_VIDEO_CHECKPOINT", checkpoint_path)
        test_csv = os.getenv("PAD_TEST_CSV", test_csv)
        behav_test_csv = os.getenv("PAD_BEHAV_TEST_CSV", behav_test_csv)
        behavior_stats_json = os.getenv("PAD_BEHAVIOR_STATS", behavior_stats_json)
        behavior_pose_model_path = os.getenv("PAD_BEHAVIOR_POSE_MODEL", behavior_pose_model_path)

        self.checkpoint_path = Path(checkpoint_path)
        self.test_csv = Path(test_csv)
        self.behav_test_csv = Path(behav_test_csv)
        self.behavior_stats_json = Path(behavior_stats_json) if behavior_stats_json else None
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.behavior_pose_model_path = Path(behavior_pose_model_path) if behavior_pose_model_path else None
        self.img_size    = int(img_size)
        self.seq_len     = int(seq_len)
        self.sample_mode = str(sample_mode)

        self.project_root = Path(__file__).resolve().parents[2]
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[VIDEO] Device: {self.device}")
        print(f"[VIDEO] Loading checkpoint: {self.checkpoint_path}")
        print(f"[VIDEO] Sampling mode: {self.sample_mode}")

        self.model, self.use_behav, self.behav_dim = self._load_model()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.frames_df        = self._load_frames_csv()
        self.behav_df         = self._load_behav_csv()
        self.behav_scaler     = self._load_behavior_scaler()
        self.behav_stats      = self._load_behavior_stats()
        self.behavior_pose_clf = self._load_behavior_pose_model()

        # ── CORRECTION 1 : Initialiser MediaPipe UNE SEULE FOIS ──────────────
        self._face_landmarker_image  = None
        self._mediapipe_module       = None
        self._rppg_landmarker_helper = None
        self._init_mediapipe_singleton()

    # ==========================================================
    # CORRECTION 1 — MEDIAPIPE SINGLETON
    # ==========================================================

    def _init_mediapipe_singleton(self):
        """
        Initialise MediaPipe une seule fois au démarrage.
        Évite la réinitialisation à chaque requête (source principale de latence).
        """
        face_model_path = self.project_root / "models" / "face_landmarker.task"

        if not face_model_path.exists():
            print(f"[VIDEO][WARN] face_landmarker.task introuvable: {face_model_path}")
            return

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

            self._mediapipe_module      = mp
            self._face_landmarker_image = vision.FaceLandmarker.create_from_options(options)
            print(f"[VIDEO] FaceLandmarker IMAGE initialisé (singleton): {face_model_path}")

        except Exception as e:
            print(f"[VIDEO][WARN] Impossible d'initialiser FaceLandmarker IMAGE: {e}")
            self._face_landmarker_image = None
            self._mediapipe_module      = None

        try:
            from src.behavior.mp_landmarks import FaceLandmarkerHelper
            self._rppg_landmarker_helper = FaceLandmarkerHelper(
                model_path=str(face_model_path)
            )
            print(f"[VIDEO] FaceLandmarkerHelper rPPG initialisé (singleton): {face_model_path}")

        except Exception as e:
            print(f"[VIDEO][WARN] Impossible d'initialiser FaceLandmarkerHelper rPPG: {e}")
            self._rppg_landmarker_helper = None

    # Méthodes d'accès aux singletons — rétrocompatibilité
    def _get_cached_face_landmarker_image(self, face_model_path: Path):
        """Retourne le singleton FaceLandmarker IMAGE — pas de réinitialisation."""
        if self._face_landmarker_image is None or self._mediapipe_module is None:
            # Fallback : réinitialiser si pas encore fait
            self._init_mediapipe_singleton()
        return self._mediapipe_module, self._face_landmarker_image

    def _get_cached_rppg_landmarker_helper(self, face_model_path: Path):
        """Retourne le singleton FaceLandmarkerHelper rPPG — pas de réinitialisation."""
        if self._rppg_landmarker_helper is None:
            self._init_mediapipe_singleton()
        return self._rppg_landmarker_helper

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
            clean[k[len("module."):] if k.startswith("module.") else k] = v
        return clean

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint vidéo introuvable: {self.checkpoint_path}")

        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)
        state_dict, cfg = self._extract_state_dict_and_config(ckpt)
        state_dict = self._clean_state_dict_keys(state_dict)

        use_behav        = bool(cfg.get("use_behav", True))
        behav_dim        = int(cfg.get("behav_dim", 15))
        behav_hidden     = int(cfg.get("behav_hidden", 16))
        temporal_pool    = cfg.get("temporal_pool", "median")
        use_gated_fusion = bool(cfg.get("use_gated_fusion", True))
        gate_hidden      = int(cfg.get("gate_hidden", 128))

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

        if len(missing) == 0 and len(unexpected) == 0:
            print("[VIDEO] Checkpoint chargé correctement.")
        else:
            print("[VIDEO][WARN] Checkpoint chargé avec différences.")
            if missing:    print("[VIDEO][WARN] Missing:", missing[:5])
            if unexpected: print("[VIDEO][WARN] Unexpected:", unexpected[:5])

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
        if "video_id" not in df.columns:
            raise ValueError("Colonne 'video_id' manquante dans test_csv")
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
            print(f"[VIDEO][WARN] Scaler introuvable: {self.scaler_path}")
            return None
        try:
            import joblib
            scaler = joblib.load(self.scaler_path)
            print(f"[VIDEO] Scaler chargé: {self.scaler_path}")
            return scaler
        except Exception as e:
            print(f"[VIDEO][WARN] Scaler non chargé: {e}")
            return None

    def _load_behavior_stats(self):
        if self.behavior_stats_json is None or not self.behavior_stats_json.exists():
            print("[VIDEO][WARN] Stats behavior+rPPG introuvables.")
            return None
        try:
            with open(self.behavior_stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)
            print(f"[VIDEO] Stats behavior+rPPG chargées: {self.behavior_stats_json}")
            return stats
        except Exception as e:
            print(f"[VIDEO][WARN] Stats non chargées: {e}")
            return None

    def _load_behavior_pose_model(self):
        if self.behavior_pose_model_path is None or not self.behavior_pose_model_path.exists():
            print("[VIDEO][WARN] Modèle behavior-pose introuvable.")
            return None
        try:
            with open(self.behavior_pose_model_path, "rb") as f:
                obj = pickle.load(f)
            print(f"[VIDEO] Modèle behavior-pose V6 chargé: {self.behavior_pose_model_path}")
            return obj
        except Exception as e:
            print(f"[VIDEO][WARN] Modèle behavior-pose non chargé: {e}")
            return None

    def _get_stat_mean_std(self, col: str):
        if self.behav_stats is None:
            return None, None
        if col in self.behav_stats and isinstance(self.behav_stats[col], dict):
            m = self.behav_stats[col].get("mean")
            s = self.behav_stats[col].get("std")
            if m is not None and s is not None:
                return float(m), float(s)
        if "mean" in self.behav_stats and "std" in self.behav_stats:
            mm = self.behav_stats.get("mean", {})
            ss = self.behav_stats.get("std", {})
            if col in mm and col in ss:
                return float(mm[col]), float(ss[col])
        return None, None

    # ==========================================================
    # SAFE STATS
    # ==========================================================

    @staticmethod
    def _safe_mean(v):   return float(np.mean(v))    if len(v) > 0 else 0.0
    @staticmethod
    def _safe_std(v):    return float(np.std(v))     if len(v) > 0 else 0.0
    @staticmethod
    def _safe_min(v):    return float(np.min(v))     if len(v) > 0 else 0.0
    @staticmethod
    def _safe_max(v):    return float(np.max(v))     if len(v) > 0 else 0.0
    @staticmethod
    def _safe_range(v):  return float(np.max(v) - np.min(v)) if len(v) > 0 else 0.0

    @staticmethod
    def _safe_delta_mean(v):
        if len(v) < 2: return 0.0
        return float(np.mean(np.abs(np.diff(np.array(v, dtype=np.float32)))))

    @staticmethod
    def _safe_autocorr(v):
        if len(v) < 3: return 0.0
        v = np.array(v, dtype=np.float32)
        if np.std(v[:-1]) < 1e-8 or np.std(v[1:]) < 1e-8: return 0.0
        c = np.corrcoef(v[:-1], v[1:])[0, 1]
        return 0.0 if np.isnan(c) else float(c)

    @staticmethod
    def _dist_2d(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _compute_ear(self, pts, eye_idxs):
        try:
            p1, p2, p3, p4, p5, p6 = [pts[i] for i in eye_idxs]
            v1 = self._dist_2d(p2, p6)
            v2 = self._dist_2d(p3, p5)
            h  = self._dist_2d(p1, p4)
            return 0.0 if h <= 1e-6 else float((v1 + v2) / (2.0 * h))
        except Exception:
            return 0.0

    # ==========================================================
    # CORRECTION 3 — BLINK COUNT ADAPTATIF
    # ==========================================================

    @staticmethod
    def _count_blinks_from_ear_sequence(ear_values: list) -> int:
        """
        Détection des clignements par seuil adaptatif relatif.
        
        CORRECTION : l'ancien seuil fixe 0.18 était trop bas pour la webcam
        (ear_mean webcam ≈ 0.22–0.33). Le nouveau seuil est calculé comme
        75% de la baseline (percentile 75 de la séquence EAR).
        
        Un clignement = chute de l'EAR sous baseline * 0.75.
        """
        if len(ear_values) < 3:
            return 0

        ear_arr  = np.array(ear_values, dtype=np.float32)
        baseline  = float(np.percentile(ear_arr, 75))
        threshold = baseline * 0.75

        blinks   = 0
        in_blink = False

        for ear in ear_arr:
            if ear < threshold:
                if not in_blink:
                    blinks  += 1
                    in_blink = True
            else:
                in_blink = False

        return blinks

    # ==========================================================
    # VIDEO READ / SAMPLING
    # ==========================================================

    def _read_video_rgb_gray_arrays(self, video_path: Path):
        """Lit la vidéo une seule fois et retourne frames RGB et grises."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

        frames_rgb  = []
        frames_gray = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        cap.release()

        if not frames_rgb:
            raise RuntimeError(f"Aucune frame lue depuis: {video_path}")

        return frames_rgb, frames_gray

    def _sample_raw_rgb_arrays(self, frames_rgb):
        frames_rgb = list(frames_rgb)
        n = len(frames_rgb)
        if n >= self.seq_len:
            start = max(0, (n - self.seq_len) // 2)
            return frames_rgb[start:start + self.seq_len]
        while len(frames_rgb) < self.seq_len:
            frames_rgb.append(frames_rgb[-1])
        return frames_rgb

    def _sample_raw_gray_arrays(self, frames_gray):
        frames_gray = list(frames_gray)
        n = len(frames_gray)
        if n >= self.seq_len:
            start = max(0, (n - self.seq_len) // 2)
            return frames_gray[start:start + self.seq_len]
        while len(frames_gray) < self.seq_len:
            frames_gray.append(frames_gray[-1])
        return frames_gray

    def _sample_raw_frames(self, frames):
        frames = list(frames)
        n = len(frames)
        if n >= self.seq_len:
            if self.sample_mode in ["consecutive", "center_consecutive", "consecutive_middle"]:
                start = max(0, (n - self.seq_len) // 2)
                return frames[start:start + self.seq_len]
            elif self.sample_mode == "uniform":
                idxs = [int(round(i*(n-1)/(self.seq_len-1))) for i in range(self.seq_len)]
                return [frames[i] for i in idxs]
            else:
                raise ValueError(f"sample_mode invalide: {self.sample_mode}")
        while len(frames) < self.seq_len:
            frames.append(frames[-1])
        return frames

    # ==========================================================
    # BEHAVIOR EXTRACTION FROM PRE-LOADED FRAMES
    # ==========================================================

    def _extract_ear_and_pose_from_frames(self, frames_rgb_sampled: list):
        """
        CORRECTION 2 : extrait EAR + pose depuis les frames déjà chargées
        en mémoire — pas de relecture vidéo.
        Utilise le singleton FaceLandmarker.
        """
        ear_values   = []
        yaw_values   = []
        pitch_values = []
        roll_values  = []
        detected     = 0
        total        = len(frames_rgb_sampled)

        mp_module   = self._mediapipe_module
        landmarker  = self._face_landmarker_image

        if mp_module is None or landmarker is None:
            print("[VIDEO][WARN] FaceLandmarker non disponible — EAR et pose à zéro.")
            return ear_values, yaw_values, pitch_values, roll_values, 0, total

        for rgb in frames_rgb_sampled:
            h, w = rgb.shape[:2]
            rgb_c = np.ascontiguousarray(rgb)

            try:
                mp_image = mp_module.Image(
                    image_format=mp_module.ImageFormat.SRGB,
                    data=rgb_c,
                )
                result = landmarker.detect(mp_image)
            except Exception:
                continue

            if not result.face_landmarks:
                continue

            detected += 1
            landmarks = result.face_landmarks[0]

            pts = {idx: (float(lm.x * w), float(lm.y * h))
                   for idx, lm in enumerate(landmarks)}

            l_ear = self._compute_ear(pts, self.LEFT_EYE)
            r_ear = self._compute_ear(pts, self.RIGHT_EYE)
            ear   = (l_ear + r_ear) / 2.0
            if ear > 0:
                ear_values.append(float(ear))

            yaw, pitch, roll = self._estimate_head_pose_from_landmarks(landmarks, w, h)
            yaw_values.append(float(yaw))
            pitch_values.append(float(pitch))
            roll_values.append(float(roll))

        return ear_values, yaw_values, pitch_values, roll_values, detected, total

    def _compute_motion_from_gray_frames(self, frames_gray_sampled: list):
        """Calcule le mouvement inter-frames depuis les frames grises déjà en mémoire."""
        motion_values = []
        for i in range(1, len(frames_gray_sampled)):
            diff = cv2.absdiff(frames_gray_sampled[i], frames_gray_sampled[i-1])
            motion_values.append(float(np.mean(diff)))
        return motion_values

    def _build_behavior_features(
        self,
        ear_values, motion_values,
        detected, total
    ) -> dict:
        """Construit le dict de features comportementales depuis les séquences calculées."""
        skipped_rate = 1.0 - (detected / total) if total > 0 else 1.0
        blink_count  = float(self._count_blinks_from_ear_sequence(ear_values))

        return {
            "ear_mean"    : self._safe_mean(ear_values),
            "ear_std"     : self._safe_std(ear_values),
            "ear_min"     : self._safe_min(ear_values),
            "ear_max"     : self._safe_max(ear_values),
            "blink_count" : blink_count,
            "motion_mean" : self._safe_mean(motion_values),
            "motion_std"  : self._safe_std(motion_values),
            "motion_max"  : self._safe_max(motion_values),
            "skipped_rate": float(skipped_rate),
        }

    def _build_behavior_pose_features(
        self,
        ear_values, motion_values,
        yaw_values, pitch_values, roll_values,
        detected, total
    ) -> dict:
        """Construit le dict de 20 features behavior-pose V6."""
        base = self._build_behavior_features(ear_values, motion_values, detected, total)
        pose_valid_rate = detected / total if total > 0 else 0.0

        base.update({
            "yaw_std"         : self._safe_std(yaw_values),
            "pitch_std"       : self._safe_std(pitch_values),
            "roll_std"        : self._safe_std(roll_values),
            "yaw_range"       : self._safe_range(yaw_values),
            "pitch_range"     : self._safe_range(pitch_values),
            "roll_range"      : self._safe_range(roll_values),
            "yaw_delta_mean"  : self._safe_delta_mean(yaw_values),
            "pitch_delta_mean": self._safe_delta_mean(pitch_values),
            "roll_delta_mean" : self._safe_delta_mean(roll_values),
            "pose_autocorr"   : self._safe_autocorr(yaw_values),
            "pose_valid_rate" : float(pose_valid_rate),
        })
        return base

    # ==========================================================
    # BEHAVIOR V3 / rPPG (legacy — garde pour CSV mode)
    # ==========================================================

    def _zero_behavior(self):
        return torch.zeros((1, self.behav_dim), dtype=torch.float32).to(self.device)

    def _get_behavior_vector(self, video_id):
        if self.behav_df is None or "video_id" not in self.behav_df.columns:
            return self._zero_behavior()
        row = self.behav_df[self.behav_df["video_id"].astype(str) == str(video_id)]
        if row.empty:
            return self._zero_behavior()
        row    = row.iloc[0]
        values = [float(row[c]) if c in self.behav_df.columns else 0.0
                  for c in self.MODEL_BEHAV_COLS]
        while len(values) < self.behav_dim: values.append(0.0)
        return torch.tensor([values[:self.behav_dim]], dtype=torch.float32).to(self.device)

    def _normalize_raw_behavior(self, raw_features: dict):
        if self.behav_dim >= 15 and self.behav_stats is not None:
            values = []
            for col in self.MODEL_BEHAV_COLS:
                value = float(raw_features.get(col, 0.0))
                mean, std = self._get_stat_mean_std(col)
                if mean is None or std is None:
                    values.append(0.0)
                    continue
                if abs(std) < 1e-8: std = 1.0
                values.append(float((value - mean) / std))
            while len(values) < self.behav_dim: values.append(0.0)
            print("[VIDEO] Normalisation V3 behavior+rPPG appliquée.")
            return values[:self.behav_dim]

        if self.behav_scaler is not None:
            try:
                arr = np.array(
                    [[float(raw_features.get(c, 0.0)) for c in self.OLD_BEHAV_COLS]],
                    dtype=np.float32,
                )
                values = [float(x) for x in self.behav_scaler.transform(arr)[0]]
                while len(values) < self.behav_dim: values.append(0.0)
                return values[:self.behav_dim]
            except Exception as e:
                print(f"[VIDEO][WARN] Normalisation scaler échouée: {e}")

        values = [float(raw_features.get(c, 0.0)) for c in self.MODEL_BEHAV_COLS]
        while len(values) < self.behav_dim: values.append(0.0)
        return values[:self.behav_dim]

    def _extract_rppg_from_raw_video(self, video_path: Path, max_frames: int = 64):
        """rPPG — inchangé, utilisé uniquement pour le quality flag."""
        from src.behavior.extract_rppg import get_cheek_roi, extract_rppg_features

        default_rppg = {
            "rppg_dominant_freq": 0.0, "rppg_hr_estimate": 0.0,
            "rppg_snr": 0.0, "rppg_signal_std": 0.0,
            "rppg_skipped_rate": 1.0, "rppg_valid": 0.0,
        }

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return default_rppg

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 1.0 or fps > 120.0: fps = 25.0
        if total_frames <= 0:
            cap.release(); return default_rppg

        start         = max(0, (total_frames - max_frames) // 2)
        selected_set  = set(range(start, min(start + max_frames, total_frames)))
        landmarker    = self._rppg_landmarker_helper

        signal_r, signal_g, signal_b = [], [], []
        n_skipped = n_used = idx = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx not in selected_set:
                idx += 1; continue
            n_used += 1
            if landmarker is not None:
                landmarks = landmarker.detect_landmarks(frame)
                if landmarks is not None:
                    h, w = frame.shape[:2]
                    roi = get_cheek_roi(landmarks, w, h)
                    if roi is not None:
                        x1, y1, x2, y2 = roi
                        patch = frame[y1:y2, x1:x2]
                        if patch.size > 0:
                            signal_b.append(float(patch[:,:,0].mean()))
                            signal_g.append(float(patch[:,:,1].mean()))
                            signal_r.append(float(patch[:,:,2].mean()))
                            idx += 1; continue
            n_skipped += 1
            idx += 1

        cap.release()

        feats = extract_rppg_features(
            signal_r=np.array(signal_r, dtype=np.float32),
            signal_g=np.array(signal_g, dtype=np.float32),
            signal_b=np.array(signal_b, dtype=np.float32),
            fps=fps, n_frames=max(n_used, 1), n_skipped=n_skipped,
        )
        print("[VIDEO] rPPG raw video extrait.")
        print("[VIDEO] rPPG:", {k: round(float(v), 4) for k, v in feats.items()})
        print(f"[VIDEO] rPPG fps utilisé: {fps:.2f}")
        return feats

    # ==========================================================
    # POSE ESTIMATION
    # ==========================================================

    def _estimate_head_pose_from_landmarks(self, landmarks, w: int, h: int):
        try:
            image_points = []
            for key in ["nose_tip","chin","left_eye_outer","right_eye_outer","left_mouth","right_mouth"]:
                idx = self.POSE_LANDMARK_IDS[key]
                lm  = landmarks[idx]
                image_points.append([float(lm.x * w), float(lm.y * h)])

            image_points  = np.array(image_points, dtype=np.float64)
            focal_length  = float(w)
            center        = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            ok, rotation_vec, _ = cv2.solvePnP(
                self.MODEL_POINTS_3D, image_points,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok: return 0.0, 0.0, 0.0

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
            return float(angles[1]), float(angles[0]), float(angles[2])

        except Exception:
            return 0.0, 0.0, 0.0

    # ==========================================================
    # BEHAVIOR POSE RELIABILITY
    # ==========================================================

    def _is_behavior_pose_reliable(self, features: dict) -> tuple:
        pose_valid_rate = float(features.get("pose_valid_rate", 0.0))
        skipped_rate    = float(features.get("skipped_rate", 1.0))
        yaw_range       = abs(float(features.get("yaw_range", 0.0)))
        pitch_range     = abs(float(features.get("pitch_range", 0.0)))
        roll_range      = abs(float(features.get("roll_range", 0.0)))
        yaw_delta       = abs(float(features.get("yaw_delta_mean", 0.0)))
        pitch_delta     = abs(float(features.get("pitch_delta_mean", 0.0)))
        roll_delta      = abs(float(features.get("roll_delta_mean", 0.0)))

        if pose_valid_rate < 0.70:
            return False, f"low_pose_valid_rate={pose_valid_rate:.3f}"
        if skipped_rate > 0.30:
            return False, f"high_skipped_rate={skipped_rate:.3f}"
        if yaw_range > 90 or pitch_range > 90 or roll_range > 90:
            return False, f"unstable_pose_range=yaw:{yaw_range:.1f},pitch:{pitch_range:.1f},roll:{roll_range:.1f}"
        if yaw_delta > 30 or pitch_delta > 30 or roll_delta > 30:
            return False, f"unstable_pose_delta=yaw:{yaw_delta:.1f},pitch:{pitch_delta:.1f},roll:{roll_delta:.1f}"
        return True, "behavior_pose_reliable"

    # ==========================================================
    # VIDEO QUALITY
    # ==========================================================

    def compute_video_quality_score(self, video_path: str) -> float:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return 0.0

        brightness_values, blur_values = [], []
        valid_frames = total_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            total_frames += 1
            if total_frames % 3 != 0: continue
            gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray) / 255.0)
            blur       = float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()) / 10.0)
            brightness_values.append(brightness)
            blur_values.append(min(max(blur, 0.0), 1.0))
            valid_frames += 1

        cap.release()
        if valid_frames == 0: return 0.0

        b_mean    = float(np.mean(brightness_values))
        b_std     = float(np.std(brightness_values))
        blur_mean = float(np.mean(blur_values))
        fvr       = valid_frames / max(total_frames, 1)

        b_balance  = min(max(1.0 - abs(b_mean - 0.5) * 2.0, 0.0), 1.0)
        b_stability= 1.0 - min(max(b_std, 0.0), 1.0)

        return float(min(max(
            0.35 * blur_mean + 0.30 * fvr + 0.20 * b_balance + 0.15 * b_stability,
            0.0), 1.0))

    # ==========================================================
    # CSV VIDEO_ID INFERENCE
    # ==========================================================

    def _resolve_frame_path(self, row):
        for col in ["path", "frame_path", "image_path", "img_path"]:
            if col in row and isinstance(row[col], str):
                p = Path(row[col])
                if p.is_absolute() and p.exists(): return p
                c = self.project_root / p
                if c.exists(): return c
                c = self.project_root / "data" / p
                if c.exists(): return c
        raise FileNotFoundError("Impossible de résoudre le chemin d'une frame.")

    def _sample_rows(self, df_video):
        df_video = df_video.copy()
        if "frame_idx" in df_video.columns:
            df_video = df_video.sort_values("frame_idx")
        n = len(df_video)
        if n <= 0: raise RuntimeError("Aucune frame disponible.")
        if n >= self.seq_len:
            if self.sample_mode in ["consecutive","center_consecutive","consecutive_middle"]:
                start = max(0, (n - self.seq_len) // 2)
                return df_video.iloc[start:start + self.seq_len]
            elif self.sample_mode == "uniform":
                idxs = [int(round(i*(n-1)/(self.seq_len-1))) for i in range(self.seq_len)]
                return df_video.iloc[idxs]
        rows = list(df_video.itertuples())
        while len(rows) < self.seq_len: rows.append(df_video.iloc[-1])
        return pd.DataFrame(rows)

    def _predict_from_csv_video_id(self, video_id):
        if self.frames_df is None:
            raise RuntimeError("frames_df non chargé.")
        video_id  = str(video_id).strip()
        df_video  = self.frames_df[self.frames_df["video_id"].astype(str) == video_id]
        if df_video.empty:
            raise FileNotFoundError(f"Aucune frame pour video_id={video_id}")
        sampled = self._sample_rows(df_video)
        images  = [self.transform(Image.open(self._resolve_frame_path(r)).convert("RGB"))
                   for _, r in sampled.iterrows()]
        seq     = torch.stack(images, dim=0).unsqueeze(0).to(self.device)
        behav   = self._get_behavior_vector(video_id)
        logits  = self.model(seq, behav) if self.use_behav else self.model(seq)
        proba   = torch.softmax(logits, dim=1)
        print(f"[VIDEO] CSV video_id={video_id} | frames utilisées={len(images)}")
        return float(proba[0, 1].item())

    # ==========================================================
    # CORRECTION 2 — predict_with_details : lecture vidéo UNIQUE
    # ==========================================================

    @torch.no_grad()
    def predict_with_details(self, video_path_or_id: str) -> dict:
        """
        Prédiction V6 complète.

        CORRECTION : la vidéo est lue UNE SEULE FOIS en mémoire.
        Les frames sont partagées entre :
        - la branche visuelle CNN+LSTM
        - l'extraction des features comportementales
        - l'extraction behavior-pose V6
        Plus de double lecture, plus de double initialisation MediaPipe.
        """
        p = Path(str(video_path_or_id))

        if p.exists() and p.suffix.lower() in self.VIDEO_EXTS:

            # ── Étape 1 : Lire la vidéo UNE SEULE FOIS ────────────────────
            frames_rgb_all, frames_gray_all = self._read_video_rgb_gray_arrays(p)
            frames_rgb  = self._sample_raw_rgb_arrays(frames_rgb_all)
            frames_gray = self._sample_raw_gray_arrays(frames_gray_all)

            video_quality_score = self.compute_video_quality_score(str(p))

            # ── Étape 2 : Extraire EAR + pose depuis frames en mémoire ────
            (ear_values, yaw_values, pitch_values, roll_values,
             detected, total) = self._extract_ear_and_pose_from_frames(frames_rgb)

            # ── Étape 3 : Calculer motion depuis frames grises en mémoire ─
            motion_values = self._compute_motion_from_gray_frames(frames_gray)

            # ── Étape 4 : Construire features comportementales ────────────
            raw_behav = self._build_behavior_features(
                ear_values, motion_values, detected, total
            )
            print("[VIDEO] Raw features:", {k: round(float(v), 4) for k, v in raw_behav.items()})

            # ── Étape 5 : rPPG quality flag (lecture séparée nécessaire) ──
            rppg_features = self._extract_rppg_from_raw_video(p)
            raw_behav.update(rppg_features)

            # ── Étape 6 : Normaliser + tensoriser ─────────────────────────
            norm_values = self._normalize_raw_behavior(raw_behav)
            while len(norm_values) < self.behav_dim: norm_values.append(0.0)
            behav_tensor = torch.tensor(
                [norm_values[:self.behav_dim]], dtype=torch.float32
            ).to(self.device)
            print("[VIDEO] Norm behavior+rPPG:", [round(float(v), 4) for v in norm_values[:self.behav_dim]])

            # ── Étape 7 : Inférence branche visuelle ──────────────────────
            pil_frames = [Image.fromarray(f) for f in frames_rgb]
            images     = [self.transform(img) for img in pil_frames]
            seq        = torch.stack(images).unsqueeze(0).to(self.device)
            print(f"[VIDEO] Raw video={p.name} | frames utilisées={len(images)}")

            logits = self.model(seq, behav_tensor) if self.use_behav else self.model(seq)
            proba  = torch.softmax(logits, dim=1)
            score_v3 = float(proba[0, 1].item())

            # ── Étape 8 : Behavior-pose V6 depuis frames déjà en mémoire ──
            bp_features = self._build_behavior_pose_features(
                ear_values, motion_values,
                yaw_values, pitch_values, roll_values,
                detected, total
            )
            print("[VIDEO] Behavior-pose V6 raw features:")
            print({k: round(float(v), 4) for k, v in bp_features.items()})

            reliable, bp_status = self._is_behavior_pose_reliable(bp_features)

            if reliable and self.behavior_pose_clf is not None:
                try:
                    x = np.array(
                        [[float(bp_features.get(c, 0.0)) for c in self.BEHAVIOR_POSE_V6_COLS]],
                        dtype=np.float32,
                    )
                    score_behavior_pose = float(
                        self.behavior_pose_clf.predict_proba(x)[0, 1]
                    )
                    print(f"[VIDEO] score_behavior_pose V6: {score_behavior_pose:.4f}")
                except Exception as e:
                    print(f"[VIDEO][WARN] score_behavior_pose échoué: {e}")
                    score_behavior_pose = None
                    bp_status = f"behavior_pose_failed: {e}"
            else:
                if not reliable:
                    print(f"[VIDEO][WARN] Behavior-pose non fiable: {bp_status}")
                score_behavior_pose = None

        else:
            # Mode CSV video_id
            score_v3            = float(self._predict_from_csv_video_id(str(video_path_or_id)))
            video_quality_score = 0.70
            score_behavior_pose = None
            bp_features         = None
            bp_status           = "csv_or_video_id_mode"

        # ── Fusion score-level ─────────────────────────────────────────────
        if score_behavior_pose is not None:
            score_final   = 0.85 * score_v3 + 0.15 * score_behavior_pose
            fusion_status = "behavior_pose_loaded"
        else:
            score_final   = score_v3
            fusion_status = "fallback_score_v3_only"

        score_final = float(min(max(score_final, 0.0), 1.0))

        return {
            "score_v3_multimodal"   : float(score_v3),
            "score_behavior_pose"   : float(score_behavior_pose) if score_behavior_pose is not None else None,
            "score_final_v6"        : score_final,
            "video_quality_score"   : float(video_quality_score),
            "behavior_pose_status"  : bp_status,
            "behavior_pose_features": bp_features if isinstance(bp_features, dict) else None,
            "fusion": {
                "type"            : "score_level_fusion_v6",
                "formula"         : "score_final_v6 = 0.85 * score_v3 + 0.15 * score_behavior_pose",
                "w_v3_multimodal" : 0.85,
                "w_behavior_pose" : 0.15,
                "status"          : fusion_status,
            },
        }

    # ==========================================================
    # PUBLIC PREDICT (rétrocompatibilité)
    # ==========================================================

    @torch.no_grad()
    def predict(self, video_path_or_id: str) -> float:
        """
        Prédiction simple — retourne seulement le score V3.
        Pour la prédiction complète V6, utiliser predict_with_details().
        """
        print(f"[INFO] Analyse vidéo: {video_path_or_id}")
        p = Path(str(video_path_or_id))

        if p.exists() and p.suffix.lower() in self.VIDEO_EXTS:
            # Lecture vidéo et inférence directe
            frames_rgb_all, frames_gray_all = self._read_video_rgb_gray_arrays(p)
            frames_rgb = self._sample_raw_rgb_arrays(frames_rgb_all)

            # Features comportementales
            frames_gray = self._sample_raw_gray_arrays(frames_gray_all)
            motion_vals = self._compute_motion_from_gray_frames(frames_gray)
            (ear_vals, _, _, _, detected, total) = self._extract_ear_and_pose_from_frames(frames_rgb)

            raw_behav = self._build_behavior_features(ear_vals, motion_vals, detected, total)
            rppg      = self._extract_rppg_from_raw_video(p)
            raw_behav.update(rppg)

            norm_values = self._normalize_raw_behavior(raw_behav)
            while len(norm_values) < self.behav_dim: norm_values.append(0.0)
            behav_tensor = torch.tensor(
                [norm_values[:self.behav_dim]], dtype=torch.float32
            ).to(self.device)

            pil_frames = [Image.fromarray(f) for f in frames_rgb]
            images     = [self.transform(img) for img in pil_frames]
            seq        = torch.stack(images).unsqueeze(0).to(self.device)
            print(f"[VIDEO] Raw video={p.name} | frames utilisées={len(images)}")

            logits = self.model(seq, behav_tensor) if self.use_behav else self.model(seq)
            proba  = torch.softmax(logits, dim=1)
            return float(proba[0, 1].item())

        # Mode CSV
        video_id = p.stem if p.suffix.lower() in self.VIDEO_EXTS else str(video_path_or_id)
        return self._predict_from_csv_video_id(video_id.strip())