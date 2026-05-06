from pathlib import Path
import math
import warnings

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
    Branche vidéo PAD.

    Supporte deux modes :
    1) vidéo brute existante (.avi/.mp4/.mov/.mkv)
       Exemple : data/demo/jury_challenge.mp4

    2) video_id présent dans un CSV de frames
       Exemple : 13_1, 13_3, axon_xxx

    Modèle utilisé par défaut :
    - modèle vidéo fine-tuné CASIA + AxonLabs.

    Flux final :
    vidéo challenge
    -> extraction T=16 frames consécutives au centre
    -> extraction features comportementales
    -> normalisation avec scaler CASIA
    -> CNN+LSTM+Behavior
    -> score spoof.
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
    ]

    FULL_BEHAV_COLS = [
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

    VIDEO_EXTS = [".avi", ".mp4", ".mov", ".mkv"]

    def __init__(
        self,
        checkpoint_path: str = r"E:\PFE_AntiSpoofing_v2\experiments\mixed_casia_axon\seed42\best_model_mixed_casia_axon.pth",
        test_csv: str = r"E:\PFE_AntiSpoofing_v2\data\mixed_casia_axon\mixed_val_frames.csv",
        behav_test_csv: str = r"E:\PFE_AntiSpoofing_v2\data\mixed_casia_axon\mixed_val_behav.csv",
        scaler_path: str = r"E:\PFE_AntiSpoofing_v2\data\processed\casia\behav\behav_scaler.pkl",
        img_size: int = 224,
        seq_len: int = 16,
        sample_mode: str = "center_consecutive",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.test_csv = Path(test_csv)
        self.behav_test_csv = Path(behav_test_csv)
        self.scaler_path = Path(scaler_path) if scaler_path else None

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
        behav_dim = int(cfg.get("behav_dim", 9))
        behav_hidden = int(cfg.get("behav_hidden", 16))
        temporal_pool = cfg.get("temporal_pool", "median")

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
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print("\n========== VIDEO CHECKPOINT LOAD ==========")
        print(f"Checkpoint      : {self.checkpoint_path}")
        print(f"use_behav       : {use_behav}")
        print(f"behav_dim       : {behav_dim}")
        print(f"behav_hidden    : {behav_hidden}")
        print(f"temporal_pool   : {temporal_pool}")
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
    # CSV / SCALER LOADING
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
    # BEHAVIOR FEATURES — CSV
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

    # ==========================================================
    # BEHAVIOR FEATURES — RAW VIDEO
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

    def _extract_behavior_raw_unscaled(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"[VIDEO][WARN] Impossible d'ouvrir la vidéo pour behavior: {video_path}")
            return {col: 0.0 for col in self.MODEL_BEHAV_COLS}

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
            return {col: 0.0 for col in self.MODEL_BEHAV_COLS}

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

            base_options = python.BaseOptions(
                model_asset_path=str(face_model_path)
            )

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
            print(f"[VIDEO][WARN] MediaPipe Tasks behavior extraction échouée: {e}")
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

    def _normalize_raw_behavior(self, raw_features: dict):
        if self.behav_scaler is None:
            print("[VIDEO][WARN] Aucun scaler behavior. Features webcam non normalisées.")
            values = [float(raw_features.get(c, 0.0)) for c in self.MODEL_BEHAV_COLS]
            return values[:self.behav_dim]

        scaler = self.behav_scaler

        try:
            if hasattr(scaler, "feature_names_in_"):
                scaler_cols = list(scaler.feature_names_in_)

                row = {}
                for col in scaler_cols:
                    row[col] = float(raw_features.get(col, 0.0))

                df = pd.DataFrame([row], columns=scaler_cols)
                arr_norm = scaler.transform(df)[0]

                norm_map = {
                    col: float(arr_norm[i])
                    for i, col in enumerate(scaler_cols)
                }

                values = [float(norm_map.get(c, 0.0)) for c in self.MODEL_BEHAV_COLS]
                return values[:self.behav_dim]

            n_features = getattr(scaler, "n_features_in_", None)

            if n_features == 9:
                arr = np.array(
                    [[float(raw_features.get(c, 0.0)) for c in self.MODEL_BEHAV_COLS]],
                    dtype=np.float32,
                )
                arr_norm = scaler.transform(arr)[0]
                return [float(x) for x in arr_norm[:self.behav_dim]]

            if n_features == 21:
                arr = np.array(
                    [[float(raw_features.get(c, 0.0)) for c in self.FULL_BEHAV_COLS]],
                    dtype=np.float32,
                )
                arr_norm = scaler.transform(arr)[0]

                norm_map = {
                    col: float(arr_norm[i])
                    for i, col in enumerate(self.FULL_BEHAV_COLS)
                }

                values = [float(norm_map.get(c, 0.0)) for c in self.MODEL_BEHAV_COLS]
                return values[:self.behav_dim]

            print(f"[VIDEO][WARN] Scaler n_features_in_ inattendu: {n_features}")
            values = [float(raw_features.get(c, 0.0)) for c in self.MODEL_BEHAV_COLS]
            return values[:self.behav_dim]

        except Exception as e:
            print(f"[VIDEO][WARN] Normalisation behavior webcam échouée: {e}")
            values = [float(raw_features.get(c, 0.0)) for c in self.MODEL_BEHAV_COLS]
            return values[:self.behav_dim]

    def _extract_behavior_from_raw_video(self, video_path: Path):
        raw_features = self._extract_behavior_raw_unscaled(video_path)
        norm_values = self._normalize_raw_behavior(raw_features)

        while len(norm_values) < self.behav_dim:
            norm_values.append(0.0)

        norm_values = norm_values[:self.behav_dim]

        print("[VIDEO] Behavior raw video extrait.")
        print("[VIDEO] Raw behavior:", {k: round(float(v), 4) for k, v in raw_features.items()})
        print("[VIDEO] Norm behavior:", [round(float(v), 4) for v in norm_values])

        return torch.tensor([norm_values], dtype=torch.float32).to(self.device)

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

    def _sample_raw_rgb_arrays(self, frames_rgb):
        if len(frames_rgb) >= self.seq_len:
            start = max(0, (len(frames_rgb) - self.seq_len) // 2)
            return frames_rgb[start:start + self.seq_len]

        while len(frames_rgb) < self.seq_len:
            frames_rgb.append(frames_rgb[-1])

        return frames_rgb

    def _sample_raw_gray_arrays(self, frames_gray):
        if len(frames_gray) >= self.seq_len:
            start = max(0, (len(frames_gray) - self.seq_len) // 2)
            return frames_gray[start:start + self.seq_len]

        while len(frames_gray) < self.seq_len:
            frames_gray.append(frames_gray[-1])

        return frames_gray

    def _predict_from_raw_video(self, video_path: Path):
        frames = self._read_raw_video_frames(video_path)
        frames = self._sample_raw_frames(frames)

        images = [self.transform(img) for img in frames]

        seq = torch.stack(images, dim=0)
        seq = seq.unsqueeze(0).to(self.device)

        behav = self._extract_behavior_from_raw_video(video_path)

        print(f"[VIDEO] Raw video={video_path.name} | frames utilisées={len(images)}")
        print("[VIDEO] Behavior raw video: features extraites + normalisées.")

        if self.use_behav:
            logits = self.model(seq, behav)
        else:
            logits = self.model(seq)

        proba = torch.softmax(logits, dim=1)
        return float(proba[0, 1].item())

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