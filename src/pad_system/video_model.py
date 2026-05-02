import os
from pathlib import Path

import cv2
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD


class VideoPADModel:
    """
    Branche vidéo PAD.

    Supporte deux modes :
    1) vidéo brute existante (.avi/.mp4/.mov/.mkv)
    2) CASIA processed : reconstruction depuis CSV + frames .jpg

    Exemple d'entrées possibles :
    - "13_1"
    - "13_1.avi"
    - r"E:\\PFE_AntiSpoofing_v2\\data\\raw\\casia\\13_1.avi"
    - r"E:\\PFE_AntiSpoofing_v2\\data\\demo\\turn_left.mp4"
    """

    def __init__(
        self,
        checkpoint_path: str = r"E:\PFE_AntiSpoofing_v2\experiments\step2_sampling\deep_behav_no_pts_consecutive\deep_behav_no_pts_consecutive_seed42\best_model.pth",
        test_csv: str = r"E:\PFE_AntiSpoofing_v2\data\processed\CASIA\splits_subject\test.csv",
        behav_test_csv: str = r"E:\PFE_AntiSpoofing_v2\data\processed\CASIA\behav\test_behav.csv",
        img_size: int = 224,
        seq_len: int = 16,
        sample_mode: str = "consecutive",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.test_csv = Path(test_csv)
        self.behav_test_csv = Path(behav_test_csv)

        self.img_size = img_size
        self.seq_len = seq_len
        self.sample_mode = sample_mode

        self.project_root = Path(__file__).resolve().parents[2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[VIDEO] Device: {self.device}")
        print(f"[VIDEO] Loading checkpoint: {self.checkpoint_path}")

        self.model, self.use_behav = self._load_model()
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

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint vidéo introuvable: {self.checkpoint_path}"
            )

        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)

        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
            cfg = ckpt.get("config", {})

        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            cfg = ckpt.get("config", {})

        else:
            state_dict = ckpt
            cfg = {}

        use_behav = bool(cfg.get("use_behav", True))

        model = CNN_LSTM_PAD(
            hidden=cfg.get("hidden", 256),
            num_layers=cfg.get("num_layers", 1),
            bidir=cfg.get("bidir", False),
            lstm_dropout=cfg.get("lstm_dropout", 0.0),
            head_dropout=0.0,
            pretrained_backbone=False,
            temporal_pool=cfg.get("temporal_pool", "median"),
            use_behav=use_behav,
            behav_dim=cfg.get("behav_dim", 9),
            behav_hidden=cfg.get("behav_hidden", 16),
        )

        if isinstance(state_dict, dict) and all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)

        print("[VIDEO] Checkpoint chargé avec succès.")
        print(f"[VIDEO] use_behav = {use_behav}")

        return model, use_behav

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
        print(f"[VIDEO] Colonnes CSV: {df.columns.tolist()}")

        return df

    def _load_behav_csv(self):
        if not self.behav_test_csv.exists():
            print(f"[VIDEO][WARN] Behavior CSV introuvable: {self.behav_test_csv}")
            return None

        df = pd.read_csv(self.behav_test_csv)

        print(f"[VIDEO] behav_test_csv chargé: {len(df)} lignes")
        print(f"[VIDEO] Colonnes behavior: {df.columns.tolist()}")

        return df

    def _resolve_frame_path(self, row):
        """
        Résout le chemin d'une frame depuis le CSV.

        Colonnes supportées :
        - path
        - frame_path
        - image_path
        - img_path
        """

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
        """
        Sampling vidéo.
        Pour respecter ton meilleur résultat Step 2, on privilégie consecutive.
        """

        df_video = df_video.copy()

        if "frame_idx" in df_video.columns:
            df_video = df_video.sort_values("frame_idx")

        n = len(df_video)

        if n >= self.seq_len:
            if self.sample_mode == "consecutive":
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

    def _get_behavior_vector(self, video_id):
        """
        Charge les features comportementales déjà pré-calculées.

        Si aucune ligne n'est trouvée :
        - retourne un vecteur zéro [1, 9]
        """

        if self.behav_df is None:
            return torch.zeros((1, 9), dtype=torch.float32).to(self.device)

        if "video_id" not in self.behav_df.columns:
            return torch.zeros((1, 9), dtype=torch.float32).to(self.device)

        row = self.behav_df[self.behav_df["video_id"].astype(str) == str(video_id)]

        if row.empty:
            print(f"[VIDEO][WARN] Aucun behavior trouvé pour video_id={video_id}. Utilisation zéro.")
            return torch.zeros((1, 9), dtype=torch.float32).to(self.device)

        row = row.iloc[0]

        candidate_cols = [
            "ear_mean",
            "ear_std",
            "blink_count",
            "motion_mean",
            "motion_std",
            "motion_max",
            "face_conf_mean",
            "face_conf_std",
            "skipped_rate",
        ]

        available = [c for c in candidate_cols if c in self.behav_df.columns]

        if len(available) < 9:
            ignore = {"video_id", "label", "subject_id", "split"}

            numeric_cols = [
                c for c in self.behav_df.columns
                if c not in ignore and pd.api.types.is_numeric_dtype(self.behav_df[c])
            ]

            available = numeric_cols[:9]

        values = []

        for c in available[:9]:
            values.append(float(row[c]))

        while len(values) < 9:
            values.append(0.0)

        behav = torch.tensor([values], dtype=torch.float32).to(self.device)

        return behav

    def _predict_from_casia_video_id(self, video_id):
        """
        Inférence à partir d'un video_id CASIA présent dans test.csv.
        Exemple : 13_1, 13_3, 10_2...
        """

        if self.frames_df is None:
            raise RuntimeError(
                "frames_df non chargé. Impossible d'utiliser le mode CASIA CSV."
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

        seq = torch.stack(images, dim=0)        # [T, 3, 224, 224]
        seq = seq.unsqueeze(0).to(self.device)  # [1, T, 3, 224, 224]

        behav = self._get_behavior_vector(video_id)

        print(f"[VIDEO] CASIA video_id={video_id} | frames utilisées={len(images)}")

        if self.use_behav:
            logits = self.model(seq, behav)
        else:
            logits = self.model(seq)

        proba = torch.softmax(logits, dim=1)
        spoof_score = proba[0, 1].item()

        return float(spoof_score)

    def _read_raw_video_frames(self, video_path):
        """
        Lit une vraie vidéo brute existante.
        """

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

    @torch.no_grad()
    def predict(self, video_path_or_id: str) -> float:
        """
        Entrée possible :
        - chemin vidéo réel existant :
          E:/.../video.mp4

        - pseudo-chemin absent :
          E:/.../13_1.avi
          → converti automatiquement en video_id = 13_1

        - video_id direct :
          13_1
        """

        print(f"[INFO] Analyse vidéo: {video_path_or_id}")

        p = Path(str(video_path_or_id))

        video_exts = [".avi", ".mp4", ".mov", ".mkv"]

        # ==========================================================
        # Cas 1 : vraie vidéo brute existante
        # ==========================================================
        if p.exists() and p.suffix.lower() in video_exts:
            frames = self._read_raw_video_frames(p)

            if len(frames) >= self.seq_len:
                start = max(0, (len(frames) - self.seq_len) // 2)
                frames = frames[start:start + self.seq_len]

            else:
                while len(frames) < self.seq_len:
                    frames.append(frames[-1])

            images = [self.transform(img) for img in frames]

            seq = torch.stack(images, dim=0).unsqueeze(0).to(self.device)
            behav = torch.zeros((1, 9), dtype=torch.float32).to(self.device)

            if self.use_behav:
                logits = self.model(seq, behav)
            else:
                logits = self.model(seq)

            proba = torch.softmax(logits, dim=1)
            spoof_score = proba[0, 1].item()

            return float(spoof_score)

        # ==========================================================
        # Cas 2 : fichier vidéo absent → convertir en video_id CASIA
        # Exemple :
        # E:/.../13_1.avi → 13_1
        # 13_1.avi       → 13_1
        # 13_1           → 13_1
        # ==========================================================
        if p.suffix.lower() in video_exts:
            video_id = p.stem
        else:
            video_id = str(video_path_or_id)

        video_id = (
            video_id.strip()
            .replace(".avi", "")
            .replace(".mp4", "")
            .replace(".mov", "")
            .replace(".mkv", "")
        )

        return self._predict_from_casia_video_id(video_id)