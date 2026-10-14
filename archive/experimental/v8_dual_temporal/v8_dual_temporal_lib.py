from pathlib import Path
import sys
import json
import pickle
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import torchvision
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BEHAV_SEQ_COLS = [
    "ear",
    "eye_closed",
    "motion_mean",
    "motion_max",
    "yaw",
    "pitch",
    "roll",
    "face_area",
    "face_center_motion",
    "frame_valid",
]


def resolve_path(p):
    p = Path(str(p))
    if p.exists():
        return p
    alt = PROJECT_ROOT / p
    if alt.exists():
        return alt
    return p


def sample_center_consecutive(g, T):
    g = g.sort_values("frame_idx").reset_index(drop=True)
    n = len(g)

    if n >= T:
        start = max(0, (n - T) // 2)
        return g.iloc[start:start + T].copy()

    last = g.iloc[[-1]].copy()
    pads = [last.copy() for _ in range(T - n)]
    return pd.concat([g] + pads, ignore_index=True)


def preprocess_image(path, img_size):
    img = cv2.imread(str(resolve_path(path)))

    if img is None:
        arr = np.zeros((img_size, img_size, 3), dtype=np.float32)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        arr = img.astype(np.float32) / 255.0

    arr = (arr - IMG_MEAN) / IMG_STD
    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    return arr


class DualTemporalDataset(Dataset):
    def __init__(
        self,
        frames_csv,
        behavior_seq_csv,
        T=16,
        img_size=224,
        scaler=None,
        fit_scaler=False,
    ):
        self.frames = pd.read_csv(frames_csv)
        self.behav = pd.read_csv(behavior_seq_csv)

        self.frames["video_id"] = self.frames["video_id"].astype(str)
        self.behav["video_id"] = self.behav["video_id"].astype(str)

        self.T = T
        self.img_size = img_size

        missing = [c for c in BEHAV_SEQ_COLS if c not in self.behav.columns]
        if missing:
            raise ValueError(f"Missing behavior sequence columns: {missing}")

        self.video_ids = sorted(self.frames["video_id"].unique().tolist())

        # Index behavior by video_id for faster access
        self.behav_groups = {
            vid: g.sort_values("frame_idx").reset_index(drop=True)
            for vid, g in self.behav.groupby("video_id")
        }

        if fit_scaler:
            values = []
            for vid in self.video_ids:
                bg = self.behav_groups.get(vid, None)
                if bg is None:
                    continue
                seq = sample_center_consecutive(bg, self.T)
                values.append(seq[BEHAV_SEQ_COLS].fillna(0.0).values.astype(np.float32))

            if len(values) == 0:
                raise RuntimeError("No behavior values found to fit scaler.")

            values = np.concatenate(values, axis=0)
            self.scaler = StandardScaler()
            self.scaler.fit(values)
        else:
            if scaler is None:
                raise ValueError("Scaler required when fit_scaler=False")
            self.scaler = scaler

    def __len__(self):
        return len(self.video_ids)

    def get_label(self, idx):
        vid = self.video_ids[idx]
        fg = self.frames[self.frames["video_id"] == vid]
        return int(fg["label"].iloc[0])

    def __getitem__(self, idx):
        vid = self.video_ids[idx]

        fg = self.frames[self.frames["video_id"] == vid].sort_values("frame_idx").reset_index(drop=True)
        fs = sample_center_consecutive(fg, self.T)

        label = int(fs["label"].iloc[0])

        imgs = []
        for _, r in fs.iterrows():
            imgs.append(preprocess_image(r["path"], self.img_size))

        x = np.stack(imgs, axis=0).astype(np.float32)

        bg = self.behav_groups.get(vid, None)

        if bg is None:
            bseq = np.zeros((self.T, len(BEHAV_SEQ_COLS)), dtype=np.float32)
        else:
            bs = sample_center_consecutive(bg, self.T)
            bseq = bs[BEHAV_SEQ_COLS].fillna(0.0).values.astype(np.float32)
            bseq = self.scaler.transform(bseq).astype(np.float32)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(bseq, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            vid,
        )


class MobileNetV3FrameEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = None
        if pretrained:
            try:
                weights = MobileNet_V3_Large_Weights.DEFAULT
            except Exception:
                weights = None

        try:
            base = mobilenet_v3_large(weights=weights)
        except Exception:
            base = mobilenet_v3_large(weights=None)

        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = base.classifier[0].in_features

    def forward(self, x):
        z = self.features(x)
        z = self.avgpool(z)
        z = torch.flatten(z, 1)
        return z


class GatedFusion(nn.Module):
    def __init__(self, video_dim, behav_dim, hidden=128):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(video_dim + behav_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, behav_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_video, z_behavior):
        z = torch.cat([z_video, z_behavior], dim=1)
        gate = self.gate(z)
        z_behavior_weighted = gate * z_behavior
        return torch.cat([z_video, z_behavior_weighted], dim=1), gate


class DualTemporalPAD(nn.Module):
    def __init__(
        self,
        behavior_dim=10,
        video_lstm_hidden=256,
        behavior_lstm_hidden=64,
        num_classes=2,
        dropout=0.5,
        pretrained_backbone=True,
    ):
        super().__init__()

        self.frame_encoder = MobileNetV3FrameEncoder(pretrained=pretrained_backbone)
        frame_dim = self.frame_encoder.out_dim

        self.video_lstm = nn.LSTM(
            input_size=frame_dim,
            hidden_size=video_lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

        self.behavior_lstm = nn.LSTM(
            input_size=behavior_dim,
            hidden_size=behavior_lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

        self.fusion = GatedFusion(
            video_dim=video_lstm_hidden,
            behav_dim=behavior_lstm_hidden,
            hidden=128,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(video_lstm_hidden + behavior_lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, behavior_seq):
        # x: [B, T, C, H, W]
        # behavior_seq: [B, T, D]
        B, T, C, H, W = x.shape

        x_flat = x.view(B * T, C, H, W)
        frame_features = self.frame_encoder(x_flat)
        frame_features = frame_features.view(B, T, -1)

        video_out, (video_h, _) = self.video_lstm(frame_features)
        z_video = video_h[-1]

        behav_out, (behav_h, _) = self.behavior_lstm(behavior_seq)
        z_behavior = behav_h[-1]

        z_fused, gate = self.fusion(z_video, z_behavior)
        logits = self.classifier(z_fused)

        return logits


def compute_metrics(y_true, y_pred, scores):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    scores = np.asarray(scores).astype(float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    real_total = tn + fp
    spoof_total = tp + fn

    bpcer = fp / real_total if real_total > 0 else 0.0
    apcer = fn / spoof_total if spoof_total > 0 else 0.0
    acer = (apcer + bpcer) / 2.0

    out = {
        "total": int(len(y_true)),
        "real_count": int(real_total),
        "spoof_count": int(spoof_total),
        "tn_real": int(tn),
        "fp_real_as_spoof": int(fp),
        "fn_spoof_as_real": int(fn),
        "tp_spoof": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spoof": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_spoof": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_spoof": float(f1_score(y_true, y_pred, zero_division=0)),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER": float(acer),
    }

    try:
        out["AUC"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["AUC"] = None

    return out
