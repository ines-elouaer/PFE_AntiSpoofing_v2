import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torchvision import models


class VideoBackboneLSTM(nn.Module):
    def __init__(
        self,
        num_classes=2,
        hidden_dim=256,
        lstm_layers=1,
        dropout=0.3,
        pretrained_backbone=False,
        temporal_pooling="median",
    ):
        super().__init__()

        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.mobilenet_v3_large(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feat_dim = backbone.classifier[0].in_features

        self.lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.temporal_pooling = temporal_pooling

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, behavior_features=None):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        x = x.view(b * t, c, h, w)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = x.view(b, t, -1)
        out, _ = self.lstm(x)

        if self.temporal_pooling == "median":
            z = out.median(dim=1).values
        else:
            z = out.mean(dim=1)

        logits = self.classifier(z)
        return logits


def load_image_as_sequence(image_path, img_size=224, seq_len=16):
    image_path = Path(image_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Image illisible : {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))

    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))

    seq = np.stack([img for _ in range(seq_len)], axis=0)
    return torch.tensor(seq, dtype=torch.float32)


def compute_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    real_real = int(cm[0, 0])
    real_spoof = int(cm[0, 1])
    spoof_real = int(cm[1, 0])
    spoof_spoof = int(cm[1, 1])

    apcer = spoof_real / max(spoof_real + spoof_spoof, 1)
    bpcer = real_spoof / max(real_real + real_spoof, 1)
    acer = (apcer + bpcer) / 2

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "confusion_matrix": {
            "REAL->REAL": real_real,
            "REAL->SPOOF": real_spoof,
            "SPOOF->REAL": spoof_real,
            "SPOOF->SPOOF": spoof_spoof,
        },
    }


def clean_state_dict(state_dict):
    cleaned = {}

    for k, v in state_dict.items():
        new_k = k

        if new_k.startswith("module."):
            new_k = new_k[len("module."):]

        if new_k.startswith("model."):
            new_k = new_k[len("model."):]

        cleaned[new_k] = v

    return cleaned


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    print("[INFO] Type checkpoint :", type(ckpt))

    if isinstance(ckpt, dict):
        print("[INFO] Clés checkpoint :", list(ckpt.keys()))

        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
            print("[INFO] Utilisation de ckpt['model_state']")
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            print("[INFO] Utilisation de ckpt['model_state_dict']")
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            print("[INFO] Utilisation de ckpt['state_dict']")
        else:
            state_dict = ckpt
            print("[WARN] Aucune clé standard trouvée, utilisation du checkpoint complet")
    else:
        state_dict = ckpt

    state_dict = clean_state_dict(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("[INFO] Checkpoint chargé :", checkpoint_path)
    print("[INFO] Missing keys :", len(missing))
    print("[INFO] Unexpected keys :", len(unexpected))

    if len(missing) > 0:
        print("[WARN] Quelques clés manquantes :", missing[:20])

    if len(unexpected) > 0:
        print("[WARN] Quelques clés inattendues :", unexpected[:20])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seq_len", type=int, default=16)

    args = parser.parse_args()

    csv_path = Path(args.csv)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[INFO] Device :", device)

    df = pd.read_csv(csv_path)

    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError("Le CSV doit contenir image_path et label.")

    model = VideoBackboneLSTM(
        num_classes=2,
        hidden_dim=256,
        lstm_layers=1,
        dropout=0.3,
        pretrained_backbone=False,
        temporal_pooling="median",
    )

    load_checkpoint(model, checkpoint_path, device)

    model.to(device)
    model.eval()

    y_true = []
    y_score = []
    image_paths = []
    errors = []

    with torch.no_grad():
        for idx, row in df.iterrows():
            image_path = row["image_path"]
            label = int(row["label"])

            try:
                seq = load_image_as_sequence(
                    image_path=image_path,
                    img_size=args.img_size,
                    seq_len=args.seq_len,
                )

                seq = seq.unsqueeze(0).to(device)

                logits = model(seq)
                prob = torch.softmax(logits, dim=1)
                spoof_score = float(prob[:, 1].detach().cpu().numpy()[0])

                y_true.append(label)
                y_score.append(spoof_score)
                image_paths.append(image_path)

            except Exception as e:
                errors.append({
                    "index": int(idx),
                    "image_path": str(image_path),
                    "error": str(e),
                })

            if (idx + 1) % 100 == 0:
                print(f"[INFO] Traité : {idx + 1}/{len(df)}")

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= args.threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_score)

    pred_df = pd.DataFrame({
        "image_path": image_paths,
        "label": y_true,
        "spoof_score": y_score,
        "prediction": y_pred,
        "label_name": ["REAL" if x == 0 else "SPOOF" for x in y_true],
        "prediction_name": ["REAL" if x == 0 else "SPOOF" for x in y_pred],
    })

    pred_df.to_csv(output_dir / "celeba_v6_predictions.csv", index=False)

    with open(output_dir / "celeba_v6_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "celeba_v6_errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)

    print("\n========== Résultats CelebA FOMA V6 ==========")
    print(json.dumps(metrics, indent=2))
    print("[INFO] Images évaluées :", len(y_true))
    print("[INFO] Erreurs images :", len(errors))
    print("[OK] Résultats sauvegardés dans :", output_dir)


if __name__ == "__main__":
    main()