import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from celeba_dataset import CelebASpoofDataset
from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD


def load_model(checkpoint_path, device):
    ckpt_path = Path(checkpoint_path)

    config = {}
    cfg_path = ckpt_path.parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    model = CNN_LSTM_PAD(
        hidden=config.get("hidden", 256),
        num_layers=config.get("num_layers", 1),
        bidir=config.get("bidir", False),
        lstm_dropout=config.get("lstm_dropout", 0.0),
        head_dropout=0.0,
        pretrained_backbone=False,
        temporal_pool=config.get("temporal_pool", "median"),
        use_behav=config.get("use_behav", True),
        behav_dim=config.get("behav_dim", 9),
        behav_hidden=config.get("behav_hidden", 16),
    ).to(device)

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "model_state" in state:
            state = state["model_state"]

    if all(k.startswith("module.") for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def collect_scores(model, csv_path, base_dir, seq_len, batch_size, device):
    ds = CelebASpoofDataset(
        csv_path=csv_path,
        base_dir=base_dir,
        seq_len=seq_len,
        augment=False,
    )

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    softmax = nn.Softmax(dim=1)
    y_true = []
    y_score = []

    for seq, behav, labels in dl:
        seq = seq.to(device)
        behav = behav.to(device)

        logits = model(seq, behav)
        probs = softmax(logits)[:, 1]

        y_true.extend(labels.numpy().tolist())
        y_score.extend(probs.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_score)


def compute_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0

    return {
        "threshold": float(threshold),
        "ACC": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_true, y_score)),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER": float(acer),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    model = load_model(args.checkpoint, device)

    y_true, y_score = collect_scores(
        model=model,
        csv_path=args.csv,
        base_dir=args.base_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    metrics = compute_metrics(y_true, y_score, args.threshold)

    print("\n==============================")
    print(f"Evaluation threshold = {args.threshold:.2f}")
    print("==============================")
    print(f"ACC   : {metrics['ACC']*100:.2f}%")
    print(f"F1    : {metrics['F1']*100:.2f}%")
    print(f"AUC   : {metrics['AUC']*100:.2f}%")
    print(f"APCER : {metrics['APCER']*100:.2f}%")
    print(f"BPCER : {metrics['BPCER']*100:.2f}%")
    print(f"ACER  : {metrics['ACER']*100:.2f}%")
    print(f"TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']} TP={metrics['TP']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("[OK] Saved:", args.out)


if __name__ == "__main__":
    main()
