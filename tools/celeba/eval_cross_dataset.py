"""
eval_cross_dataset.py
=====================

Évaluation CelebA-Spoof / Cross-dataset.

Rôles :
1) CASIA -> CelebA avant fine-tuning
2) CelebA fine-tuned -> CelebA
3) Mixed FT -> CelebA

Protocoles threshold :
- fixed05 : threshold = 0.50
- valopt  : cherche meilleur threshold sur CelebA VAL puis applique sur CelebA TEST

Entrée modèle :
- seq   : [B, T, 3, 224, 224]
- behav : [B, 9] = zeros pour CelebA
- label : 0=real, 1=spoof
"""

import os
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.celeba_dataset import CelebASpoofDataset
from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD


def compute_pad_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = 0.0

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER": float(acer),
        "Accuracy": float(acc),
        "F1": float(f1),
        "AUC": float(auc),
    }


def find_config_near_checkpoint(checkpoint_path: Path):
    candidates = [
        checkpoint_path.parent / "config.json",
        checkpoint_path.parent / "train_config.json",
        checkpoint_path.parent / "args.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_model(checkpoint_path, device):
    ckpt_path = Path(checkpoint_path)

    config = {}
    cfg_path = find_config_near_checkpoint(ckpt_path)

    if cfg_path is not None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[OK] Config trouvée: {cfg_path}")
    else:
        print("[WARN] Aucun fichier config trouvé près du checkpoint. Paramètres par défaut utilisés.")

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
    )

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "model_state" in state:
            state = state["model_state"]

    if all(k.startswith("module.") for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    print(f"[OK] Checkpoint chargé: {ckpt_path}")
    return model


@torch.no_grad()
def run_inference(model, dataloader, device, threshold=0.50):
    softmax = nn.Softmax(dim=1)

    all_labels = []
    all_preds = []
    all_scores = []

    for seq, behav, labels in dataloader:
        seq = seq.to(device)
        behav = behav.to(device)
        labels = labels.to(device)

        logits = model(seq, behav)
        probs = softmax(logits)[:, 1]
        preds = (probs >= threshold).long()

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_scores.extend(probs.cpu().numpy().tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores),
    )


def make_loader(csv_path, base_dir, seq_len, batch_size):
    dataset = CelebASpoofDataset(
        csv_path=csv_path,
        base_dir=base_dir,
        seq_len=seq_len,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def find_best_threshold_on_val(model, val_csv, base_dir, seq_len, batch_size, device):
    val_loader = make_loader(
        csv_path=val_csv,
        base_dir=base_dir,
        seq_len=seq_len,
        batch_size=batch_size,
    )

    y_val, _, s_val = run_inference(
        model=model,
        dataloader=val_loader,
        device=device,
        threshold=0.50,
    )

    best = None

    for th in np.arange(0.01, 1.00, 0.01):
        pred_val = (s_val >= th).astype(int)
        metrics = compute_pad_metrics(y_val, pred_val, s_val)
        metrics["threshold"] = float(round(th, 4))

        if best is None:
            best = metrics
        else:
            better = (
                metrics["ACER"] < best["ACER"]
                or (
                    metrics["ACER"] == best["ACER"]
                    and metrics["F1"] > best["F1"]
                )
                or (
                    metrics["ACER"] == best["ACER"]
                    and metrics["F1"] == best["F1"]
                    and abs(metrics["threshold"] - 0.50) < abs(best["threshold"] - 0.50)
                )
            )

            if better:
                best = metrics

    return best


def save_confusion_csv(path, tn, fp, fn, tp):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "pred_real", "pred_attack"])
        w.writerow(["real", tn, fp])
        w.writerow(["attack", fn, tp])


def save_score_distribution_csv(path, y_true, y_score):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "label", "score"])
        for i, (yt, ys) in enumerate(zip(y_true, y_score)):
            w.writerow([i, int(yt), float(ys)])


def save_threshold_sweep_csv(path, y_true, y_score):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "Accuracy", "F1", "AUC", "APCER", "BPCER", "ACER", "TN", "FP", "FN", "TP"])

        for th in np.arange(0.01, 1.00, 0.01):
            y_pred = (y_score >= th).astype(int)
            m = compute_pad_metrics(y_true, y_pred, y_score)
            w.writerow([
                float(round(th, 4)),
                m["Accuracy"],
                m["F1"],
                m["AUC"],
                m["APCER"],
                m["BPCER"],
                m["ACER"],
                m["TN"],
                m["FP"],
                m["FN"],
                m["TP"],
            ])


def pretty_print_metrics(title, metrics, threshold_protocol, threshold_source):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Threshold protocol : {threshold_protocol}")
    print(f"Threshold source   : {threshold_source}")
    print(f"Threshold used     : {metrics['threshold']:.4f}")
    print("-" * 60)
    print(f"Accuracy : {metrics['Accuracy'] * 100:.2f}%")
    print(f"F1       : {metrics['F1'] * 100:.2f}%")
    print(f"AUC      : {metrics['AUC'] * 100:.2f}%")
    print(f"APCER    : {metrics['APCER'] * 100:.2f}%")
    print(f"BPCER    : {metrics['BPCER'] * 100:.2f}%")
    print(f"ACER     : {metrics['ACER'] * 100:.2f}%")
    print("-" * 60)
    print(f"TN={metrics['TN']} | FP={metrics['FP']} | FN={metrics['FN']} | TP={metrics['TP']}")
    print("=" * 60)
    print("Note: CelebA est image-based, donc behav=zeros et séquence synthétique.\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True, help="Chemin vers celeba_ft_test.csv")
    parser.add_argument("--base_dir", required=True, help="Racine CelebA_Spoof/Data")

    parser.add_argument("--val_csv", default=None, help="Chemin vers celeba_ft_val.csv requis si valopt")
    parser.add_argument(
        "--threshold_protocol",
        choices=["fixed05", "valopt"],
        default="fixed05",
        help="fixed05 = threshold 0.5 ; valopt = meilleur threshold sur VAL puis appliqué sur TEST",
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--out_dir", default="reports/cross_dataset/celeba_eval")

    args = parser.parse_args()

    if args.threshold_protocol == "valopt" and args.val_csv is None:
        raise ValueError("--val_csv est obligatoire avec --threshold_protocol valopt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.checkpoint, device)

    best_val = None

    if args.threshold_protocol == "fixed05":
        th_used = 0.50
        threshold_source = "fixed05"

    else:
        print("[INFO] Recherche du meilleur threshold sur CelebA VAL...")
        best_val = find_best_threshold_on_val(
            model=model,
            val_csv=args.val_csv,
            base_dir=args.base_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
        )

        th_used = float(best_val["threshold"])
        threshold_source = "valopt"

        print("\n[VALOPT] Meilleur threshold trouvé sur VAL")
        print(f"  threshold = {th_used:.4f}")
        print(f"  VAL ACER  = {best_val['ACER'] * 100:.2f}%")
        print(f"  VAL F1    = {best_val['F1'] * 100:.2f}%")

    print("[INFO] Inférence TEST en cours...")
    test_loader = make_loader(
        csv_path=args.csv,
        base_dir=args.base_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    y_true, y_pred, y_score = run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=th_used,
    )

    metrics = compute_pad_metrics(y_true, y_pred, y_score)
    metrics["threshold"] = float(th_used)

    pretty_print_metrics(
        title="Résultats — CelebA Evaluation",
        metrics=metrics,
        threshold_protocol=args.threshold_protocol,
        threshold_source=threshold_source,
    )

    result = {
        "checkpoint": args.checkpoint,
        "test_csv": args.csv,
        "val_csv": args.val_csv,
        "threshold_protocol": args.threshold_protocol,
        "threshold_source": threshold_source,
        "threshold_used": float(th_used),
        "best_val": best_val,
        "test_metrics": metrics,
        "note": "CelebA image-based: behav=zeros and synthetic repeated-frame sequence.",
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    save_confusion_csv(
        os.path.join(args.out_dir, "confusion_matrix.csv"),
        metrics["TN"],
        metrics["FP"],
        metrics["FN"],
        metrics["TP"],
    )

    save_score_distribution_csv(
        os.path.join(args.out_dir, "score_distribution.csv"),
        y_true,
        y_score,
    )

    save_threshold_sweep_csv(
        os.path.join(args.out_dir, "threshold_sweep.csv"),
        y_true,
        y_score,
    )

    print(f"[OK] Résultats sauvegardés dans: {args.out_dir}")


if __name__ == "__main__":
    main()