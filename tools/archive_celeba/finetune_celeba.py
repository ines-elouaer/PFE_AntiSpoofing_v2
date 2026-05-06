"""
finetune_celeba.py
==================
Expérience 4 — Fine-tuning contrôlé sur subset CelebA.

Stratégie anti-overfitting :
  - Geler backbone + LSTM + behav_mlp
  - Entraîner seulement model.head
  - LR = 1e-4, max 10 epochs, early stopping patience=3
  - Vérifier que CASIA ne régresse pas (catastrophic forgetting check)

Usage :
  python tools\celeba\finetune_celeba.py `
    --checkpoint "experiments\...\best_model.pth" `
    --train_csv  "data\celeba\celeba_ft_train.csv" `
    --val_csv    "data\celeba\celeba_ft_val.csv" `
    --base_dir   "E:\CelebA-Spoof\CelebA_Spoof\Data" `
    --out_dir    "experiments\celeba_finetune\seed42"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from celeba_dataset import CelebASpoofDataset
from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD


# ─── Métriques ────────────────────────────────────────────────────────────────
def compute_acer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return (apcer + bpcer) / 2.0, apcer, bpcer


# ─── Chargement modèle ────────────────────────────────────────────────────────
def load_model(checkpoint_path, device):
    ckpt_path = Path(checkpoint_path)
    config = {}
    cfg = ckpt_path.parent / "config.json"
    if cfg.exists():
        with open(cfg) as f:
            config = json.load(f)

    model = CNN_LSTM_PAD(
        hidden              = config.get("hidden", 256),
        num_layers          = config.get("num_layers", 1),
        bidir               = config.get("bidir", False),
        lstm_dropout        = config.get("lstm_dropout", 0.0),
        head_dropout        = 0.0,
        pretrained_backbone = False,
        temporal_pool       = config.get("temporal_pool", "median"),
        use_behav           = config.get("use_behav", True),
        behav_dim           = config.get("behav_dim", 9),
        behav_hidden        = config.get("behav_hidden", 16),
    )

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "model_state_dict" in state: state = state["model_state_dict"]
    elif "model_state"     in state: state = state["model_state"]
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device)
    print(f"[OK] Checkpoint : {ckpt_path}")
    return model


# ─── Gel du modèle ────────────────────────────────────────────────────────────
def freeze_except_head(model):
    """
    Gèle tout sauf model.head.
    Le head = couche de décision finale [256→128→2].
    C'est la seule partie qu'on adapte à CelebA.
    """
    # Geler backbone CNN
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Geler LSTM
    for p in model.lstm.parameters():
        p.requires_grad = False

    # Geler behav_mlp si présent
    if model.behav_mlp is not None:
        for p in model.behav_mlp.parameters():
            p.requires_grad = False

    # Laisser le head libre
    for p in model.head.parameters():
        p.requires_grad = True
    try:
        for p in model.backbone.features[-1].parameters():
           p.requires_grad = True
        print("[Unfreeze] Dernier bloc backbone dégelé.")
    except Exception as e:
        print("[WARN] Impossible de dégeler le dernier bloc backbone:", e)

    # Afficher ce qui est entraînable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Freeze] Paramètres entraînables : {trainable:,} / {total:,} "
          f"({trainable/total*100:.2f}%)")


# ─── Évaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, y_true, y_pred = [], [], []

    for seq, behav, labels in loader:
        seq, behav, labels = seq.to(device), behav.to(device), labels.to(device)
        logits = model(seq, behav)
        loss   = criterion(logits, labels)
        losses.append(loss.item())

        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    acer, apcer, bpcer = compute_acer(np.array(y_true), np.array(y_pred))
    return np.mean(losses), acer, apcer, bpcer


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_csv",  required=True)
    parser.add_argument("--val_csv",    required=True)
    parser.add_argument("--base_dir",   required=True)
    parser.add_argument("--out_dir",    default="experiments/celeba_finetune")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--patience",   type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--seq_len",    type=int,   default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Expérience 4 — Fine-tuning head uniquement | LR={args.lr} | patience={args.patience}\n")

    # Datasets
    train_ds = CelebASpoofDataset(args.train_csv, args.base_dir,
                                   args.seq_len, augment=True)
    val_ds   = CelebASpoofDataset(args.val_csv,   args.base_dir,
                                   args.seq_len, augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)

    # Modèle
    model = load_model(args.checkpoint, device)
    freeze_except_head(model)

    # Loss + optimizer sur le head seulement
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # ── Boucle d'entraînement ─────────────────────────────────────────────
    best_acer      = float("inf")
    best_state     = None
    no_improve     = 0
    history        = []

    print(f"{'Epoch':>5} {'TrainLoss':>10} {'ValLoss':>9} "
          f"{'ACER%':>7} {'APCER%':>7} {'BPCER%':>7} {'Status':>10}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_losses = []
        for seq, behav, labels in train_dl:
            seq, behav, labels = seq.to(device), behav.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seq, behav)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # Val
        val_loss, acer, apcer, bpcer = evaluate(model, val_dl, device, criterion)
        scheduler.step(val_loss)

        # Early stopping
        status = ""
        if acer < best_acer:
            best_acer  = acer
            best_state = deepcopy(model.state_dict())
            no_improve = 0
            status     = "✓ best"
            torch.save(best_state,
                       os.path.join(args.out_dir, "best_model_celeba_ft.pth"))
        else:
            no_improve += 1
            status = f"({no_improve}/{args.patience})"

        print(f"{epoch:>5} {train_loss:>10.4f} {val_loss:>9.4f} "
              f"{acer*100:>7.2f} {apcer*100:>7.2f} {bpcer*100:>7.2f} {status:>10}")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "ACER": acer,
            "APCER": apcer, "BPCER": bpcer,
        })

        if no_improve >= args.patience:
            print(f"\nEarly stopping à l'epoch {epoch} (patience={args.patience})")
            break

    # Sauvegarder historique
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nMeilleur ACER val : {best_acer*100:.2f}%")
    print(f"Modèle sauvegardé : {args.out_dir}/best_model_celeba_ft.pth")
    print("\nProchaine étape : lancer eval_cross_dataset.py avec le nouveau checkpoint")
    print("pour comparer avant/après fine-tuning.")


if __name__ == "__main__":
    main()
