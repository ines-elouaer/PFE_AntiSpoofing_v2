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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.celeba_dataset import CelebASpoofDataset
from src.deep_learning.datasets_sequence import CASIASequenceDataset
from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD


def compute_acer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0

    return acer, apcer, bpcer, tn, fp, fn, tp


def unpack_batch(batch, device):
    """
    Compatible avec:
    - CelebA: (seq, behav, label)
    - CASIA: formats variables
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Batch inattendu: {type(batch)}")

    if len(batch) == 3:
        seq, x2, x3 = batch
        if torch.is_tensor(x2) and torch.is_tensor(x3) and x3.ndim <= 1:
            behav = x2
            labels = x3
        else:
            labels = x2
            behav = torch.zeros(seq.size(0), 9, dtype=torch.float32)

    elif len(batch) == 2:
        seq, labels = batch
        behav = torch.zeros(seq.size(0), 9, dtype=torch.float32)

    else:
        seq = batch[0]
        labels = None
        behav = None

        for item in batch[1:]:
            if torch.is_tensor(item):
                if item.ndim == 2 and item.shape[-1] == 9 and behav is None:
                    behav = item
                elif item.ndim <= 1 and labels is None:
                    labels = item

        if labels is None:
            raise ValueError("Impossible d'identifier le label dans le batch.")

        if behav is None:
            behav = torch.zeros(seq.size(0), 9, dtype=torch.float32)

    return seq.to(device), behav.to(device), labels.to(device).long()


def load_checkpoint_with_config(checkpoint_path, device):
    ckpt_path = Path(checkpoint_path)

    config = {}
    cfg_path = ckpt_path.parent / "config.json"

    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[OK] Config trouvée: {cfg_path}")
    else:
        print("[WARN] config.json introuvable, paramètres par défaut utilisés.")

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

    print(f"[OK] Checkpoint chargé: {ckpt_path}")
    return model, config


def freeze_for_mixed_ft(model, unfreeze_last_backbone=False):
    """
    Stratégie:
    - Backbone gelé par défaut
    - LSTM gelé
    - behav_mlp entraînable
    - head entraînable
    - option: dégeler le dernier bloc backbone
    """

    for p in model.backbone.parameters():
        p.requires_grad = False

    if unfreeze_last_backbone:
        try:
            for p in model.backbone.features[-1].parameters():
                p.requires_grad = True
            print("[Unfreeze] Dernier bloc backbone dégelé.")
        except Exception:
            print("[WARN] Impossible de dégeler model.backbone.features[-1].")

    for p in model.lstm.parameters():
        p.requires_grad = False

    if model.behav_mlp is not None:
        for p in model.behav_mlp.parameters():
            p.requires_grad = True

    for p in model.head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(
        f"[Freeze] Paramètres entraînables : {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )


@torch.no_grad()
def evaluate_dataset(model, loader, device, criterion):
    model.eval()

    losses = []
    y_true = []
    y_pred = []

    for batch in loader:
        seq, behav, labels = unpack_batch(batch, device)

        logits = model(seq, behav)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        losses.append(loss.item())
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    acer, apcer, bpcer, tn, fp, fn, tp = compute_acer(
        np.array(y_true), np.array(y_pred)
    )

    return {
        "loss": float(np.mean(losses)),
        "ACER": float(acer),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def train_one_epoch_mixed(
    model,
    casia_loader,
    celeba_loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    casia_iter = iter(casia_loader)
    celeba_iter = iter(celeba_loader)

    n_steps = min(len(casia_loader), len(celeba_loader))
    losses = []

    for _ in range(n_steps):
        # CASIA batch
        batch = next(casia_iter)
        seq, behav, labels = unpack_batch(batch, device)

        optimizer.zero_grad()
        logits = model(seq, behav)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # CelebA batch
        batch = next(celeba_iter)
        seq, behav, labels = unpack_batch(batch, device)

        optimizer.zero_grad()
        logits = model(seq, behav)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--celeba_train_csv", required=True)
    parser.add_argument("--celeba_val_csv", required=True)
    parser.add_argument("--celeba_base_dir", required=True)

    parser.add_argument("--casia_train_csv", required=True)
    parser.add_argument("--casia_val_csv", required=True)
    parser.add_argument("--casia_behav_train_csv", required=True)
    parser.add_argument("--casia_behav_val_csv", required=True)

    parser.add_argument("--out_dir", default="experiments/mixed_ft_v2/seed42")

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--unfreeze_last_backbone",
        action="store_true",
        help="Dégeler le dernier bloc du backbone MobileNetV3.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print("Expérience 5 v2 — Mixed Fine-Tuning CASIA + CelebA")
    print(f"LR={args.lr} | epochs={args.epochs} | patience={args.patience}\n")

    # =========================
    # CelebA datasets
    # =========================
    celeba_train_ds = CelebASpoofDataset(
        csv_path=args.celeba_train_csv,
        base_dir=args.celeba_base_dir,
        seq_len=args.seq_len,
        augment=True,
    )

    celeba_val_ds = CelebASpoofDataset(
        csv_path=args.celeba_val_csv,
        base_dir=args.celeba_base_dir,
        seq_len=args.seq_len,
        augment=False,
    )

    celeba_train_dl = DataLoader(
        celeba_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    celeba_val_dl = DataLoader(
        celeba_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # =========================
    # CASIA datasets
    # =========================
    casia_train_ds = CASIASequenceDataset(
        csv_path=args.casia_train_csv,
        T=args.seq_len,
        img_size=224,
        aug_mode="strong",
        sample_mode="consecutive",
        seed=args.seed,
        behav_csv=args.casia_behav_train_csv,
    )

    casia_val_ds = CASIASequenceDataset(
        csv_path=args.casia_val_csv,
        T=args.seq_len,
        img_size=224,
        aug_mode="none",
        sample_mode="center_consecutive",
        seed=args.seed,
        behav_csv=args.casia_behav_val_csv,
    )

    casia_train_dl = DataLoader(
        casia_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    casia_val_dl = DataLoader(
        casia_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # =========================
    # Model
    # =========================
    model, config = load_checkpoint_with_config(args.checkpoint, device)

    freeze_for_mixed_ft(
        model,
        unfreeze_last_backbone=args.unfreeze_last_backbone,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )

    best_mixed_acer = float("inf")
    no_improve = 0
    history = []

    print(
        f"{'Epoch':>5} {'TrainLoss':>10} "
        f"{'CASIA_ACER':>11} {'CelebA_ACER':>12} "
        f"{'Mixed_ACER':>11} {'Status':>10}"
    )
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_mixed(
            model=model,
            casia_loader=casia_train_dl,
            celeba_loader=celeba_train_dl,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        casia_val = evaluate_dataset(
            model=model,
            loader=casia_val_dl,
            device=device,
            criterion=criterion,
        )

        celeba_val = evaluate_dataset(
            model=model,
            loader=celeba_val_dl,
            device=device,
            criterion=criterion,
        )

        mixed_acer = (casia_val["ACER"] + celeba_val["ACER"]) / 2.0

        if mixed_acer < best_mixed_acer:
            best_mixed_acer = mixed_acer
            no_improve = 0
            status = "✓ best"

            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, "best_model_mixed_ft.pth"),
            )

            with open(
                os.path.join(args.out_dir, "best_metrics_val.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "mixed_ACER": mixed_acer,
                        "casia_val": casia_val,
                        "celeba_val": celeba_val,
                        "lr": args.lr,
                        "class_weights": [1.0, 1.5],
                        "unfreeze_last_backbone": args.unfreeze_last_backbone,
                    },
                    f,
                    indent=2,
                )
        else:
            no_improve += 1
            status = f"({no_improve}/{args.patience})"

        print(
            f"{epoch:>5} {train_loss:>10.4f} "
            f"{casia_val['ACER'] * 100:>11.2f} "
            f"{celeba_val['ACER'] * 100:>12.2f} "
            f"{mixed_acer * 100:>11.2f} {status:>10}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "mixed_ACER": mixed_acer,
                "casia_val": casia_val,
                "celeba_val": celeba_val,
            }
        )

        if no_improve >= args.patience:
            print(f"\nEarly stopping à l'epoch {epoch}")
            break

    with open(
        os.path.join(args.out_dir, "history.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history, f, indent=2)

    print(f"\nMeilleur mixed_ACER val: {best_mixed_acer * 100:.2f}%")
    print(f"Checkpoint sauvegardé: {args.out_dir}/best_model_mixed_ft.pth")


if __name__ == "__main__":
    main()