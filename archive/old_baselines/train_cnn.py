# src/deep_learning/train_cnn.py
import os
import json
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from tqdm import tqdm

from src.deep_learning.datasets_frame import CASIAFrameDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # determinism (reduce variance between runs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(num_classes=2):
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Returns:
      val_loss, val_acc, val_f1_attack
    where attack class = 1
    """
    model.eval()
    ce = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    total_loss = 0.0

    # confusion for attack=1
    tp = fp = fn = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = ce(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += x.size(0)

        # attack F1 (positive class = 1)
        tp += int(((pred == 1) & (y == 1)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())

    val_loss = total_loss / max(total, 1)
    val_acc = correct / max(total, 1)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    val_f1_attack = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return val_loss, val_acc, val_f1_attack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--aug_mode", type=str, default="strong", choices=["none", "soft", "strong"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_csv", type=str, default=r"data\processed\CASIA\splits_subject\train.csv")
    parser.add_argument("--val_csv", type=str, default=r"data\processed\CASIA\splits_subject\val.csv")
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- Experiment folder
    if args.out_dir:
        exp_dir = args.out_dir
    else:
        exp_dir = fr"experiments\exp_cnn_casia_mnv3large_{args.aug_mode}_seed{args.seed}"
    os.makedirs(exp_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Save config (reproducibility)
    config = {
        "date": str(datetime.now()),
        "task": "CASIA PAD - CNN baseline (frame-level training, video-level eval later)",
        "backbone": "mobilenet_v3_large",
        "aug_mode_train": args.aug_mode,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "selection_metric": "val_f1_attack",
        "attack_class": 1,
        "seed": args.seed,
    }
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Datasets / loaders
    train_ds = CASIAFrameDataset(args.train_csv, img_size=args.img_size, aug_mode=args.aug_mode)
    val_ds = CASIAFrameDataset(args.val_csv, img_size=args.img_size, aug_mode="none")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model / optim
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    # Logging
    history_csv = os.path.join(exp_dir, "history.csv")
    with open(history_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_f1_attack,lr\n")

    best_val_f1_attack = -1.0
    best_path = os.path.join(exp_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total = 0
        correct = 0
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

            pbar.set_postfix(
                loss=running_loss / max(total, 1),
                acc=correct / max(total, 1),
                lr=opt.param_groups[0]["lr"],
            )

        scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc, val_f1_attack = evaluate(model, val_loader, device)
        print(f"Val: loss={val_loss:.4f} acc={val_acc:.4f} f1_attack={val_f1_attack:.4f}")

        with open(history_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                f"{val_loss:.6f},{val_acc:.6f},{val_f1_attack:.6f},"
                f"{opt.param_groups[0]['lr']:.8f}\n"
            )

        # Best selection by val_f1_attack
        if val_f1_attack > best_val_f1_attack:
            best_val_f1_attack = val_f1_attack
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_val_f1_attack": best_val_f1_attack,
                    "config": config,
                },
                best_path,
            )
            print(f"Saved best model (val_f1_attack={best_val_f1_attack:.4f})")

    print("Done. Best val_f1_attack:", best_val_f1_attack)
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()