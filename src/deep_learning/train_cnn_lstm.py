import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.deep_learning.datasets_sequence import CASIASequenceDataset
from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean().item()

    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return acc, f1


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)

        total_loss += loss.item() * x.size(0)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    y_cat = torch.cat(all_y, dim=0)

    acc, f1 = metrics_from_logits(logits_cat, y_cat)
    return {"loss": total_loss / len(y_cat), "acc": acc, "f1": f1}


def make_balanced_sampler(ds: CASIASequenceDataset) -> WeightedRandomSampler:

    labels = [ds.labels_by_vid[vid] for vid in ds.video_ids] 
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)  
    class_weights = 1.0 / np.maximum(class_counts, 1.0)             
    sample_weights = [class_weights[y] for y in labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


@dataclass
class TrainConfig:
    train_csv: str = r"data\processed\CASIA\splits_subject\train.csv"
    val_csv: str   = r"data\processed\CASIA\splits_subject\val.csv"
    out_dir: str   = r"experiments\exp4_casia_mnv3_cnn_lstm_pro"

    T: int = 16
    img_size: int = 224
    train_aug: str = "strong"
    val_aug: str = "none"

 
    train_sample_mode: str = "random_clip"
    val_sample_mode: str = "uniform"

    
    batch_size: int = 4
    epochs: int = 14
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 0

    hidden: int = 256
    num_layers: int = 1
    bidir: bool = False
    temporal_pool: str = "median"  

    seed: int = 42
    freeze_backbone_epochs: int = 2
    use_amp: bool = True

   
    use_balanced_sampler: bool = True  


def main():
    cfg = TrainConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)

    
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = CASIASequenceDataset(
        cfg.train_csv, T=cfg.T, img_size=cfg.img_size,
        aug_mode=cfg.train_aug, sample_mode=cfg.train_sample_mode, seed=cfg.seed
    )
    val_ds = CASIASequenceDataset(
        cfg.val_csv, T=cfg.T, img_size=cfg.img_size,
        aug_mode=cfg.val_aug, sample_mode=cfg.val_sample_mode, seed=cfg.seed
    )

    if cfg.use_balanced_sampler:
        sampler = make_balanced_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = CNN_LSTM_PAD(
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        bidir=cfg.bidir,
        temporal_pool=cfg.temporal_pool,
        pretrained_backbone=True,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.startswith("cuda")))
    ce = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_path = os.path.join(cfg.out_dir, "best_model.pth")

    hist_path = os.path.join(cfg.out_dir, "history.csv")
    with open(hist_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,train_f1,val_loss,val_acc,val_f1,lr,freeze\n")

    for epoch in range(1, cfg.epochs + 1):

        freeze = epoch <= cfg.freeze_backbone_epochs
        model.freeze_backbone(freeze)

        model.train()
        total_loss = 0.0
        all_logits, all_y = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = ce(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item() * x.size(0)
            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())

            logits_cat = torch.cat(all_logits, dim=0)
            y_cat = torch.cat(all_y, dim=0)
            train_acc, train_f1 = metrics_from_logits(logits_cat, y_cat)
            pbar.set_postfix(loss=total_loss/len(y_cat), acc=train_acc, f1=train_f1, lr=opt.param_groups[0]["lr"], freeze=("Y" if freeze else "N"))

        scheduler.step()

     
        logits_cat = torch.cat(all_logits, dim=0)
        y_cat = torch.cat(all_y, dim=0)
        train_acc, train_f1 = metrics_from_logits(logits_cat, y_cat)
        train_loss = total_loss / len(y_cat)

        val_m = evaluate(model, val_loader, device)
        print(f"VAL: loss={val_m['loss']:.4f} acc={val_m['acc']:.4f} f1={val_m['f1']:.4f}")

        with open(hist_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{train_f1:.6f},{val_m['loss']:.6f},{val_m['acc']:.6f},{val_m['f1']:.6f},{opt.param_groups[0]['lr']:.8f},{int(freeze)}\n")


        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, best_path)
            print("Saved best model:", best_path)

    print("Done. Best val F1:", best_val_f1)


if __name__ == "__main__":
    main()
