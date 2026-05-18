from __future__ import annotations
import os
import json
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, List

import pandas as pd
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def split_batch(batch):
    if len(batch) == 3:
        x, y, vid = batch
        behav = None
    else:
        x, y, vid, behav = batch
    return x, y, vid, behav


@torch.no_grad()
def evaluate(model, loader, device, cfg: TrainConfig | None = None) -> dict:
    model.eval()

    if cfg is not None:
        class_weights = torch.tensor(
            [cfg.loss_real_weight, cfg.loss_spoof_weight],
            dtype=torch.float32,
            device=device,
        )
        ce = nn.CrossEntropyLoss(weight=class_weights)
    else:
        ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    for batch in loader:
        x, y, _, behav = split_batch(batch)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        behav = None if behav is None else behav.to(device, non_blocking=True)

        logits = model(x, behav)
        loss = ce(logits, y)

        total_loss += loss.item() * x.size(0)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    y_cat = torch.cat(all_y, dim=0)

    acc, f1 = metrics_from_logits(logits_cat, y_cat)
    return {"loss": total_loss / len(y_cat), "acc": acc, "f1": f1}
def load_hard_examples(hard_real_csv: str | None = None, hard_spoof_csv: str | None = None) -> dict:
    """
    Charge les hard examples.

    Format attendu :
        video_id,hard_type,sample_weight

    Exemple :
        LOCAL_REAL__p03_samsung_normal_t02.mp4,REAL_AS_SPOOF,3.0
        AXON__axon_06d326f921582e,SPOOF_AS_REAL,3.0
    """

    hard_weights = {}

    for csv_path in [hard_real_csv, hard_spoof_csv]:
        if csv_path is None:
            continue

        p = Path(csv_path)
        if not p.exists():
            print(f"[WARN] Hard examples introuvable, ignoré: {p}")
            continue

        df = pd.read_csv(p)

        if "video_id" not in df.columns:
            raise ValueError(f"Le fichier {p} doit contenir une colonne video_id")

        if "sample_weight" not in df.columns:
            df["sample_weight"] = 3.0

        for _, row in df.iterrows():
            vid = str(row["video_id"])
            hard_weights[vid] = float(row["sample_weight"])

        print(f"[INFO] Hard examples chargés depuis {p}: {len(df)}")

    return hard_weights


def make_hard_balanced_sampler(
    ds: CASIASequenceDataset,
    hard_real_csv: str | None = None,
    hard_spoof_csv: str | None = None,
    real_sampling_weight: float = 1.5,
    spoof_sampling_weight: float = 1.0,
) -> WeightedRandomSampler:
    """
    Sampler amélioré.

    Objectif :
    - renforcer légèrement la classe REAL pour réduire REAL -> SPOOF ;
    - renforcer les hard examples REAL -> SPOOF ;
    - renforcer aussi les hard examples SPOOF -> REAL pour préserver la sécurité.

    Poids final :
        class_weight * hard_weight

    Exemple :
        REAL normal      : 1.5
        SPOOF normal     : 1.0
        REAL->SPOOF hard : 1.5 * 3 = 4.5
        SPOOF->REAL hard : 1.0 * 3 = 3.0
    """

    hard_weights = load_hard_examples(hard_real_csv, hard_spoof_csv)

    sample_weights = []
    matched_hard = 0

    for vid in ds.video_ids:
        label = int(ds.labels_by_vid[vid])

        if label == 0:
            weight = float(real_sampling_weight)
        else:
            weight = float(spoof_sampling_weight)

        if str(vid) in hard_weights:
            weight *= float(hard_weights[str(vid)])
            matched_hard += 1

        sample_weights.append(weight)

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    labels = [int(ds.labels_by_vid[vid]) for vid in ds.video_ids]
    real_count = sum(1 for y in labels if y == 0)
    spoof_count = sum(1 for y in labels if y == 1)

    print("\n========== SAMPLER HARD BALANCED ==========")
    print(f"Total train videos         : {len(ds.video_ids)}")
    print(f"REAL count                 : {real_count}")
    print(f"SPOOF count                : {spoof_count}")
    print(f"Hard examples in CSV       : {len(hard_weights)}")
    print(f"Hard examples matched train: {matched_hard}")
    print(f"REAL sampling weight       : {real_sampling_weight}")
    print(f"SPOOF sampling weight      : {spoof_sampling_weight}")
    print(f"Min sample weight          : {sample_weights.min().item():.4f}")
    print(f"Max sample weight          : {sample_weights.max().item():.4f}")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

def make_optimizer(model, lr_backbone: float, lr_head: float, weight_decay: float):
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})

    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def set_freeze_backbone_compat(model: nn.Module, freeze: bool):
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone(freeze)  # type: ignore
        return

    if freeze:
        if hasattr(model, "freeze_all_backbone"):
            model.freeze_all_backbone()  # type: ignore
        else:
            for name, p in model.named_parameters():
                if name.startswith("backbone."):
                    p.requires_grad = False
    else:
        if hasattr(model, "unfreeze_all_backbone"):
            model.unfreeze_all_backbone()  # type: ignore
        else:
            for name, p in model.named_parameters():
                if name.startswith("backbone."):
                    p.requires_grad = True


@dataclass
class TrainConfig:
    train_csv: str = r"data\processed\CASIA\splits_subject\train.csv"
    val_csv: str = r"data\processed\CASIA\splits_subject\val.csv"
    out_dir: str = r"experiments\exp4_casia_mnv3_cnn_lstm_pro"

    T: int = 16
    img_size: int = 224
    train_aug: str = "strong"
    val_aug: str = "none"

    # Bloc B
    train_sample_mode: str = "uniform"            # uniform | random_clip | consecutive | center_consecutive
    val_sample_mode: str = "uniform"              # uniform | random_clip | consecutive | center_consecutive

    batch_size: int = 8

    epochs: int = 14
    lr: float = 2e-4
    weight_decay: float = 1e-4

    num_workers: int = 2

    hidden: int = 256
    num_layers: int = 1
    bidir: bool = False
    temporal_pool: str = "median"

    seed: int = 42
    freeze_backbone_epochs: int = 2
    use_amp: bool = True
    use_balanced_sampler: bool = True

    use_pts: bool = True

    phase1_epochs: int = 3
    phase2_epochs: int = 6
    phase3_epochs: int = 5

    unfreeze_last_k: int = 4

    lr_head_p1: float = 2e-4
    lr_bb_p1: float = 0.0

    lr_head_p2: float = 1e-4
    lr_bb_p2: float = 1e-5

    lr_head_p3: float = 5e-5
    lr_bb_p3: float = 5e-6

    use_behav: bool = True
    behav_dim: int = 9
    behav_hidden: int = 16
    behav_train_csv: str = r"data\processed\CASIA\behav\train_behav.csv"
    behav_val_csv: str = r"data\processed\CASIA\behav\val_behav.csv"

    # Gated / attention fusion
    use_gated_fusion: bool = True
    gate_hidden: int = 128

    # Hard examples
    hard_real_csv: str = r"data\hard_examples\hard_real_samples.csv"
    hard_spoof_csv: str = r"data\hard_examples\hard_spoof_samples.csv"

    # Sampling weights
    real_sampling_weight: float = 1.5
    spoof_sampling_weight: float = 1.0

    # Loss weights: label 0 = REAL, label 1 = SPOOF
    loss_real_weight: float = 1.2
    loss_spoof_weight: float = 1.0

def run_phase(
    phase_name: str,
    model: CNN_LSTM_PAD,
    train_loader,
    val_loader,
    device: str,
    cfg: TrainConfig,
    epochs: int,
    lr_bb: float,
    lr_head: float,
    mode: str,
    hist_f,
):
    if mode == "head_only":
        model.freeze_all_backbone()
    elif mode == "last_k":
        model.freeze_all_backbone()
        model.unfreeze_last_k_backbone_blocks(cfg.unfreeze_last_k)
    elif mode == "all":
        model.unfreeze_all_backbone()
    else:
        raise ValueError(mode)

    opt = make_optimizer(model, lr_backbone=lr_bb, lr_head=lr_head, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.startswith("cuda")))
    class_weights = torch.tensor(
        [cfg.loss_real_weight, cfg.loss_spoof_weight],
        dtype=torch.float32,
        device=device,
    )
    ce = nn.CrossEntropyLoss(weight=class_weights)

    print("\n========== LOSS WEIGHTS ==========")
    print(f"REAL loss weight : {cfg.loss_real_weight}")
    print(f"SPOOF loss weight: {cfg.loss_spoof_weight}")

    best_val_f1 = -1.0
    best_path = os.path.join(cfg.out_dir, f"best_{phase_name}.pth")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_logits, all_y = [], []

        pbar = tqdm(train_loader, desc=f"{phase_name} {ep}/{epochs}")
        for batch in pbar:
            x, y, _, behav = split_batch(batch)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            behav = None if behav is None else behav.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x, behav)
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

            pbar.set_postfix(
                loss=total_loss / len(y_cat),
                acc=train_acc,
                f1=train_f1,
                lr_bb=lr_bb,
                lr_head=lr_head,
                mode=mode,
            )

        scheduler.step()

        logits_cat = torch.cat(all_logits, dim=0)
        y_cat = torch.cat(all_y, dim=0)
        train_acc, train_f1 = metrics_from_logits(logits_cat, y_cat)
        train_loss = total_loss / len(y_cat)

        val_m = evaluate(model, val_loader, device, cfg)

        print(
            f"[{phase_name}] train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
            f"val_loss={val_m['loss']:.4f} acc={val_m['acc']:.4f} f1={val_m['f1']:.4f} | "
            f"lr_bb={lr_bb} lr_head={lr_head}"
        )

        hist_f.write(
            f"{phase_name},{ep},{train_loss:.6f},{train_acc:.6f},{train_f1:.6f},"
            f"{val_m['loss']:.6f},{val_m['acc']:.6f},{val_m['f1']:.6f},"
            f"{lr_bb:.8f},{lr_head:.8f},{mode}\n"
        )
        hist_f.flush()

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, best_path)

    print(f"{phase_name} done. Best val_f1={best_val_f1:.4f} saved={best_path}")
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--use_pts", type=int, default=1)
    parser.add_argument("--unfreeze_last_k", type=int, default=None)

    parser.add_argument("--use_behav", type=int, default=None, help="1=use behavior features, 0=deep only")
    parser.add_argument("--behav_train_csv", type=str, default=None)
    parser.add_argument("--behav_val_csv", type=str, default=None)
    parser.add_argument("--behav_hidden", type=int, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--behav_dim", type=int, default=None)
    parser.add_argument("--use_gated_fusion", type=int, default=None, help="1=gated fusion, 0=concat fusion")
    parser.add_argument("--gate_hidden", type=int, default=None)

    parser.add_argument("--hard_real_csv", type=str, default=None)
    parser.add_argument("--hard_spoof_csv", type=str, default=None)

    parser.add_argument("--real_sampling_weight", type=float, default=None)
    parser.add_argument("--spoof_sampling_weight", type=float, default=None)

    parser.add_argument("--loss_real_weight", type=float, default=None)
    parser.add_argument("--loss_spoof_weight", type=float, default=None)

    # Bloc B
    parser.add_argument(
        "--train_sample_mode",
        type=str,
        default=None,
        choices=["uniform", "random_clip", "consecutive", "center_consecutive"],
    )
    parser.add_argument(
        "--val_sample_mode",
        type=str,
        default=None,
        choices=["uniform", "random_clip", "consecutive", "center_consecutive"],
    )

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.seed = args.seed
    cfg.use_pts = bool(args.use_pts)

    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.unfreeze_last_k is not None:
        cfg.unfreeze_last_k = args.unfreeze_last_k
    if args.train_csv is not None:
        cfg.train_csv = args.train_csv
    if args.val_csv is not None:
        cfg.val_csv = args.val_csv

    if args.use_behav is not None:
        cfg.use_behav = bool(args.use_behav)
    if args.behav_train_csv is not None:
        cfg.behav_train_csv = args.behav_train_csv
    if args.behav_val_csv is not None:
        cfg.behav_val_csv = args.behav_val_csv
    if args.behav_hidden is not None:
        cfg.behav_hidden = args.behav_hidden
    if args.behav_dim is not None:
        cfg.behav_dim = args.behav_dim
    if args.use_gated_fusion is not None:
        cfg.use_gated_fusion = bool(args.use_gated_fusion)
    if args.gate_hidden is not None:
        cfg.gate_hidden = args.gate_hidden

    if args.hard_real_csv is not None:
        cfg.hard_real_csv = args.hard_real_csv
    if args.hard_spoof_csv is not None:
        cfg.hard_spoof_csv = args.hard_spoof_csv

    if args.real_sampling_weight is not None:
        cfg.real_sampling_weight = args.real_sampling_weight
    if args.spoof_sampling_weight is not None:
        cfg.spoof_sampling_weight = args.spoof_sampling_weight

    if args.loss_real_weight is not None:
        cfg.loss_real_weight = args.loss_real_weight
    if args.loss_spoof_weight is not None:
        cfg.loss_spoof_weight = args.loss_spoof_weight

    if args.train_sample_mode is not None:
        cfg.train_sample_mode = args.train_sample_mode
    if args.val_sample_mode is not None:
        cfg.val_sample_mode = args.val_sample_mode

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("use_pts:", cfg.use_pts)
    print("use_behav:", cfg.use_behav)
    print("train_sample_mode:", cfg.train_sample_mode)
    print("val_sample_mode:", cfg.val_sample_mode)
    print("use_gated_fusion:", cfg.use_gated_fusion)
    print("gate_hidden:", cfg.gate_hidden)
    print("hard_real_csv:", cfg.hard_real_csv)
    print("hard_spoof_csv:", cfg.hard_spoof_csv)
    print("real_sampling_weight:", cfg.real_sampling_weight)
    print("spoof_sampling_weight:", cfg.spoof_sampling_weight)
    print("loss_real_weight:", cfg.loss_real_weight)
    print("loss_spoof_weight:", cfg.loss_spoof_weight)
    train_ds = CASIASequenceDataset(
        cfg.train_csv,
        T=cfg.T,
        img_size=cfg.img_size,
        aug_mode=cfg.train_aug,
        sample_mode=cfg.train_sample_mode,
        seed=cfg.seed,
        behav_csv=(cfg.behav_train_csv if cfg.use_behav else None),
    )
    val_ds = CASIASequenceDataset(
        cfg.val_csv,
        T=cfg.T,
        img_size=cfg.img_size,
        aug_mode=cfg.val_aug,
        sample_mode=cfg.val_sample_mode,
        seed=cfg.seed,
        behav_csv=(cfg.behav_val_csv if cfg.use_behav else None),
    )

    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    if cfg.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    if cfg.use_balanced_sampler:
        sampler = make_hard_balanced_sampler(
            train_ds,
            hard_real_csv=cfg.hard_real_csv,
            hard_spoof_csv=cfg.hard_spoof_csv,
            real_sampling_weight=cfg.real_sampling_weight,
            spoof_sampling_weight=cfg.spoof_sampling_weight,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = CNN_LSTM_PAD(
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        bidir=cfg.bidir,
        temporal_pool=cfg.temporal_pool,
        pretrained_backbone=True,
        use_behav=cfg.use_behav,
        behav_dim=cfg.behav_dim,
        behav_hidden=cfg.behav_hidden,
        use_gated_fusion=cfg.use_gated_fusion,
        gate_hidden=cfg.gate_hidden,
    ).to(device)

    hist_path = os.path.join(cfg.out_dir, "history.csv")

    if cfg.use_pts:
        with open(hist_path, "w", encoding="utf-8") as f:
            f.write("phase,epoch,train_loss,train_acc,train_f1,val_loss,val_acc,val_f1,lr_bb,lr_head,mode\n")

            print("\n=== PTS Phase 1: head only ===")
            run_phase(
                "p1_head_only",
                model,
                train_loader,
                val_loader,
                device,
                cfg,
                cfg.phase1_epochs,
                cfg.lr_bb_p1,
                cfg.lr_head_p1,
                "head_only",
                f,
            )

            print("\n=== PTS Phase 2: unfreeze last k blocks ===")
            run_phase(
                "p2_last_k",
                model,
                train_loader,
                val_loader,
                device,
                cfg,
                cfg.phase2_epochs,
                cfg.lr_bb_p2,
                cfg.lr_head_p2,
                "last_k",
                f,
            )

            print("\n=== PTS Phase 3: unfreeze all ===")
            best_p3 = run_phase(
                "p3_all",
                model,
                train_loader,
                val_loader,
                device,
                cfg,
                cfg.phase3_epochs,
                cfg.lr_bb_p3,
                cfg.lr_head_p3,
                "all",
                f,
            )

        final_path = os.path.join(cfg.out_dir, "best_model.pth")
        ckpt = torch.load(best_p3, map_location=device)
        torch.save(ckpt, final_path)
        print("\nPTS Training Finished. Final model saved:", final_path)
        return

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.startswith("cuda")))
    class_weights = torch.tensor(
        [cfg.loss_real_weight, cfg.loss_spoof_weight],
        dtype=torch.float32,
        device=device,
    )

    ce = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    best_path = os.path.join(cfg.out_dir, "best_model.pth")

    with open(hist_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,train_f1,val_loss,val_acc,val_f1,lr,freeze\n")

        for epoch in range(1, cfg.epochs + 1):
            freeze = epoch <= cfg.freeze_backbone_epochs
            set_freeze_backbone_compat(model, freeze)

            model.train()
            total_loss = 0.0
            all_logits, all_y = [], []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
            for batch in pbar:
                x, y, _, behav = split_batch(batch)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                behav = None if behav is None else behav.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    logits = model(x, behav)
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

                pbar.set_postfix(
                    loss=total_loss / len(y_cat),
                    acc=train_acc,
                    f1=train_f1,
                    lr=float(opt.param_groups[0]["lr"]),
                    freeze=int(freeze),
                )

            scheduler.step()

            logits_cat = torch.cat(all_logits, dim=0)
            y_cat = torch.cat(all_y, dim=0)
            train_acc, train_f1 = metrics_from_logits(logits_cat, y_cat)
            train_loss = total_loss / len(y_cat)

            val_m = evaluate(model, val_loader, device, cfg)
            print(
                f"[baseline] epoch={epoch} train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
                f"val_loss={val_m['loss']:.4f} acc={val_m['acc']:.4f} f1={val_m['f1']:.4f}"
            )

            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},{train_f1:.6f},"
                f"{val_m['loss']:.6f},{val_m['acc']:.6f},{val_m['f1']:.6f},"
                f"{float(opt.param_groups[0]['lr']):.8f},{int(freeze)}\n"
            )
            f.flush()

            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, best_path)

    print(f"Baseline done. Best val_f1={best_val_f1:.4f} saved={best_path}")


if __name__ == "__main__":
    main()