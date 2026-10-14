from pathlib import Path
import sys
import argparse
import json
import pickle
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.v8.v8_dual_temporal_lib import (
    DualTemporalDataset,
    DualTemporalPAD,
    BEHAV_SEQ_COLS,
    compute_metrics,
)

NUM_WORKERS = 0


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_labels(ds):
    labels = []
    for i in range(len(ds)):
        labels.append(ds.get_label(i))
    return np.asarray(labels, dtype=int)


def build_sampler(labels):
    real_count = np.sum(labels == 0)
    spoof_count = np.sum(labels == 1)

    counts = np.array([real_count, spoof_count], dtype=np.float64)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_loss(labels, device):
    real_count = np.sum(labels == 0)
    spoof_count = np.sum(labels == 1)
    total = len(labels)

    w_real = total / (2.0 * real_count)
    w_spoof = total / (2.0 * spoof_count)

    weights = torch.tensor([w_real, w_spoof], dtype=torch.float32).to(device)

    print("Class counts REAL/SPOOF:", int(real_count), int(spoof_count))
    print("Loss weights:", weights.detach().cpu().numpy())

    return nn.CrossEntropyLoss(weight=weights)


def compute_metrics_from_scores(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)
    return compute_metrics(y_true, pred, scores)


def find_best_threshold(y_true, scores):
    best = None

    for thr in np.linspace(0.05, 0.95, 181):
        m = compute_metrics_from_scores(y_true, scores, float(thr))

        # Objectif PAD :
        # 1. minimiser ACER
        # 2. minimiser APCER
        # 3. maximiser F1 spoof
        key = (m["ACER"], m["APCER"], -m["f1_spoof"])

        if best is None or key < best["key"]:
            best = {
                "threshold": float(thr),
                "metrics": m,
                "key": key,
            }

    return best


def set_trainable_phase(model, phase):
    for p in model.parameters():
        p.requires_grad = False

    if phase == 1:
        # On entraîne seulement la branche comportementale temporelle + fusion + classifier.
        keywords = [
            "behavior_lstm",
            "fusion",
            "classifier",
        ]

    elif phase == 2:
        # On ajoute le LSTM vidéo, mais on garde le backbone MobileNetV3 gelé.
        keywords = [
            "video_lstm",
            "behavior_lstm",
            "fusion",
            "classifier",
        ]

    elif phase == 3:
        # Fine-tuning complet optionnel.
        for p in model.parameters():
            p.requires_grad = True
        keywords = None

    else:
        raise ValueError("phase must be 1, 2, or 3")

    if keywords is not None:
        for name, p in model.named_parameters():
            if any(k in name for k in keywords):
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"\\n========== PHASE {phase} TRAINABLE ==========")
    print("Trainable params:", trainable)
    print("Total params    :", total)
    print("Ratio           :", trainable / max(total, 1))

    print("Trainable modules:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("  ", name)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    y_all = []
    pred_all = []
    score_all = []

    total_loss = 0.0
    total_samples = 0

    for x, behavior_seq, y, vids in loader:
        x = x.to(device)
        behavior_seq = behavior_seq.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x, behavior_seq)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        probs = torch.softmax(logits.detach(), dim=1)
        scores = probs[:, 1]
        pred = torch.argmax(logits.detach(), dim=1)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs

        y_all.extend(y.detach().cpu().numpy().tolist())
        pred_all.extend(pred.detach().cpu().numpy().tolist())
        score_all.extend(scores.detach().cpu().numpy().tolist())

    metrics = compute_metrics(y_all, pred_all, score_all)
    metrics["loss"] = float(total_loss / max(total_samples, 1))

    return metrics, np.asarray(y_all), np.asarray(score_all)


@torch.no_grad()
def predict(model, loader, device, threshold=0.5):
    model.eval()

    rows = []
    y_all = []
    pred_all = []
    score_all = []

    for x, behavior_seq, y, vids in loader:
        x = x.to(device)
        behavior_seq = behavior_seq.to(device)

        logits = model(x, behavior_seq)
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1]
        pred = (scores >= threshold).long()

        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()
        score_np = scores.cpu().numpy()

        for vid, yy, pp, ss in zip(vids, y_np, pred_np, score_np):
            rows.append({
                "video_id": str(vid),
                "label": int(yy),
                "pred_label": int(pp),
                "score_spoof": float(ss),
            })

        y_all.extend(y_np.tolist())
        pred_all.extend(pred_np.tolist())
        score_all.extend(score_np.tolist())

    metrics = compute_metrics(y_all, pred_all, score_all)
    metrics["threshold"] = float(threshold)

    return pd.DataFrame(rows), metrics


def train_phase(
    model,
    phase,
    train_loader,
    val_loader,
    criterion,
    device,
    epochs,
    lr,
    patience,
    out_dir,
    global_best,
):
    set_trainable_phase(model, phase)

    params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    wait = 0
    history = []

    local_best_acer = 999.0

    for epoch in range(1, epochs + 1):
        train_m, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_m, val_y, val_scores = run_epoch(model, val_loader, criterion, None, device, train=False)

        best_thr = find_best_threshold(val_y, val_scores)
        val_m_best = best_thr["metrics"]
        val_thr = best_thr["threshold"]

        row = {
            "phase": phase,
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],

            "train_ACER_05": train_m["ACER"],
            "val_ACER_05": val_m["ACER"],

            "val_ACER_best_thr": val_m_best["ACER"],
            "val_APCER_best_thr": val_m_best["APCER"],
            "val_BPCER_best_thr": val_m_best["BPCER"],
            "val_F1_best_thr": val_m_best["f1_spoof"],
            "val_AUC": val_m["AUC"],
            "best_threshold": val_thr,
        }

        history.append(row)

        print(
            f"PHASE {phase} | Epoch {epoch:03d} | "
            f"train_ACER@0.5={train_m['ACER']:.4f} | "
            f"val_ACER@0.5={val_m['ACER']:.4f} | "
            f"val_ACER@best={val_m_best['ACER']:.4f} | "
            f"thr={val_thr:.3f} | "
            f"val_APCER={val_m_best['APCER']:.4f} | "
            f"val_BPCER={val_m_best['BPCER']:.4f} | "
            f"val_AUC={val_m['AUC']}"
        )

        # Local early stopping sur ACER validation avec seuil optimisé
        current_acer = val_m_best["ACER"]

        if current_acer < local_best_acer:
            local_best_acer = current_acer
            wait = 0
        else:
            wait += 1

        # Global best toutes phases confondues
        key = (
            val_m_best["ACER"],
            val_m_best["APCER"],
            -val_m_best["f1_spoof"],
        )

        if global_best["key"] is None or key < global_best["key"]:
            global_best["key"] = key
            global_best["phase"] = phase
            global_best["epoch"] = epoch
            global_best["threshold"] = val_thr
            global_best["val_metrics"] = copy.deepcopy(val_m_best)
            global_best["state_dict"] = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state_dict": global_best["state_dict"],
                    "best_threshold_from_val": global_best["threshold"],
                    "best_phase": global_best["phase"],
                    "best_epoch": global_best["epoch"],
                    "best_val_metrics": global_best["val_metrics"],
                },
                out_dir / "best_model_v8_dual_temporal_v2.pth",
            )

            print(
                f"[GLOBAL BEST] phase={phase} epoch={epoch} "
                f"ACER={val_m_best['ACER']:.4f} threshold={val_thr:.3f}"
            )

        if wait >= patience:
            print(f"[EARLY STOP] phase {phase}")
            break

    pd.DataFrame(history).to_csv(out_dir / f"history_phase{phase}.csv", index=False, encoding="utf-8")

    return model, global_best


def save_split(out_dir, split_name, pred_df, metrics):
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(split_dir / "predictions.csv", index=False, encoding="utf-8")

    with open(split_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "predictions": str(split_dir / "predictions.csv"),
        "metrics": str(split_dir / "metrics.json"),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_frames", required=True)
    parser.add_argument("--val_frames", required=True)
    parser.add_argument("--test_frames", required=True)

    parser.add_argument("--train_behavior_seq", required=True)
    parser.add_argument("--val_behavior_seq", required=True)
    parser.add_argument("--test_behavior_seq", required=True)

    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--video_lstm_hidden", type=int, default=256)
    parser.add_argument("--behavior_lstm_hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.6)

    parser.add_argument("--phase1_epochs", type=int, default=15)
    parser.add_argument("--phase2_epochs", type=int, default=5)
    parser.add_argument("--phase3_epochs", type=int, default=0)

    parser.add_argument("--phase1_lr", type=float, default=0.0005)
    parser.add_argument("--phase2_lr", type=float, default=0.0001)
    parser.add_argument("--phase3_lr", type=float, default=0.00001)

    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    train_ds = DualTemporalDataset(
        args.train_frames,
        args.train_behavior_seq,
        T=args.T,
        img_size=args.img_size,
        fit_scaler=True,
    )

    val_ds = DualTemporalDataset(
        args.val_frames,
        args.val_behavior_seq,
        T=args.T,
        img_size=args.img_size,
        scaler=train_ds.scaler,
    )

    test_ds = DualTemporalDataset(
        args.test_frames,
        args.test_behavior_seq,
        T=args.T,
        img_size=args.img_size,
        scaler=train_ds.scaler,
    )

    with open(out_dir / "behavior_scaler.pkl", "wb") as f:
        pickle.dump(train_ds.scaler, f)

    train_labels = get_labels(train_ds)
    sampler = build_sampler(train_labels)

    # Loader avec sampler uniquement pour entraînement
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # Loader sans sampler pour évaluation réelle train
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = DualTemporalPAD(
        behavior_dim=len(BEHAV_SEQ_COLS),
        video_lstm_hidden=args.video_lstm_hidden,
        behavior_lstm_hidden=args.behavior_lstm_hidden,
        dropout=args.dropout,
        pretrained_backbone=True,
    ).to(device)

    criterion = build_loss(train_labels, device)

    config = vars(args)
    config["behavior_cols"] = BEHAV_SEQ_COLS
    config["notes"] = {
        "version": "v8_dual_temporal_v2",
        "global_best_checkpoint": True,
        "best_threshold_from_validation": True,
        "phase3_default_disabled": True,
        "behavior_lstm_hidden_reduced": True,
    }

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    global_best = {
        "key": None,
        "phase": None,
        "epoch": None,
        "threshold": 0.5,
        "val_metrics": None,
        "state_dict": None,
    }

    if args.phase1_epochs > 0:
        model, global_best = train_phase(
            model=model,
            phase=1,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epochs=args.phase1_epochs,
            lr=args.phase1_lr,
            patience=args.patience,
            out_dir=out_dir,
            global_best=global_best,
        )

    if args.phase2_epochs > 0:
        model, global_best = train_phase(
            model=model,
            phase=2,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epochs=args.phase2_epochs,
            lr=args.phase2_lr,
            patience=args.patience,
            out_dir=out_dir,
            global_best=global_best,
        )

    if args.phase3_epochs > 0:
        model, global_best = train_phase(
            model=model,
            phase=3,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epochs=args.phase3_epochs,
            lr=args.phase3_lr,
            patience=args.patience,
            out_dir=out_dir,
            global_best=global_best,
        )

    if global_best["state_dict"] is None:
        raise RuntimeError("No global best model was saved.")

    # Recharger le meilleur modèle global toutes phases confondues
    model.load_state_dict(global_best["state_dict"])
    best_threshold = float(global_best["threshold"])

    print("\\n========== GLOBAL BEST MODEL ==========")
    print("Best phase:", global_best["phase"])
    print("Best epoch:", global_best["epoch"])
    print("Best threshold:", best_threshold)
    print("Best val metrics:")
    print(json.dumps(global_best["val_metrics"], indent=2))

    train_pred, train_metrics = predict(model, train_eval_loader, device, threshold=best_threshold)
    val_pred, val_metrics = predict(model, val_loader, device, threshold=best_threshold)
    test_pred, test_metrics = predict(model, test_loader, device, threshold=best_threshold)

    outputs = {
        "train": save_split(out_dir, "mixed_train", train_pred, train_metrics),
        "val": save_split(out_dir, "mixed_val", val_pred, val_metrics),
        "test": save_split(out_dir, "mixed_test", test_pred, test_metrics),
    }

    report = {
        "model": str(out_dir / "best_model_v8_dual_temporal_v2.pth"),
        "best_phase": global_best["phase"],
        "best_epoch": global_best["epoch"],
        "best_threshold_from_val": best_threshold,
        "best_val_metrics_during_training": global_best["val_metrics"],
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "outputs": outputs,
    }

    with open(out_dir / "final_metrics_v8_v2.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\\n========== V8.2 FINAL TEST METRICS ==========")
    print(json.dumps(test_metrics, indent=2))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
