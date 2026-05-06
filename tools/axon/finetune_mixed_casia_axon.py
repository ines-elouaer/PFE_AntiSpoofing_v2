from pathlib import Path
import sys
import json
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD
from src.deep_learning.datasets_sequence import CASIASequenceDataset


# ==========================================================
# CONFIG
# ==========================================================

SEED = 42

T = 16
IMG_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0

USE_BEHAV = True
BEHAV_DIM = 9
BEHAV_HIDDEN = 16  

TEMPORAL_POOL = "median"

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 5

LR_PHASE1 = 1e-4
LR_PHASE2 = 1e-5

WEIGHT_DECAY = 1e-4

MIN_COUNT_SAMPLER = 10

CHECKPOINT_CASIA = (
    PROJECT_ROOT
    / "experiments"
    / "step2_sampling"
    / "deep_behav_no_pts_consecutive"
    / "deep_behav_no_pts_consecutive_seed42"
    / "best_model.pth"
)

MIXED_DIR = PROJECT_ROOT / "data" / "mixed_casia_axon"

TRAIN_FRAMES = MIXED_DIR / "mixed_train_frames.csv"
VAL_FRAMES = MIXED_DIR / "mixed_val_frames.csv"
TRAIN_BEHAV = MIXED_DIR / "mixed_train_behav.csv"
VAL_BEHAV = MIXED_DIR / "mixed_val_behav.csv"

OUT_DIR = PROJECT_ROOT / "experiments" / "mixed_casia_axon" / "seed42"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUT_DIR / "best_model_mixed_casia_axon.pth"
LAST_MODEL_PATH = OUT_DIR / "last_model_mixed_casia_axon.pth"
HISTORY_PATH = OUT_DIR / "training_history.json"
CONFIG_PATH = OUT_DIR / "config.json"


# ==========================================================
# UTILS
# ==========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_state(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {path}")

    ckpt = torch.load(str(path), map_location="cpu")

    if isinstance(ckpt, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model_state",
            "model",
            "net",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"[INFO] Checkpoint state chargé depuis la clé: {key}")
                return ckpt[key], ckpt

    if isinstance(ckpt, dict):
        print("[WARN] Aucune clé standard trouvée, tentative d'utiliser le checkpoint comme state_dict.")
        return ckpt, ckpt

    raise RuntimeError(f"Format checkpoint non reconnu: {type(ckpt)}")

def clean_state_dict_keys(state_dict):
    """
    Corrige les clés si le modèle a été sauvegardé avec DataParallel.
    """
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    return new_state


def create_model(device):
    """
    Crée le modèle CNN+LSTM+Behavior et charge le checkpoint CASIA.
    """

    model = CNN_LSTM_PAD(
    hidden=256,
    num_layers=1,
    bidir=False,
    lstm_dropout=0.2,
    head_dropout=0.5,
    pretrained_backbone=True,
    temporal_pool=TEMPORAL_POOL,
    use_behav=USE_BEHAV,
    behav_dim=BEHAV_DIM,
    behav_hidden=BEHAV_HIDDEN,
)

    state_dict, raw_ckpt = load_checkpoint_state(CHECKPOINT_CASIA)
    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("\n========== CHECKPOINT LOAD ==========")
    print(f"Checkpoint      : {CHECKPOINT_CASIA}")
    print(f"Missing keys    : {len(missing)}")
    print(f"Unexpected keys : {len(unexpected)}")

    if len(missing) > 0:
        print("Missing examples:", missing[:10])

    if len(unexpected) > 0:
        print("Unexpected examples:", unexpected[:10])

    if len(missing) > 20 or len(unexpected) > 20:
        print("\n[WARN] Beaucoup de clés manquantes/inattendues.")
        print("[WARN] Le checkpoint n'est peut-être pas chargé correctement.")
        print("[WARN] Vérifie l'architecture du modèle et la clé du checkpoint.")

    else:
        print("[OK] Checkpoint CASIA chargé correctement.")

    model = model.to(device)
    return model

def build_dataset(csv_path, behav_path, aug_mode, sample_mode):
    return CASIASequenceDataset(
        csv_path=str(csv_path),
        T=T,
        img_size=IMG_SIZE,
        aug_mode=aug_mode,
        sample_mode=sample_mode,
        seed=SEED,
        behav_csv=str(behav_path),
    )


def load_video_meta(frames_csv: Path):
    df = pd.read_csv(frames_csv)

    video_df = (
        df.sort_values(["video_id", "frame_idx"])
        .groupby("video_id")
        .first()
        .reset_index()
    )

    return video_df


def build_stratified_sampler(dataset, frames_csv: Path, min_count: int = 10):
    """
    Sampler stratifié :
    - distingue domain
    - distingue REAL
    - distingue chaque attack_type SPOOF

    Objectif :
    éviter que le modèle voie seulement les attaques fréquentes.
    """

    video_df = load_video_meta(frames_csv)
    meta_by_vid = video_df.set_index("video_id").to_dict(orient="index")

    groups = []
    labels = []

    for vid in dataset.video_ids:
        meta = meta_by_vid.get(vid, {})

        label = int(meta.get("label", dataset.labels_by_vid[vid]))
        domain = str(meta.get("domain", "UNKNOWN"))
        attack_type = str(meta.get("attack_type", "unknown"))

        labels.append(label)

        if label == 0:
            group = f"{domain}::REAL"
        else:
            group = f"{domain}::SPOOF::{attack_type}"

        groups.append(group)

    counts = Counter(groups)

    weights = []

    for group in groups:
        count = counts[group]
        weight = 1.0 / max(count, min_count)
        weights.append(weight)

    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )

    print("\n========== STRATIFIED SAMPLER GROUPS ==========")
    for k, v in sorted(counts.items(), key=lambda x: x[0]):
        print(f"{k:<35}: {v}")

    return sampler, counts


def compute_class_weights(dataset, device):
    labels = [dataset.labels_by_vid[vid] for vid in dataset.video_ids]
    counts = Counter(labels)

    n_real = counts.get(0, 1)
    n_spoof = counts.get(1, 1)
    total = n_real + n_spoof

    weight_real = total / (2.0 * n_real)
    weight_spoof = total / (2.0 * n_spoof)

    weights = torch.tensor([weight_real, weight_spoof], dtype=torch.float32).to(device)

    print("\n========== CLASS WEIGHTS ==========")
    print(f"REAL count  : {n_real}")
    print(f"SPOOF count : {n_spoof}")
    print(f"weight REAL : {weight_real:.4f}")
    print(f"weight SPOOF: {weight_spoof:.4f}")

    return weights


def set_trainable_phase1(model):
    """
    Phase 1 :
    - backbone gelé
    - LSTM + behavior MLP + head entraînés
    """
    for p in model.parameters():
        p.requires_grad = True

    model.freeze_all_backbone()

    print("\n========== PHASE 1 TRAINABLE PARAMS ==========")
    print_trainable_summary(model)


def set_trainable_phase2(model, last_k: int = 3):
    """
    Phase 2 :
    - derniers blocs du backbone dégelés
    - LSTM + behavior MLP + head entraînés
    """
    for p in model.parameters():
        p.requires_grad = True

    model.unfreeze_last_k_backbone_blocks(k=last_k)

    print("\n========== PHASE 2 TRAINABLE PARAMS ==========")
    print_trainable_summary(model)


def print_trainable_summary(model):
    total = 0
    trainable = 0

    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {total - trainable:,}")


def make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )


def compute_pad_metrics(y_true, y_pred, scores):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    scores = np.array(scores).astype(float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

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
        "precision_spoof": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_spoof": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_spoof": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER": float(acer),
    }

    try:
        out["AUC"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["AUC"] = None

    return out


def unpack_batch(batch, device):
    """
    Ton dataset retourne :
    - sans behavior : x, y, vid
    - avec behavior : x, y, vid, behav
    """

    if len(batch) == 4:
        x, y, vid, behav = batch
        x = x.to(device)
        y = y.to(device)
        behav = behav.to(device)
        return x, y, vid, behav

    if len(batch) == 3:
        x, y, vid = batch
        x = x.to(device)
        y = y.to(device)
        return x, y, vid, None

    raise RuntimeError(f"Batch format non supporté: len={len(batch)}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        x, y, _, behav = unpack_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x, behav=behav)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_y = []
    all_pred = []
    all_scores = []
    all_vids = []

    for batch in loader:
        x, y, vids, behav = unpack_batch(batch, device)

        logits = model(x, behav=behav)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1]
        pred = torch.argmax(logits, dim=1)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        all_y.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())
        all_scores.extend(scores.detach().cpu().numpy().tolist())
        all_vids.extend(list(vids))

    metrics = compute_pad_metrics(all_y, all_pred, all_scores)
    metrics["loss"] = float(total_loss / max(1, total_samples))

    pred_df = pd.DataFrame({
        "video_id": all_vids,
        "label": all_y,
        "pred_label": all_pred,
        "score_spoof": all_scores,
    })

    return metrics, pred_df


def save_checkpoint(path, model, optimizer, epoch, phase, metrics, config):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "phase": phase,
        "metrics": metrics,
        "config": config,
    }

    torch.save(ckpt, str(path))


def print_metrics(prefix, metrics):
    print(
        f"{prefix} | "
        f"loss={metrics['loss']:.4f} | "
        f"acc={metrics['accuracy']:.4f} | "
        f"APCER={metrics['APCER']:.4f} | "
        f"BPCER={metrics['BPCER']:.4f} | "
        f"ACER={metrics['ACER']:.4f} | "
        f"AUC={metrics['AUC']}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():
    set_seed(SEED)
    device = get_device()

    print("========== CONFIG ==========")
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Device           : {device}")
    print(f"Checkpoint CASIA : {CHECKPOINT_CASIA}")
    print(f"Train frames     : {TRAIN_FRAMES}")
    print(f"Val frames       : {VAL_FRAMES}")
    print(f"Train behav      : {TRAIN_BEHAV}")
    print(f"Val behav        : {VAL_BEHAV}")
    print(f"Output dir       : {OUT_DIR}")

    for p in [CHECKPOINT_CASIA, TRAIN_FRAMES, VAL_FRAMES, TRAIN_BEHAV, VAL_BEHAV]:
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable: {p}")

    config = {
        "seed": SEED,
        "T": T,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "use_behav": USE_BEHAV,
        "behav_dim": BEHAV_DIM,
        "behav_hidden": BEHAV_HIDDEN,
        "temporal_pool": TEMPORAL_POOL,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "lr_phase1": LR_PHASE1,
        "lr_phase2": LR_PHASE2,
        "weight_decay": WEIGHT_DECAY,
        "checkpoint_casia": str(CHECKPOINT_CASIA),
        "train_frames": str(TRAIN_FRAMES),
        "val_frames": str(VAL_FRAMES),
        "train_behav": str(TRAIN_BEHAV),
        "val_behav": str(VAL_BEHAV),
        "sampler": "domain_label_attack_type_stratified",
        "class_weights": True,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n========== DATASETS ==========")

    train_ds = build_dataset(
    csv_path=TRAIN_FRAMES,
    behav_path=TRAIN_BEHAV,
    aug_mode="none",
    sample_mode="center_consecutive",
    )

    val_ds = build_dataset(
        csv_path=VAL_FRAMES,
        behav_path=VAL_BEHAV,
        aug_mode="none",
        sample_mode="center_consecutive",
    )

    print(f"Train videos : {len(train_ds)}")
    print(f"Val videos   : {len(val_ds)}")

    sampler, sampler_counts = build_stratified_sampler(
        dataset=train_ds,
        frames_csv=TRAIN_FRAMES,
        min_count=MIN_COUNT_SAMPLER,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    class_weights = compute_class_weights(train_ds, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = create_model(device)

    best_acer = 999.0
    best_epoch = -1
    best_phase = None

    history = []

    # ======================================================
    # PHASE 1
    # ======================================================

    print("\n" + "=" * 70)
    print("PHASE 1 — BACKBONE GELÉ")
    print("=" * 70)

    set_trainable_phase1(model)
    optimizer = make_optimizer(model, lr=LR_PHASE1)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_pred_df = evaluate(model, val_loader, criterion, device)

        val_metrics["train_loss"] = float(train_loss)
        val_metrics["epoch"] = int(epoch)
        val_metrics["phase"] = "phase1_backbone_frozen"

        history.append(val_metrics)

        print_metrics(
            prefix=f"[PHASE 1][EPOCH {epoch}/{PHASE1_EPOCHS}] train_loss={train_loss:.4f}",
            metrics=val_metrics,
        )

        val_pred_df.to_csv(
            OUT_DIR / f"val_predictions_phase1_epoch{epoch}.csv",
            index=False,
            encoding="utf-8",
        )

        if val_metrics["ACER"] < best_acer:
            best_acer = val_metrics["ACER"]
            best_epoch = epoch
            best_phase = "phase1_backbone_frozen"

            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                phase=best_phase,
                metrics=val_metrics,
                config=config,
            )

            print(f"[OK] Nouveau meilleur modèle sauvegardé | ACER={best_acer:.4f}")

    # ======================================================
    # PHASE 2
    # ======================================================

    print("\n" + "=" * 70)
    print("PHASE 2 — UNFREEZE DERNIERS BLOCS CNN")
    print("=" * 70)

    set_trainable_phase2(model, last_k=3)
    optimizer = make_optimizer(model, lr=LR_PHASE2)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_pred_df = evaluate(model, val_loader, criterion, device)

        global_epoch = PHASE1_EPOCHS + epoch

        val_metrics["train_loss"] = float(train_loss)
        val_metrics["epoch"] = int(global_epoch)
        val_metrics["phase"] = "phase2_unfreeze_last3"

        history.append(val_metrics)

        print_metrics(
            prefix=f"[PHASE 2][EPOCH {epoch}/{PHASE2_EPOCHS}] train_loss={train_loss:.4f}",
            metrics=val_metrics,
        )

        val_pred_df.to_csv(
            OUT_DIR / f"val_predictions_phase2_epoch{epoch}.csv",
            index=False,
            encoding="utf-8",
        )

        if val_metrics["ACER"] < best_acer:
            best_acer = val_metrics["ACER"]
            best_epoch = global_epoch
            best_phase = "phase2_unfreeze_last3"

            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                epoch=global_epoch,
                phase=best_phase,
                metrics=val_metrics,
                config=config,
            )

            print(f"[OK] Nouveau meilleur modèle sauvegardé | ACER={best_acer:.4f}")

    save_checkpoint(
        path=LAST_MODEL_PATH,
        model=model,
        optimizer=optimizer,
        epoch=PHASE1_EPOCHS + PHASE2_EPOCHS,
        phase="last",
        metrics=history[-1],
        config=config,
    )

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print("\n========== TRAINING FINISHED ==========")
    print(f"Best model : {BEST_MODEL_PATH}")
    print(f"Last model : {LAST_MODEL_PATH}")
    print(f"History    : {HISTORY_PATH}")
    print(f"Best ACER  : {best_acer:.4f}")
    print(f"Best epoch : {best_epoch}")
    print(f"Best phase : {best_phase}")


if __name__ == "__main__":
    main()