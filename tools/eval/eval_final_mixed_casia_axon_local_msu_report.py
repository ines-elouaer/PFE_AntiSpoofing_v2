from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD
from src.deep_learning.datasets_sequence import CASIASequenceDataset

T = 16
IMG_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0

USE_BEHAV = True
BEHAV_DIM = 9
BEHAV_HIDDEN = 16
TEMPORAL_POOL = "median"


def resolve_path(p):
    p = Path(p)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_state(path: Path):
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model_state", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"[INFO] Checkpoint state chargé depuis la clé: {key}")
                return ckpt[key]
        return ckpt
    raise RuntimeError(f"Format checkpoint non reconnu: {type(ckpt)}")


def clean_state_dict_keys(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def create_model(checkpoint_path: Path, device):
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

    state = clean_state_dict_keys(load_checkpoint_state(checkpoint_path))
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("\n========== CHECKPOINT LOAD ==========")
    print("Checkpoint     :", checkpoint_path)
    print("Missing keys   :", len(missing))
    print("Unexpected keys:", len(unexpected))

    model = model.to(device)
    model.eval()
    return model


def build_dataset(frames_csv, behav_csv):
    return CASIASequenceDataset(
        csv_path=str(frames_csv),
        T=T,
        img_size=IMG_SIZE,
        aug_mode="none",
        sample_mode="center_consecutive",
        seed=42,
        behav_csv=str(behav_csv),
    )


def unpack_batch(batch, device):
    if len(batch) == 4:
        x, y, vids, behav = batch
        return x.to(device), y.to(device), list(vids), behav.to(device)
    if len(batch) == 3:
        x, y, vids = batch
        return x.to(device), y.to(device), list(vids), None
    raise RuntimeError(f"Batch format non supporté: {len(batch)}")


def compute_metrics(y_true, y_pred, scores):
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
        "precision_spoof": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_spoof": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_spoof": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER": float(acer),
    }

    try:
        out["AUC"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["AUC"] = None

    return out


@torch.no_grad()
def predict(model, loader, device):
    criterion = nn.CrossEntropyLoss()

    all_vids, all_y, all_pred, all_scores = [], [], [], []
    total_loss = 0.0
    total_samples = 0

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

        all_vids.extend(vids)
        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_scores.extend(scores.cpu().numpy().tolist())

    pred_df = pd.DataFrame({
        "video_id": all_vids,
        "label": all_y,
        "pred_label": all_pred,
        "score_spoof": all_scores,
    })

    metrics = compute_metrics(all_y, all_pred, all_scores)
    metrics["loss"] = float(total_loss / max(1, total_samples))

    return pred_df, metrics


def load_meta(frames_csv):
    frames = pd.read_csv(frames_csv)
    meta = (
        frames.sort_values(["video_id", "frame_idx"])
        .groupby("video_id")
        .first()
        .reset_index()
    )

    needed = [
        "video_id", "original_video_id", "label_name", "subject_id",
        "device_id", "condition", "attack_type", "domain",
        "source_dataset", "split"
    ]

    for c in needed:
        if c not in meta.columns:
            meta[c] = "unknown"

    return meta[needed]


def subgroup_metrics(pred_df, group_col):
    rows = []

    for val, g in pred_df.groupby(group_col):
        m = compute_metrics(g["label"], g["pred_label"], g["score_spoof"])
        m[group_col] = val
        m["mean_score"] = float(g["score_spoof"].mean())
        m["min_score"] = float(g["score_spoof"].min())
        m["max_score"] = float(g["score_spoof"].max())
        rows.append(m)

    if not rows:
        return pd.DataFrame()

    cols = [group_col, "total", "real_count", "spoof_count", "accuracy", "APCER", "BPCER", "ACER", "AUC", "tn_real", "fp_real_as_spoof", "fn_spoof_as_real", "tp_spoof", "mean_score", "min_score", "max_score"]
    return pd.DataFrame(rows)[cols]


def attack_summary(pred_df):
    return subgroup_metrics(pred_df, "attack_type").rename(columns={"attack_type": "attack_type"})


def save_block(name, df, out_root):
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(df["label"], df["pred_label"], df["score_spoof"])
    attack = attack_summary(df)

    df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")
    attack.to_csv(out_dir / "errors_by_attack_type.csv", index=False, encoding="utf-8")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append(f"========== {name.upper()} FINAL EVALUATION ==========")
    for k, v in metrics.items():
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("========== CONFUSION DETAILS ==========")
    lines.append(f"TN REAL correct  : {metrics['tn_real']}")
    lines.append(f"FP REAL as SPOOF : {metrics['fp_real_as_spoof']}")
    lines.append(f"FN SPOOF as REAL : {metrics['fn_spoof_as_real']}")
    lines.append(f"TP SPOOF correct : {metrics['tp_spoof']}")

    lines.append("")
    lines.append("========== ATTACK TYPE SUMMARY ==========")
    lines.append(attack.to_string(index=False) if len(attack) else "No attack summary.")

    (out_dir / "results.txt").write_text("\n".join(lines), encoding="utf-8")

    return {
        "metrics": metrics,
        "outputs": {
            "predictions": str(out_dir / "predictions.csv"),
            "metrics": str(out_dir / "metrics.json"),
            "attack_analysis": str(out_dir / "errors_by_attack_type.csv"),
            "results_txt": str(out_dir / "results.txt"),
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--behav_csv", required=True)
    parser.add_argument("--out_root", required=True)

    args = parser.parse_args()

    model_path = resolve_path(args.model)
    frames_csv = resolve_path(args.frames_csv)
    behav_csv = resolve_path(args.behav_csv)
    out_root = resolve_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for p in [model_path, frames_csv, behav_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable: {p}")

    print("========== FINAL EVAL CONFIG ==========")
    print("Model     :", model_path)
    print("Frames CSV:", frames_csv)
    print("Behav CSV :", behav_csv)
    print("Out root  :", out_root)

    device = get_device()
    print("Device    :", device)

    ds = build_dataset(frames_csv, behav_csv)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_model(model_path, device)
    pred_df, global_metrics = predict(model, loader, device)

    meta = load_meta(frames_csv)
    pred_df = pred_df.merge(meta, on="video_id", how="left")
    pred_df["pred_label_name"] = pred_df["pred_label"].map({0: "REAL", 1: "SPOOF"})
    pred_df["decision_correct"] = pred_df["label"] == pred_df["pred_label"]

    results = {
        "model": str(model_path),
        "mixed_test": None,
        "source_datasets": {},
        "outputs": {},
    }

    mixed_block = save_block("mixed_test", pred_df, out_root)
    results["mixed_test"] = mixed_block["metrics"]
    results["outputs"]["mixed_test"] = mixed_block["outputs"]

    name_map = {
        "CASIA": "casia_test",
        "AXON": "axon_test",
        "LOCAL_REAL": "local_real_test",
        "MSU_MFSD": "msu_mfsd_test",
    }

    for source, g in pred_df.groupby("source_dataset"):
        block_name = name_map.get(source, f"{source.lower()}_test")
        block = save_block(block_name, g.copy(), out_root)
        results["source_datasets"][source] = block["metrics"]
        results["outputs"][block_name] = block["outputs"]

    by_source = subgroup_metrics(pred_df, "source_dataset")
    by_device = subgroup_metrics(pred_df, "device_id")
    by_attack = subgroup_metrics(pred_df, "attack_type")

    by_source.to_csv(out_root / "metrics_by_source_dataset.csv", index=False, encoding="utf-8")
    by_device.to_csv(out_root / "metrics_by_device.csv", index=False, encoding="utf-8")
    by_attack.to_csv(out_root / "metrics_by_attack_type.csv", index=False, encoding="utf-8")

    results["outputs"]["metrics_by_source_dataset"] = str(out_root / "metrics_by_source_dataset.csv")
    results["outputs"]["metrics_by_device"] = str(out_root / "metrics_by_device.csv")
    results["outputs"]["metrics_by_attack_type"] = str(out_root / "metrics_by_attack_type.csv")

    with open(out_root / "comparison_before_after.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n========== MIXED TEST FINAL METRICS ==========")
    for k, v in results["mixed_test"].items():
        print(f"{k}: {v}")

    print("\n========== BY SOURCE_DATASET ==========")
    print(by_source.to_string(index=False))

    print("\n========== BY DEVICE ==========")
    print(by_device.to_string(index=False))

    print("\n========== ERRORS ==========")
    errors = pred_df[~pred_df["decision_correct"]].copy()
    if len(errors) == 0:
        print("Aucune erreur.")
    else:
        cols = [
            "video_id", "original_video_id", "label_name", "pred_label_name",
            "score_spoof", "source_dataset", "device_id", "subject_id", "attack_type"
        ]
        print(errors[cols].sort_values(["source_dataset", "score_spoof"]).to_string(index=False))

    print("\n[OK] Évaluation finale terminée.")
    print("Report root:", out_root)


if __name__ == "__main__":
    main()