from pathlib import Path
import sys
import json
import warnings
import csv

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


warnings.filterwarnings("ignore")


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

MIXED_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "mixed_casia_axon"
    / "seed42"
    / "best_model_mixed_casia_axon.pth"
)

AXON_TEST_FRAMES = (
    PROJECT_ROOT
    / "data"
    / "axon_prepared"
    / "manifests"
    / "splits"
    / "axon_video_frames_test.csv"
)

AXON_TEST_BEHAV = (
    PROJECT_ROOT
    / "data"
    / "axon_prepared"
    / "manifests"
    / "splits"
    / "axon_video_behav_test.csv"
)

CASIA_TEST_FRAMES_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "casia" / "splits_subject" / "test.csv",
    PROJECT_ROOT / "data" / "processed" / "casia" / "splits_subject" / "casia_test.csv",
    PROJECT_ROOT / "data" / "processed" / "casia" / "splits" / "test.csv",
    PROJECT_ROOT / "data" / "processed" / "casia" / "test.csv",
]

CASIA_TEST_BEHAV_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "casia" / "behav" / "test_behav.csv",
    PROJECT_ROOT / "data" / "processed" / "casia" / "behav" / "behav_test.csv",
    PROJECT_ROOT / "data" / "processed" / "casia" / "behav" / "casia_test_behav.csv",
    PROJECT_ROOT / "data" / "processed" / "casia" / "behav" / "test.csv",
]

OUT_DIR = PROJECT_ROOT / "reports" / "mixed_casia_axon_final_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# HELPERS
# ==========================================================

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


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
        print("[WARN] Aucune clé standard trouvée, tentative checkpoint complet.")
        return ckpt, ckpt

    raise RuntimeError(f"Format checkpoint non reconnu: {type(ckpt)}")


def clean_state_dict_keys(state_dict):
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    return new_state


def create_model(device):
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

    state_dict, raw_ckpt = load_checkpoint_state(MIXED_MODEL_PATH)
    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("\n========== CHECKPOINT LOAD ==========")
    print(f"Checkpoint      : {MIXED_MODEL_PATH}")
    print(f"Missing keys    : {len(missing)}")
    print(f"Unexpected keys : {len(unexpected)}")

    if len(missing) > 0:
        print("Missing examples:", missing[:10])

    if len(unexpected) > 0:
        print("Unexpected examples:", unexpected[:10])

    if len(missing) == 0 and len(unexpected) == 0:
        print("[OK] Modèle mixte chargé correctement.")
    else:
        print("[WARN] Checkpoint chargé avec clés manquantes/inattendues.")

    model = model.to(device)
    model.eval()
    return model, raw_ckpt


def compute_pad_metrics(y_true, y_pred, scores):
    """
    label 0 = REAL
    label 1 = SPOOF

    APCER = spoof accepté à tort = spoof prédit REAL
    BPCER = réel rejeté à tort = real prédit SPOOF
    ACER = moyenne(APCER, BPCER)
    """

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

    metrics = {
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
        metrics["AUC"] = float(roc_auc_score(y_true, scores))
    except Exception:
        metrics["AUC"] = None

    return metrics


def banking_decision_video(score: float):
    score = float(score)

    if score < 0.30:
        return "ACCEPT"

    if score < 0.60:
        return "RETRY"

    return "REJECT"


def load_video_meta(frames_csv: Path):
    df = pd.read_csv(frames_csv)

    video_df = (
        df.sort_values(["video_id", "frame_idx"])
        .groupby("video_id")
        .first()
        .reset_index()
    )

    meta_by_vid = {}

    for _, row in video_df.iterrows():
        vid = str(row["video_id"])

        meta_by_vid[vid] = {
            "video_id": vid,
            "label": int(row["label"]),
            "label_name": str(row.get("label_name", "REAL" if int(row["label"]) == 0 else "SPOOF")),
            "attack_type": str(row.get("attack_type", "unknown")),
            "level": str(row.get("level", "unknown")),
            "domain": str(row.get("domain", "UNKNOWN")),
            "subject_id": str(row.get("subject_id", "unknown")),
        }

    return meta_by_vid


def build_dataset(frames_csv: Path, behav_csv: Path):
    return CASIASequenceDataset(
        csv_path=str(frames_csv),
        T=T,
        img_size=IMG_SIZE,
        aug_mode="none",
        sample_mode="center_consecutive",
        seed=SEED,
        behav_csv=str(behav_csv),
    )


def unpack_batch(batch, device):
    if len(batch) == 4:
        x, y, vids, behav = batch
        return x.to(device), y.to(device), list(vids), behav.to(device)

    if len(batch) == 3:
        x, y, vids = batch
        return x.to(device), y.to(device), list(vids), None

    raise RuntimeError(f"Batch format non supporté: len={len(batch)}")


@torch.no_grad()
def evaluate_dataset(
    model,
    frames_csv: Path,
    behav_csv: Path,
    split_name: str,
    device,
):
    if not frames_csv.exists():
        raise FileNotFoundError(f"Frames CSV introuvable: {frames_csv}")

    if not behav_csv.exists():
        raise FileNotFoundError(f"Behavior CSV introuvable: {behav_csv}")

    print("\n" + "=" * 70)
    print(f"EVALUATION : {split_name}")
    print("=" * 70)
    print(f"Frames CSV : {frames_csv}")
    print(f"Behav CSV  : {behav_csv}")

    ds = build_dataset(frames_csv, behav_csv)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    meta_by_vid = load_video_meta(frames_csv)

    all_y = []
    all_pred = []
    all_scores = []
    rows = []

    for batch_idx, batch in enumerate(loader, start=1):
        x, y, vids, behav = unpack_batch(batch, device)

        logits = model(x, behav=behav)
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_cpu = y.detach().cpu().numpy().tolist()
        pred_cpu = preds.detach().cpu().numpy().tolist()
        score_cpu = scores.detach().cpu().numpy().tolist()

        for vid, yt, yp, sc in zip(vids, y_cpu, pred_cpu, score_cpu):
            meta = meta_by_vid.get(str(vid), {})

            rows.append({
                "split": split_name,
                "video_id": str(vid),
                "label": int(yt),
                "label_name": "REAL" if int(yt) == 0 else "SPOOF",
                "pred_label": int(yp),
                "pred_label_name": "REAL" if int(yp) == 0 else "SPOOF",
                "score_spoof": float(sc),
                "banking_decision": banking_decision_video(float(sc)),
                "correct": bool(int(yt) == int(yp)),
                "attack_type": meta.get("attack_type", "unknown"),
                "level": meta.get("level", "unknown"),
                "domain": meta.get("domain", "UNKNOWN"),
                "subject_id": meta.get("subject_id", "unknown"),
            })

        all_y.extend(y_cpu)
        all_pred.extend(pred_cpu)
        all_scores.extend(score_cpu)

        if batch_idx % 25 == 0:
            print(f"[INFO] Batch {batch_idx}/{len(loader)} traité")

    pred_df = pd.DataFrame(rows)
    metrics = compute_pad_metrics(all_y, all_pred, all_scores)

    # Analyse par attack_type
    attack_rows = []

    if "attack_type" in pred_df.columns:
        for attack_type, g in pred_df.groupby("attack_type"):
            yt = g["label"].astype(int).values
            yp = g["pred_label"].astype(int).values
            sc = g["score_spoof"].astype(float).values

            m = compute_pad_metrics(yt, yp, sc)

            attack_rows.append({
                "attack_type": attack_type,
                "n": int(len(g)),
                "accuracy": m["accuracy"],
                "APCER": m["APCER"],
                "BPCER": m["BPCER"],
                "ACER": m["ACER"],
                "f1_spoof": m["f1_spoof"],
                "mean_score": float(np.mean(sc)),
                "min_score": float(np.min(sc)),
                "max_score": float(np.max(sc)),
            })

    attack_df = pd.DataFrame(attack_rows)

    if len(attack_df) > 0:
        attack_df = attack_df.sort_values("n", ascending=False)

    return metrics, pred_df, attack_df


def save_eval_outputs(split_name, metrics, pred_df, attack_df):
    split_dir = OUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    pred_path = split_dir / "predictions.csv"
    metrics_path = split_dir / "metrics.json"
    attack_path = split_dir / "errors_by_attack_type.csv"
    results_path = split_dir / "results.txt"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if attack_df is not None and len(attack_df) > 0:
        attack_df.to_csv(attack_path, index=False, encoding="utf-8")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"========== {split_name.upper()} FINAL EVALUATION ==========\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\n========== CONFUSION DETAILS ==========\n")
        f.write(f"TN REAL correct  : {metrics['tn_real']}\n")
        f.write(f"FP REAL as SPOOF : {metrics['fp_real_as_spoof']}\n")
        f.write(f"FN SPOOF as REAL : {metrics['fn_spoof_as_real']}\n")
        f.write(f"TP SPOOF correct : {metrics['tp_spoof']}\n")

        if attack_df is not None and len(attack_df) > 0:
            f.write("\n========== ATTACK TYPE SUMMARY ==========\n")
            f.write(attack_df.to_string(index=False))

    return {
        "predictions": str(pred_path),
        "metrics": str(metrics_path),
        "attack_analysis": str(attack_path),
        "results_txt": str(results_path),
    }


def print_metrics(title, metrics):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

    keys = [
        "total",
        "real_count",
        "spoof_count",
        "tn_real",
        "fp_real_as_spoof",
        "fn_spoof_as_real",
        "tp_spoof",
        "accuracy",
        "precision_spoof",
        "recall_spoof",
        "f1_spoof",
        "APCER",
        "BPCER",
        "ACER",
        "AUC",
    ]

    for k in keys:
        print(f"{k:<22}: {metrics.get(k)}")


def main():
    device = get_device()

    casia_test_frames = first_existing(CASIA_TEST_FRAMES_CANDIDATES)
    casia_test_behav = first_existing(CASIA_TEST_BEHAV_CANDIDATES)

    if casia_test_frames is None:
        raise FileNotFoundError(
            "CASIA test frames introuvable. Vérifie les chemins dans CASIA_TEST_FRAMES_CANDIDATES."
        )

    if casia_test_behav is None:
        raise FileNotFoundError(
            "CASIA test behavior introuvable. Vérifie les chemins dans CASIA_TEST_BEHAV_CANDIDATES."
        )

    print("========== CONFIG ==========")
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Device             : {device}")
    print(f"Mixed model        : {MIXED_MODEL_PATH}")
    print(f"Axon test frames   : {AXON_TEST_FRAMES}")
    print(f"Axon test behav    : {AXON_TEST_BEHAV}")
    print(f"CASIA test frames  : {casia_test_frames}")
    print(f"CASIA test behav   : {casia_test_behav}")
    print(f"Output dir         : {OUT_DIR}")

    for p in [MIXED_MODEL_PATH, AXON_TEST_FRAMES, AXON_TEST_BEHAV, casia_test_frames, casia_test_behav]:
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable: {p}")

    model, raw_ckpt = create_model(device)

    axon_metrics, axon_pred_df, axon_attack_df = evaluate_dataset(
        model=model,
        frames_csv=AXON_TEST_FRAMES,
        behav_csv=AXON_TEST_BEHAV,
        split_name="axon_test",
        device=device,
    )

    print_metrics("AXON TEST METRICS", axon_metrics)

    axon_outputs = save_eval_outputs(
        split_name="axon_test",
        metrics=axon_metrics,
        pred_df=axon_pred_df,
        attack_df=axon_attack_df,
    )

    casia_metrics, casia_pred_df, casia_attack_df = evaluate_dataset(
        model=model,
        frames_csv=casia_test_frames,
        behav_csv=casia_test_behav,
        split_name="casia_test",
        device=device,
    )

    print_metrics("CASIA TEST METRICS", casia_metrics)

    casia_outputs = save_eval_outputs(
        split_name="casia_test",
        metrics=casia_metrics,
        pred_df=casia_pred_df,
        attack_df=casia_attack_df,
    )

    comparison = {
        "model": str(MIXED_MODEL_PATH),
        "axon_test": axon_metrics,
        "casia_test": casia_metrics,
        "baseline_before_finetuning": {
            "axon_full_casia_checkpoint": {
                "accuracy": 0.7693498452012384,
                "APCER": 0.20483870967741935,
                "BPCER": 0.8461538461538461,
                "ACER": 0.5254962779156327,
                "AUC": 0.47332506203473945,
                "note": "Baseline CASIA checkpoint evaluated on full Axon video set before mixed fine-tuning.",
            }
        },
        "outputs": {
            "axon_test": axon_outputs,
            "casia_test": casia_outputs,
        },
    }

    comparison_path = OUT_DIR / "comparison_before_after.json"

    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print("\n========== FINAL OUTPUTS ==========")
    print(f"Axon predictions  : {axon_outputs['predictions']}")
    print(f"Axon metrics      : {axon_outputs['metrics']}")
    print(f"CASIA predictions : {casia_outputs['predictions']}")
    print(f"CASIA metrics     : {casia_outputs['metrics']}")
    print(f"Comparison JSON   : {comparison_path}")

    print("\n[OK] Évaluation finale modèle mixte terminée.")


if __name__ == "__main__":
    main()