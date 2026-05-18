from pathlib import Path
import sys
import json
import argparse

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


# ==========================================================
# CONFIG
# ==========================================================

T = 16
IMG_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0

USE_BEHAV = True
BEHAV_DIM = 9
BEHAV_HIDDEN = 16
TEMPORAL_POOL = "median"

FEATURE_COLS = [
    "ear_mean",
    "ear_std",
    "ear_min",
    "ear_max",
    "blink_count",
    "motion_mean",
    "motion_std",
    "motion_max",
    "skipped_rate",
]


# ==========================================================
# MODEL UTILS
# ==========================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_state(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {path}")

    ckpt = torch.load(str(path), map_location="cpu")

    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model_state", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"[INFO] Checkpoint state chargé depuis la clé: {key}")
                return ckpt[key], ckpt

    if isinstance(ckpt, dict):
        print("[WARN] Aucune clé standard trouvée, utilisation directe du dict.")
        return ckpt, ckpt

    raise RuntimeError(f"Format checkpoint non reconnu: {type(ckpt)}")


def clean_state_dict_keys(state_dict):
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    return new_state


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

    state_dict, raw_ckpt = load_checkpoint_state(checkpoint_path)
    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("\n========== CHECKPOINT LOAD ==========")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"use_behav       : {USE_BEHAV}")
    print(f"behav_dim       : {BEHAV_DIM}")
    print(f"behav_hidden    : {BEHAV_HIDDEN}")
    print(f"temporal_pool   : {TEMPORAL_POOL}")
    print(f"Missing keys    : {len(missing)}")
    print(f"Unexpected keys : {len(unexpected)}")

    if missing:
        print("Missing examples:", missing[:10])

    if unexpected:
        print("Unexpected examples:", unexpected[:10])

    model = model.to(device)
    model.eval()

    return model


# ==========================================================
# DATASET / METRICS
# ==========================================================
def build_dataset(frames_csv: Path, behav_csv: Path):
    print("\n========== DATASET BUILD ==========")
    print(f"Frames CSV used : {frames_csv}")
    print(f"Behav CSV used  : {behav_csv}")

    ds = CASIASequenceDataset(
        csv_path=str(frames_csv),
        T=T,
        img_size=IMG_SIZE,
        aug_mode="none",
        sample_mode="center_consecutive",
        seed=42,
        behav_csv=str(behav_csv),
    )

    return ds
def unpack_batch(batch, device):
    """
    Récupère un batch du DataLoader et force la compatibilité
    avec le modèle évalué.

    Le checkpoint mixed_casia_axon_local_msu historique attend
    un vecteur comportemental de dimension 9 :

    - ear_mean
    - ear_std
    - ear_min
    - ear_max
    - blink_count
    - motion_mean
    - motion_std
    - motion_max
    - skipped_rate

    Si le Dataset renvoie 15 features, on garde uniquement
    les 9 premières, qui correspondent aux features historiques.
    """

    if len(batch) == 4:
        x, y, vids, behav = batch

        x = x.to(device)
        y = y.to(device)
        behav = behav.to(device)

        if behav is not None:
            if behav.shape[1] < BEHAV_DIM:
                raise RuntimeError(
                    f"Behavior dim invalide: reçu {behav.shape[1]}, "
                    f"attendu au minimum {BEHAV_DIM}"
                )

            if behav.shape[1] > BEHAV_DIM:
                print(
                    f"[WARN] Behavior dim reçue = {behav.shape[1]} ; "
                    f"réduction à BEHAV_DIM = {BEHAV_DIM}"
                )
                behav = behav[:, :BEHAV_DIM]

        return x, y, list(vids), behav

    if len(batch) == 3:
        x, y, vids = batch
        return x.to(device), y.to(device), list(vids), None

    raise RuntimeError(f"Batch format non supporté: len={len(batch)}")
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


@torch.no_grad()
def predict_model(model, loader, device):
    criterion = nn.CrossEntropyLoss()

    all_vids = []
    all_y = []
    all_pred = []
    all_scores = []

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        x, y, vids, behav = unpack_batch(batch, device)
        if total_samples == 0 and behav is not None:
            print("\n========== FIRST BATCH DEBUG ==========")
            print(f"x shape     : {tuple(x.shape)}")
            print(f"y shape     : {tuple(y.shape)}")
            print(f"behav shape : {tuple(behav.shape)}")

        logits = model(x, behav=behav)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1]
        pred = torch.argmax(logits, dim=1)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        all_vids.extend(vids)
        all_y.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())
        all_scores.extend(scores.detach().cpu().numpy().tolist())

    pred_df = pd.DataFrame(
        {
            "video_id": all_vids,
            "label": all_y,
            "pred_label": all_pred,
            "score_spoof": all_scores,
        }
    )

    metrics = compute_pad_metrics(all_y, all_pred, all_scores)
    metrics["loss"] = float(total_loss / max(1, total_samples))

    return pred_df, metrics


def load_video_meta(frames_csv: Path):
    frames = pd.read_csv(frames_csv)

    meta = (
        frames.sort_values(["video_id", "frame_idx"])
        .groupby("video_id")
        .first()
        .reset_index()
    )

    wanted_cols = [
        "video_id",
        "original_video_id",
        "label_name",
        "subject_id",
        "device_id",
        "condition",
        "attack_type",
        "domain",
        "source_dataset",
        "split",
    ]

    for col in wanted_cols:
        if col not in meta.columns:
            meta[col] = "unknown"

    return meta[wanted_cols].copy()
def prepare_behav_csv_for_model(behav_csv: Path, out_root: Path) -> Path:
    """
    Prépare un fichier behavior compatible avec le modèle chargé.

    Le modèle mixed_casia_axon_local_msu historique attend BEHAV_DIM = 9.
    Si le CSV contient plus de colonnes, par exemple rPPG ou pose,
    on garde uniquement les 9 features utilisées à l'entraînement.
    """

    df = pd.read_csv(behav_csv)

    required_cols = ["video_id"] + FEATURE_COLS
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Colonnes behavior manquantes dans {behav_csv}: {missing}"
        )

    keep_cols = ["video_id"]

    if "label" in df.columns:
        keep_cols.append("label")

    keep_cols += FEATURE_COLS

    filtered = df[keep_cols].copy()

    out_root.mkdir(parents=True, exist_ok=True)
    filtered_path = out_root / "mixed_test_behav_filtered_9_features.csv"

    filtered.to_csv(filtered_path, index=False, encoding="utf-8")

    print("\n========== BEHAVIOR CSV FILTER ==========")
    print(f"Original behav CSV : {behav_csv}")
    print(f"Filtered behav CSV : {filtered_path}")
    print(f"Features used      : {FEATURE_COLS}")
    print(f"BEHAV_DIM expected : {BEHAV_DIM}")
    print(f"Filtered shape     : {filtered.shape}")

    return filtered_path

def load_baseline_metrics(path: Path) -> dict:
    """
    Charge les métriques baseline depuis un fichier JSON.

    Objectif :
    éviter les métriques hardcodées dans le script d'évaluation.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier baseline metrics introuvable: {path}\n"
            "Créer le fichier reports/baseline_before_finetuning/metrics.json "
            "ou passer un autre chemin avec --baseline_metrics_json."
        )

    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    required = {"accuracy", "APCER", "BPCER", "ACER", "AUC"}
    missing = required - set(metrics.keys())

    if missing:
        raise ValueError(
            f"Métriques manquantes dans baseline JSON: {sorted(missing)}"
        )

    return metrics


# ==========================================================
# REPORTS
# ==========================================================

def attack_type_summary(pred_df: pd.DataFrame):
    rows = []

    for attack_type, g in pred_df.groupby("attack_type"):
        metrics = compute_pad_metrics(
            g["label"].values,
            g["pred_label"].values,
            g["score_spoof"].values,
        )

        row = {
            "attack_type": attack_type,
            "n": int(len(g)),
            "accuracy": metrics["accuracy"],
            "APCER": metrics["APCER"],
            "BPCER": metrics["BPCER"],
            "ACER": metrics["ACER"],
            "f1_spoof": metrics["f1_spoof"],
            "mean_score": float(g["score_spoof"].mean()),
            "min_score": float(g["score_spoof"].min()),
            "max_score": float(g["score_spoof"].max()),
        }

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["ACER", "n"],
        ascending=[False, False],
    )


def subgroup_summary(pred_df: pd.DataFrame, group_col: str):
    rows = []

    for group_value, g in pred_df.groupby(group_col):
        metrics = compute_pad_metrics(
            g["label"].values,
            g["pred_label"].values,
            g["score_spoof"].values,
        )

        row = {
            group_col: group_value,
            **metrics,
            "mean_score": float(g["score_spoof"].mean()),
            "min_score": float(g["score_spoof"].min()),
            "max_score": float(g["score_spoof"].max()),
        }

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(group_col)


def write_results_txt(
    out_path: Path,
    title: str,
    metrics: dict,
    attack_summary: pd.DataFrame,
):
    lines = []

    lines.append(f"========== {title} FINAL EVALUATION ==========")

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

    if attack_summary is not None and len(attack_summary):
        lines.append(attack_summary.to_string(index=False))
    else:
        lines.append("No attack type summary available.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_eval_block(name: str, pred_df: pd.DataFrame, out_root: Path):
    """
    Crée un dossier comme l'ancien rapport :
    reports/.../{name}/
      - predictions.csv
      - metrics.json
      - errors_by_attack_type.csv
      - results.txt
    """

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_pad_metrics(
        pred_df["label"].values,
        pred_df["pred_label"].values,
        pred_df["score_spoof"].values,
    )

    attack_summary = attack_type_summary(pred_df)

    predictions_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics.json"
    attack_path = out_dir / "errors_by_attack_type.csv"
    results_path = out_dir / "results.txt"

    pred_df.to_csv(predictions_path, index=False, encoding="utf-8")
    attack_summary.to_csv(attack_path, index=False, encoding="utf-8")

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    write_results_txt(
        out_path=results_path,
        title=name.upper(),
        metrics=metrics,
        attack_summary=attack_summary,
    )

    return {
        "name": name,
        "metrics": metrics,
        "outputs": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "attack_analysis": str(attack_path),
            "results_txt": str(results_path),
        },
    }


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=str(
            PROJECT_ROOT
            / "experiments"
            / "mixed_casia_axon_local_msu"
            / "seed42"
            / "best_model_mixed_casia_axon_local_msu.pth"
        ),
    )

    parser.add_argument(
        "--frames_csv",
        default=str(
            PROJECT_ROOT
            / "data"
            / "mixed_casia_axon_local_msu"
            / "mixed_test_frames.csv"
        ),
    )

    parser.add_argument(
        "--behav_csv",
        default=str(
            PROJECT_ROOT
            / "data"
            / "mixed_casia_axon_local_msu"
            / "mixed_test_behav.csv"
        ),
    )

    parser.add_argument(
        "--out_root",
        default=str(
            PROJECT_ROOT
            / "reports"
            / "mixed_casia_axon_local_msu_final_eval"
        ),
    )

    parser.add_argument(
        "--baseline_metrics_json",
        default=str(
            PROJECT_ROOT
            / "reports"
            / "baseline_before_finetuning"
            / "metrics.json"
        ),
        help="Chemin vers les métriques baseline avant fine-tuning.",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    frames_csv = Path(args.frames_csv)
    behav_csv = Path(args.behav_csv)
    out_root = Path(args.out_root)
    baseline_metrics_json = Path(args.baseline_metrics_json)

    out_root.mkdir(parents=True, exist_ok=True)

    for p in [model_path, frames_csv, behav_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable: {p}")

    baseline_before_finetuning = load_baseline_metrics(baseline_metrics_json)

    print("========== FINAL REPORT EVAL CONFIG ==========")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model        : {model_path}")
    print(f"Frames CSV   : {frames_csv}")
    print(f"Behav CSV    : {behav_csv}")
    print(f"Out root     : {out_root}")
    print(f"Baseline JSON: {baseline_metrics_json}")

    device = get_device()
    print(f"Device       : {device}")

    filtered_behav_csv = prepare_behav_csv_for_model(behav_csv, out_root)
  
    # Important : à partir d'ici, on remplace behav_csv par le fichier filtré
    behav_csv = filtered_behav_csv

    dataset = build_dataset(frames_csv, behav_csv)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_model(model_path, device)

    pred_df, global_metrics = predict_model(model, loader, device)

    meta_df = load_video_meta(frames_csv)

    pred_df = pred_df.merge(meta_df, on="video_id", how="left")
    pred_df["pred_label_name"] = pred_df["pred_label"].map({0: "REAL", 1: "SPOOF"})
    pred_df["decision_correct"] = pred_df["label"] == pred_df["pred_label"]

    all_results = {
        "model": str(model_path),
        "mixed_test": None,
        "by_source_dataset": {},
        "by_device": {},
        "outputs": {},
    }

    # 1. Bloc global mixed_test
    mixed_block = save_eval_block("mixed_test", pred_df, out_root)
    all_results["mixed_test"] = mixed_block["metrics"]
    all_results["outputs"]["mixed_test"] = mixed_block["outputs"]

    # 2. Blocs par source_dataset
    source_name_map = {
        "CASIA": "casia_test",
        "AXON": "axon_test",
        "LOCAL_REAL": "local_real_test",
        "MSU_MFSD": "msu_mfsd_test",
    }

    for source, source_df in pred_df.groupby("source_dataset"):
        block_name = source_name_map.get(source, f"{str(source).lower()}_test")
        block = save_eval_block(block_name, source_df.copy(), out_root)

        all_results["by_source_dataset"][source] = block["metrics"]
        all_results["outputs"][block_name] = block["outputs"]

    # 3. Résumés CSV globaux par source / device / attack
    by_source = subgroup_summary(pred_df, "source_dataset")
    by_device = subgroup_summary(pred_df, "device_id")
    by_attack = subgroup_summary(pred_df, "attack_type")

    by_source_path = out_root / "metrics_by_source_dataset.csv"
    by_device_path = out_root / "metrics_by_device.csv"
    by_attack_path = out_root / "metrics_by_attack_type.csv"

    by_source.to_csv(by_source_path, index=False, encoding="utf-8")
    by_device.to_csv(by_device_path, index=False, encoding="utf-8")
    by_attack.to_csv(by_attack_path, index=False, encoding="utf-8")

    all_results["outputs"]["metrics_by_source_dataset"] = str(by_source_path)
    all_results["outputs"]["metrics_by_device"] = str(by_device_path)
    all_results["outputs"]["metrics_by_attack_type"] = str(by_attack_path)

    # 4. Comparison before/after sans métriques hardcodées
    comparison = {
        "model": str(model_path),

        "baseline_before_finetuning": baseline_before_finetuning,

        "after_finetuning": {
            "mixed_test": all_results["mixed_test"],
            "source_datasets": all_results["by_source_dataset"],
        },

        "notes": {
            "baseline_reference": str(baseline_metrics_json),
            "current_model": (
                "Fine-tuned from mixed_casia_axon using LOCAL_REAL "
                "and MSU_MFSD real videos."
            ),
            "purpose": (
                "Evaluate robustness after adapting to webcam/mobile "
                "real capture domain."
            ),
        },

        "outputs": all_results["outputs"],
    }

    comparison_path = out_root / "comparison_before_after.json"

    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # 5. Console report
    print("\n========== MIXED TEST FINAL METRICS ==========")

    for k, v in all_results["mixed_test"].items():
        print(f"{k}: {v}")

    print("\n========== BASELINE BEFORE FINETUNING ==========")

    for k, v in baseline_before_finetuning.items():
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
        print(
            errors[
                [
                    "video_id",
                    "original_video_id",
                    "label_name",
                    "pred_label_name",
                    "score_spoof",
                    "source_dataset",
                    "device_id",
                    "subject_id",
                    "attack_type",
                ]
            ]
            .sort_values(["source_dataset", "score_spoof"])
            .to_string(index=False)
        )

    print("\n========== SAVED ==========")
    print(f"Report root               : {out_root}")
    print(f"Comparison JSON           : {comparison_path}")
    print(f"Metrics by source dataset : {by_source_path}")
    print(f"Metrics by device         : {by_device_path}")
    print(f"Metrics by attack type    : {by_attack_path}")
    print("\n[OK] Rapport final généré.")


if __name__ == "__main__":
    main()