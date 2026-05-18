from pathlib import Path
import json
import inspect
import warnings
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.pad_system.video_model import VideoPADModel


warnings.filterwarnings("ignore")


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_paths(project_root: Path):
    config_path = project_root / "configs" / "paths.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    axon_prepared_dir = Path(cfg["axon_prepared_dir"])

    if not axon_prepared_dir.is_absolute():
        axon_prepared_dir = (project_root / axon_prepared_dir).resolve()

    return axon_prepared_dir


def banking_decision_video(score: float) -> str:
    """
    Politique vidéo actuelle :
    score proche de 0 -> REAL
    score proche de 1 -> SPOOF
    """
    score = float(score)

    if score < 0.30:
        return "ACCEPT"

    if score < 0.60:
        return "RETRY"

    return "REJECT"


def label_from_score(score: float, threshold: float = 0.5) -> int:
    """
    label prédit :
    0 = REAL
    1 = SPOOF
    """
    return 1 if float(score) >= threshold else 0


def compute_pad_metrics(y_true, y_pred, scores):
    """
    Convention :
    label 0 = REAL / bona fide
    label 1 = SPOOF / attack

    APCER = attaques acceptées à tort = spoof prédit real
    BPCER = réels rejetés à tort = real prédit spoof
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


def patch_video_model_with_axon_data(
    model,
    frames_df: pd.DataFrame,
    behav_df: pd.DataFrame,
    frames_csv_path: Path,
    behav_csv_path: Path,
):
    """
    Ton VideoPADModel a été créé pour CASIA.
    Selon son implémentation exacte, les noms d'attributs peuvent varier.
    Cette fonction injecte les données Axon dans les noms d'attributs probables.

    L'objectif est que :
    model.predict(video_id)
    puisse retrouver les frames Axon par video_id.
    """

    # CSV paths possibles
    possible_test_csv_attrs = [
        "test_csv",
        "test_csv_path",
        "csv_path",
        "frames_csv",
        "frames_csv_path",
    ]

    possible_behav_csv_attrs = [
        "behav_test_csv",
        "behav_test_csv_path",
        "behavior_csv",
        "behavior_csv_path",
        "behav_csv",
        "behav_csv_path",
    ]

    for attr in possible_test_csv_attrs:
        try:
            setattr(model, attr, str(frames_csv_path))
        except Exception:
            pass

    for attr in possible_behav_csv_attrs:
        try:
            setattr(model, attr, str(behav_csv_path))
        except Exception:
            pass

    # DataFrames possibles
    possible_test_df_attrs = [
        "test_df",
        "df_test",
        "frames_df",
        "test_csv_df",
        "csv_df",
        "metadata_df",
    ]

    possible_behav_df_attrs = [
        "behav_test_df",
        "behavior_df",
        "behav_df",
        "behav_features_df",
    ]

    for attr in possible_test_df_attrs:
        try:
            setattr(model, attr, frames_df)
        except Exception:
            pass

    for attr in possible_behav_df_attrs:
        try:
            setattr(model, attr, behav_df)
        except Exception:
            pass

    # Cas fréquent : certains modèles gardent un dict video_id -> rows
    try:
        setattr(model, "video_ids", sorted(frames_df["video_id"].unique().tolist()))
    except Exception:
        pass

    return model


def instantiate_video_model(
    checkpoint_path: Path = None,
    frames_csv_path: Path = None,
    behav_csv_path: Path = None,
):
    """
    Instanciation robuste de VideoPADModel.
    On essaie d'abord avec arguments si la classe les accepte,
    sinon on tombe sur VideoPADModel() puis patch manuel.
    """

    sig = inspect.signature(VideoPADModel)
    params = sig.parameters

    kwargs = {}

    if checkpoint_path is not None:
        for name in ["checkpoint_path", "ckpt_path", "model_path"]:
            if name in params:
                kwargs[name] = str(checkpoint_path)

    if frames_csv_path is not None:
        for name in ["test_csv", "test_csv_path", "csv_path", "frames_csv_path"]:
            if name in params:
                kwargs[name] = str(frames_csv_path)

    if behav_csv_path is not None:
        for name in ["behav_test_csv", "behav_test_csv_path", "behavior_csv_path"]:
            if name in params:
                kwargs[name] = str(behav_csv_path)

    if "use_behav" in params:
        kwargs["use_behav"] = True

    try:
        return VideoPADModel(**kwargs)
    except TypeError:
        return VideoPADModel()


def predict_one_video(model, video_id: str):
    """
    Appel robuste.
    Ton modèle a déjà predict(...).
    Si besoin, on utilise la méthode privée connue.
    """

    if hasattr(model, "predict"):
        return float(model.predict(video_id))

    if hasattr(model, "_predict_from_casia_video_id"):
        return float(model._predict_from_casia_video_id(video_id))

    raise RuntimeError("Aucune méthode predict compatible trouvée dans VideoPADModel.")


def main():
    project_root = get_project_root()
    axon_prepared_dir = load_paths(project_root)

    frames_csv_path = axon_prepared_dir / "manifests" / "axon_video_frames_manifest.csv"
    behav_csv_path = axon_prepared_dir / "manifests" / "axon_video_behav_norm.csv"

    out_dir = project_root / "reports" / "axon_video_casia_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics.json"
    results_txt_path = out_dir / "results.txt"
    errors_by_attack_path = out_dir / "errors_by_attack_type.csv"

    checkpoint_path = (
        project_root
        / "experiments"
        / "step2_sampling"
        / "deep_behav_no_pts_consecutive"
        / "deep_behav_no_pts_consecutive_seed42"
        / "best_model.pth"
    )

    if not frames_csv_path.exists():
        raise FileNotFoundError(
            f"Frames manifest introuvable: {frames_csv_path}\n"
            "Lance d'abord: python tools\\axon\\prepare_axon_video_frames.py"
        )

    if not behav_csv_path.exists():
        raise FileNotFoundError(
            f"Behavior norm CSV introuvable: {behav_csv_path}\n"
            "Lance d'abord: python tools\\axon\\extract_axon_video_behavior.py"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint CASIA introuvable: {checkpoint_path}")

    print("========== CONFIG ==========")
    print(f"Project root       : {project_root}")
    print(f"Frames CSV         : {frames_csv_path}")
    print(f"Behavior CSV       : {behav_csv_path}")
    print(f"Checkpoint CASIA   : {checkpoint_path}")
    print(f"Output dir         : {out_dir}")

    frames_df = pd.read_csv(frames_csv_path)
    behav_df = pd.read_csv(behav_csv_path)

    required_frame_cols = {
        "path",
        "label",
        "label_name",
        "video_id",
        "frame_idx",
        "attack_type",
        "level",
    }

    missing = required_frame_cols - set(frames_df.columns)
    if missing:
        raise RuntimeError(f"Colonnes manquantes dans frames CSV: {missing}")

    print("\n========== DATA ==========")
    print(f"Frame rows     : {len(frames_df)}")
    print(f"Unique videos  : {frames_df['video_id'].nunique()}")
    print("\nLabels videos:")
    print(frames_df.drop_duplicates("video_id")["label_name"].value_counts())

    print("\n========== LOAD MODEL ==========")
    model = instantiate_video_model(
        checkpoint_path=checkpoint_path,
        frames_csv_path=frames_csv_path,
        behav_csv_path=behav_csv_path,
    )

    model = patch_video_model_with_axon_data(
        model=model,
        frames_df=frames_df,
        behav_df=behav_df,
        frames_csv_path=frames_csv_path,
        behav_csv_path=behav_csv_path,
    )

    video_meta = (
        frames_df.sort_values(["video_id", "frame_idx"])
        .groupby("video_id")
        .first()
        .reset_index()
    )

    rows = []
    errors = []

    print("\n========== EVALUATION CASIA → AXON VIDEO ==========")
    print(f"Videos à évaluer: {len(video_meta)}")

    for idx, row in video_meta.iterrows():
        video_id = str(row["video_id"])
        label = int(row["label"])
        label_name = str(row["label_name"])
        attack_type = str(row.get("attack_type", "unknown"))
        level = str(row.get("level", "unknown"))

        try:
            score = predict_one_video(model, video_id)
            pred_label = label_from_score(score, threshold=0.5)
            pred_label_name = "SPOOF" if pred_label == 1 else "REAL"
            correct = int(pred_label == label)
            decision = banking_decision_video(score)

            rows.append({
                "video_id": video_id,
                "label": label,
                "label_name": label_name,
                "attack_type": attack_type,
                "level": level,
                "score_spoof": float(score),
                "pred_label": int(pred_label),
                "pred_label_name": pred_label_name,
                "banking_decision": decision,
                "correct": bool(correct),
                "error": "",
            })

        except Exception as e:
            errors.append({
                "video_id": video_id,
                "label": label,
                "label_name": label_name,
                "attack_type": attack_type,
                "level": level,
                "error": str(e),
            })

            rows.append({
                "video_id": video_id,
                "label": label,
                "label_name": label_name,
                "attack_type": attack_type,
                "level": level,
                "score_spoof": np.nan,
                "pred_label": -1,
                "pred_label_name": "ERROR",
                "banking_decision": "ERROR",
                "correct": False,
                "error": str(e),
            })

        if (idx + 1) % 25 == 0:
            print(f"[INFO] Progression: {idx + 1}/{len(video_meta)} vidéos")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(predictions_path, index=False, encoding="utf-8")

    valid_df = pred_df[pred_df["pred_label"].isin([0, 1])].copy()

    if len(valid_df) == 0:
        raise RuntimeError(
            "Aucune prédiction valide. Vérifie que VideoPADModel utilise bien les CSV Axon."
        )

    y_true = valid_df["label"].astype(int).values
    y_pred = valid_df["pred_label"].astype(int).values
    scores = valid_df["score_spoof"].astype(float).values

    metrics = compute_pad_metrics(y_true, y_pred, scores)

    metrics["evaluated_videos"] = int(len(pred_df))
    metrics["valid_predictions"] = int(len(valid_df))
    metrics["prediction_errors"] = int(len(errors))
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["frames_csv"] = str(frames_csv_path)
    metrics["behavior_csv"] = str(behav_csv_path)
    metrics["threshold"] = 0.5
    metrics["sampling_mode"] = "consecutive_middle"
    metrics["behavior_scaler"] = "CASIA_train_behav_scaler"

    # Erreurs par attack_type
    attack_rows = []

    for attack_type, g in valid_df.groupby("attack_type"):
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

    attack_df = pd.DataFrame(attack_rows).sort_values("n", ascending=False)
    attack_df.to_csv(errors_by_attack_path, index=False, encoding="utf-8")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("========== CASIA CHECKPOINT → AXON VIDEO EVALUATION ==========\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\n========== CONFUSION DETAILS ==========\n")
        f.write(f"TN REAL correct        : {metrics['tn_real']}\n")
        f.write(f"FP REAL as SPOOF       : {metrics['fp_real_as_spoof']}\n")
        f.write(f"FN SPOOF as REAL       : {metrics['fn_spoof_as_real']}\n")
        f.write(f"TP SPOOF correct       : {metrics['tp_spoof']}\n")

        f.write("\n========== ATTACK TYPE SUMMARY ==========\n")
        f.write(attack_df.to_string(index=False))

    print("\n========== SUMMARY ==========")
    for k, v in metrics.items():
        print(f"{k:<25}: {v}")

    print("\n========== ATTACK TYPE SUMMARY ==========")
    print(attack_df)

    print("\n========== OUTPUTS ==========")
    print(f"Predictions     : {predictions_path}")
    print(f"Metrics         : {metrics_path}")
    print(f"Results txt     : {results_txt_path}")
    print(f"Attack analysis : {errors_by_attack_path}")

    print("\n[OK] Évaluation CASIA → Axon vidéo terminée.")


if __name__ == "__main__":
    main()