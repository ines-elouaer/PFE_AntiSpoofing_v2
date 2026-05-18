from pathlib import Path
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


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

    "yaw_std",
    "pitch_std",
    "roll_std",
    "yaw_range",
    "pitch_range",
    "roll_range",
    "yaw_delta_mean",
    "pitch_delta_mean",
    "roll_delta_mean",
    "pose_autocorr",
    "pose_valid_rate",
]


def compute_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

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
        "precision_spoof": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_spoof": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_spoof": float(f1_score(y_true, y_pred, zero_division=0)),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER": float(acer),
        "threshold": float(threshold),
    }

    try:
        out["AUC"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["AUC"] = None

    return out


def find_best_threshold(y_true, scores):
    best = None

    for thr in np.linspace(0.05, 0.95, 181):
        m = compute_metrics(y_true, scores, threshold=float(thr))

        # Priorité PAD :
        # 1. minimiser ACER
        # 2. minimiser APCER
        # 3. minimiser BPCER
        key = (m["ACER"], m["APCER"], m["BPCER"])

        if best is None or key < best["key"]:
            best = {
                "threshold": float(thr),
                "metrics": m,
                "key": key,
            }

    return best


def load_xy(csv_path):
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise ValueError(f"Colonne label absente dans {path}. Colonnes: {df.columns.tolist()}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]

    if missing:
        raise ValueError(
            f"Colonnes behavior-pose manquantes dans {path}: {missing}\\n"
            f"Colonnes disponibles: {df.columns.tolist()}"
        )

    x = df[FEATURE_COLS].fillna(0.0).astype(float).values
    y = df["label"].astype(int).values

    return df, x, y


def predict_scores(model, x):
    return model.predict_proba(x)[:, 1]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)

    parser.add_argument("--out_dir", default="reports/fusion_v6_behavior_pose")
    parser.add_argument("--model_name", default="behavior_pose_clf.pkl")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, x_train, y_train = load_xy(args.train_csv)
    val_df, x_val, y_val = load_xy(args.val_csv)
    test_df, x_test, y_test = load_xy(args.test_csv)

    print("========== DATA ==========")
    print("Train:", x_train.shape, "real:", int((y_train == 0).sum()), "spoof:", int((y_train == 1).sum()))
    print("Val  :", x_val.shape, "real:", int((y_val == 0).sum()), "spoof:", int((y_val == 1).sum()))
    print("Test :", x_test.shape, "real:", int((y_test == 0).sum()), "spoof:", int((y_test == 1).sum()))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    model.fit(x_train, y_train)

    val_scores = predict_scores(model, x_val)
    test_scores = predict_scores(model, x_test)
    train_scores = predict_scores(model, x_train)

    best = find_best_threshold(y_val, val_scores)
    best_thr = best["threshold"]

    train_metrics = compute_metrics(y_train, train_scores, threshold=best_thr)
    val_metrics = compute_metrics(y_val, val_scores, threshold=best_thr)
    test_metrics = compute_metrics(y_test, test_scores, threshold=best_thr)

    report = {
        "feature_cols": FEATURE_COLS,
        "best_threshold_from_val": best_thr,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "notes": {
            "model": "StandardScaler + LogisticRegression(class_weight=balanced)",
            "label_convention": "0=REAL, 1=SPOOF",
            "score_convention": "score close to 1 means SPOOF",
        },
    }

    model_path = out_dir / args.model_name

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(out_dir / "behavior_pose_clf_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Sauvegarde des prédictions pour vérification
    for split_name, df, scores, y in [
        ("train", train_df, train_scores, y_train),
        ("val", val_df, val_scores, y_val),
        ("test", test_df, test_scores, y_test),
    ]:
        pred = df.copy()
        pred["score_behavior_pose"] = scores
        pred["pred_label"] = (scores >= best_thr).astype(int)
        pred.to_csv(out_dir / f"{split_name}_behavior_pose_predictions.csv", index=False, encoding="utf-8")

    print("\\n========== SAVED ==========")
    print("Model:", model_path)
    print("Metrics:", out_dir / "behavior_pose_clf_metrics.json")
    print("\\n========== TEST METRICS ==========")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
