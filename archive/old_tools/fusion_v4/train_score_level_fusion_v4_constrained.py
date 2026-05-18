from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


BEHAV_COLS = [
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

RPPG_COLS = [
    "rppg_dominant_freq",
    "rppg_hr_estimate",
    "rppg_snr",
    "rppg_signal_std",
    "rppg_skipped_rate",
    "rppg_valid",
]


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "video_id" not in df.columns:
        raise ValueError(f"video_id missing in {path}. Columns={df.columns.tolist()}")
    df["video_id"] = df["video_id"].astype(str)
    return df


def clean_visual_predictions(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"video_id", "label", "score_spoof"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{name} visual predictions missing columns: {missing}")

    out = df[["video_id", "label", "score_spoof"]].copy()
    out["video_id"] = out["video_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out = out.rename(columns={"score_spoof": "score_visual"})

    return out


def merge_visual_and_features(visual_df: pd.DataFrame, feature_df: pd.DataFrame, name: str) -> pd.DataFrame:
    feature_df = feature_df.copy()
    feature_df["video_id"] = feature_df["video_id"].astype(str)

    # On garde label depuis predictions.csv pour éviter label_x / label_y
    if "label" in feature_df.columns:
        feature_df = feature_df.drop(columns=["label"])

    out = visual_df.merge(feature_df, on="video_id", how="left")

    missing_values = out.isna().sum().sum()
    if missing_values > 0:
        print(f"[WARN] {name}: {missing_values} missing values after merge. Filling with 0.")
        out = out.fillna(0.0)

    return out


def available_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def train_logistic_branch(train_df: pd.DataFrame, cols: list[str], branch_name: str):
    used_cols = available_cols(train_df, cols)

    if len(used_cols) == 0:
        raise ValueError(f"No usable feature columns for branch {branch_name}")

    if "label" not in train_df.columns:
        raise ValueError("train_df must contain label")

    X = train_df[used_cols].fillna(0.0).values
    y = train_df["label"].astype(int).values

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    clf.fit(X, y)

    return clf, used_cols


def predict_branch_score(clf, df: pd.DataFrame, used_cols: list[str]) -> np.ndarray:
    X = df[used_cols].fillna(0.0).values
    return clf.predict_proba(X)[:, 1]


def metrics_from_scores(y_true, scores, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    real_count = tn + fp
    spoof_count = fn + tp

    apcer = fn / spoof_count if spoof_count else 0.0
    bpcer = fp / real_count if real_count else 0.0
    acer = (apcer + bpcer) / 2.0

    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")

    return {
        "total": int(len(y_true)),
        "real_count": int(real_count),
        "spoof_count": int(spoof_count),
        "tn_real": int(tn),
        "fp_real_as_spoof": int(fp),
        "fn_spoof_as_real": int(fn),
        "tp_spoof": int(tp),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_spoof": float(precision_score(y_true, pred, zero_division=0)),
        "recall_spoof": float(recall_score(y_true, pred, zero_division=0)),
        "f1_spoof": float(f1_score(y_true, pred, zero_division=0)),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER": float(acer),
        "AUC": float(auc),
        "threshold": float(threshold),
    }


def fuse_scores(df: pd.DataFrame, weights: tuple[float, float, float]) -> np.ndarray:
    w_visual, w_behavior, w_rppg = weights

    score = (
        w_visual * df["score_visual"].values +
        w_behavior * df["score_behavior"].values +
        w_rppg * df["score_rppg"].values
    )

    return np.clip(score, 0.0, 1.0)


def grid_search_weights_constrained(
    val_df: pd.DataFrame,
    step: float = 0.05,
    min_visual: float = 0.60,
    min_behavior: float = 0.15,
    max_rppg: float = 0.25,
):
    y_val = val_df["label"].astype(int).values
    values = np.round(np.arange(0.0, 1.00001, step), 2)

    rows = []
    best = None

    for w_visual in values:
        for w_behavior in values:
            w_rppg = round(1.0 - w_visual - w_behavior, 2)

            if w_rppg < 0:
                continue

            # Contraintes métier/professionnelles
            if w_visual < min_visual:
                continue

            if w_behavior < min_behavior:
                continue

            if w_rppg > max_rppg:
                continue

            weights = (float(w_visual), float(w_behavior), float(w_rppg))
            fused = fuse_scores(val_df, weights)
            m = metrics_from_scores(y_val, fused, threshold=0.5)

            row = {
                "w_visual": weights[0],
                "w_behavior": weights[1],
                "w_rppg": weights[2],
                **m,
            }

            rows.append(row)

            # Objectif société :
            # 1. minimiser ACER
            # 2. minimiser APCER pour sécurité
            # 3. maximiser F1
            key = (m["ACER"], m["APCER"], -m["f1_spoof"])

            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "weights": weights,
                    "metrics": m,
                }

    if best is None:
        raise RuntimeError("No valid weight combination found. Relax constraints.")

    return best, pd.DataFrame(rows)


def apply_policy(score_final: float, motion_max: float = 0.0, rppg_valid: float = 1.0):
    if score_final < 0.35:
        decision = "ACCEPT"
        reason = "low_spoof_score"
    elif score_final >= 0.75:
        decision = "REJECT"
        reason = "high_spoof_score"
    else:
        decision = "RETRY"
        reason = "gray_zone"

    # Si la vidéo est instable, on évite ACCEPT direct
    if decision == "ACCEPT" and motion_max > 2.0:
        decision = "RETRY"
        reason = "unstable_motion"

    # rppg_valid est normalisé. Une valeur très faible signale un rPPG peu fiable.
    if decision == "ACCEPT" and rppg_valid < -1.0:
        decision = "RETRY"
        reason = "weak_rppg_quality"

    return decision, reason


def add_policy(df: pd.DataFrame) -> pd.DataFrame:
    decisions = []
    reasons = []

    for _, row in df.iterrows():
        motion_max = float(row["motion_max"]) if "motion_max" in df.columns else 0.0
        rppg_valid = float(row["rppg_valid"]) if "rppg_valid" in df.columns else 1.0

        decision, reason = apply_policy(
            score_final=float(row["score_fusion_v4"]),
            motion_max=motion_max,
            rppg_valid=rppg_valid,
        )

        decisions.append(decision)
        reasons.append(reason)

    df = df.copy()
    df["decision_v4"] = decisions
    df["decision_reason"] = reasons

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_behav", required=True)
    parser.add_argument("--val_behav", required=True)
    parser.add_argument("--test_behav", required=True)

    parser.add_argument("--val_visual_preds", required=True)
    parser.add_argument("--test_visual_preds", required=True)

    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--grid_step", type=float, default=0.05)

    parser.add_argument("--min_visual", type=float, default=0.60)
    parser.add_argument("--min_behavior", type=float, default=0.15)
    parser.add_argument("--max_rppg", type=float, default=0.25)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_behav = read_csv(args.train_behav)
    val_behav = read_csv(args.val_behav)
    test_behav = read_csv(args.test_behav)

    if "label" not in train_behav.columns:
        raise ValueError("train_behav must contain label")

    train_behav["label"] = train_behav["label"].astype(int)

    val_visual = clean_visual_predictions(read_csv(args.val_visual_preds), "VAL")
    test_visual = clean_visual_predictions(read_csv(args.test_visual_preds), "TEST")

    val_df = merge_visual_and_features(val_visual, val_behav, "VAL")
    test_df = merge_visual_and_features(test_visual, test_behav, "TEST")

    print("========== DATA CHECK ==========")
    print("Train behavior rows:", len(train_behav), "videos:", train_behav["video_id"].nunique())
    print("Val rows:", len(val_df), "videos:", val_df["video_id"].nunique())
    print("Test rows:", len(test_df), "videos:", test_df["video_id"].nunique())

    behavior_clf, behavior_cols = train_logistic_branch(train_behav, BEHAV_COLS, "behavior")
    rppg_clf, rppg_cols = train_logistic_branch(train_behav, RPPG_COLS, "rppg")

    val_df["score_behavior"] = predict_branch_score(behavior_clf, val_df, behavior_cols)
    val_df["score_rppg"] = predict_branch_score(rppg_clf, val_df, rppg_cols)

    test_df["score_behavior"] = predict_branch_score(behavior_clf, test_df, behavior_cols)
    test_df["score_rppg"] = predict_branch_score(rppg_clf, test_df, rppg_cols)

    best, grid = grid_search_weights_constrained(
        val_df,
        step=args.grid_step,
        min_visual=args.min_visual,
        min_behavior=args.min_behavior,
        max_rppg=args.max_rppg,
    )

    best_weights = best["weights"]

    print("\n========== BEST CONSTRAINED WEIGHTS ON VAL ==========")
    print("weights:", best_weights)
    print(json.dumps(best["metrics"], indent=2))

    grid.to_csv(out_dir / "fusion_weight_grid_val_constrained.csv", index=False, encoding="utf-8")

    y_test = test_df["label"].astype(int).values

    test_df["score_fusion_v4"] = fuse_scores(test_df, best_weights)
    test_df["pred_fusion_v4"] = (test_df["score_fusion_v4"] >= 0.5).astype(int)
    test_df = add_policy(test_df)

    metrics = {
        "visual_only": metrics_from_scores(y_test, test_df["score_visual"].values),
        "behavior_only": metrics_from_scores(y_test, test_df["score_behavior"].values),
        "rppg_only": metrics_from_scores(y_test, test_df["score_rppg"].values),
        "fusion_v4_constrained": metrics_from_scores(y_test, test_df["score_fusion_v4"].values),
        "best_weights_from_val": {
            "w_visual": best_weights[0],
            "w_behavior": best_weights[1],
            "w_rppg": best_weights[2],
        },
        "constraints": {
            "min_visual": args.min_visual,
            "min_behavior": args.min_behavior,
            "max_rppg": args.max_rppg,
        },
        "used_features": {
            "behavior_cols": behavior_cols,
            "rppg_cols": rppg_cols,
        },
    }

    with (out_dir / "fusion_v4_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    val_df.to_csv(out_dir / "val_scores_visual_behavior_rppg.csv", index=False, encoding="utf-8")
    test_df.to_csv(out_dir / "test_scores_fusion_v4.csv", index=False, encoding="utf-8")

    with (out_dir / "behavior_clf.pkl").open("wb") as f:
        pickle.dump(behavior_clf, f)

    with (out_dir / "rppg_clf.pkl").open("wb") as f:
        pickle.dump(rppg_clf, f)

    print("\n========== TEST METRICS ==========")
    print(json.dumps(metrics, indent=2))
    print("\nSaved outputs in:", out_dir)


if __name__ == "__main__":
    main()
