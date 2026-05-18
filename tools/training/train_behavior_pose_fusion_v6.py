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


BEHAV_POSE_COLS = [
    # EAR / blink
    "ear_mean",
    "ear_std",
    "ear_min",
    "ear_max",
    "blink_count",

    # optical flow / motion
    "motion_mean",
    "motion_std",
    "motion_max",
    "skipped_rate",

    # head pose dynamics
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


def read_csv(path):
    df = pd.read_csv(path)
    if "video_id" not in df.columns:
        raise ValueError(f"video_id missing in {path}")
    df["video_id"] = df["video_id"].astype(str)
    return df


def clean_visual_predictions(df, name):
    required = {"video_id", "label", "score_spoof"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{name} visual predictions missing columns: {missing}")

    out = df[["video_id", "label", "score_spoof"]].copy()
    out["video_id"] = out["video_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out = out.rename(columns={"score_spoof": "score_visual"})

    return out


def merge_visual_and_features(visual_df, feature_df):
    feature_df = feature_df.copy()

    if "label" in feature_df.columns:
        feature_df = feature_df.drop(columns=["label"])

    out = visual_df.merge(feature_df, on="video_id", how="left").fillna(0.0)
    return out


def available_cols(df, cols):
    return [c for c in cols if c in df.columns]


def train_behavior_pose_clf(train_df):
    used_cols = available_cols(train_df, BEHAV_POSE_COLS)

    if "label" not in train_df.columns:
        raise ValueError("train_df must contain label")

    if len(used_cols) == 0:
        raise ValueError("No behavior-pose columns found.")

    X = train_df[used_cols].fillna(0.0).values
    y = train_df["label"].astype(int).values

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    clf.fit(X, y)

    return clf, used_cols


def predict_score(clf, df, used_cols):
    X = df[used_cols].fillna(0.0).values
    return clf.predict_proba(X)[:, 1]


def metrics_from_scores(y_true, scores, threshold=0.5):
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


def fuse_scores(df, w_visual, w_behavior):
    return np.clip(
        w_visual * df["score_visual"].values +
        w_behavior * df["score_behavior_pose"].values,
        0.0,
        1.0,
    )


def search_weights(val_df, step=0.05, min_behavior=0.15, max_behavior=0.35):
    y = val_df["label"].astype(int).values

    best = None
    rows = []

    for w_behavior in np.round(np.arange(min_behavior, max_behavior + 0.0001, step), 2):
        w_visual = round(1.0 - w_behavior, 2)

        scores = fuse_scores(val_df, w_visual, w_behavior)
        m = metrics_from_scores(y, scores)

        row = {
            "w_visual": w_visual,
            "w_behavior_pose": float(w_behavior),
            **m,
        }

        rows.append(row)

        key = (m["ACER"], m["APCER"], -m["f1_spoof"])

        if best is None or key < best["key"]:
            best = {
                "key": key,
                "weights": (w_visual, float(w_behavior)),
                "metrics": m,
            }

    return best, pd.DataFrame(rows)


def add_policy(df):
    decisions = []
    reasons = []

    for _, row in df.iterrows():
        score = float(row["score_fusion_v6"])

        if score < 0.35:
            decision = "ACCEPT"
            reason = "low_spoof_score"
        elif score >= 0.75:
            decision = "REJECT"
            reason = "high_spoof_score"
        else:
            decision = "RETRY"
            reason = "gray_zone"

        # Si pose peu fiable, éviter accept direct
        if decision == "ACCEPT" and "pose_valid_rate" in df.columns:
            if float(row["pose_valid_rate"]) < 0.5:
                decision = "RETRY"
                reason = "weak_pose_quality"

        decisions.append(decision)
        reasons.append(reason)

    df = df.copy()
    df["decision_v6"] = decisions
    df["decision_reason"] = reasons

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_behavior_pose", required=True)
    parser.add_argument("--val_behavior_pose", required=True)
    parser.add_argument("--test_behavior_pose", required=True)

    parser.add_argument("--val_visual_preds", required=True)
    parser.add_argument("--test_visual_preds", required=True)

    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--min_behavior", type=float, default=0.15)
    parser.add_argument("--max_behavior", type=float, default=0.35)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_csv(args.train_behavior_pose)
    val_feat = read_csv(args.val_behavior_pose)
    test_feat = read_csv(args.test_behavior_pose)

    val_visual = clean_visual_predictions(read_csv(args.val_visual_preds), "VAL")
    test_visual = clean_visual_predictions(read_csv(args.test_visual_preds), "TEST")

    val_df = merge_visual_and_features(val_visual, val_feat)
    test_df = merge_visual_and_features(test_visual, test_feat)

    clf, used_cols = train_behavior_pose_clf(train_df)

    val_df["score_behavior_pose"] = predict_score(clf, val_df, used_cols)
    test_df["score_behavior_pose"] = predict_score(clf, test_df, used_cols)

    best, grid = search_weights(
        val_df,
        step=args.step,
        min_behavior=args.min_behavior,
        max_behavior=args.max_behavior,
    )

    w_visual, w_behavior = best["weights"]

    test_df["score_fusion_v6"] = fuse_scores(test_df, w_visual, w_behavior)
    test_df["pred_fusion_v6"] = (test_df["score_fusion_v6"] >= 0.5).astype(int)
    test_df = add_policy(test_df)

    y_test = test_df["label"].astype(int).values

    metrics = {
        "visual_only": metrics_from_scores(y_test, test_df["score_visual"].values),
        "behavior_pose_only": metrics_from_scores(y_test, test_df["score_behavior_pose"].values),
        "fusion_v6": metrics_from_scores(y_test, test_df["score_fusion_v6"].values),
        "best_weights_from_val": {
            "w_visual": w_visual,
            "w_behavior_pose": w_behavior,
        },
        "val_best_metrics": best["metrics"],
        "used_behavior_pose_features": used_cols,
    }

    with (out_dir / "fusion_v6_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    test_df.to_csv(out_dir / "test_scores_fusion_v6.csv", index=False, encoding="utf-8")
    grid.to_csv(out_dir / "fusion_v6_weight_grid_val.csv", index=False, encoding="utf-8")

    with (out_dir / "behavior_pose_clf.pkl").open("wb") as f:
        pickle.dump(clf, f)

    print(json.dumps(metrics, indent=2))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
