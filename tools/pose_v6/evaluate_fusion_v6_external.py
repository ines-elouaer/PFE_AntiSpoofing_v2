from pathlib import Path
import argparse
import json
import pickle
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


BEHAV_POSE_COLS = [
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


def read_csv(path):
    df = pd.read_csv(path)
    if "video_id" not in df.columns:
        raise ValueError(f"video_id missing in {path}")
    df["video_id"] = df["video_id"].astype(str)
    return df


def clean_visual_predictions(df):
    required = {"video_id", "label", "score_spoof"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"external visual predictions missing columns: {missing}")

    out = df[["video_id", "label", "score_spoof"]].copy()
    out["video_id"] = out["video_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out = out.rename(columns={"score_spoof": "score_visual"})

    return out


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


def predict_behavior_pose(clf, df, used_cols):
    missing = [c for c in used_cols if c not in df.columns]
    if missing:
        print("[WARN] Missing behavior-pose columns:", missing)
        for c in missing:
            df[c] = 0.0

    X = df[used_cols].fillna(0.0).values
    return clf.predict_proba(X)[:, 1]


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

    parser.add_argument("--external_visual_preds", required=True)
    parser.add_argument("--external_behavior_pose", required=True)
    parser.add_argument("--fusion_v6_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    visual = clean_visual_predictions(read_csv(args.external_visual_preds))
    behavior_pose = read_csv(args.external_behavior_pose)

    if "label" in behavior_pose.columns:
        behavior_pose = behavior_pose.drop(columns=["label"])

    df = visual.merge(behavior_pose, on="video_id", how="left").fillna(0.0)

    fusion_v6_dir = Path(args.fusion_v6_dir)

    with open(fusion_v6_dir / "behavior_pose_clf.pkl", "rb") as f:
        behavior_pose_clf = pickle.load(f)

    with open(fusion_v6_dir / "fusion_v6_metrics.json", "r", encoding="utf-8") as f:
        fusion_info = json.load(f)

    used_cols = fusion_info["used_behavior_pose_features"]
    weights = fusion_info["best_weights_from_val"]

    w_visual = float(weights["w_visual"])
    w_behavior_pose = float(weights["w_behavior_pose"])

    df["score_behavior_pose"] = predict_behavior_pose(
        behavior_pose_clf,
        df,
        used_cols,
    )

    df["score_fusion_v6"] = np.clip(
        w_visual * df["score_visual"].values +
        w_behavior_pose * df["score_behavior_pose"].values,
        0.0,
        1.0,
    )

    df["pred_fusion_v6"] = (df["score_fusion_v6"] >= 0.5).astype(int)
    df = add_policy(df)

    y = df["label"].astype(int).values

    metrics = {
        "visual_only_external": metrics_from_scores(y, df["score_visual"].values),
        "behavior_pose_only_external": metrics_from_scores(y, df["score_behavior_pose"].values),
        "fusion_v6_external": metrics_from_scores(y, df["score_fusion_v6"].values),
        "weights_used": {
            "w_visual": w_visual,
            "w_behavior_pose": w_behavior_pose,
        },
        "policy_counts": df["decision_v6"].value_counts().to_dict(),
        "used_behavior_pose_features": used_cols,
    }

    df.to_csv(out_dir / "external_scores_fusion_v6.csv", index=False, encoding="utf-8")

    with (out_dir / "external_fusion_v6_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
