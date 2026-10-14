from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


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


def read_csv(path):
    df = pd.read_csv(path)
    df["video_id"] = df["video_id"].astype(str)
    return df


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
    }


def predict_branch(clf, df, cols):
    used = [c for c in cols if c in df.columns]
    X = df[used].fillna(0.0).values
    return clf.predict_proba(X)[:, 1]


def apply_policy(score_final, motion_max=0.0, rppg_valid=1.0):
    if score_final < 0.35:
        decision = "ACCEPT"
        reason = "low_spoof_score"
    elif score_final >= 0.75:
        decision = "REJECT"
        reason = "high_spoof_score"
    else:
        decision = "RETRY"
        reason = "gray_zone"

    if decision == "ACCEPT" and motion_max > 2.0:
        decision = "RETRY"
        reason = "unstable_motion"

    if decision == "ACCEPT" and rppg_valid < -1.0:
        decision = "RETRY"
        reason = "weak_rppg_quality"

    return decision, reason


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--external_visual_preds", required=True)
    parser.add_argument("--external_behav", required=True)
    parser.add_argument("--fusion_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    visual = read_csv(args.external_visual_preds)
    behav = read_csv(args.external_behav)

    required = {"video_id", "label", "score_spoof"}
    missing = required - set(visual.columns)
    if missing:
        raise ValueError(f"external_visual_preds missing columns: {missing}")

    visual = visual[["video_id", "label", "score_spoof"]].copy()
    visual = visual.rename(columns={"score_spoof": "score_visual"})

    if "label" in behav.columns:
        behav = behav.drop(columns=["label"])

    df = visual.merge(behav, on="video_id", how="left").fillna(0.0)

    fusion_dir = Path(args.fusion_dir)

    with open(fusion_dir / "behavior_clf.pkl", "rb") as f:
        behavior_clf = pickle.load(f)

    with open(fusion_dir / "rppg_clf.pkl", "rb") as f:
        rppg_clf = pickle.load(f)

    with open(fusion_dir / "fusion_v4_metrics.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    weights = info["best_weights_from_val"]
    w_visual = float(weights["w_visual"])
    w_behavior = float(weights["w_behavior"])
    w_rppg = float(weights["w_rppg"])

    df["score_behavior"] = predict_branch(behavior_clf, df, BEHAV_COLS)
    df["score_rppg"] = predict_branch(rppg_clf, df, RPPG_COLS)

    df["score_fusion_v4"] = np.clip(
        w_visual * df["score_visual"].values +
        w_behavior * df["score_behavior"].values +
        w_rppg * df["score_rppg"].values,
        0.0,
        1.0,
    )

    df["pred_fusion_v4"] = (df["score_fusion_v4"] >= 0.5).astype(int)

    decisions = []
    reasons = []

    for _, row in df.iterrows():
        motion_max = float(row["motion_max"]) if "motion_max" in df.columns else 0.0
        rppg_valid = float(row["rppg_valid"]) if "rppg_valid" in df.columns else 1.0
        d, r = apply_policy(float(row["score_fusion_v4"]), motion_max, rppg_valid)
        decisions.append(d)
        reasons.append(r)

    df["decision_v4"] = decisions
    df["decision_reason"] = reasons

    y = df["label"].astype(int).values

    metrics = {
        "visual_only_external": metrics_from_scores(y, df["score_visual"].values),
        "behavior_only_external": metrics_from_scores(y, df["score_behavior"].values),
        "rppg_only_external": metrics_from_scores(y, df["score_rppg"].values),
        "fusion_v4_external": metrics_from_scores(y, df["score_fusion_v4"].values),
        "weights_used": {
            "w_visual": w_visual,
            "w_behavior": w_behavior,
            "w_rppg": w_rppg,
        },
        "policy_counts": df["decision_v4"].value_counts().to_dict(),
    }

    df.to_csv(out_dir / "external_scores_fusion_v4.csv", index=False, encoding="utf-8")

    with (out_dir / "external_fusion_v4_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
