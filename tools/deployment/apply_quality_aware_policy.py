from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def quality_level(q):
    if q >= 0.70:
        return "GOOD"
    elif q >= 0.45:
        return "MEDIUM"
    else:
        return "LOW"

def decide(score, q):
    """
    Politique quality-aware finale.

    Convention :
    - score proche de 0 => REAL probable
    - score proche de 1 => SPOOF probable
    """

    score = float(score)
    q = float(q)

    level = quality_level(q)

    if level == "GOOD":
        if score < 0.35:
            return "ACCEPT", "good_quality_low_spoof_score"
        elif score < 0.75:
            return "RETRY", "good_quality_gray_zone"
        else:
            return "REJECT", "good_quality_high_spoof_score"

    if level == "MEDIUM":
        if score < 0.30:
            return "ACCEPT", "medium_quality_low_spoof_score"
        elif score < 0.80:
            return "RETRY", "medium_quality_gray_zone"
        else:
            return "REJECT", "medium_quality_high_spoof_score"

    # LOW quality
    if score < 0.25:
        return "ACCEPT", "low_quality_low_spoof_score"
    elif score < 0.85:
        return "RETRY", "low_quality_retry_zone"
    else:
        return "REJECT", "low_quality_high_spoof_score"

def summarize_policy(df):
    total = len(df)

    counts = df["decision_quality_aware"].value_counts().to_dict()

    real = df[df["label"] == 0]
    spoof = df[df["label"] == 1]

    summary = {
        "total": int(total),
        "decision_counts": {k: int(v) for k, v in counts.items()},
        "real_count": int(len(real)),
        "spoof_count": int(len(spoof)),

        "real_accept": int((real["decision_quality_aware"] == "ACCEPT").sum()),
        "real_retry": int((real["decision_quality_aware"] == "RETRY").sum()),
        "real_reject": int((real["decision_quality_aware"] == "REJECT").sum()),

        "spoof_accept": int((spoof["decision_quality_aware"] == "ACCEPT").sum()),
        "spoof_retry": int((spoof["decision_quality_aware"] == "RETRY").sum()),
        "spoof_reject": int((spoof["decision_quality_aware"] == "REJECT").sum()),

        "real_reject_rate": float((real["decision_quality_aware"] == "REJECT").mean()) if len(real) else 0.0,
        "real_retry_rate": float((real["decision_quality_aware"] == "RETRY").mean()) if len(real) else 0.0,
        "spoof_accept_rate": float((spoof["decision_quality_aware"] == "ACCEPT").mean()) if len(spoof) else 0.0,
        "spoof_retry_rate": float((spoof["decision_quality_aware"] == "RETRY").mean()) if len(spoof) else 0.0,
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", required=True)
    parser.add_argument("--quality_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores_csv)
    quality = pd.read_csv(args.quality_csv)

    scores["video_id"] = scores["video_id"].astype(str)
    quality["video_id"] = quality["video_id"].astype(str)

    if "score_fusion_v6" not in scores.columns:
        raise ValueError("scores_csv must contain score_fusion_v6")

    if "video_quality_score" not in quality.columns:
        raise ValueError("quality_csv must contain video_quality_score")

    if "label" in quality.columns:
        quality = quality.drop(columns=["label"])

    df = scores.merge(quality, on="video_id", how="left")

    df["video_quality_score"] = df["video_quality_score"].fillna(0.5)
    df["quality_level"] = df["video_quality_score"].apply(quality_level)

    decisions = []
    reasons = []

    for _, row in df.iterrows():
        d, r = decide(
            score=float(row["score_fusion_v6"]),
            q=float(row["video_quality_score"])
        )
        decisions.append(d)
        reasons.append(r)

    df["decision_quality_aware"] = decisions
    df["decision_reason_quality_aware"] = reasons

    summary = summarize_policy(df)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False, encoding="utf-8")

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Saved:", out_csv)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
