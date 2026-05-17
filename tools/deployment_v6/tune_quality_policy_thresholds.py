from pathlib import Path
import argparse
import json
import itertools
import pandas as pd


def quality_level(q):
    if q >= 0.70:
        return "GOOD"
    elif q >= 0.45:
        return "MEDIUM"
    else:
        return "LOW"


def decide(score, q, params):
    level = quality_level(q)

    if level == "GOOD":
        a = params["good_accept"]
        r = params["good_reject"]
    elif level == "MEDIUM":
        a = params["medium_accept"]
        r = params["medium_reject"]
    else:
        a = params["low_accept"]
        r = params["low_reject"]

    if score < a:
        return "ACCEPT"
    elif score < r:
        return "RETRY"
    else:
        return "REJECT"


def summarize(df, decision_col):
    real = df[df["label"] == 0]
    spoof = df[df["label"] == 1]

    return {
        "real_accept": int((real[decision_col] == "ACCEPT").sum()),
        "real_retry": int((real[decision_col] == "RETRY").sum()),
        "real_reject": int((real[decision_col] == "REJECT").sum()),
        "spoof_accept": int((spoof[decision_col] == "ACCEPT").sum()),
        "spoof_retry": int((spoof[decision_col] == "RETRY").sum()),
        "spoof_reject": int((spoof[decision_col] == "REJECT").sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", required=True)
    parser.add_argument("--quality_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_spoof_accept", type=int, default=1)
    parser.add_argument("--max_spoof_retry", type=int, default=10)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores_csv)
    quality = pd.read_csv(args.quality_csv)

    scores["video_id"] = scores["video_id"].astype(str)
    quality["video_id"] = quality["video_id"].astype(str)

    if "label" in quality.columns:
        quality = quality.drop(columns=["label"])

    df = scores.merge(quality, on="video_id", how="left")
    df["video_quality_score"] = df["video_quality_score"].fillna(0.5)

    rows = []
    best = None

    good_accept_values = [0.30, 0.35, 0.40]
    good_reject_values = [0.70, 0.75, 0.80]

    medium_accept_values = [0.25, 0.30, 0.35]
    medium_reject_values = [0.75, 0.80, 0.85]

    low_accept_values = [0.20, 0.25, 0.30]
    low_reject_values = [0.80, 0.85, 0.90]

    for ga, gr, ma, mr, la, lr in itertools.product(
        good_accept_values,
        good_reject_values,
        medium_accept_values,
        medium_reject_values,
        low_accept_values,
        low_reject_values,
    ):
        if ga >= gr or ma >= mr or la >= lr:
            continue

        params = {
            "good_accept": ga,
            "good_reject": gr,
            "medium_accept": ma,
            "medium_reject": mr,
            "low_accept": la,
            "low_reject": lr,
        }

        decisions = [
            decide(float(row["score_fusion_v6"]), float(row["video_quality_score"]), params)
            for _, row in df.iterrows()
        ]

        tmp = df.copy()
        tmp["decision_tuned"] = decisions
        s = summarize(tmp, "decision_tuned")

        row = {
            **params,
            **s,
        }
        rows.append(row)

        # Contraintes sécurité
        if s["spoof_accept"] > args.max_spoof_accept:
            continue

        if s["spoof_retry"] > args.max_spoof_retry:
            continue

        # Objectif : réduire real_reject, puis réduire spoof_accept, puis spoof_retry
        key = (
            s["real_reject"],
            s["spoof_accept"],
            s["spoof_retry"],
            -s["spoof_reject"],
        )

        if best is None or key < best["key"]:
            best = {
                "key": key,
                "params": params,
                "summary": s,
                "df": tmp,
            }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "quality_policy_grid.csv", index=False, encoding="utf-8")

    if best is None:
        print("No valid policy found under constraints.")
        return

    best["df"].to_csv(out_dir / "external_scores_fusion_v6_quality_policy_tuned.csv", index=False, encoding="utf-8")

    report = {
        "best_params": best["params"],
        "best_summary": best["summary"],
        "constraints": {
            "max_spoof_accept": args.max_spoof_accept,
            "max_spoof_retry": args.max_spoof_retry,
        },
    }

    with (out_dir / "quality_policy_tuned_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
