from pathlib import Path
import argparse
import json
import pandas as pd


def summarize(df, decision_col):
    real = df[df["label"] == 0]
    spoof = df[df["label"] == 1]

    return {
        "decision_column": decision_col,
        "total": int(len(df)),

        "decision_counts": {
            k: int(v) for k, v in df[decision_col].value_counts().to_dict().items()
        },

        "real_count": int(len(real)),
        "spoof_count": int(len(spoof)),

        "real_accept": int((real[decision_col] == "ACCEPT").sum()),
        "real_retry": int((real[decision_col] == "RETRY").sum()),
        "real_reject": int((real[decision_col] == "REJECT").sum()),

        "spoof_accept": int((spoof[decision_col] == "ACCEPT").sum()),
        "spoof_retry": int((spoof[decision_col] == "RETRY").sum()),
        "spoof_reject": int((spoof[decision_col] == "REJECT").sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_csv", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.policy_csv)

    summaries = {}

    if "decision_v6" in df.columns:
        summaries["old_policy"] = summarize(df, "decision_v6")

    if "decision_quality_aware" in df.columns:
        summaries["quality_aware_policy"] = summarize(df, "decision_quality_aware")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(json.dumps(summaries, indent=2))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
