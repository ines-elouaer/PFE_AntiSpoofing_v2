from pathlib import Path
import argparse
import subprocess
import sys
import pandas as pd


def run_one(args, video_id, out_path, target_class):
    cmd = [
        sys.executable,
        "tools/explainability/gradcam_final_model_sequence.py",
        "--model", args.model,
        "--frames_csv", args.frames_csv,
        "--behav_csv", args.behav_csv,
        "--video_id", str(video_id),
        "--out", str(out_path),
        "--target_class", str(target_class),
        "--temporal_pool", args.temporal_pool,
        "--behav_dim", str(args.behav_dim),
        "--behav_hidden", str(args.behav_hidden),
        "--use_gated_fusion", str(args.use_gated_fusion),
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        print("[ERROR]", video_id)
        print(result.stderr)
        print(result.stdout)
    else:
        print("[OK]", out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--behav_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=3)

    parser.add_argument("--temporal_pool", default="mean")
    parser.add_argument("--behav_dim", type=int, default=15)
    parser.add_argument("--behav_hidden", type=int, default=16)
    parser.add_argument("--use_gated_fusion", type=int, default=1)

    args = parser.parse_args()

    df = pd.read_csv(args.predictions)

    required = {"video_id", "label", "pred_label", "score_spoof"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in predictions.csv: {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "name": "real_to_real",
            "df": df[(df.label == 0) & (df.pred_label == 0)].sort_values("score_spoof", ascending=True).head(args.top_k),
            "target_class": 0,
        },
        {
            "name": "spoof_to_spoof",
            "df": df[(df.label == 1) & (df.pred_label == 1)].sort_values("score_spoof", ascending=False).head(args.top_k),
            "target_class": 1,
        },
        {
            "name": "real_to_spoof",
            "df": df[(df.label == 0) & (df.pred_label == 1)].sort_values("score_spoof", ascending=False).head(args.top_k),
            "target_class": 1,
        },
        {
            "name": "spoof_to_real",
            "df": df[(df.label == 1) & (df.pred_label == 0)].sort_values("score_spoof", ascending=True).head(args.top_k),
            "target_class": 0,
        },
    ]

    rows = []

    for case in cases:
        case_name = case["name"]
        case_dir = out_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"========== {case_name} ==========")

        for _, row in case["df"].iterrows():
            video_id = row["video_id"]
            score = float(row["score_spoof"])
            safe_id = str(video_id).replace("\\\\", "_").replace("/", "_").replace(":", "_")

            out_path = case_dir / f"{safe_id}_score_{score:.4f}_final_gradcam.jpg"

            run_one(args, video_id, out_path, case["target_class"])

            rows.append({
                "case": case_name,
                "video_id": video_id,
                "score_spoof": score,
                "target_class": case["target_class"],
                "gradcam_path": str(out_path),
            })

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "final_model_gradcam_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print("\nSaved:", summary_path)


if __name__ == "__main__":
    main()
