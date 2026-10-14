from pathlib import Path
import argparse
import subprocess
import sys
import pandas as pd


def run_one(args, video_id, out_path, target_class, frame_position):
    cmd = [
        sys.executable,
        "tools/explainability/gradcam_final_model_sequence.py",
        "--model", args.model,
        "--frames_csv", args.frames_csv,
        "--behav_csv", args.behav_csv,
        "--video_id", str(video_id),
        "--out", str(out_path),
        "--target_class", str(target_class),
        "--frame_position", str(frame_position),
        "--temporal_pool", args.temporal_pool,
        "--behav_dim", str(args.behav_dim),
        "--behav_hidden", str(args.behav_hidden),
        "--use_gated_fusion", str(args.use_gated_fusion),
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        print("[ERROR]", video_id, "frame_position=", frame_position)
        print(result.stderr)
        print(result.stdout)
    else:
        print("[OK]", out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--behav_csv", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--positions", default="2,5,8,11,14")

    parser.add_argument("--temporal_pool", default="mean")
    parser.add_argument("--behav_dim", type=int, default=15)
    parser.add_argument("--behav_hidden", type=int, default=16)
    parser.add_argument("--use_gated_fusion", type=int, default=1)

    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    positions = [int(x.strip()) for x in args.positions.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, row in summary.iterrows():
        case = row["case"]
        video_id = row["video_id"]
        score = float(row["score_spoof"])
        target_class = int(row["target_class"])

        safe_id = str(video_id).replace("\\\\", "_").replace("/", "_").replace(":", "_")
        case_dir = out_dir / str(case)
        case_dir.mkdir(parents=True, exist_ok=True)

        for pos in positions:
            out_path = case_dir / f"{safe_id}_score_{score:.4f}_pos_{pos}_gradcam.jpg"

            run_one(args, video_id, out_path, target_class, pos)

            rows.append({
                "case": case,
                "video_id": video_id,
                "score_spoof": score,
                "target_class": target_class,
                "frame_position": pos,
                "gradcam_path": str(out_path),
            })

    out = pd.DataFrame(rows)
    out_path = out_dir / "multiframe_gradcam_summary.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
