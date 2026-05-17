from pathlib import Path
import argparse
import subprocess
import time
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--behav_csv", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--eval_script", default="tools/eval/eval_final_mixed_casia_axon_local_msu_report.py")
    args = parser.parse_args()

    frames = pd.read_csv(args.frames_csv)
    n_videos = frames["video_id"].nunique()

    cmd = [
        "python",
        args.eval_script,
        "--model", args.model,
        "--frames_csv", args.frames_csv,
        "--behav_csv", args.behav_csv,
        "--out_root", args.out_root,
    ]

    print("========== RUNTIME MEASUREMENT ==========")
    print("Videos:", n_videos)
    print("Command:")
    print(" ".join(cmd))

    start = time.perf_counter()

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    end = time.perf_counter()
    elapsed = end - start

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Evaluation script failed.")

    avg_per_video = elapsed / n_videos if n_videos else 0.0
    videos_per_second = n_videos / elapsed if elapsed > 0 else 0.0

    runtime = {
        "n_videos": int(n_videos),
        "total_seconds": round(elapsed, 4),
        "avg_seconds_per_video": round(avg_per_video, 6),
        "videos_per_second": round(videos_per_second, 4),
    }

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "runtime_summary.json").open("w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)

    print("\n========== RUNTIME SUMMARY ==========")
    print(json.dumps(runtime, indent=2))
    print("Saved:", out_dir / "runtime_summary.json")


if __name__ == "__main__":
    main()