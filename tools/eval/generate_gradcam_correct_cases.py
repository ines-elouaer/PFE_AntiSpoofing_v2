from pathlib import Path
import argparse
import subprocess
import pandas as pd

import sys

def select_center_frame(frames_df, video_id):
    rows = frames_df[frames_df["video_id"].astype(str) == str(video_id)].copy()

    if len(rows) == 0:
        return None

    rows = rows.sort_values("frame_idx").reset_index(drop=True)
    center_idx = len(rows) // 2

    return rows.iloc[center_idx]["path"]


def run_gradcam(image_path, out_path, target_class):
    cmd = [
    sys.executable,
    "tools/explainability/gradcam_mobilenet_frame.py",
    "--image", str(image_path),
    "--out", str(out_path),
    "--target_class", str(target_class),
]

    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        print("[ERROR]", image_path)
        print(result.stderr)
    else:
        print("[OK]", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    frames = pd.read_csv(args.frames_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # REAL correctement détectés : label=0, pred=0
    # On prend les scores spoof les plus faibles.
    real_as_real = pred[
        (pred["label"] == 0) & (pred["pred_label"] == 0)
    ].copy().sort_values("score_spoof", ascending=True).head(args.top_k)

    # SPOOF correctement détectés : label=1, pred=1
    # On prend les scores spoof les plus élevés.
    spoof_as_spoof = pred[
        (pred["label"] == 1) & (pred["pred_label"] == 1)
    ].copy().sort_values("score_spoof", ascending=False).head(args.top_k)

    summary_rows = []

    print("========== REAL -> REAL ==========")
    for _, row in real_as_real.iterrows():
        video_id = row["video_id"]
        score = float(row["score_spoof"])

        frame_path = select_center_frame(frames, video_id)
        if frame_path is None:
            print("[WARN] No frame found:", video_id)
            continue

        safe_id = str(video_id).replace("\\", "_").replace("/", "_").replace(":", "_")
        out_path = out_dir / "real_as_real" / f"{safe_id}_score_{score:.4f}_gradcam.jpg"

        # target_class=1 ici garde une visualisation "spoof-like" comme tes premiers tests.
        # Pour REAL-like, tu peux aussi tester target_class=0.
        run_gradcam(frame_path, out_path, target_class=1)

        summary_rows.append({
            "case": "REAL->REAL",
            "video_id": video_id,
            "score_spoof": score,
            "frame_path": frame_path,
            "gradcam_path": str(out_path),
        })

    print("\n========== SPOOF -> SPOOF ==========")
    for _, row in spoof_as_spoof.iterrows():
        video_id = row["video_id"]
        score = float(row["score_spoof"])

        frame_path = select_center_frame(frames, video_id)
        if frame_path is None:
            print("[WARN] No frame found:", video_id)
            continue

        safe_id = str(video_id).replace("\\", "_").replace("/", "_").replace(":", "_")
        out_path = out_dir / "spoof_as_spoof" / f"{safe_id}_score_{score:.4f}_gradcam.jpg"

        run_gradcam(frame_path, out_path, target_class=1)

        summary_rows.append({
            "case": "SPOOF->SPOOF",
            "video_id": video_id,
            "score_spoof": score,
            "frame_path": frame_path,
            "gradcam_path": str(out_path),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "gradcam_correct_cases_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print("\nSaved summary:", summary_path)


if __name__ == "__main__":
    main()