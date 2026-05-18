from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd


def compute_brightness(gray):
    return float(np.mean(gray) / 255.0)


def compute_blur(gray):
    value = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.log1p(max(value, 0.0)) / 10.0)


def compute_motion(prev_gray, gray):
    if prev_gray is None:
        return 0.0, 0.0

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag)), float(np.max(mag))


def build_sequences(frames_csv, out_csv):
    df = pd.read_csv(frames_csv)

    required = {"video_id", "frame_idx", "path", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in frames_csv: {missing}. Columns={df.columns.tolist()}")

    df["video_id"] = df["video_id"].astype(str)
    df = df.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

    rows = []

    for video_id, g in df.groupby("video_id"):
        g = g.sort_values("frame_idx").reset_index(drop=True)

        prev_gray = None

        for _, r in g.iterrows():
            img_path = str(r["path"])
            img = cv2.imread(img_path)

            if img is None:
                rows.append({
                    "video_id": video_id,
                    "frame_idx": int(r["frame_idx"]),
                    "label": int(r["label"]),
                    "brightness": 0.0,
                    "blur": 0.0,
                    "motion_mean": 0.0,
                    "motion_max": 0.0,
                    "frame_valid": 0.0,
                })
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (128, 128))

            brightness = compute_brightness(gray_small)
            blur = compute_blur(gray_small)
            motion_mean, motion_max = compute_motion(prev_gray, gray_small)

            prev_gray = gray_small

            rows.append({
                "video_id": video_id,
                "frame_idx": int(r["frame_idx"]),
                "label": int(r["label"]),
                "brightness": brightness,
                "blur": blur,
                "motion_mean": motion_mean,
                "motion_max": motion_max,
                "frame_valid": 1.0,
            })

    out = pd.DataFrame(rows)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")

    print("Saved:", out_path)
    print("Rows:", len(out))
    print("Videos:", out["video_id"].nunique())
    print(out.head().to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    build_sequences(args.frames_csv, args.out_csv)


if __name__ == "__main__":
    main()
