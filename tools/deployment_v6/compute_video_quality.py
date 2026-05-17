from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd


def resolve_path(p):
    p = Path(str(p))
    if p.exists():
        return p

    alt = Path.cwd() / p
    if alt.exists():
        return alt

    return p


def frame_quality(img):
    if img is None:
        return {
            "brightness": 0.0,
            "blur": 0.0,
            "valid": 0.0,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray) / 255.0)
    blur = float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()) / 10.0)

    return {
        "brightness": brightness,
        "blur": blur,
        "valid": 1.0,
    }


def normalize_quality_score(df):
    """
    Score simple entre 0 et 1 :
    - blur_mean élevé = mieux
    - brightness proche de 0.5 = mieux
    - valid_rate élevé = mieux
    - brightness_std trop élevé = moins bien
    """

    out = df.copy()

    blur = out["blur_mean"].clip(0, 1)
    valid = out["frame_valid_rate"].clip(0, 1)

    brightness_balance = 1.0 - (out["brightness_mean"] - 0.5).abs() * 2.0
    brightness_balance = brightness_balance.clip(0, 1)

    lighting_stability = 1.0 - out["brightness_std"].clip(0, 1)

    out["video_quality_score"] = (
        0.35 * blur +
        0.30 * valid +
        0.20 * brightness_balance +
        0.15 * lighting_stability
    ).clip(0, 1)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--max_frames_per_video", type=int, default=64)
    args = parser.parse_args()

    frames = pd.read_csv(args.frames_csv)

    required = {"video_id", "frame_idx", "path", "label"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Columns={frames.columns.tolist()}")

    frames["video_id"] = frames["video_id"].astype(str)
    frames = frames.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

    rows = []

    for video_id, g in frames.groupby("video_id"):
        g = g.sort_values("frame_idx").reset_index(drop=True)

        if len(g) > args.max_frames_per_video:
            idx = np.linspace(0, len(g) - 1, args.max_frames_per_video).astype(int)
            g = g.iloc[idx].copy()

        values = []
        label = int(g["label"].iloc[0])

        for _, r in g.iterrows():
            img_path = resolve_path(r["path"])
            img = cv2.imread(str(img_path))
            values.append(frame_quality(img))

        q = pd.DataFrame(values)

        rows.append({
            "video_id": video_id,
            "label": label,
            "brightness_mean": float(q["brightness"].mean()),
            "brightness_std": float(q["brightness"].std() if len(q) > 1 else 0.0),
            "blur_mean": float(q["blur"].mean()),
            "blur_std": float(q["blur"].std() if len(q) > 1 else 0.0),
            "frame_valid_rate": float(q["valid"].mean()),
            "n_frames_quality": int(len(q)),
        })

        print(f"[OK] {video_id} | frames={len(q)}")

    out = pd.DataFrame(rows)
    out = normalize_quality_score(out)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")

    print("\\nSaved:", out_path)
    print(out[[
        "video_id",
        "video_quality_score",
        "brightness_mean",
        "blur_mean",
        "frame_valid_rate"
    ]].head().to_string(index=False))


if __name__ == "__main__":
    main()
