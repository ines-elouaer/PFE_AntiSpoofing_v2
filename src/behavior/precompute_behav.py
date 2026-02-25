import csv
from pathlib import Path
from collections import defaultdict

from .mp_landmarks import FaceLandmarkerHelper, video_features_from_signals
from .extract_from_frames import extract_ear_and_motion_from_frames


def read_rows(csv_path: str):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
          
            row["frame_idx"] = int(row["frame_idx"])
            row["label"] = int(row["label"])
            rows.append(row)
    return rows


def precompute_split(
    split_csv: str,
    out_csv: str,
    every_n: int = 1,
    max_frames: int = 600,
    model_path: str = "models/face_landmarker.task",
):
    rows = read_rows(split_csv)

    by_vid = defaultdict(list)
    for r in rows:
        by_vid[r["video_id"]].append(r)

    landmarker = FaceLandmarkerHelper(model_path=model_path)

    out_rows = []
    for vid, frs in by_vid.items():
        frs = sorted(frs, key=lambda x: x["frame_idx"])
        ears, motions, skipped, used = extract_ear_and_motion_from_frames(
            frs, landmarker, every_n=every_n, max_frames=max_frames
        )

        feats = video_features_from_signals(ears, motions)
        skipped_rate = float(skipped) / float(max(used, 1))

        vid_label = int(frs[0]["label"]) if frs else -1

        out_rows.append({
            "video_id": vid,
            "label": vid_label,          
            **feats,
            "skipped_rate": skipped_rate,
        })

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    if not out_rows:
        cols = ["video_id", "label"] + list(video_features_from_signals([], []).keys()) + ["skipped_rate"]
        with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
        print(f"Saved: {out_csv} | videos: 0")
        return

    cols = list(out_rows[0].keys())
    with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Saved: {out_csv} | videos: {len(out_rows)}")


def main():
    splits = [
        ("train", r"data\processed\casia\splits_subject\train.csv", r"data\processed\casia\behav\train_behav.csv"),
        ("val",   r"data\processed\casia\splits_subject\val.csv",   r"data\processed\casia\behav\val_behav.csv"),
        ("test",  r"data\processed\casia\splits_subject\test.csv",  r"data\processed\casia\behav\test_behav.csv"),
    ]

    for name, in_csv, out_csv in splits:
        print(f"\n=== Precompute BEHAV: {name} ===")
        precompute_split(
            split_csv=in_csv,
            out_csv=out_csv,
            every_n=1,
            max_frames=600,
            model_path="models/face_landmarker.task",
        )


if __name__ == "__main__":
    main()