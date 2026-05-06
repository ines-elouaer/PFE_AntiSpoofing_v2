import re
import csv
import random
from pathlib import Path
from collections import defaultdict

RANDOM_SEED = 42


PATTERN = re.compile(
    r"^(?P<vid>\d+_\d+)\.avi_(?P<frame>\d+)_(?P<label>real|fake)\.jpg$",
    re.IGNORECASE
)

def parse_filename(name: str):
    m = PATTERN.match(name)
    if not m:
        return None
    video_id = m.group("vid")           
    frame_idx = int(m.group("frame"))   
    label_str = m.group("label").lower()
    label = 0 if label_str == "real" else 1
    return video_id, frame_idx, label

def write_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "video_id", "frame_idx"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main(img_dir: str, out_dir: str, train_ratio=0.8, val_ratio=0.1):
    random.seed(RANDOM_SEED)

    img_dir = Path(img_dir)
    out_dir = Path(out_dir)

    rows_by_video = defaultdict(list)

    for p in img_dir.glob("*.jpg"):
        parsed = parse_filename(p.name)
        if parsed is None:
            continue
        video_id, frame_idx, label = parsed
        rows_by_video[video_id].append({
            "path": str(p.resolve()),
            "label": label,
            "video_id": video_id,
            "frame_idx": frame_idx
        })

    video_ids = sorted(rows_by_video.keys())
    if not video_ids:
        raise RuntimeError(f"Aucune image CASIA valide trouvée dans: {img_dir}")

    random.shuffle(video_ids)

    n = len(video_ids)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_vids = set(video_ids[:n_train])
    val_vids   = set(video_ids[n_train:n_train + n_val])
    test_vids  = set(video_ids[n_train + n_val:])

    def rows_for(vids):
        out = []
        for vid in sorted(vids):
            out.extend(sorted(rows_by_video[vid], key=lambda r: r["frame_idx"]))
        return out

    train_rows = rows_for(train_vids)
    val_rows   = rows_for(val_vids)
    test_rows  = rows_for(test_vids)

    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)
    write_csv(out_dir / "test.csv", test_rows)

    print(" Splits CASIA créés (GroupSplit par video_id)")
    print(f"Total videos: {n}")
    print(f"Train videos: {len(train_vids)} | images: {len(train_rows)}")
    print(f"Val   videos: {len(val_vids)} | images: {len(val_rows)}")
    print(f"Test  videos: {len(test_vids)} | images: {len(test_rows)}")

if __name__ == "__main__":
    IMG_DIR = r"data\processed\casia\all_img\color"
    OUT_DIR = r"data\processed\casia\splits"
    main(IMG_DIR, OUT_DIR)
