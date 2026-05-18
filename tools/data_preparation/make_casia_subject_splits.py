import re
import csv
import random
from pathlib import Path
from collections import defaultdict

RANDOM_SEED = 42

PATTERN = re.compile(
    r"^(?P<subj>\d+)_(?P<vid>\d+)\.avi_(?P<frame>\d+)_(?P<label>real|fake)\.jpg$",
    re.IGNORECASE
)

def parse_filename(name):
    m = PATTERN.match(name)
    if not m:
        return None

    subject_id = m.group("subj")
    video_id = f"{m.group('subj')}_{m.group('vid')}"
    frame_idx = int(m.group("frame"))
    label = 0 if m.group("label").lower() == "real" else 1

    return subject_id, video_id, frame_idx, label


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "label", "subject_id", "video_id", "frame_idx"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    img_dir = Path("data/processed/CASIA/all_img/color")
    out_dir = Path("data/processed/CASIA/splits_subject")

    random.seed(RANDOM_SEED)

    rows_by_subject = defaultdict(list)

    for p in img_dir.glob("*.jpg"):
        parsed = parse_filename(p.name)
        if parsed is None:
            continue

        subject_id, video_id, frame_idx, label = parsed

        rows_by_subject[subject_id].append({
            "path": str(p.resolve()),
            "label": label,
            "subject_id": subject_id,
            "video_id": video_id,
            "frame_idx": frame_idx
        })

    subjects = list(rows_by_subject.keys())
    random.shuffle(subjects)

    n = len(subjects)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_sub = set(subjects[:n_train])
    val_sub = set(subjects[n_train:n_train+n_val])
    test_sub = set(subjects[n_train+n_val:])

    def collect(subset):
        rows = []
        for s in subset:
            rows.extend(rows_by_subject[s])
        return rows

    write_csv(out_dir / "train.csv", collect(train_sub))
    write_csv(out_dir / "val.csv", collect(val_sub))
    write_csv(out_dir / "test.csv", collect(test_sub))

    print("Subject-split created")
    print("Train subjects:", len(train_sub))
    print("Val subjects:", len(val_sub))
    print("Test subjects:", len(test_sub))


if __name__ == "__main__":
    main()
