import csv
from pathlib import Path

def read_video_ids(csv_path: str):
    vids = set()
    with Path(csv_path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            vids.add(row["video_id"])
    return vids

def main():
    train_csv = r"data\processed\casia\splits\train.csv"
    val_csv   = r"data\processed\casia\splits\val.csv"
    test_csv  = r"data\processed\casia\splits\test.csv"

    tr = read_video_ids(train_csv)
    va = read_video_ids(val_csv)
    te = read_video_ids(test_csv)

    print("Overlap train/val :", len(tr & va))
    print("Overlap train/test:", len(tr & te))
    print("Overlap val/test  :", len(va & te))

    if (tr & va) or (tr & te) or (va & te):
        print("Leakage détecté (mêmes video_id dans plusieurs splits)")
    else:
        print("OK: aucun video_id partagé entre train/val/test")

if __name__ == "__main__":
    main()
