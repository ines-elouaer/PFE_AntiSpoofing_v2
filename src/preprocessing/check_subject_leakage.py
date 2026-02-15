import csv
from pathlib import Path

def read_subjects(csv_path: str):
    s = set()
    with Path(csv_path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s.add(row["subject_id"])
    return s

def main():
    train_csv = r"data\processed\CASIA\splits_subject\train.csv"
    val_csv   = r"data\processed\CASIA\splits_subject\val.csv"
    test_csv  = r"data\processed\CASIA\splits_subject\test.csv"

    tr = read_subjects(train_csv)
    va = read_subjects(val_csv)
    te = read_subjects(test_csv)

    print("Overlap train/val :", len(tr & va))
    print("Overlap train/test:", len(tr & te))
    print("Overlap val/test  :", len(va & te))

    if (tr & va) or (tr & te) or (va & te):
        print("Leakage subject détecté")
    else:
        print("OK: aucun subject_id partagé")

if __name__ == "__main__":
    main()
