from pathlib import Path
import pandas as pd


def load_video_ids_from_frames(frames_csv):
    p = Path(frames_csv)
    if not p.exists():
        raise FileNotFoundError(f"Train frames CSV introuvable: {p}")

    df = pd.read_csv(p)

    if "video_id" not in df.columns:
        raise ValueError(f"Colonne video_id introuvable dans {p}. Colonnes: {df.columns.tolist()}")

    return set(df["video_id"].astype(str).unique())


def load_hard_ids(path):
    p = Path(path)
    if not p.exists():
        print(f"[WARN] fichier introuvable: {p}")
        return set()

    df = pd.read_csv(p)

    if "video_id" not in df.columns:
        raise ValueError(f"Colonne video_id introuvable dans {p}. Colonnes: {df.columns.tolist()}")

    return set(df["video_id"].astype(str).unique())


def main():
    train_csv = Path("data/mixed_casia_axon_local_msu/mixed_train_frames.csv")
    hard_real_csv = Path("data/hard_examples/hard_real_samples.csv")
    hard_spoof_csv = Path("data/hard_examples/hard_spoof_samples.csv")

    train_ids = load_video_ids_from_frames(train_csv)
    hard_real = load_hard_ids(hard_real_csv)
    hard_spoof = load_hard_ids(hard_spoof_csv)

    hard_real_found = train_ids & hard_real
    hard_real_missing = hard_real - train_ids

    hard_spoof_found = train_ids & hard_spoof
    hard_spoof_missing = hard_spoof - train_ids

    print("========== CHECK HARD EXAMPLES ==========")
    print(f"Train videos              : {len(train_ids)}")
    print(f"Hard REAL in CSV          : {len(hard_real)}")
    print(f"Hard SPOOF in CSV         : {len(hard_spoof)}")

    print("\\n========== HARD REAL ==========")
    print(f"Found in train            : {len(hard_real_found)}")
    print(f"Not found in train        : {len(hard_real_missing)}")
    if hard_real_found:
        print("\\nHard REAL found:")
        for vid in sorted(hard_real_found):
            print("  ", vid)
    if hard_real_missing:
        print("\\nHard REAL not found:")
        for vid in sorted(hard_real_missing):
            print("  ", vid)

    print("\\n========== HARD SPOOF ==========")
    print(f"Found in train            : {len(hard_spoof_found)}")
    print(f"Not found in train        : {len(hard_spoof_missing)}")
    if hard_spoof_found:
        print("\\nHard SPOOF found:")
        for vid in sorted(hard_spoof_found):
            print("  ", vid)
    if hard_spoof_missing:
        print("\\nHard SPOOF not found:")
        for vid in sorted(hard_spoof_missing):
            print("  ", vid)

    out_dir = Path("reports/hard_examples_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for vid in sorted(hard_real):
        rows.append({
            "video_id": vid,
            "hard_type": "REAL_AS_SPOOF",
            "found_in_train": vid in train_ids,
        })

    for vid in sorted(hard_spoof):
        rows.append({
            "video_id": vid,
            "hard_type": "SPOOF_AS_REAL",
            "found_in_train": vid in train_ids,
        })

    out = pd.DataFrame(rows)
    out_path = out_dir / "hard_examples_train_check.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    print("\\nSaved:", out_path)


if __name__ == "__main__":
    main()
