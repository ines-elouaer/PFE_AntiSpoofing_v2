import argparse
from pathlib import Path
import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_label_from_path(path: Path):
    """
    Convention :
    0 = REAL
    1 = SPOOF

    Le script cherche les mots live/real/genuine pour REAL
    et spoof/fake/attack pour SPOOF dans le chemin.
    """
    parts = [p.lower() for p in path.parts]

    real_words = {"live", "real", "genuine", "bonafide", "bona_fide"}
    spoof_words = {"spoof", "fake", "attack", "print", "replay", "mask"}

    if any(p in real_words for p in parts):
        return 0

    if any(p in spoof_words for p in parts):
        return 1

    path_str = str(path).lower()

    if any(w in path_str for w in real_words):
        return 0

    if any(w in path_str for w in spoof_words):
        return 1

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dossier racine CelebA réel")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--n_real", type=int, default=1000)
    parser.add_argument("--n_fake", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    output_csv = Path(args.output_csv)

    if not root.exists():
        raise FileNotFoundError(f"Dossier introuvable : {root}")

    rows = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in IMAGE_EXTS:
            continue

        label = infer_label_from_path(p)

        if label is None:
            continue

        rows.append({
            "image_path": str(p),
            "label": label,
            "class_name": "REAL" if label == 0 else "SPOOF",
            "source_root": str(root)
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "Aucune image labellisée trouvée. Vérifie que les dossiers contiennent "
            "des mots comme live/real ou spoof/fake."
        )

    print("[INFO] Images détectées :")
    print(df["class_name"].value_counts())

    real_df = df[df["label"] == 0]
    spoof_df = df[df["label"] == 1]

    if len(real_df) < args.n_real:
        raise ValueError(f"REAL insuffisant : {len(real_df)} disponible, {args.n_real} demandé.")

    if len(spoof_df) < args.n_fake:
        raise ValueError(f"SPOOF insuffisant : {len(spoof_df)} disponible, {args.n_fake} demandé.")

    real_sample = real_df.sample(n=args.n_real, random_state=args.seed)
    spoof_sample = spoof_df.sample(n=args.n_fake, random_state=args.seed)

    final_df = pd.concat([real_sample, spoof_sample], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False, encoding="utf-8")

    print("\n[OK] CSV externe CelebA créé :", output_csv)
    print(final_df["class_name"].value_counts())
    print(final_df.head())


if __name__ == "__main__":
    main()