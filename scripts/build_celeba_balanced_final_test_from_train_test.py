import argparse
from pathlib import Path
import pandas as pd


def normalize_label(value):
    """
    Convention :
    0 = REAL
    1 = SPOOF / FAKE
    """
    if pd.isna(value):
        return None

    v = str(value).strip().lower()

    real_values = {
        "0", "real", "live", "bona_fide", "bonafide",
        "genuine", "true", "authentic"
    }

    spoof_values = {
        "1", "fake", "spoof", "attack", "presentation_attack",
        "false", "pa"
    }

    if v in real_values:
        return 0

    if v in spoof_values:
        return 1

    try:
        f = float(v)
        if f == 0:
            return 0
        if f == 1:
            return 1
    except Exception:
        pass

    return None


def fix_path(path_value, root_dir):
    p = Path(str(path_value))

    if p.is_absolute():
        return str(p)

    return str(root_dir / p)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--root_dir", default="")

    parser.add_argument("--n_real", type=int, default=1000)
    parser.add_argument("--n_fake", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    output_csv = Path(args.output_csv)
    root_dir = Path(args.root_dir) if args.root_dir else None

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print("[INFO] Colonnes train :", list(train_df.columns))
    print("[INFO] Colonnes test  :", list(test_df.columns))

    required_cols = {"image_path", "label"}

    if not required_cols.issubset(train_df.columns):
        raise ValueError("Le CSV train doit contenir les colonnes image_path et label.")

    if not required_cols.issubset(test_df.columns):
        raise ValueError("Le CSV test doit contenir les colonnes image_path et label.")

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["source_csv"] = "train"
    test_df["source_csv"] = "test"

    df = pd.concat([train_df, test_df], ignore_index=True)

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    if root_dir is not None:
        df["image_path"] = df["image_path"].apply(lambda p: fix_path(p, root_dir))

    # Supprimer les doublons éventuels
    before_drop = len(df)
    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    after_drop = len(df)

    print("[INFO] Total avant suppression doublons :", before_drop)
    print("[INFO] Total après suppression doublons :", after_drop)

    real_df = df[df["label"] == 0].copy()
    fake_df = df[df["label"] == 1].copy()

    print("[INFO] REAL disponibles :", len(real_df))
    print("[INFO] FAKE/SPOOF disponibles :", len(fake_df))

    if len(real_df) < args.n_real:
        raise ValueError(
            f"REAL insuffisant : {len(real_df)} disponible, {args.n_real} demandé."
        )

    if len(fake_df) < args.n_fake:
        raise ValueError(
            f"FAKE/SPOOF insuffisant : {len(fake_df)} disponible, {args.n_fake} demandé."
        )

    real_sample = real_df.sample(n=args.n_real, random_state=args.seed)
    fake_sample = fake_df.sample(n=args.n_fake, random_state=args.seed)

    final_df = pd.concat([real_sample, fake_sample], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    final_df["class_name"] = final_df["label"].map({0: "REAL", 1: "SPOOF"})

    # Garder les colonnes utiles
    keep_cols = ["image_path", "label", "class_name", "source_csv"]

    if "split" in final_df.columns:
        keep_cols.append("split")

    if "person_id" in final_df.columns:
        keep_cols.append("person_id")

    final_df = final_df[keep_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False, encoding="utf-8")

    print("\n[OK] Dataset CelebA final équilibré créé :")
    print(output_csv)

    print("\n[INFO] Distribution finale :")
    print(final_df["class_name"].value_counts())

    print("\n[INFO] Origine des images utilisées :")
    print(pd.crosstab(final_df["source_csv"], final_df["class_name"]))


if __name__ == "__main__":
    main()