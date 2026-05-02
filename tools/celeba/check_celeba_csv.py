import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "celeba"

files = [
    DATA_DIR / "celeba_full.csv",
    DATA_DIR / "celeba_subset_8000.csv",
    DATA_DIR / "celeba_8000_ft_train.csv",
    DATA_DIR / "celeba_8000_ft_val.csv",
    DATA_DIR / "celeba_8000_ft_test.csv",
]

for path in files:
    print("\n" + "=" * 70)
    print(path)

    if not path.exists():
        print("❌ Fichier introuvable")
        continue

    df = pd.read_csv(path)

    print("Nb lignes:", len(df))
    print("Colonnes:", list(df.columns))

    if "label" in df.columns:
        print("\nLabels:")
        print(df["label"].value_counts(dropna=False))

    if "person_id" in df.columns:
        print("\nNb identities:", df["person_id"].nunique())

    if "image_path" in df.columns:
        print("\nExemples chemins:")
        print(df["image_path"].head(3))

    print("\nNull values:")
    print(df.isnull().sum())

    print("\nDoublons image_path:", df["image_path"].duplicated().sum() if "image_path" in df.columns else "N/A")