from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "celeba"
    data_dir.mkdir(parents=True, exist_ok=True)

    subset_csv = data_dir / "celeba_subset_8000.csv"
    if not subset_csv.exists():
        raise FileNotFoundError(f"Fichier introuvable: {subset_csv}")

    df = pd.read_csv(subset_csv)

    print("Total images:", len(df))
    print("Nb identities:", df["person_id"].nunique())

    ids = df["person_id"].unique()

    train_ids, temp_ids = train_test_split(
        ids,
        test_size=0.2,
        random_state=42
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=42
    )

    train_df = df[df["person_id"].isin(train_ids)].copy()
    val_df = df[df["person_id"].isin(val_ids)].copy()
    test_df = df[df["person_id"].isin(test_ids)].copy()

    print("\nTrain images:", len(train_df), "| identities:", train_df["person_id"].nunique())
    print("Val images:", len(val_df), "| identities:", val_df["person_id"].nunique())
    print("Test images:", len(test_df), "| identities:", test_df["person_id"].nunique())

    print("\nTrain labels:\n", train_df["label"].value_counts())
    print("\nVal labels:\n", val_df["label"].value_counts())
    print("\nTest labels:\n", test_df["label"].value_counts())

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    print("\nOverlap train/val:", len(train_set & val_set))
    print("Overlap train/test:", len(train_set & test_set))
    print("Overlap val/test:", len(val_set & test_set))

    train_out = data_dir / "celeba_8000_ft_train.csv"
    val_out = data_dir / "celeba_8000_ft_val.csv"
    test_out = data_dir / "celeba_8000_ft_test.csv"
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print("\nFichiers créés ✔")
    print(train_out)
    print(val_out)
    print(test_out)


if __name__ == "__main__":
    main()