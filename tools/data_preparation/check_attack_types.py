from pathlib import Path
import pandas as pd


CSV_PATHS = [
    # adapte les chemins selon ton projet
    Path("data/casia_prepared/manifests/train.csv"),
    Path("data/casia_prepared/manifests/val.csv"),
    Path("data/casia_prepared/manifests/test.csv"),

    Path("data/axon_prepared/manifests/axon_video_frames_manifest.csv"),

    Path("data/local_real_prepared/manifests/local_real_frames_manifest.csv"),

    Path("data/msu_mfsd_prepared/manifests/msu_real_16_frames_manifest.csv"),

    Path("data/external_real_fake/manifests/external_frames_manifest.csv"),
]


def show_basic_info(csv_path: Path):
    if not csv_path.exists():
        print(f"\n[SKIP] Introuvable : {csv_path}")
        return

    print("\n" + "=" * 90)
    print(f"CSV : {csv_path}")
    print("=" * 90)

    df = pd.read_csv(csv_path)

    print("\nColonnes disponibles :")
    print(list(df.columns))

    # Nombre de lignes
    print(f"\nNombre de lignes : {len(df)}")

    # Nombre de vidéos si video_id existe
    if "video_id" in df.columns:
        print(f"Nombre de vidéos : {df['video_id'].nunique()}")

    # Distribution des labels
    if "label_name" in df.columns:
        print("\nDistribution label_name :")
        print(df.drop_duplicates("video_id")["label_name"].value_counts() if "video_id" in df.columns else df["label_name"].value_counts())

    elif "label" in df.columns:
        print("\nDistribution label :")
        print(df.drop_duplicates("video_id")["label"].value_counts() if "video_id" in df.columns else df["label"].value_counts())

    # Distribution des types d'attaque
    if "attack_type" in df.columns:
        print("\nTypes d'attaque disponibles :")
        video_df = df.drop_duplicates("video_id") if "video_id" in df.columns else df
        print(video_df["attack_type"].fillna("unknown").value_counts())

    else:
        print("\n[INFO] Colonne attack_type absente.")
        print("On peut essayer d'inférer les attaques depuis path ou video_id.")

        # Inférence simple depuis les chemins
        if "path" in df.columns:
            paths = df["path"].astype(str).str.lower()

            possible_keywords = [
                "print",
                "photo",
                "cut",
                "replay",
                "video",
                "screen",
                "mobile",
                "mask",
                "silicone",
                "paper",
                "cylinder",
                "wrapped",
                "fake",
                "spoof",
                "real",
            ]

            print("\nMots-clés trouvés dans les chemins :")
            for kw in possible_keywords:
                count = paths.str.contains(kw, regex=False).sum()
                if count > 0:
                    print(f"{kw}: {count} lignes")

    # Si source_dataset existe
    if "source_dataset" in df.columns:
        print("\nDistribution par source_dataset :")
        video_df = df.drop_duplicates("video_id") if "video_id" in df.columns else df
        print(video_df["source_dataset"].value_counts())

    # Tableau croisé source_dataset / attack_type
    if "source_dataset" in df.columns and "attack_type" in df.columns:
        video_df = df.drop_duplicates("video_id") if "video_id" in df.columns else df
        print("\nTableau source_dataset x attack_type :")
        print(pd.crosstab(video_df["source_dataset"], video_df["attack_type"].fillna("unknown")))

    # Tableau label / attack_type
    if "label_name" in df.columns and "attack_type" in df.columns:
        video_df = df.drop_duplicates("video_id") if "video_id" in df.columns else df
        print("\nTableau label_name x attack_type :")
        print(pd.crosstab(video_df["label_name"], video_df["attack_type"].fillna("unknown")))


def main():
    for csv_path in CSV_PATHS:
        show_basic_info(csv_path)


if __name__ == "__main__":
    main()