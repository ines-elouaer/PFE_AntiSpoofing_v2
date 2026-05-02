from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data" / "celeba"

full_csv = data_dir / "celeba_full.csv"
N_PER_CLASS = 4000  # 4000 real + 4000 spoof = 8000 total
subset_csv = data_dir / "celeba_subset_8000.csv"
df = pd.read_csv(full_csv)

df_train = df[df["split"] == "train"].copy()

print("Train total:", len(df_train))
print(df_train["label"].value_counts())

df_real = df_train[df_train["label"] == 0]
df_spoof = df_train[df_train["label"] == 1]

df_real_sample = df_real.sample(n=N_PER_CLASS, random_state=42)
df_spoof_sample = df_spoof.sample(n=N_PER_CLASS, random_state=42)
df_subset = pd.concat([df_real_sample, df_spoof_sample], axis=0)
df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

print("Subset total:", len(df_subset))
print(df_subset["label"].value_counts())

df_subset.to_csv(subset_csv, index=False)
print(f"Fichier créé ✔ : {subset_csv}")