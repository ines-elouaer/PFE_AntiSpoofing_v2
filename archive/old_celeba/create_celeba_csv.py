import os
from pathlib import Path
import pandas as pd

base_path = r"E:\CelebA-Spoof\CelebA_Spoof\Data"

project_root = Path(__file__).resolve().parents[2]
out_dir = project_root / "data" / "celeba"
out_dir.mkdir(parents=True, exist_ok=True)

data = []

for split in ["train", "test"]:
    split_path = os.path.join(base_path, split)

    if not os.path.exists(split_path):
        print(f"{split_path} n'existe pas")
        continue

    for person_id in os.listdir(split_path):
        person_path = os.path.join(split_path, person_id)

        if not os.path.isdir(person_path):
            continue

        live_path = os.path.join(person_path, "live")
        if os.path.exists(live_path):
            for file in os.listdir(live_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(live_path, file)
                    rel_path = os.path.relpath(full_path, base_path)
                    data.append([rel_path, 0, split, person_id])

        spoof_path = os.path.join(person_path, "spoof")
        if os.path.exists(spoof_path):
            for file in os.listdir(spoof_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(spoof_path, file)
                    rel_path = os.path.relpath(full_path, base_path)
                    data.append([rel_path, 1, split, person_id])

df = pd.DataFrame(data, columns=["image_path", "label", "split", "person_id"])

print("Total images trouvées :", len(df))
print(df["label"].value_counts(dropna=False))
print(df["split"].value_counts(dropna=False))

out_path = out_dir / "celeba_full.csv"
df.to_csv(out_path, index=False)

print(f"Fichier créé ✔ : {out_path}")
