# pipelines/train_image_baseline.py
import os
import csv
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

CSV_PATH = os.path.join("results", "image_demo", "image_features.csv")

def load_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV introuvable: {csv_path} (lance pipelines/demo_image_batch.py d'abord)")

    X, y, files = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            files.append(row["file"])
            label = row["label"].strip().upper()
            y.append(1 if label == "REAL" else 0)

            feats = [
                float(row["lap_var"]),
                float(row["mean"]),
                float(row["std"]),
                float(row["rb_diff"]),
                float(row["rg_diff"]),
                float(row["gb_diff"]),
            ]
            X.append(feats)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), files

def save_cm_png(y_true, y_pred, out_path, labels=("FAKE", "REAL")):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    csv_path = os.path.join(base, CSV_PATH)

    X, y, files = load_csv(csv_path)

    print("[INFO] Total images:", len(y))
    print("[INFO] REAL:", int((y == 1).sum()), "| FAKE:", int((y == 0).sum()))

    if len(np.unique(y)) < 2:
        raise RuntimeError("Dataset invalide: il faut au moins REAL et FAKE.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, preds))

    print("\n=== Report ===")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))

    results_dir = os.path.join(base, "results", "image_baseline")
    os.makedirs(results_dir, exist_ok=True)
    save_cm_png(y_test, preds, os.path.join(results_dir, "confusion_matrix.png"))

    out_model = os.path.join(base, "trained_models", "image_baseline_model.joblib")
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump(model, out_model)

    print(f"\n[OK] Modèle sauvegardé: {out_model}")
    print(f"[OK] Confusion matrix saved: {os.path.join(results_dir, 'confusion_matrix.png')}")
