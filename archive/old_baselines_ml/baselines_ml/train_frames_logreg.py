
import os
import sys
import glob
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from features import frame_features  


VALID_EXT = (".jpg", ".jpeg", ".png")


def list_images_recursive(folder):

    pattern = os.path.join(folder, "**", "*")
    paths = sorted(glob.glob(pattern, recursive=True))
    return [p for p in paths if p.lower().endswith(VALID_EXT) and os.path.isfile(p)]


def load_dataset(frames_root):
    X, y, files = [], [], []

    real_dir = os.path.join(frames_root, "real")
    fake_dir = os.path.join(frames_root, "fake")

    real_imgs = list_images_recursive(real_dir)
    fake_imgs = list_images_recursive(fake_dir)

    used_real = used_fake = skipped = 0

   
    for p in real_imgs:
        img = cv2.imread(p)
        if img is None:
            skipped += 1
            continue

        feat = frame_features(img)
        if feat is None:
            skipped += 1
            continue

        X.append(feat)
        y.append(1)
        
        files.append(os.path.relpath(p, frames_root))
        used_real += 1

   
    for p in fake_imgs:
        img = cv2.imread(p)
        if img is None:
            skipped += 1
            continue

        feat = frame_features(img)
        if feat is None:
            skipped += 1
            continue

        X.append(feat)
        y.append(0)
        files.append(os.path.relpath(p, frames_root))
        used_fake += 1

    if len(X) == 0:
        return np.empty((0, 6), dtype=np.float32), np.array([], dtype=np.int32), files, used_real, used_fake, skipped

    return np.vstack(X), np.array(y, dtype=np.int32), files, used_real, used_fake, skipped


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

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    frames_root = os.path.join(BASE_DIR, "data", "frames")

   
    print("[INFO] frames_root =", frames_root)
    print("[INFO] exists real_dir =", os.path.exists(os.path.join(frames_root, "real")))
    print("[INFO] exists fake_dir =", os.path.exists(os.path.join(frames_root, "fake")))

    X, y, files, used_real, used_fake, skipped = load_dataset(frames_root)
    print(f"[INFO] Used REAL: {used_real} | Used FAKE: {used_fake} | Total used: {len(y)} | Skipped: {skipped}")

    if len(y) == 0 or len(np.unique(y)) < 2:
        raise RuntimeError(
            "Dataset invalide: il faut des frames REAL et FAKE détectées.\n"
            "Vérifie que tu as bien des .jpg dans data/frames/real/** et data/frames/fake/**"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, preds))

    print("\n=== Report ===")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"], zero_division=0))

    results_dir = os.path.join(BASE_DIR, "results", "frames_baseline")
    os.makedirs(results_dir, exist_ok=True)
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    save_cm_png(y_test, preds, cm_path)

    out_model = os.path.join(BASE_DIR, "trained_models", "baseline_model.joblib")
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump(model, out_model)

    print(f"\n[OK] Modèle sauvegardé: {out_model}")
    print(f"[OK] Confusion matrix saved: {cm_path}")
