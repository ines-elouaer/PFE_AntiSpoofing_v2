
import os
import sys
import glob
import csv
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# --- Fix imports (si lancé depuis pipelines/ ou avec -m) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.mp_landmarks import (  # noqa: E402
    FaceLandmarkerHelper,
    extract_ear_and_motion_from_video,
    video_features_from_signals
)


def list_videos(folder):
    exts = (".mp4", ".avi", ".mov", ".mkv")
    paths = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in paths if p.lower().endswith(exts) and os.path.isfile(p)]


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
    real_dir = os.path.join(BASE_DIR, "data", "videos", "real")
    fake_dir = os.path.join(BASE_DIR, "data", "videos", "fake")

    out_dir = os.path.join(BASE_DIR, "results", "video_temporal")
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, "video_features.csv")
    report_out = os.path.join(out_dir, "report.txt")
    cm_out = os.path.join(out_dir, "confusion_matrix.png")

    landmarker = FaceLandmarkerHelper(os.path.join(BASE_DIR, "models", "face_landmarker.task"))

    rows = []
    for label, folder in [("REAL", real_dir), ("FAKE", fake_dir)]:
        vids = list_videos(folder)
        if not vids:
            print(f"[WARN] No videos in: {folder}")
        for vp in vids:
            ears, motions, fps, skipped = extract_ear_and_motion_from_video(
                vp, landmarker, every_n=1, max_frames=600
            )
            feats = video_features_from_signals(ears, motions)
            row = {
                "file": os.path.basename(vp),
                "label": label,
                "fps": float(fps),
                "skipped_frames": int(skipped),
                **feats
            }
            rows.append(row)
            print(f"[OK] {label} video -> {os.path.basename(vp)} | ears={len(ears)} motions={len(motions)} skipped={skipped}")

    if not rows:
        raise SystemExit("[ERROR] No videos found in data/videos/real and data/videos/fake")

    # Save CSV
    keys = list(rows[0].keys())
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] Saved: {csv_out}")

    # Train
    feature_cols = [k for k in keys if k not in ("file", "label")]
    X = np.array([[r[c] for c in feature_cols] for r in rows], dtype=np.float32)
    y = np.array([1 if r["label"] == "REAL" else 0 for r in rows], dtype=np.int32)

    print("[INFO] Videos:", len(y), "| REAL:", int((y == 1).sum()), "| FAKE:", int((y == 0).sum()))
    if len(np.unique(y)) < 2:
        raise RuntimeError("Need both REAL and FAKE videos to train.")

    model = LogisticRegression(max_iter=2000)

    # règle : si peu de vidéos, split instable
    if len(y) < 6:
        print("[WARN] Trop peu de vidéos pour une évaluation fiable (train/test).")
        print("[WARN] On entraîne sur TOUT et on affiche les prédictions (pas de F1/CM fiable).")

        model.fit(X, y)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        lines = []
        for i, r in enumerate(rows):
            line = f"{r['file']} true={r['label']} pred={'REAL' if preds[i]==1 else 'FAKE'} proba_real={probs[i]:.3f}"
            print("  -", line)
            lines.append(line)

        with open(report_out, "w", encoding="utf-8") as f:
            f.write("Too few videos => trained on ALL (no reliable split)\n")
            f.write("\n".join(lines) + "\n")

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        cm = confusion_matrix(y_test, preds)
        rep = classification_report(y_test, preds, target_names=["FAKE", "REAL"], zero_division=0)

        print("\n=== Confusion Matrix ===")
        print(cm)

        print("\n=== Report (precision/recall/F1) ===")
        print(rep)

        save_cm_png(y_test, preds, cm_out)
        print(f"[OK] Confusion matrix saved: {cm_out}")

        with open(report_out, "w", encoding="utf-8") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write("Report:\n")
            f.write(rep + "\n")

    model_out = os.path.join(BASE_DIR, "trained_models", "video_temporal_model.joblib")
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_out)

    print(f"\n[OK] Model saved: {model_out}")
    print(f"[OK] Report saved: {report_out}")


