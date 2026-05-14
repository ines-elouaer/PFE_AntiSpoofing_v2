from pathlib import Path
import sys
import time

import cv2

ROOT = Path(r"E:\PFE_AntiSpoofing_v2")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pad_system.video_model import VideoPADModel
from src.pad_system.banking_adapter import banking_decision, label_from_decision


OUTPUT_VIDEO = ROOT / "data" / "demo" / "webcam_direct_model_test.mp4"

CHECKPOINT_PATH = (
    ROOT
    / "experiments"
    / "mixed_casia_axon_local_msu"
    / "seed42"
    / "best_model_mixed_casia_axon_local_msu.pth"
)


def record_webcam_video(output_path: Path, duration_sec: int = 5, camera_id: int = 0):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam. Essaie camera_id=1.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    if fps <= 0:
        fps = 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("Webcam ouverte.")
    print("Enregistrement direct pour le modèle PAD...")
    print("Appuie sur 'q' pour arrêter avant la fin.")

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        elapsed = time.time() - start_time
        remaining = max(0, duration_sec - int(elapsed))

        display = frame.copy()

        cv2.putText(
            display,
            "TEST DIRECT MODELE PAD",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            f"Temps restant: {remaining}s",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        cv2.imshow("Webcam - Test direct modele PAD", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if elapsed >= duration_sec:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Vidéo webcam enregistrée : {output_path}")


def main():
    print("\n==========================================================")
    print("TEST WEBCAM DIRECT - MODELE PAD SANS CHALLENGE")
    print("==========================================================")

    print("Checkpoint :", CHECKPOINT_PATH)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {CHECKPOINT_PATH}")

    # 1. Capture webcam directe
    record_webcam_video(
        output_path=OUTPUT_VIDEO,
        duration_sec=5,
        camera_id=0,
    )

    # 2. Chargement du modèle
    model = VideoPADModel(
        checkpoint_path=str(CHECKPOINT_PATH),
        sample_mode="center_consecutive",
    )

    # 3. Prédiction directe
    score = float(model.predict(str(OUTPUT_VIDEO)))

    decision = banking_decision(score, profile="video")
    label = label_from_decision(decision)

    print("\n---------------- RESULTAT WEBCAM DIRECT ----------------")
    print("Vidéo testée :", OUTPUT_VIDEO)
    print("Score spoof :", round(score, 4))
    print("Label       :", label)
    print("Décision    :", decision)

    if score < 0.30:
        interpretation = "Le modèle considère la vidéo webcam comme plutôt REELLE."
    elif score < 0.60:
        interpretation = "Le modèle considère la vidéo webcam comme AMBIGUË."
    else:
        interpretation = "Le modèle considère la vidéo webcam comme probablement SPOOF."

    print("Interprétation :", interpretation)
    print("==========================================================")
    print("FIN TEST WEBCAM DIRECT")
    print("==========================================================")


if __name__ == "__main__":
    main()