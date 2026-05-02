from pathlib import Path
import time
import random

import cv2

from src.pad_system.router import PADRouter


ROOT = Path(r"E:\PFE_AntiSpoofing_v2")
DEMO_DIR = ROOT / "data" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = DEMO_DIR / "webcam_challenge.mp4"


CHALLENGE_MESSAGES = {
    "TURN_LEFT": "Tournez la tete a gauche",
    "TURN_RIGHT": "Tournez la tete a droite",
    "BLINK": "Clignez les yeux",
    "SMILE": "Souriez",
}


def record_webcam_challenge(
    output_path: Path,
    challenge: str,
    duration_sec: int = 4,
    camera_id: int = 0,
):
    """
    Enregistre une courte vidéo webcam avec instruction affichée.
    """

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            "Impossible d'ouvrir la webcam. Essaie camera_id=1 si camera_id=0 ne marche pas."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    instruction = CHALLENGE_MESSAGES.get(challenge, challenge)

    print("\n====================================")
    print("CHALLENGE ACTIF ALEATOIRE")
    print("Challenge :", challenge)
    print("Instruction :", instruction)
    print("La capture commence maintenant.")
    print("Appuie sur 'q' pour arrêter avant la fin.")
    print("====================================\n")

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
            f"Challenge: {challenge}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            instruction,
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            f"Temps restant: {remaining}s",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        cv2.imshow("PAD Active Liveness Challenge", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if elapsed >= duration_sec:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"[OK] Vidéo enregistrée: {output_path}")


if __name__ == "__main__":
    # Challenge choisi aléatoirement à chaque exécution
    challenge = random.choice(["TURN_LEFT", "TURN_RIGHT", "BLINK", "SMILE"])

    record_webcam_challenge(
        output_path=OUTPUT_VIDEO,
        challenge=challenge,
        duration_sec=4,
        camera_id=0,
    )

    router = PADRouter(enable_liveness=True)

    result = router.analyze(
        str(OUTPUT_VIDEO),
        challenge=challenge,
        enable_liveness=True,
    )

    print("\n=== RESULTAT FINAL WEBCAM LIVENESS ===")
    print(result)