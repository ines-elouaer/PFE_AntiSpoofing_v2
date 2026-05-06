from pathlib import Path
import time

import cv2

from src.pad_system.router import PADRouter


ROOT = Path(r"E:\PFE_AntiSpoofing_v2")
DEMO_DIR = ROOT / "data" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = DEMO_DIR / "webcam_challenge.mp4"


def record_webcam_challenge(
    output_path: Path,
    challenge_type: str,
    instruction: str,
    duration_sec: int = 4,
    camera_id: int = 0,
):
    """
    Enregistre une courte vidéo webcam avec instruction affichée.

    Nouveau flux :
    - le challenge n'est plus choisi directement dans ce fichier ;
    - il est créé par ChallengeManager via router.create_video_challenge().
    """

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            "Impossible d'ouvrir la webcam. Essaie camera_id=1 si camera_id=0 ne marche pas."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    # Sécurité minimale si fps invalide.
    if fps <= 0:
        fps = 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("\n====================================")
    print("CHALLENGE VIDEO DIRECT")
    print("Challenge    :", challenge_type)
    print("Instruction  :", instruction)
    print("Durée capture:", duration_sec, "secondes")
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
            f"Challenge: {challenge_type}",
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
        cv2.imshow("PAD Video Challenge", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if elapsed >= duration_sec:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"[OK] Vidéo enregistrée: {output_path}")


if __name__ == "__main__":
    print("\n==========================================")
    print("TEST WEBCAM LIVENESS - FLUX VIDEO DIRECT")
    print("==========================================")

    router = PADRouter(
        enable_liveness=True,
        load_video_model=True,
        challenge_ttl_seconds=60,
    )

    # 1. Créer challenge via le routeur
    challenge_payload = router.create_video_challenge(
        source="test_webcam_liveness"
    )

    session_id = challenge_payload["session_id"]
    challenge_id = challenge_payload["challenge_id"]
    challenge_type = challenge_payload["challenge_type"]
    instruction = challenge_payload["instruction"]

    print("\n[1] Challenge créé")
    print("session_id     :", session_id)
    print("challenge_id   :", challenge_id)
    print("challenge_type :", challenge_type)
    print("instruction    :", instruction)

    # 2. Capturer vidéo webcam
    print("\n[2] Capture vidéo challenge")
    record_webcam_challenge(
        output_path=OUTPUT_VIDEO,
        challenge_type=challenge_type,
        instruction=instruction,
        duration_sec=4,
        camera_id=0,
    )

    # 3. Analyser vidéo avec challenge_id
    print("\n[3] Analyse vidéo challenge")
    result = router.analyze(
        str(OUTPUT_VIDEO),
        enable_liveness=True,
        challenge_id=challenge_id,
        require_challenge_validation=True,
    )

    print("\n=== RESULTAT FINAL WEBCAM LIVENESS ===")
    print(result)