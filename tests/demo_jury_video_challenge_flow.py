from pathlib import Path
import sys
import time

ROOT = Path(r"E:\PFE_AntiSpoofing_v2")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from src.pad_system.router import PADRouter


DEMO_DIR = ROOT / "data" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = DEMO_DIR / "jury_video_challenge.mp4"
def print_section(title: str):
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def record_webcam_challenge(
    output_path: Path,
    challenge_type: str,
    instruction: str,
    duration_sec: int = 4,
    camera_id: int = 0,
):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            "Impossible d'ouvrir la webcam. Essaie camera_id=1 si camera_id=0 ne marche pas."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    if fps <= 0:
        fps = 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("Une fenêtre webcam va s'ouvrir.")
    print("Instruction :", instruction)
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
            "PAD VIDEO CHALLENGE",
            (30, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            f"Challenge: {challenge_type}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            instruction,
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            f"Temps restant: {remaining}s",
            (30, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        cv2.imshow("Demo Jury - PAD Video Challenge", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if elapsed >= duration_sec:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Vidéo enregistrée : {output_path}")


def main():
    print_section("DEMO PAD BANCAIRE - VIDEO CHALLENGE DIRECT")

    print("\n[0] Initialisation du système")
    print("-" * 58)
    print("Chargement des modules en cours...")
    print("Modèle utilisé : V6 final — gated rPPG + behavior-pose")

    router = PADRouter(
        enable_liveness=True,
        load_video_model=True,
        challenge_ttl_seconds=60,
    )

    print("Système prêt.")

    print("\n[1] Création du challenge vidéo")
    print("-" * 58)

    challenge_payload = router.create_video_challenge(
        source="jury_demo_video_challenge"
    )

    session_id = challenge_payload["session_id"]
    challenge_id = challenge_payload["challenge_id"]
    challenge_type = challenge_payload["challenge_type"]
    instruction = challenge_payload["instruction"]
    expires_at = challenge_payload["expires_at"]

    print("Session ID   :", session_id)
    print("Challenge ID :", challenge_id)
    print("Challenge    :", challenge_type)
    print("Instruction  :", instruction)
    print("Expiration   :", expires_at)

    print("\n[2] Capture vidéo challenge")
    print("-" * 58)

    record_webcam_challenge(
        output_path=OUTPUT_VIDEO,
        challenge_type=challenge_type,
        instruction=instruction,
        duration_sec=4,
        camera_id=0,
    )

    print("\n[3] Analyse vidéo challenge")
    print("-" * 58)

    result = router.analyze(
        str(OUTPUT_VIDEO),
        enable_liveness=True,
        challenge_id=challenge_id,
        require_challenge_validation=True,
    )

    liveness = result.get("liveness", {})

    print("Validation entrée :", result.get("validation", {}).get("reason"))
    print("Liveness          :", liveness.get("status"))
    print("Raison liveness   :", liveness.get("reason"))
    print("Score vidéo       :", result.get("score"))
    print("Qualité vidéo     :", result.get("video_quality_score"))
    print("Détails modèle    :", result.get("model_details"))
    print("Label vidéo       :", result.get("label"))
    print("Décision finale   :", result.get("decision"))
    print("Action suivante   :", result.get("next_action"))
    print("Message           :", result.get("message"))

    print_section("FIN DE LA DEMO")


if __name__ == "__main__":
    main()