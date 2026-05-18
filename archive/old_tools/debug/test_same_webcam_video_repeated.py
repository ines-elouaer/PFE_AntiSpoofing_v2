from pathlib import Path
import sys
import json
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pad_system.video_model import VideoPADModel


VIDEO_PATH = PROJECT_ROOT / "data" / "demo" / "jury_challenge.mp4"

OUT_DIR = PROJECT_ROOT / "reports" / "debug_same_webcam_video"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "same_video_repeated_predictions.csv"
OUT_JSON = OUT_DIR / "same_video_repeated_summary.json"


def banking_decision_video(score: float) -> str:
    score = float(score)

    if score < 0.30:
        return "ACCEPT"

    if score < 0.60:
        return "RETRY"

    return "REJECT"


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Vidéo introuvable: {VIDEO_PATH}\n"
            "Lance d'abord demo_jury_mobile_flow.py pour créer data/demo/jury_challenge.mp4"
        )

    print("========== TEST SAME WEBCAM VIDEO ==========")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Video        : {VIDEO_PATH}")
    print(f"Output CSV   : {OUT_CSV}")
    print(f"Output JSON  : {OUT_JSON}")

    print("\n[INFO] Chargement du modèle vidéo...")
    model = VideoPADModel()

    rows = []

    n_runs = 10

    print(f"\n[INFO] Même vidéo analysée {n_runs} fois.\n")

    for i in range(1, n_runs + 1):
        start = time.time()

        score = float(model.predict(str(VIDEO_PATH)))

        elapsed_ms = (time.time() - start) * 1000.0
        decision = banking_decision_video(score)

        row = {
            "run": i,
            "video_path": str(VIDEO_PATH),
            "score_spoof": score,
            "decision": decision,
            "processing_time_ms": elapsed_ms,
        }

        rows.append(row)

        print(
            f"Run {i:02d} | "
            f"score={score:.6f} | "
            f"decision={decision:<6} | "
            f"time={elapsed_ms:.1f} ms"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    scores = df["score_spoof"].astype(float).values

    summary = {
        "video_path": str(VIDEO_PATH),
        "n_runs": int(n_runs),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_range": float(np.max(scores) - np.min(scores)),
        "decisions": df["decision"].value_counts().to_dict(),
        "output_csv": str(OUT_CSV),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== SUMMARY ==========")
    for k, v in summary.items():
        print(f"{k:<20}: {v}")

    print("\n========== INTERPRETATION ==========")

    if summary["score_range"] < 0.001:
        print("[OK] Le score est stable sur le même fichier vidéo.")
        print("Donc le modèle est déterministe. Les variations viennent surtout de la capture webcam.")
    elif summary["score_range"] < 0.02:
        print("[OK] Petite variation acceptable.")
        print("Le pipeline est globalement stable.")
    else:
        print("[WARN] Le score change beaucoup sur le même fichier.")
        print("Il faut vérifier l'inférence : sampling, extraction behavior, preprocessing ou modèle en mode train.")

    print("\n[OK] Test terminé.")


if __name__ == "__main__":
    main()