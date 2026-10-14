import argparse
import json
import sys
from pathlib import Path

import requests


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def pretty(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def check_response(response, expected_status=200):
    if response.status_code != expected_status:
        print("[ERROR] Status code inattendu")
        print("Expected:", expected_status)
        print("Got     :", response.status_code)
        try:
            pretty(response.json())
        except Exception:
            print(response.text)
        sys.exit(1)

    try:
        return response.json()
    except Exception:
        return response.text


def test_health(base_url: str):
    print_section("1. TEST /health")

    response = requests.get(f"{base_url}/health", timeout=20)
    data = check_response(response)

    pretty(data)

    if data.get("status") != "ok":
        print("[ERROR] /health ne retourne pas status=ok")
        sys.exit(1)

    print("[OK] /health fonctionne.")


def test_model_info(base_url: str):
    print_section("2. TEST /model/info")

    response = requests.get(f"{base_url}/model/info", timeout=20)
    data = check_response(response)

    pretty(data)

    required_keys = [
        "model_name",
        "api_version",
        "video_checkpoint",
        "behavior_pose_model",
        "face_landmarker_model",
        "fusion_formula",
        "decisions",
    ]

    for key in required_keys:
        if key not in data:
            print(f"[ERROR] Clé manquante dans /model/info: {key}")
            sys.exit(1)

    print("[OK] /model/info fonctionne.")
    return data


def test_challenge_start(base_url: str, challenge_type: str):
    print_section("3. TEST /pad/challenge/start")

    payload = {
        "challenge_type": challenge_type
    }

    response = requests.post(
        f"{base_url}/pad/challenge/start",
        json=payload,
        timeout=20,
    )
    data = check_response(response)

    pretty(data)

    challenge_id = data.get("challenge_id")

    if not challenge_id:
        print("[ERROR] challenge_id absent.")
        sys.exit(1)

    if data.get("challenge_type") != challenge_type:
        print("[ERROR] challenge_type retourné différent.")
        sys.exit(1)

    print("[OK] Challenge créé.")
    return challenge_id


def test_challenge_status(base_url: str, challenge_id: str):
    print_section("4. TEST /pad/challenge/{challenge_id}")

    response = requests.get(
        f"{base_url}/pad/challenge/{challenge_id}",
        timeout=20,
    )
    data = check_response(response)

    pretty(data)

    if not data.get("is_valid"):
        print("[ERROR] Challenge non valide.")
        sys.exit(1)

    print("[OK] Challenge valide.")


def test_analyze_video_without_liveness(base_url: str, video_path: Path):
    print_section("5. TEST /pad/analyze-video SANS liveness")

    if not video_path.exists():
        print(f"[ERROR] Vidéo introuvable: {video_path}")
        sys.exit(1)

    with video_path.open("rb") as f:
        files = {
            "video": (
                video_path.name,
                f,
                "video/mp4",
            )
        }

        data = {
            "challenge_id": "",
            "enable_liveness": "false",
            "require_challenge_validation": "false",
        }

        response = requests.post(
            f"{base_url}/pad/analyze-video",
            files=files,
            data=data,
            timeout=300,
        )

    result = check_response(response)

    pretty(result)

    if result.get("model_called") is not True:
        print("[ERROR] Le modèle V6 n'a pas été appelé.")
        sys.exit(1)

    if "model_details" not in result or result["model_details"] is None:
        print("[ERROR] model_details absent.")
        sys.exit(1)

    model_details = result["model_details"]

    required_model_keys = [
        "score_v3_multimodal",
        "score_final_v6",
        "video_quality_score",
        "fusion",
    ]

    for key in required_model_keys:
        if key not in model_details:
            print(f"[ERROR] Clé manquante dans model_details: {key}")
            sys.exit(1)

    decision = result.get("decision")

    if decision not in ["ACCEPT", "RETRY", "REJECT"]:
        print(f"[ERROR] Décision invalide: {decision}")
        sys.exit(1)

    print("[OK] Analyse vidéo sans liveness fonctionne.")
    print(f"[RESULT] decision={decision} | score={result.get('score')}")


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test pour PAD Banking API."
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="URL de base de l'API.",
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Chemin vers une vidéo de test.",
    )

    parser.add_argument(
        "--challenge-type",
        default="SMILE",
        choices=["SMILE", "BLINK", "TURN_LEFT"],
        help="Type de challenge à créer pour le test challenge.",
    )

    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    video_path = Path(args.video)

    print_section("PAD BANKING API — SMOKE TEST")
    print("Base URL :", base_url)
    print("Video    :", video_path)

    test_health(base_url)
    test_model_info(base_url)
    challenge_id = test_challenge_start(base_url, args.challenge_type)
    test_challenge_status(base_url, challenge_id)
    test_analyze_video_without_liveness(base_url, video_path)

    print_section("SMOKE TEST TERMINÉ")
    print("[OK] API opérationnelle pour la phase de déploiement.")


if __name__ == "__main__":
    main()