from pathlib import Path
import csv

from src.pad_system.router import PADRouter


ROOT = Path(r"E:\PFE_AntiSpoofing_v2")
STATIC_DIR = ROOT / "data" / "demo_static"
OUT_DIR = ROOT / "reports" / "static_image_branch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "static_image_results.csv"


def collect_images(folder: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    images = []

    for ext in exts:
        images.extend(folder.rglob(ext))

    return sorted(images)


def expected_label_from_folder(path: Path):
    parts = [p.lower() for p in path.parts]

    if "real" in parts:
        return "REAL"

    if "spoof" in parts:
        return "SPOOF"

    return "UNKNOWN"


def main():
    router = PADRouter(enable_liveness=False)

    images = collect_images(STATIC_DIR)

    if not images:
        print(f"[ERROR] Aucune image trouvée dans : {STATIC_DIR}")
        print("Ajoute des images dans :")
        print(STATIC_DIR / "real")
        print(STATIC_DIR / "spoof")
        return

    results = []

    print(f"[INFO] Nombre d'images trouvées : {len(images)}")

    for img_path in images:
        expected = expected_label_from_folder(img_path)

        print("\n======================================")
        print(f"[TEST IMAGE] {img_path}")
        print(f"[EXPECTED] {expected}")

        try:
            result = router.analyze(str(img_path))
        except Exception as e:
            print(f"[ERROR] {e}")
            results.append({
                "path": str(img_path),
                "expected": expected,
                "score": None,
                "label": "ERROR",
                "decision": "ERROR",
                "correct": False,
                "error": str(e),
            })
            continue

        score = result.get("score")
        label = result.get("label")
        decision = result.get("decision")

        correct = None
        if expected in ["REAL", "SPOOF"]:
            correct = (label == expected)

        print(result)
        print(f"[CORRECT] {correct}")

        results.append({
            "path": str(img_path),
            "expected": expected,
            "score": score,
            "label": label,
            "decision": decision,
            "correct": correct,
            "error": "",
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "expected",
                "score",
                "label",
                "decision",
                "correct",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    valid = [r for r in results if r["expected"] in ["REAL", "SPOOF"] and r["label"] != "ERROR"]

    if valid:
        acc = sum(1 for r in valid if r["correct"]) / len(valid)
        print("\n======================================")
        print("[SUMMARY]")
        print(f"Images évaluées : {len(valid)}")
        print(f"Accuracy simple : {acc:.4f}")
        print(f"CSV sauvegardé : {OUTPUT_CSV}")
    else:
        print("\n[WARN] Aucun résultat valide avec label attendu.")


if __name__ == "__main__":
    main()