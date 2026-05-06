from pathlib import Path
import json
import csv
import sys
from collections import Counter


def load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "paths.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    axon_prepared_dir = Path(cfg["axon_prepared_dir"])
    if not axon_prepared_dir.is_absolute():
        axon_prepared_dir = (project_root / axon_prepared_dir).resolve()
    else:
        axon_prepared_dir = axon_prepared_dir.resolve()

    return project_root, axon_prepared_dir


def read_manifest(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest introuvable: {csv_path}")

    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row["label"] = int(row["label"])
            rows.append(row)

    return rows


def industrial_image_gate(score: float):
    """
    Politique industrielle stricte pour image statique.

    score proche de 0 => REAL
    score proche de 1 => SPOOF

    - score < 0.10        => ACCEPT
    - 0.10 <= score < 0.60 => VIDEO_CHALLENGE_REQUIRED
    - score >= 0.60       => REJECT
    """

    score = float(score)

    if score < 0.10:
        return "ACCEPT", "NONE"

    if score < 0.60:
        return "VIDEO_CHALLENGE_REQUIRED", "REQUEST_VIDEO_CHALLENGE"

    return "REJECT", "NONE"


def compute_gate_metrics(results):
    total = len(results)

    real = [r for r in results if r["label"] == 0]
    spoof = [r for r in results if r["label"] == 1]

    real_count = len(real)
    spoof_count = len(spoof)

    real_accept = sum(1 for r in real if r["decision"] == "ACCEPT")
    real_challenge = sum(1 for r in real if r["decision"] == "VIDEO_CHALLENGE_REQUIRED")
    real_reject = sum(1 for r in real if r["decision"] == "REJECT")

    spoof_accept = sum(1 for r in spoof if r["decision"] == "ACCEPT")
    spoof_challenge = sum(1 for r in spoof if r["decision"] == "VIDEO_CHALLENGE_REQUIRED")
    spoof_reject = sum(1 for r in spoof if r["decision"] == "REJECT")

    def div(a, b):
        return a / b if b else 0.0

    metrics = {
        "total": total,
        "real_count": real_count,
        "spoof_count": spoof_count,

        "real_accept": real_accept,
        "real_challenge": real_challenge,
        "real_reject": real_reject,

        "spoof_accept_danger": spoof_accept,
        "spoof_challenge": spoof_challenge,
        "spoof_reject": spoof_reject,

        "real_accept_rate": div(real_accept, real_count),
        "real_challenge_rate": div(real_challenge, real_count),
        "real_reject_rate": div(real_reject, real_count),

        "spoof_dangerous_accept_rate": div(spoof_accept, spoof_count),
        "spoof_challenge_rate": div(spoof_challenge, spoof_count),
        "spoof_reject_rate": div(spoof_reject, spoof_count),

        "global_challenge_rate": div(
            real_challenge + spoof_challenge,
            total,
        ),
    }

    return metrics


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    project_root, axon_prepared_dir = load_config()

    sys.path.insert(0, str(project_root))

    from src.pad_system.image_model import ImagePADModel

    static_manifest = axon_prepared_dir / "manifests" / "axon_static_manifest.csv"

    out_dir = project_root / "reports" / "axon_industrial_image_gate"
    predictions_csv = out_dir / "industrial_gate_predictions.csv"
    metrics_json = out_dir / "industrial_gate_metrics.json"
    dangerous_csv = out_dir / "dangerous_accepts.csv"
    challenge_csv = out_dir / "video_challenge_required.csv"

    print("========== CONFIG ==========")
    print(f"Project root     : {project_root}")
    print(f"Axon prepared    : {axon_prepared_dir}")
    print(f"Static manifest  : {static_manifest}")
    print(f"Output dir       : {out_dir}")

    rows = read_manifest(static_manifest)
    print(f"[INFO] Images Axon statiques à évaluer : {len(rows)}")

    model = ImagePADModel()

    results = []

    for i, row in enumerate(rows, start=1):
        rel_path = row["prepared_relative_path"]
        img_path = (project_root / rel_path).resolve()

        label = int(row["label"])
        label_name = row.get("label_name", "REAL" if label == 0 else "SPOOF")
        attack_type = row.get("attack_type", "")
        level = row.get("level", "")
        source = row.get("source", "")

        print("\n----------------------------------------")
        print(f"[{i}/{len(rows)}] {img_path}")
        print(f"Expected={label_name} | attack_type={attack_type}")

        error = ""

        try:
            if not img_path.exists():
                raise FileNotFoundError(f"Image introuvable: {img_path}")

            score = float(model.predict(str(img_path)))
            decision, next_action = industrial_image_gate(score)

            if decision == "ACCEPT":
                output_label = "REAL"
            elif decision == "REJECT":
                output_label = "SPOOF"
            else:
                output_label = "UNCERTAIN"

            print(
                f"score={score:.4f} | decision={decision} | "
                f"next_action={next_action}"
            )

        except Exception as e:
            score = None
            decision = "ERROR"
            next_action = "ERROR"
            output_label = "ERROR"
            error = str(e)
            print(f"[ERROR] {error}")

        results.append({
            "prepared_relative_path": rel_path,
            "origin_raw_relative_path": row.get("origin_raw_relative_path", ""),
            "label": label,
            "label_name": label_name,
            "attack_type": attack_type,
            "level": level,
            "source": source,
            "score_spoof": "" if score is None else round(score, 6),
            "decision": decision,
            "next_action": next_action,
            "output_label": output_label,
            "error": error,
        })

    valid_results = [r for r in results if r["decision"] != "ERROR"]
    metrics = compute_gate_metrics(valid_results)

    decision_counts = Counter(r["decision"] for r in valid_results)
    spoof_decision_counts = Counter(
        r["decision"] for r in valid_results if r["label"] == 1
    )
    real_decision_counts = Counter(
        r["decision"] for r in valid_results if r["label"] == 0
    )

    metrics["decision_counts"] = dict(decision_counts)
    metrics["spoof_decision_counts"] = dict(spoof_decision_counts)
    metrics["real_decision_counts"] = dict(real_decision_counts)
    metrics["errors"] = len(results) - len(valid_results)

    dangerous = [
        r for r in valid_results
        if r["label"] == 1 and r["decision"] == "ACCEPT"
    ]

    challenge_required = [
        r for r in valid_results
        if r["decision"] == "VIDEO_CHALLENGE_REQUIRED"
    ]

    save_csv(predictions_csv, results)
    save_csv(dangerous_csv, dangerous)
    save_csv(challenge_csv, challenge_required)
    save_json(metrics_json, metrics)

    print("\n========== SUMMARY INDUSTRIAL IMAGE GATE ==========")
    print(f"Total                      : {metrics['total']}")
    print(f"REAL                       : {metrics['real_count']}")
    print(f"SPOOF                      : {metrics['spoof_count']}")
    print()
    print("REAL decisions:")
    print(f"  ACCEPT                   : {metrics['real_accept']}")
    print(f"  VIDEO_CHALLENGE_REQUIRED : {metrics['real_challenge']}")
    print(f"  REJECT                   : {metrics['real_reject']}")
    print()
    print("SPOOF decisions:")
    print(f"  REJECT                   : {metrics['spoof_reject']}")
    print(f"  VIDEO_CHALLENGE_REQUIRED : {metrics['spoof_challenge']}")
    print(f"  ACCEPT danger            : {metrics['spoof_accept_danger']}")
    print()
    print(f"Spoof dangerous accept rate: {metrics['spoof_dangerous_accept_rate']:.6f}")
    print(f"Spoof challenge rate       : {metrics['spoof_challenge_rate']:.6f}")
    print(f"Spoof reject rate          : {metrics['spoof_reject_rate']:.6f}")
    print(f"Global challenge rate      : {metrics['global_challenge_rate']:.6f}")
    print()
    print("========== OUTPUTS ==========")
    print(f"Predictions                : {predictions_csv}")
    print(f"Metrics                    : {metrics_json}")
    print(f"Dangerous accepts          : {dangerous_csv}")
    print(f"Video challenge required   : {challenge_csv}")


if __name__ == "__main__":
    main()
