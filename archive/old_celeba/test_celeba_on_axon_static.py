from pathlib import Path
import json
import csv
import argparse
import random
import sys


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

    return project_root, axon_prepared_dir, cfg


def read_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest statique introuvable: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["label"] = int(row["label"])

    return rows


def limit_rows(rows, max_samples=None, limit_per_label=None, seed=42):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)

    if limit_per_label is not None:
        kept = []
        counts = {0: 0, 1: 0}

        for row in rows:
            y = int(row["label"])

            if counts[y] < limit_per_label:
                kept.append(row)
                counts[y] += 1

            if counts[0] >= limit_per_label and counts[1] >= limit_per_label:
                break

        rows = kept

    if max_samples is not None and len(rows) > max_samples:
        rows = rows[:max_samples]

    return rows


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def compute_metrics(results):
    """
    Label convention:
    0 = REAL / bona fide
    1 = SPOOF / attack

    pred_label_th = 0 / 1 using threshold.
    """

    total = len(results)

    tp = sum(1 for r in results if r["label"] == 1 and r["pred_label"] == 1)
    tn = sum(1 for r in results if r["label"] == 0 and r["pred_label"] == 0)
    fp = sum(1 for r in results if r["label"] == 0 and r["pred_label"] == 1)
    fn = sum(1 for r in results if r["label"] == 1 and r["pred_label"] == 0)

    real_count = tn + fp
    spoof_count = tp + fn

    accuracy = safe_div(tp + tn, total)

    precision_spoof = safe_div(tp, tp + fp)
    recall_spoof = safe_div(tp, tp + fn)
    f1_spoof = safe_div(2 * precision_spoof * recall_spoof, precision_spoof + recall_spoof)

    # PAD metrics
    # APCER = attack classified as bona fide = FN / spoof_count
    # BPCER = bona fide classified as attack = FP / real_count
    apcer = safe_div(fn, spoof_count)
    bpcer = safe_div(fp, real_count)
    acer = (apcer + bpcer) / 2.0

    return {
        "total": total,
        "real_count": real_count,
        "spoof_count": spoof_count,
        "tp_spoof": tp,
        "tn_real": tn,
        "fp_real_as_spoof": fp,
        "fn_spoof_as_real": fn,
        "accuracy": accuracy,
        "precision_spoof": precision_spoof,
        "recall_spoof": recall_spoof,
        "f1_spoof": f1_spoof,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
    }


def banking_decision_from_score(score, t_accept=0.20, t_reject=0.60):
    if score < t_accept:
        return "ACCEPT"

    if score < t_reject:
        return "RETRY"

    return "REJECT"

def label_from_banking(decision):
    if decision == "ACCEPT":
        return "REAL"

    if decision == "REJECT":
        return "SPOOF"

    return "UNCERTAIN"


def save_predictions(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prepared_relative_path",
        "origin_raw_relative_path",
        "label",
        "label_name",
        "attack_type",
        "level",
        "source",
        "score_spoof",
        "pred_label",
        "pred_label_name",
        "banking_decision",
        "banking_label",
        "correct_th",
        "error",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metrics(path: Path, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_results_txt(path: Path, metrics: dict, args):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Evaluation CelebA image model on AxonLabs static subset\n")
        f.write("=" * 70 + "\n\n")

        f.write("Parameters\n")
        f.write("-" * 70 + "\n")
        f.write(f"threshold         : {args.threshold}\n")
        f.write("banking_t_accept : 0.20\n")
        f.write("banking_t_reject : 0.60\n")
        f.write(f"max_samples       : {args.max_samples}\n")
        f.write(f"limit_per_label   : {args.limit_per_label}\n")
        f.write(f"seed              : {args.seed}\n\n")

        f.write("Metrics\n")
        f.write("-" * 70 + "\n")

        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k:25s}: {v:.6f}\n")
            else:
                f.write(f"{k:25s}: {v}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Seuil score_spoof pour pred_label: score >= threshold => SPOOF",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Nombre max total d'images à évaluer.",
    )

    parser.add_argument(
        "--limit_per_label",
        type=int,
        default=None,
        help="Nombre max par label REAL/SPOOF. Ex: 50 => 50 real + 50 spoof.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    project_root, axon_prepared_dir, cfg = load_config()

    # Permet d'importer src même si le script est lancé depuis tools/axon
    sys.path.insert(0, str(project_root))

    from src.pad_system.image_model import ImagePADModel

    static_manifest = axon_prepared_dir / "manifests" / "axon_static_manifest.csv"

    out_dir = project_root / "reports" / "axon_static_celeba_eval"
    predictions_csv = out_dir / "predictions.csv"
    metrics_json = out_dir / "metrics.json"
    results_txt = out_dir / "results.txt"

    print("========== CONFIG ==========")
    print(f"Project root     : {project_root}")
    print(f"Axon prepared    : {axon_prepared_dir}")
    print(f"Static manifest  : {static_manifest}")
    print(f"Output dir       : {out_dir}")

    rows = read_rows(static_manifest)
    rows = limit_rows(
        rows,
        max_samples=args.max_samples,
        limit_per_label=args.limit_per_label,
        seed=args.seed,
    )

    if not rows:
        raise RuntimeError("Aucune image à évaluer dans axon_static_manifest.csv")

    print(f"[INFO] Images à évaluer: {len(rows)}")

    model = ImagePADModel()

    prediction_rows = []
    metric_rows = []

    for i, row in enumerate(rows, start=1):
        rel_path = row["prepared_relative_path"]
        img_path = (project_root / rel_path).resolve()

        expected_label = int(row["label"])
        expected_label_name = row.get("label_name", "REAL" if expected_label == 0 else "SPOOF")

        print("\n----------------------------------------")
        print(f"[{i}/{len(rows)}] {img_path}")
        print(f"Expected: {expected_label_name} | attack_type={row.get('attack_type')}")

        error = ""

        try:
            if not img_path.exists():
                raise FileNotFoundError(f"Image introuvable: {img_path}")

            score = float(model.predict(str(img_path)))

            pred_label = 1 if score >= args.threshold else 0
            pred_label_name = "SPOOF" if pred_label == 1 else "REAL"

            banking_decision = banking_decision_from_score(score)
            banking_label = label_from_banking(banking_decision)

            correct = pred_label == expected_label

            print(
                f"score={score:.4f} | pred={pred_label_name} | "
                f"banking={banking_decision} | correct={correct}"
            )

        except Exception as e:
            score = None
            pred_label = None
            pred_label_name = "ERROR"
            banking_decision = "ERROR"
            banking_label = "ERROR"
            correct = False
            error = str(e)

            print(f"[ERROR] {error}")

        out_row = {
            "prepared_relative_path": row.get("prepared_relative_path", ""),
            "origin_raw_relative_path": row.get("origin_raw_relative_path", ""),
            "label": expected_label,
            "label_name": expected_label_name,
            "attack_type": row.get("attack_type", ""),
            "level": row.get("level", ""),
            "source": row.get("source", ""),
            "score_spoof": "" if score is None else round(score, 6),
            "pred_label": "" if pred_label is None else pred_label,
            "pred_label_name": pred_label_name,
            "banking_decision": banking_decision,
            "banking_label": banking_label,
            "correct_th": correct,
            "error": error,
        }

        prediction_rows.append(out_row)

        if pred_label is not None:
            metric_rows.append({
                "label": expected_label,
                "pred_label": pred_label,
            })

    metrics = compute_metrics(metric_rows)

    metrics["threshold"] = args.threshold
    metrics["evaluated_rows"] = len(rows)
    metrics["valid_metric_rows"] = len(metric_rows)
    metrics["errors"] = len(rows) - len(metric_rows)

    save_predictions(predictions_csv, prediction_rows)
    save_metrics(metrics_json, metrics)
    save_results_txt(results_txt, metrics, args)

    print("\n========== SUMMARY ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.6f}")
        else:
            print(f"{k:25s}: {v}")

    print("\n========== OUTPUTS ==========")
    print(f"Predictions: {predictions_csv}")
    print(f"Metrics    : {metrics_json}")
    print(f"Results    : {results_txt}")


if __name__ == "__main__":
    main()
