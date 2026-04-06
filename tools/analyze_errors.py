import argparse
import csv
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def load_scores(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    scores = data["test_scores"]
    threshold_used = float(data["threshold_used"]["th"])
    protocol_name = data.get("protocol_name", "unknown_protocol")
    model_name = data.get("model_name", "unknown_model")
    return data, scores, threshold_used, protocol_name, model_name


def build_error_rows(scores: dict, threshold: float):
    rows = []
    for vid, d in scores.items():
        label = int(d["label"])
        score = float(d["score"])
        pred = 1 if score >= threshold else 0

        if label == 0 and pred == 1:
            error_type = "FP_real_as_attack"
        elif label == 1 and pred == 0:
            error_type = "FN_attack_as_real"
        else:
            error_type = "correct"

        rows.append({
            "video_id": vid,
            "label": label,
            "score": score,
            "prediction": pred,
            "threshold": threshold,
            "error_type": error_type,
        })

    return rows


def summarize_rows(rows):
    counts = Counter(r["error_type"] for r in rows)
    total = len(rows)
    correct = counts.get("correct", 0)
    fp = counts.get("FP_real_as_attack", 0)
    fn = counts.get("FN_attack_as_real", 0)

    return {
        "total_videos": total,
        "correct": correct,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": correct / total if total else 0.0,
    }


def save_csv(rows, out_path: Path, fieldnames=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def save_json(obj, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_error_only_csv(rows, out_path: Path):
    err_rows = [r for r in rows if r["error_type"] != "correct"]
    fieldnames = ["video_id", "label", "score", "prediction", "threshold", "error_type"]
    save_csv(err_rows, out_path, fieldnames=fieldnames)


def save_barplot(summary, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["correct", "false_positive", "false_negative"]
    values = [
        summary.get("correct", 0),
        summary.get("false_positive", 0),
        summary.get("false_negative", 0),
    ]
    colors = ["steelblue", "firebrick", "darkorange"]

    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(labels, values, color=colors)

    plt.title(title)
    plt.ylabel("Number of videos")
    plt.ylim(0, max(values) + 2 if values else 1)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def rows_to_map(rows):
    return {r["video_id"]: r for r in rows}


def compare_rows(rows_a, rows_b):
    map_a = rows_to_map(rows_a)
    map_b = rows_to_map(rows_b)

    vids = sorted(set(map_a.keys()) & set(map_b.keys()))
    out_rows = []

    for vid in vids:
        a = map_a[vid]
        b = map_b[vid]

        a_err = a["error_type"]
        b_err = b["error_type"]

        if a_err != "correct" and b_err == "correct":
            effect = "b_improves_over_a"
        elif a_err == "correct" and b_err != "correct":
            effect = "b_worse_than_a"
        elif a_err != "correct" and b_err != "correct":
            effect = "both_wrong"
        else:
            effect = "both_correct"

        out_rows.append({
            "video_id": vid,
            "label": a["label"],
            "score_a": a["score"],
            "pred_a": a["prediction"],
            "error_a": a_err,
            "score_b": b["score"],
            "pred_b": b["prediction"],
            "error_b": b_err,
            "comparison": effect,
        })

    return out_rows


def summarize_comparison(rows):
    counts = Counter(r["comparison"] for r in rows)
    return {
        "both_correct": counts.get("both_correct", 0),
        "b_improves_over_a": counts.get("b_improves_over_a", 0),
        "b_worse_than_a": counts.get("b_worse_than_a", 0),
        "both_wrong": counts.get("both_wrong", 0),
    }


def save_comparison_plot(summary, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Both correct", "B improves over A", "B worse than A", "Both wrong"]
    values = [
        summary.get("both_correct", 0),
        summary.get("b_improves_over_a", 0),
        summary.get("b_worse_than_a", 0),
        summary.get("both_wrong", 0),
    ]
    colors = ["steelblue", "seagreen", "firebrick", "gray"]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values, color=colors)

    plt.title(title)
    plt.ylabel("Number of videos")
    plt.ylim(0, max(values) + 2 if values else 1)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_side_by_side_error_plot(summary_a, summary_b, out_path: Path, model_a_name: str, model_b_name: str, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["correct", "false_positive", "false_negative"]
    values_a = [
        summary_a.get("correct", 0),
        summary_a.get("false_positive", 0),
        summary_a.get("false_negative", 0),
    ]
    values_b = [
        summary_b.get("correct", 0),
        summary_b.get("false_positive", 0),
        summary_b.get("false_negative", 0),
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4.5))
    bars_a = plt.bar(x - width / 2, values_a, width, label=model_a_name)
    bars_b = plt.bar(x + width / 2, values_b, width, label=model_b_name)

    plt.xticks(x, labels)
    plt.ylabel("Number of videos")
    plt.title(title)
    plt.legend()

    ymax = max(values_a + values_b) if (values_a + values_b) else 1
    plt.ylim(0, ymax + 2)

    for bars in [bars_a, bars_b]:
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.05,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def analyze_single(json_path: Path, out_dir: Path):
    data, scores, threshold_used, protocol_name, model_name = load_scores(json_path)
    rows = build_error_rows(scores, threshold_used)
    summary = summarize_rows(rows)
    summary["model_name"] = model_name
    summary["protocol_name"] = protocol_name
    summary["threshold_used"] = threshold_used

    save_csv(rows, out_dir / "all_predictions.csv")
    save_error_only_csv(rows, out_dir / "only_errors.csv")
    save_json(summary, out_dir / "summary.json")
    save_barplot(summary, out_dir / "error_plot.png", f"{model_name} - {protocol_name}")

    return rows, summary, model_name, protocol_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_a", required=True, help="First model test_scores.json")
    parser.add_argument("--json_b", default=None, help="Second model test_scores.json (optional)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_a, summary_a, model_a, protocol_a = analyze_single(Path(args.json_a), out_dir / "model_a")

    if args.json_b is None:
        print(f"Saved single-model analysis in: {out_dir}")
        return

    rows_b, summary_b, model_b, protocol_b = analyze_single(Path(args.json_b), out_dir / "model_b")

    comp_rows = compare_rows(rows_a, rows_b)
    comp_summary = summarize_comparison(comp_rows)
    comp_summary["model_a"] = model_a
    comp_summary["model_b"] = model_b
    comp_summary["protocol_a"] = protocol_a
    comp_summary["protocol_b"] = protocol_b

    save_csv(comp_rows, out_dir / "comparison.csv")
    save_json(comp_summary, out_dir / "comparison_summary.json")
    save_comparison_plot(comp_summary, out_dir / "comparison_plot.png", f"{model_a} vs {model_b}")
    save_side_by_side_error_plot(
        summary_a=summary_a,
        summary_b=summary_b,
        out_path=out_dir / "side_by_side_error_plot.png",
        model_a_name=model_a,
        model_b_name=model_b,
        title=f"{model_a} vs {model_b} - error breakdown",
    )

    print(f"Saved full comparison analysis in: {out_dir}")


if __name__ == "__main__":
    main()