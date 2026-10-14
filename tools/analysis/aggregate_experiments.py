import argparse
import json
from pathlib import Path
from math import sqrt
from collections import defaultdict

METRICS = ["ACC", "F1", "APCER", "BPCER", "ACER", "AUC"]


def mean_std(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": []}
    m = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": m, "std": 0.0, "values": values}
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return {"mean": m, "std": sqrt(var), "values": values}


def load_run(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    model_name = data.get("model_name", "unknown_model")
    protocol_name = data.get("protocol_name", "unknown_protocol")
    threshold_protocol = data.get("threshold_protocol", "unknown")
    threshold_val = data.get("threshold_val", {})
    threshold_used = data.get("threshold_used", {})
    metrics = data.get("metrics_test", {})
    cfg = data.get("config", {})

    return {
        "exp_dir": str(json_path.parent),
        "seed": cfg.get("seed", None),
        "model_name": model_name,
        "protocol_name": protocol_name,
        "threshold_protocol": threshold_protocol,
        "threshold_val_th": threshold_val.get("th", None),
        "threshold_used_th": threshold_used.get("th", None),
        "threshold_source": threshold_used.get("source", None),
        "ACC": metrics.get("ACC", None),
        "F1": metrics.get("F1", None),
        "APCER": metrics.get("APCER", None),
        "BPCER": metrics.get("BPCER", None),
        "ACER": metrics.get("ACER", None),
        "AUC": metrics.get("AUC", None),
        "TN": metrics.get("TN", None),
        "FP": metrics.get("FP", None),
        "FN": metrics.get("FN", None),
        "TP": metrics.get("TP", None),
    }


def summarize_group(rows):
    out = {}
    for k in METRICS:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = mean_std(vals)

    out["n_runs"] = len(rows)
    out["seeds"] = [r["seed"] for r in rows]
    out["threshold_protocols"] = sorted(set(r["threshold_protocol"] for r in rows))
    out["threshold_sources"] = sorted(set(r["threshold_source"] for r in rows if r["threshold_source"] is not None))
    out["threshold_used_values"] = [r["threshold_used_th"] for r in rows]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root experiments folder")
    ap.add_argument("--out_json", default=None, help="Output summary json")
    args = ap.parse_args()

    root = Path(args.root)
    json_paths = sorted(root.rglob("test_scores.json"))
    if not json_paths:
        raise RuntimeError(f"No test_scores.json found under {root}")

    rows = [load_run(p) for p in json_paths]

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["protocol_name"]].append(r)

    report = {}
    for protocol_name, group_rows in grouped.items():
        report[protocol_name] = summarize_group(group_rows)

    report["runs"] = rows

    out_path = Path(args.out_json) if args.out_json else (root / "aggregate_summary.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    print("\n===== Aggregation summary =====")
    for protocol_name, stats in report.items():
        if protocol_name == "runs":
            continue

        print(f"\n[{protocol_name}]")
        print(f"n_runs: {stats['n_runs']}")
        print(f"threshold_protocols: {stats['threshold_protocols']}")
        print(f"threshold_sources: {stats['threshold_sources']}")

        for k in METRICS:
            d = stats[k]
            print(f"{k}: {d['mean']:.4f} ± {d['std']:.4f}")


if __name__ == "__main__":
    main()