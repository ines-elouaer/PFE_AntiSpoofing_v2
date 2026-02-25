
import argparse
import json
from pathlib import Path
from math import sqrt

def confusion(video_scores: dict, th: float):
    tn = fp = fn = tp = 0
    for d in video_scores.values():
        y = int(d["label"])
        score = float(d["score"])
        pred = 1 if score >= th else 0
        if y == 0 and pred == 0: tn += 1
        elif y == 0 and pred == 1: fp += 1
        elif y == 1 and pred == 0: fn += 1
        elif y == 1 and pred == 1: tp += 1
    return tn, fp, fn, tp

def metrics(video_scores: dict, th: float):
    tn, fp, fn, tp = confusion(video_scores, th)
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total else 0.0

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    # label 1 = attack
    apcer = fn / (tp + fn) if (tp + fn) else 0.0
    bpcer = fp / (tn + fp) if (tn + fp) else 0.0
    acer = 0.5 * (apcer + bpcer)

    return {
        "ACC": acc,
        "F1": f1,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }

def mean_std(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": []}
    m = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": m, "std": 0.0, "values": values}
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return {"mean": m, "std": sqrt(var), "values": values}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=r"experiments\pts",
                    help="Folder containing multiple PTS runs (searches for **/test_scores.json).")
    ap.add_argument("--out", type=str, default=None,
                    help="Output json path. Default: <root>/pts_summary.json")
    ap.add_argument("--name", type=str, default="PTS_CNN_LSTM",
                    help="Name of the model key in JSON.")
    ap.add_argument("--require_seed", type=int, default=0,
                    help="If 1, skip runs that don't have seed in config.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    json_paths = sorted(root.rglob("test_scores.json"))
    if not json_paths:
        raise RuntimeError(f"No test_scores.json found under: {root}")

    rows = []
    for p in json_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        th = float(data.get("threshold_val", {}).get("th", 0.5))
        test_scores = data["test_scores"]

        m = metrics(test_scores, th)

        cfg = data.get("config", {})
        seed = cfg.get("seed", None)
        if args.require_seed and seed is None:
            continue

        rows.append({
            "exp_dir": str(p.parent),
            "seed": seed,
            "th": th,
            **m
        })

    out = {args.name: {}}
    for key in ["ACC", "F1", "APCER", "BPCER", "ACER"]:
        vals = [r[key] for r in rows]
        out[args.name][key] = mean_std(vals)

    out["runs"] = rows

    out_path = Path(args.out) if args.out else (root / "pts_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Saved:", out_path)
    print(f"\n{args.name} mean±std over {len(rows)} runs")
    for k in ["ACER", "F1", "ACC", "APCER", "BPCER"]:
        d = out[args.name][k]
        print(f"{k}: mean={d['mean']:.4f} std={d['std']:.4f} values={[round(x,4) for x in d['values']]}")

if __name__ == "__main__":
    main()