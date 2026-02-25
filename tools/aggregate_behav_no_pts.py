
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


METRICS = ["ACC", "F1", "APCER", "BPCER", "ACER", "AUC"]

# Parse results.txt produced by eval_cnn_lstm_video_level.py
PATTERNS = {
    "ACC": re.compile(r"Video Accuracy:\s*([0-9.]+)", re.IGNORECASE),
    "F1": re.compile(r"F1:\s*([0-9.]+)", re.IGNORECASE),
    "APCER": re.compile(r"APCER:\s*([0-9.]+)", re.IGNORECASE),
    "BPCER": re.compile(r"BPCER:\s*([0-9.]+)", re.IGNORECASE),
    "ACER": re.compile(r"ACER:\s*([0-9.]+)", re.IGNORECASE),
    "AUC": re.compile(r"ROC-AUC:\s*([0-9.]+)", re.IGNORECASE),
    "TH": re.compile(r"Threshold\s*\(from VAL\):\s*([0-9.]+)", re.IGNORECASE),
}


def read_results_txt(exp_dir: Path) -> Dict:
    p = exp_dir / "results.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing results.txt in {exp_dir}")

    txt = p.read_text(encoding="utf-8", errors="ignore")
    out: Dict[str, float] = {}

    # threshold (optional)
    mth = PATTERNS["TH"].search(txt)
    if mth:
        out["th"] = float(mth.group(1))

    for k in METRICS:
        m = PATTERNS[k].search(txt)
        if not m:
            raise RuntimeError(f"Cannot parse {k} from {p}")
        out[k] = float(m.group(1))
    return out


def summarize(values: List[float]) -> Dict:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "std": float(arr.std(ddof=0)) if len(arr) else 0.0,
        "values": [float(x) for x in arr.tolist()],
    }


def aggregate_group(name: str, exp_dirs: List[Path]) -> Tuple[Dict, List[Dict]]:
    runs = []
    for d in exp_dirs:
        m = read_results_txt(d)
        seed = None
        mseed = re.search(r"seed(\d+)", d.name, re.IGNORECASE)
        if mseed:
            seed = int(mseed.group(1))

        runs.append({
            "exp_dir": str(d.resolve()),
            "seed": seed,
            **m
        })

  
    summary = {}
    for k in METRICS:
        summary[k] = summarize([r[k] for r in runs])

    return {name: summary}, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=r"experiments\behav_no_pts", help="Folder containing the 6 runs")
    ap.add_argument("--out_json", default=r"experiments\behav_no_pts\aggregate_behav_no_pts.json")
    args = ap.parse_args()

    base = Path(args.base_dir)

    deep_dirs = [
        base / "deep_seed42",
        base / "deep_seed43",
        base / "deep_seed44",
    ]
    behav_dirs = [
        base / "deep_behav_seed42",
        base / "deep_behav_seed43",
        base / "deep_behav_seed44",
    ]

    for d in deep_dirs + behav_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing run directory: {d}")

    deep_sum, deep_runs = aggregate_group("Deep_only_no_PTS", deep_dirs)
    behav_sum, behav_runs = aggregate_group("Deep_plus_Behav_no_PTS", behav_dirs)

    out = {
        **deep_sum,
        **behav_sum,
        "runs": {
            "deep_only": deep_runs,
            "deep_plus_behav": behav_runs,
        }
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

   
    def fmt(x): return f"{x:.4f}"
    print("\n================ Aggregation (mean ± std) ================")
    for group_name, group in [("Deep_only_no_PTS", out["Deep_only_no_PTS"]),
                              ("Deep_plus_Behav_no_PTS", out["Deep_plus_Behav_no_PTS"])]:
        print(f"\n[{group_name}]")
        for k in METRICS:
            print(f"  {k}: {fmt(group[k]['mean'])} ± {fmt(group[k]['std'])}   {group[k]['values']}")

    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()