import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

METRIC_PATTERNS = {
    "ACC": re.compile(r"Video Accuracy:\s*([0-9.]+)"),
    "F1": re.compile(r"F1:\s*([0-9.]+)"),
    "APCER": re.compile(r"APCER:\s*([0-9.]+)"),
    "BPCER": re.compile(r"BPCER:\s*([0-9.]+)"),
    "ACER": re.compile(r"ACER:\s*([0-9.]+)"),
    "AUC": re.compile(r"ROC-AUC:\s*([0-9.]+)"),
    "TH": re.compile(r"Threshold \(from VAL\):\s*([0-9.]+)"),
    "TN": re.compile(r"TN=([0-9]+)"),
    "FP": re.compile(r"FP=([0-9]+)"),
    "FN": re.compile(r"FN=([0-9]+)"),
    "TP": re.compile(r"TP=([0-9]+)"),
}

def try_read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_results_txt(text: str) -> Dict:
    out: Dict[str, float] = {}
    for k, pat in METRIC_PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        val = m.group(1)
        if k in ("TN", "FP", "FN", "TP"):
            out[k] = int(val)
        else:
            out[k] = float(val)
    return out

def parse_scores_json(path: str) -> Dict:
   
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    out: Dict = {}

    th = None
    if isinstance(j.get("threshold_val"), dict) and "th" in j["threshold_val"]:
        th = j["threshold_val"]["th"]
    if th is not None:
        out["TH"] = float(th)

    
    cfg = j.get("config", {})
    if isinstance(cfg, dict):
        out["use_behav"] = bool(cfg.get("use_behav", False))
        out["seed"] = cfg.get("seed", None)

    return out

def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(var)

def summarize_runs(runs: List[Dict]) -> Dict:
    keys = ["ACC", "F1", "APCER", "BPCER", "ACER", "AUC"]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r and isinstance(r[k], (int, float))]
        m, s = mean_std([float(v) for v in vals])
        out[k] = {"mean": m, "std": s, "values": vals}
    return out


def find_exp_dirs(root: str) -> List[str]:
   
    exp_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        fn = set(filenames)
        if ("best_model.pth" in fn) or ("results.txt" in fn) or ("test_scores.json" in fn):
            exp_dirs.append(dirpath)
    exp_dirs = sorted(set(exp_dirs))
    return exp_dirs

@dataclass
class Run:
    exp_dir: str
    seed: Optional[int]
    th: Optional[float]
    ACC: Optional[float]
    F1: Optional[float]
    APCER: Optional[float]
    BPCER: Optional[float]
    ACER: Optional[float]
    AUC: Optional[float]
    TN: Optional[int]
    FP: Optional[int]
    FN: Optional[int]
    TP: Optional[int]
    use_behav: Optional[bool]

def load_one_run(exp_dir: str) -> Dict:
    out: Dict = {"exp_dir": exp_dir}

    txt = try_read_text(os.path.join(exp_dir, "results.txt"))
    if txt:
        out.update(parse_results_txt(txt))

    js_path = os.path.join(exp_dir, "test_scores.json")
    if os.path.exists(js_path):
        out.update(parse_scores_json(js_path))

    cfg_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "seed" in cfg:
                out["seed"] = cfg["seed"]
            if "use_behav" in cfg:
                out["use_behav"] = bool(cfg["use_behav"])
        except Exception:
            pass

   
    if "TH" not in out and "th" in out:
        out["TH"] = out["th"]

    return out

def to_run_dict(d: Dict) -> Dict:
    # Keep only relevant fields (and make sure types are jsonable)
    fields = ["exp_dir", "seed", "TH", "ACC", "F1", "APCER", "BPCER", "ACER", "AUC", "TN", "FP", "FN", "TP", "use_behav"]
    o = {}
    for f in fields:
        if f in d:
            o[f if f != "TH" else "th"] = d[f]
    return o

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root folder containing experiment subfolders")
    parser.add_argument("--group", default="auto", choices=["auto", "deep_only", "deep_plus_behav"], help="How to group runs")
    parser.add_argument("--out_json", default=None, help="Optional output json path")
    args = parser.parse_args()

    exp_dirs = find_exp_dirs(args.root)
    if not exp_dirs:
        raise SystemExit(f"No experiments found under: {args.root}")

    runs = [load_one_run(d) for d in exp_dirs]

    
    deep_only = []
    deep_behav = []

    for r in runs:
        use_behav = r.get("use_behav", None)

        if args.group == "deep_only":
            deep_only.append(r)
        elif args.group == "deep_plus_behav":
            deep_behav.append(r)
        else:
            
            if use_behav is True:
                deep_behav.append(r)
            else:
                deep_only.append(r)

    report = {}
    if deep_only:
        report["Deep_only"] = summarize_runs(deep_only)
    if deep_behav:
        report["Deep_plus_Behav"] = summarize_runs(deep_behav)

    report["runs"] = {
        "deep_only": [to_run_dict(r) for r in deep_only],
        "deep_plus_behav": [to_run_dict(r) for r in deep_behav],
    }

    print(json.dumps(report, indent=2))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved: {args.out_json}")

if __name__ == "__main__":
    main()