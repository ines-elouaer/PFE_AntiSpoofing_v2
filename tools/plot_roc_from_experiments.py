# tools/plot_roc_from_experiments.py
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import roc_curve, auc
except ImportError:
    raise SystemExit("❌ sklearn requis: pip install scikit-learn")

# ---------- IO ----------
def load_scores(exp_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    p = exp_dir / "test_scores.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    test_scores = data["test_scores"]  # {vid: {label, score}}
    y_true, y_score = [], []
    for _, d in test_scores.items():
        y_true.append(int(d["label"]))
        y_score.append(float(d["score"]))
    return np.asarray(y_true), np.asarray(y_score)


# ---------- ROC mean±std over seeds ----------
def roc_mean_std(exp_dirs: List[Path], grid_points: int = 400) -> Dict:
    fpr_grid = np.linspace(0.0, 1.0, grid_points)

    tprs = []
    aucs = []
    for d in exp_dirs:
        y_true, y_score = load_scores(d)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        tpr_i = np.interp(fpr_grid, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
        aucs.append(roc_auc)

    tprs = np.stack(tprs, axis=0)
    return {
        "fpr": fpr_grid,
        "tpr_mean": tprs.mean(axis=0),
        "tpr_std": tprs.std(axis=0),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "aucs": [float(x) for x in aucs],
    }


def parse_models_args(model_args: List[str]) -> Dict[str, List[Path]]:
    models = {}
    for s in model_args:
        if "=" not in s:
            raise ValueError(f"Bad --model format: {s}")
        name, paths = s.split("=", 1)
        exp_dirs = [Path(p.strip()) for p in paths.split(",") if p.strip()]
        models[name.strip()] = exp_dirs
    return models


# ---------- Paper-style plot ----------
def paper_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "grid.alpha": 0.25,
    })


def plot_roc(models: Dict[str, List[Path]], out_png: Path, title: str):
    paper_style()
    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_subplot(111)

    # Chance line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.4, alpha=0.7, label="Chance")

    for name, exp_dirs in models.items():
        stats = roc_mean_std(exp_dirs)
        fpr = stats["fpr"]
        tpr_m = stats["tpr_mean"]
        tpr_s = stats["tpr_std"]

        label = f"{name}  (AUC={stats['auc_mean']:.3f}±{stats['auc_std']:.3f})"
        ax.plot(fpr, tpr_m, label=label)
        ax.fill_between(
            fpr,
            np.clip(tpr_m - tpr_s, 0, 1),
            np.clip(tpr_m + tpr_s, 0, 1),
            alpha=0.12,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(title)
    ax.grid(True)

    # Put legend inside but clean
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"✅ Saved: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[],
                    help="Name=dir1,dir2,dir3  (repeatable)")
    ap.add_argument("--out_png", type=str, default="reports/plots/roc_paper.png")
    ap.add_argument("--title", type=str, default="ROC Curves (mean ± std over 3 seeds)")
    args = ap.parse_args()

    if not args.model:
        raise SystemExit("❌ Provide at least one --model")

    models = parse_models_args(args.model)
    plot_roc(models, Path(args.out_png), args.title)


if __name__ == "__main__":
    main()