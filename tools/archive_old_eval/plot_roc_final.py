

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ─── Chargement ──────────────────────────────────────────────────────

def load_scores(json_path: Path):
    """Charge labels et scores depuis un test_scores.json."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    test_scores = data["test_scores"]
    y_true, y_score = [], []
    for _, d in test_scores.items():
        y_true.append(int(d["label"]))
        y_score.append(float(d["score"]))
    return np.array(y_true), np.array(y_score)


def roc_mean_std(seed_dirs, n_grid=400):
    """Calcule ROC mean±std sur plusieurs seeds."""
    fpr_grid = np.linspace(0.0, 1.0, n_grid)
    tprs, aucs = [], []

    for d in seed_dirs:
        p = Path(d)
        if not p.exists():
            print(f"  [WARN] Fichier introuvable: {p}")
            continue
        y_true, y_score = load_scores(p)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(roc_auc)

    if not tprs:
        return None

    tprs = np.stack(tprs)
    return {
        "fpr":      fpr_grid,
        "tpr_mean": tprs.mean(axis=0),
        "tpr_std":  tprs.std(axis=0),
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs)),
        "aucs":     aucs,
    }


# ─── Configuration modèles ───────────────────────────────────────────

MODELS = {
    "CNN seul": {
        "seeds": [
            "reports/eval/step1/cnn/valopt/exp_cnn_seed42/test_scores.json",
            "reports/eval/step1/cnn/valopt/exp_cnn_seed43/test_scores.json",
            "reports/eval/step1/cnn/valopt/exp_cnn_seed44/test_scores.json",
        ],
        "color": "#E07A5F",
        "ls": ":",
        "lw": 1.5,
    },
    "CNN + LSTM": {
        "seeds": [
            "reports/eval/step1/cnn_lstm/valopt/exp_cnn_lstm_seed42/test_scores.json",
            "reports/eval/step1/cnn_lstm/valopt/exp_cnn_lstm_seed43/test_scores.json",
            "reports/eval/step1/cnn_lstm/valopt/exp_cnn_lstm_seed44/test_scores.json",
        ],
        "color": "#F4D03F",
        "ls": "--",
        "lw": 1.8,
    },
    "Deep+Behav Uniform": {
        "seeds": [
            "reports/eval/step2/deep_behav_no_pts/uniform/deep_behav_no_pts_uniform_seed42/test_scores.json",
            "reports/eval/step2/deep_behav_no_pts/uniform/deep_behav_no_pts_uniform_seed43/test_scores.json",
            "reports/eval/step2/deep_behav_no_pts/uniform/deep_behav_no_pts_uniform_seed44/test_scores.json",
        ],
        "color": "#6C8EBF",
        "ls": "-.",
        "lw": 1.8,
    },
    "Deep+Behav Consecutive ★": {
        "seeds": [
            "reports/eval/step2/deep_behav_no_pts/valopt/deep_behav_no_pts_consecutive_seed42/test_scores.json",
            "reports/eval/step2/deep_behav_no_pts/valopt/deep_behav_no_pts_consecutive_seed43/test_scores.json",
            "reports/eval/step2/deep_behav_no_pts/valopt/deep_behav_no_pts_consecutive_seed44/test_scores.json",
        ],
        "color": "#1B998B",
        "ls": "-",
        "lw": 2.8,
    },
}


# ─── Plot ────────────────────────────────────────────────────────────

def plot_roc(out_png: str, title: str, dark_mode: bool = True):
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    if dark_mode:
        bg      = "#0D1B2A"
        ax_bg   = "#112240"
        text_c  = "#CCD6F6"
        grid_c  = "#1B3A5C"
        legend_fc = "#0D1B2A"
    else:
        bg = ax_bg = text_c = grid_c = legend_fc = None

    fig, ax = plt.subplots(figsize=(8, 6.5))
    if dark_mode:
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(ax_bg)

    # Chance
    ax.plot([0, 1], [0, 1], "--", color="#445577" if dark_mode else "gray",
            lw=1.2, alpha=0.6, label="Aléatoire (AUC = 0.500)")

    for name, cfg in MODELS.items():
        stats = roc_mean_std(cfg["seeds"])
        if stats is None:
            print(f"  [SKIP] {name} — données manquantes")
            continue

        fpr      = stats["fpr"]
        tpr_mean = stats["tpr_mean"]
        tpr_std  = stats["tpr_std"]
        label = (f"{name}  "
                 f"(AUC = {stats['auc_mean']:.3f} ± {stats['auc_std']:.3f})")

        ax.plot(fpr, tpr_mean, color=cfg["color"],
                ls=cfg["ls"], lw=cfg["lw"], label=label)
        ax.fill_between(
            fpr,
            np.clip(tpr_mean - tpr_std, 0, 1),
            np.clip(tpr_mean + tpr_std, 0, 1),
            color=cfg["color"], alpha=0.10,
        )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate (FPR) = BPCER",
                  color=text_c if dark_mode else "black", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR) = 1 - APCER",
                  color=text_c if dark_mode else "black", fontsize=12)
    ax.set_title(title,
                 color="white" if dark_mode else "black",
                 fontsize=14, pad=12)

    if dark_mode:
        ax.tick_params(colors=text_c)
        for sp in ax.spines.values():
            sp.set_color(grid_c)
        ax.grid(True, color=grid_c, alpha=0.5)
    else:
        ax.grid(True, alpha=0.3)

    legend = ax.legend(
        loc="lower right",
        framealpha=0.85,
        facecolor=legend_fc if dark_mode else "white",
        edgecolor=grid_c if dark_mode else "gray",
        labelcolor=text_c if dark_mode else "black",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight",
                facecolor=bg if dark_mode else "white")
    plt.close()
    print(f"[OK] ROC sauvegardée : {out_png}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ROC curves — PAD project")
    parser.add_argument("--out_png", default=r"reports\plots\roc_final.png")
    parser.add_argument("--title",   default="Courbes ROC — Comparaison des modèles\nCASIA-FASD test set (mean ± std, 3 seeds)")
    parser.add_argument("--light",   action="store_true", help="Mode clair (pour mémoire)")
    args = parser.parse_args()

    print("Génération des courbes ROC...")
    plot_roc(args.out_png, args.title, dark_mode=not args.light)


if __name__ == "__main__":
    main()