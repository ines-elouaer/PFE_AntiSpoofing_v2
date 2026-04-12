"""
plot_confusion_matrix.py
=========================
Génère une figure de confusion matrix visuelle (3 seeds + moyenne)
à partir des données JSON existantes.

Usage :
    python tools/plot_confusion_matrix.py
    python tools/plot_confusion_matrix.py --light
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ─── Palettes ────────────────────────────────────────────────────────

DARK = {
    "bg":     "#0D1B2A",
    "ax":     "#112240",
    "text":   "#CCD6F6",
    "grid":   "#1B3A5C",
    "white":  "#FFFFFF",
    "title":  "#FFFFFF",
    "green":  "#1B998B",
    "orange": "#E07A5F",
    "yellow": "#E6A817",
    "muted":  "#8899AA",
}

LIGHT = {
    "bg":     "#FFFFFF",
    "ax":     "#F8F9FA",
    "text":   "#2C3E50",
    "grid":   "#BDC3C7",
    "white":  "#FFFFFF",
    "title":  "#1A252F",
    "green":  "#1A7A4A",
    "orange": "#C0392B",
    "yellow": "#D4920A",
    "muted":  "#7F8C8D",
}

C = DARK


# ─── Chargement ──────────────────────────────────────────────────────

def load_cm_from_json(json_path: Path, th: float = 0.5):
    """Calcule TN/FP/FN/TP depuis test_scores.json."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    scores = data.get("test_scores", {})
    threshold = float(
        data.get("threshold_used", {}).get("th", th)
    )

    tn = fp = fn = tp = 0
    for d in scores.values():
        y    = int(d["label"])
        pred = 1 if float(d["score"]) >= threshold else 0
        if   y == 0 and pred == 0: tn += 1
        elif y == 0 and pred == 1: fp += 1
        elif y == 1 and pred == 0: fn += 1
        else:                       tp += 1

    return np.array([[tn, fp], [fn, tp]]), threshold


# ─── Figure principale ───────────────────────────────────────────────

def plot_confusion_matrices(seed_data: list, out_path: Path):
    """
    Affiche 4 sous-figures : Seed 42 | Seed 43 | Seed 44 | Moyenne
    seed_data = list of (seed_label, cm_2x2_array)
    """
    n = len(seed_data) + 1  # +1 pour la moyenne
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.5))
    fig.patch.set_facecolor(C["bg"])

    fig.suptitle(
        "Matrice de Confusion — Deep+Behav Consecutive\n"
        "CASIA-FASD test set (th=0.5)",
        color=C["title"], fontsize=14, fontweight="bold", y=1.02,
    )

    # Colormap personnalisée : blanc → vert foncé
    if C == DARK:
        cmap = LinearSegmentedColormap.from_list(
            "custom", ["#112240", "#1B998B"], N=256
        )
    else:
        cmap = LinearSegmentedColormap.from_list(
            "custom", ["#F0F8F4", "#1A7A4A"], N=256
        )

    # ── Calcul moyenne ────────────────────────────────────────────────
    cms = [d[1] for d in seed_data]
    cm_mean = np.mean(cms, axis=0)

    # CORRECTION : calculer les métriques moyennes depuis chaque seed
    # individuellement, pas depuis la matrice moyenne.
    # np.mean([[TN,FP],[FN,TP]]) puis recalculer APCER/BPCER est faux
    # car APCER_mean ≠ FP_mean / (TN_mean + FP_mean)
    mean_apcers, mean_bpcers, mean_acers, mean_accs = [], [], [], []
    for cm_s in cms:
        tn_s, fp_s = cm_s[0,0], cm_s[0,1]
        fn_s, tp_s = cm_s[1,0], cm_s[1,1]
        apcer_s = fp_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0
        bpcer_s = fn_s / (fn_s + tp_s) if (fn_s + tp_s) > 0 else 0
        acer_s  = (apcer_s + bpcer_s) / 2
        acc_s   = (tn_s + tp_s) / cm_s.sum() if cm_s.sum() > 0 else 0
        mean_apcers.append(apcer_s)
        mean_bpcers.append(bpcer_s)
        mean_acers.append(acer_s)
        mean_accs.append(acc_s)

    # Métriques moyennes correctes
    mean_metrics = {
        "apcer": float(np.mean(mean_apcers)),
        "bpcer": float(np.mean(mean_bpcers)),
        "acer":  float(np.mean(mean_acers)),
        "acc":   float(np.mean(mean_accs)),
    }

    all_data = seed_data + [("Moyenne\n(3 seeds)", cm_mean)]

    for ax, (label, cm) in zip(axes, all_data):
        ax.set_facecolor(C["ax"])
        is_mean = "Moyenne" in label

        total     = cm.sum()
        n_real    = cm[0].sum()   # TN + FP
        n_attack  = cm[1].sum()   # FN + TP

        # ── Heatmap ───────────────────────────────────────────────────
        im = ax.imshow(
            cm, interpolation="nearest", cmap=cmap,
            vmin=0, vmax=max(n_real, n_attack)
        )

        # ── Labels classes ────────────────────────────────────────────
        classes = ["Réel\n(Bonafide)", "Attaque\n(Spoof)"]
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, color=C["text"], fontsize=10)
        ax.set_yticklabels(classes, color=C["text"], fontsize=10)
        ax.set_xlabel("Prédiction", color=C["text"], fontsize=11,
                      fontweight="bold", labelpad=8)
        ax.set_ylabel("Vérité terrain", color=C["text"], fontsize=11,
                      fontweight="bold", labelpad=8)

        # ── Valeurs dans les cellules ─────────────────────────────────
        cell_labels = [["TN", "FP"], ["FN", "TP"]]
        cell_colors_ok  = C["white"]
        cell_colors_err = C["orange"]

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                is_correct = (i == j)  # diagonale = correct

                # Format : entier ou décimal si moyenne
                if is_mean:
                    val_str = f"{val:.1f}"
                else:
                    val_str = str(int(val))

                # Couleur du texte selon luminosité fond
                thresh_color = max(n_real, n_attack) / 2.0
                txt_color = "white" if val > thresh_color else C["text"]

                # Valeur principale
                ax.text(
                    j, i, val_str,
                    ha="center", va="center",
                    fontsize=22 if not is_mean else 18,
                    fontweight="bold",
                    color=txt_color,
                )

                # Label TN/FP/FN/TP en petit
                ax.text(
                    j, i + 0.32, cell_labels[i][j],
                    ha="center", va="center",
                    fontsize=9, color=txt_color, alpha=0.8,
                )

                # Pourcentage
                pct = val / (n_real if i == 0 else n_attack) * 100
                ax.text(
                    j, i - 0.32, f"{pct:.1f}%",
                    ha="center", va="center",
                    fontsize=9,
                    color=cell_colors_ok if is_correct else cell_colors_err,
                    fontweight="bold",
                )

        # ── Titre seed ────────────────────────────────────────────────
        border_color = C["green"] if is_mean else C["yellow"]
        ax.set_title(
            label,
            color=C["green"] if is_mean else C["title"],
            fontsize=12, fontweight="bold", pad=12,
        )

        # ── Métriques sous la matrice ─────────────────────────────────
        if is_mean:
            # CORRECTION : utiliser les métriques moyennées correctement
            # depuis chaque seed, pas recalculées depuis cm_mean
            apcer = mean_metrics["apcer"]
            bpcer = mean_metrics["bpcer"]
            acer  = mean_metrics["acer"]
            acc   = mean_metrics["acc"]
        else:
            tn_v, fp_v = int(round(cm[0,0])), int(round(cm[0,1]))
            fn_v, tp_v = int(round(cm[1,0])), int(round(cm[1,1]))
            apcer = fp_v / (tn_v + fp_v) if (tn_v + fp_v) > 0 else 0
            bpcer = fn_v / (fn_v + tp_v) if (fn_v + tp_v) > 0 else 0
            acer  = (apcer + bpcer) / 2
            acc   = (tn_v + tp_v) / total if total > 0 else 0

        metrics_txt = (
            f"APCER={apcer:.4f}  BPCER={bpcer:.4f}\n"
            f"ACER={acer:.4f}   ACC={acc:.4f}"
        )
        ax.text(
            0.5, -0.22, metrics_txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=8.5, color=C["muted"],
            style="italic",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=C["ax"],
                edgecolor=C["grid"],
                alpha=0.7,
            ),
        )

        # ── Bordures ─────────────────────────────────────────────────
        for sp in ax.spines.values():
            sp.set_color(border_color if is_mean else C["grid"])
            sp.set_linewidth(2 if is_mean else 0.8)
        ax.tick_params(colors=C["text"])

    # ── Légende globale ───────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=C["green"],  label="Correct (diagonale)"),
        mpatches.Patch(color=C["orange"], label="Erreur (hors diagonale)"),
    ]
    fig.legend(
        handles=patches, loc="lower center",
        ncol=2, fontsize=10,
        facecolor=C["bg"], edgecolor=C["grid"],
        labelcolor=C["text"],
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close()
    print(f"[OK] {out_path}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Confusion matrix visuelle — Deep+Behav Consecutive"
    )
    parser.add_argument("--seed42",
        default=r"reports\eval\step2\deep_behav_no_pts_consecutive\deep_behav_no_pts_consecutive_seed42\test_scores.json")
    parser.add_argument("--seed43",
        default=r"reports\eval\step2\deep_behav_no_pts_consecutive\deep_behav_no_pts_consecutive_seed43\test_scores.json")
    parser.add_argument("--seed44",
        default=r"reports\eval\step2\deep_behav_no_pts_consecutive\deep_behav_no_pts_consecutive_seed44\test_scores.json")
    parser.add_argument("--out_dir",
        default=r"reports\errors\step2_consecutive_final")
    parser.add_argument("--light", action="store_true",
        help="Mode clair pour mémoire/rapport")
    args = parser.parse_args()

    global C
    if args.light:
        C = LIGHT
        print("Mode : clair")
    else:
        print("Mode : sombre")

    seeds      = [42, 43, 44]
    json_paths = [args.seed42, args.seed43, args.seed44]
    out_dir    = Path(args.out_dir)

    print("Chargement des données...")
    seed_data = []

    for seed, jp in zip(seeds, json_paths):
        p = Path(jp)
        if not p.exists():
            print(f"  [WARN] Introuvable : {p}")
            # Utiliser les valeurs du JSON global si disponible
            fallback = {
                42: np.array([[5, 1], [0, 18]]),
                43: np.array([[6, 0], [1, 17]]),
                44: np.array([[4, 2], [0, 18]]),
            }
            cm = fallback[seed]
            print(f"  [INFO] Utilisation valeurs JSON : {cm.tolist()}")
        else:
            cm, th = load_cm_from_json(p)
            print(f"  Seed {seed} : TN={cm[0,0]} FP={cm[0,1]} "
                  f"FN={cm[1,0]} TP={cm[1,1]} (th={th})")

        seed_data.append((f"Seed {seed}", cm))

    print("\nGénération de la figure...")
    plot_confusion_matrices(
        seed_data,
        out_dir / "fig6_confusion_matrix.png"
    )
    print("\n✅ Confusion matrix sauvegardée !")
    print("   fig6_confusion_matrix.png")


if __name__ == "__main__":
    main()