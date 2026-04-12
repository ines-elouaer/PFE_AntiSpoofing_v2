"""
analyze_errors_final.py
========================
Analyse d'erreurs complète sur le meilleur modèle :
Deep+Behav Consecutive — Step 2

Ce script :
  1. Charge les test_scores.json des 3 seeds
  2. Identifie les FP et FN avec leurs scores exacts
  3. Sauvegarde les CSV d'erreurs
  4. Génère les figures : FP/FN par seed + vidéos difficiles + distribution

Usage depuis E:\PFE_AntiSpoofing_v2 :

    python tools/analyze_errors_final.py

Ou avec chemins personnalisés :
    python tools/analyze_errors_final.py ^
        --seed42 reports/eval/step2/deep_behav_no_pts/valopt/deep_behav_no_pts_consecutive_seed42/test_scores.json ^
        --seed43 reports/eval/step2/deep_behav_no_pts/valopt/deep_behav_no_pts_consecutive_seed43/test_scores.json ^
        --seed44 reports/eval/step2/deep_behav_no_pts/valopt/deep_behav_no_pts_consecutive_seed44/test_scores.json ^
        --out_dir reports/errors/step2_consecutive_final
"""

import argparse
import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Chargement ──────────────────────────────────────────────────────

def load_scores(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    scores     = data["test_scores"]
    threshold  = float(data["threshold_used"]["th"])
    model_name = data.get("model_name", "unknown")
    protocol   = data.get("protocol_name", "unknown")
    return scores, threshold, model_name, protocol


def build_rows(scores: dict, threshold: float):
    rows = []
    for vid, d in scores.items():
        label = int(d["label"])
        score = float(d["score"])
        pred  = 1 if score >= threshold else 0

        if   label == 0 and pred == 1: etype = "FP_real_as_attack"
        elif label == 1 and pred == 0: etype = "FN_attack_as_real"
        else:                           etype = "correct"

        rows.append({
            "video_id":   vid,
            "label":      label,
            "score":      round(score, 6),
            "prediction": pred,
            "threshold":  threshold,
            "error_type": etype,
        })
    return rows


def summarize(rows):
    c = Counter(r["error_type"] for r in rows)
    n = len(rows)
    return {
        "total":    n,
        "correct":  c.get("correct", 0),
        "FP":       c.get("FP_real_as_attack", 0),
        "FN":       c.get("FN_attack_as_real", 0),
        "accuracy": round(c.get("correct", 0) / n, 4) if n else 0,
    }


# ─── Sauvegarde CSV ──────────────────────────────────────────────────

def save_csv(rows, path: Path, only_errors=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r for r in rows if r["error_type"] != "correct"] if only_errors else rows
    if not data:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)


# ─── Figures ─────────────────────────────────────────────────────────

DARK = {
    "bg":      "#0D1B2A",
    "ax":      "#112240",
    "text":    "#CCD6F6",
    "grid":    "#1B3A5C",
    "legend":  "#0D1B2A",
    "green":   "#1B998B",
    "red":     "#C0392B",
    "orange":  "#E07A5F",
    "yellow":  "#E6A817",
    "blue":    "#6C8EBF",
    "white":   "#FFFFFF",
    "title":   "#FFFFFF",
}

LIGHT = {
    "bg":      "#FFFFFF",
    "ax":      "#F8F9FA",
    "text":    "#2C3E50",
    "grid":    "#BDC3C7",
    "legend":  "#FFFFFF",
    "green":   "#27AE60",
    "red":     "#2980B9",
    "orange":  "#D4920A",
    "yellow":  "#D4920A",
    "blue":    "#2980B9",
    "white":   "#2C3E50",
    "title":   "#1A252F",
}

# Palette active — changée par --light
C = DARK


def fig_fp_fn_per_seed(summaries, seeds, out_path: Path):
    """Barplot FP vs FN par seed."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["ax"])

    x  = np.arange(len(seeds))
    w  = 0.35
    fp = [s["FP"] for s in summaries]
    fn = [s["FN"] for s in summaries]

    b1 = ax.bar(x - w/2, fp, w, color=C["red"],    alpha=0.85,
                label="FP — Réel détecté comme Attaque")
    b2 = ax.bar(x + w/2, fn, w, color=C["yellow"], alpha=0.85,
                label="FN — Attaque détectée comme Réel")

    for bar, val in [(b, v) for bars, vals in [(b1, fp), (b2, fn)]
                    for b, v in zip(bars, vals)]:
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    str(val), ha="center", color=C["white"],
                    fontsize=13, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], color=C["text"], fontsize=11)
    ax.set_ylabel("Nombre d'erreurs", color=C["text"], fontsize=11)
    ax.set_title(
        "Faux Positifs vs Faux Négatifs — Deep+Behav Consecutive\n"
        f"Total test = {summaries[0]['total']} vidéos par seed",
        color=C["white"], fontsize=12,
    )
    ax.set_ylim(0, max(max(fp), max(fn)) + 2)
    ax.tick_params(colors=C["text"])
    for sp in ax.spines.values(): sp.set_color(C["grid"])
    ax.legend(
        framealpha=0.3, facecolor=C["legend"],
        edgecolor=C["grid"], labelcolor=C["text"], fontsize=10,
    )

    # Moyenne
    avg_err = sum(s["FP"] + s["FN"] for s in summaries) / len(summaries)
    ax.text(0.5, 0.92, f"Moyenne : {avg_err:.1f} erreur(s) / {summaries[0]['total']} vidéos",
            transform=ax.transAxes, ha="center", color=C["text"],
            fontsize=9, style="italic")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  [OK] {out_path}")


def fig_hard_videos(all_rows_by_seed, seeds, threshold, out_path: Path):
    """Barplot des vidéos mal classées avec leurs scores exacts."""
    # Collecter toutes les erreurs
    errors = []
    for seed, rows in zip(seeds, all_rows_by_seed):
        for r in rows:
            if r["error_type"] != "correct":
                errors.append({
                    "label": f"{r['video_id']}\n({r['error_type'][:2]} s{seed})",
                    "score": r["score"],
                    "etype": r["error_type"],
                })

    if not errors:
        print("  [INFO] Aucune erreur à afficher.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(errors) * 2.2), 5))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["ax"])

    labels = [e["label"] for e in errors]
    scores = [e["score"] for e in errors]
    colors = [C["red"] if e["etype"].startswith("FP") else C["yellow"]
              for e in errors]

    bars = ax.bar(labels, scores, color=colors, alpha=0.85,
                  edgecolor=C["bg"], linewidth=1)
    ax.axhline(y=threshold, color=C["yellow"], ls="--", lw=2,
               alpha=0.8, label=f"Seuil décision (th={threshold})")

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.015,
                f"{score:.4f}", ha="center",
                color=C["white"], fontsize=10, fontweight="bold")

    ax.set_ylabel("Score d'attaque", color=C["text"], fontsize=11)
    ax.set_title(
        "Vidéos mal classées — Scores exacts\n"
        "(Rouge = FP : réel → attaque | Jaune = FN : attaque → réel)",
        color=C["white"], fontsize=12,
    )
    ax.set_ylim(0, 1.12)
    ax.tick_params(colors=C["text"])
    ax.set_xticklabels(labels, color=C["text"], fontsize=9)
    for sp in ax.spines.values(): sp.set_color(C["grid"])
    ax.legend(
        framealpha=0.3, facecolor=C["legend"],
        edgecolor=C["grid"], labelcolor=C["text"], fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  [OK] {out_path}")


def fig_score_distribution(all_rows_by_seed, seeds, threshold, out_path: Path):
    """Distribution des scores réel vs attaque pour chaque seed."""
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 5),
                             sharey=False)
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle(
        "Distribution des scores — Deep+Behav Consecutive\n"
        "Séparation réel vs attaque par seed",
        color=C["white"], fontsize=13, y=1.02,
    )

    bins = np.linspace(0, 1, 20)

    for ax, seed, rows in zip(axes, seeds, all_rows_by_seed):
        ax.set_facecolor(C["ax"])
        real_s = [r["score"] for r in rows if r["label"] == 0]
        atk_s  = [r["score"] for r in rows if r["label"] == 1]

        ax.hist(real_s, bins=bins, alpha=0.8, color=C["green"],
                label=f"Réel (n={len(real_s)})", edgecolor=C["bg"])
        ax.hist(atk_s,  bins=bins, alpha=0.8, color=C["orange"],
                label=f"Attaque (n={len(atk_s)})", edgecolor=C["bg"])
        ax.axvline(x=threshold, color=C["yellow"], ls="--",
                   lw=2, label=f"th={threshold}")

        ax.set_title(f"Seed {seed}", color=C["green"], fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("Score d'attaque", color=C["text"], fontsize=10)
        ax.set_ylabel("Vidéos", color=C["text"], fontsize=10)
        ax.tick_params(colors=C["text"])
        for sp in ax.spines.values(): sp.set_color(C["grid"])
        ax.legend(
            framealpha=0.3, facecolor=C["legend"],
            edgecolor=C["grid"], labelcolor=C["text"], fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  [OK] {out_path}")


def fig_score_means(all_rows_by_seed, seeds, out_path: Path):
    """Score moyen réel vs attaque par seed — montre la séparation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["ax"])

    x  = np.arange(len(seeds))
    w  = 0.35
    real_means = [np.mean([r["score"] for r in rows if r["label"] == 0])
                  for rows in all_rows_by_seed]
    atk_means  = [np.mean([r["score"] for r in rows if r["label"] == 1])
                  for rows in all_rows_by_seed]

    b1 = ax.bar(x - w/2, real_means, w, color=C["green"],  alpha=0.85,
                label="Réel (score moyen)")
    b2 = ax.bar(x + w/2, atk_means,  w, color=C["orange"], alpha=0.85,
                label="Attaque (score moyen)")

    ax.axhline(y=0.5, color=C["yellow"], ls="--", lw=1.5, alpha=0.7)

    for bars, vals in [(b1, real_means), (b2, atk_means)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.015,
                    f"{v:.3f}", ha="center",
                    color=C["white"], fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds],
                       color=C["text"], fontsize=11)
    ax.set_ylabel("Score moyen", color=C["text"], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Score moyen par classe — 3 seeds\n"
        "(Idéal : Réel → 0 | Attaque → 1)",
        color=C["white"], fontsize=12,
    )
    ax.tick_params(colors=C["text"])
    for sp in ax.spines.values(): sp.set_color(C["grid"])
    ax.legend(
        framealpha=0.3, facecolor=C["legend"],
        edgecolor=C["grid"], labelcolor=C["text"], fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  [OK] {out_path}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse d'erreurs — Deep+Behav Consecutive")
    parser.add_argument("--seed42",
        default=r"reports\eval\step2\deep_behav_no_pts\valopt\deep_behav_no_pts_consecutive_seed42\test_scores.json")
    parser.add_argument("--seed43",
        default=r"reports\eval\step2\deep_behav_no_pts\valopt\deep_behav_no_pts_consecutive_seed43\test_scores.json")
    parser.add_argument("--seed44",
        default=r"reports\eval\step2\deep_behav_no_pts\valopt\deep_behav_no_pts_consecutive_seed44\test_scores.json")
    parser.add_argument("--out_dir",
        default=r"reports\errors\step2_consecutive_final")
    parser.add_argument("--light", action="store_true",
        help="Mode clair (fond blanc) pour mémoire et rapport")
    args = parser.parse_args()

    # Switcher vers palette claire si demandé
    global C
    if args.light:
        C = LIGHT
        print("Mode : clair (fond blanc)")
    else:
        print("Mode : sombre (fond noir)")

    seeds     = [42, 43, 44]
    json_paths = [args.seed42, args.seed43, args.seed44]
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Chargement des données...")
    all_rows = []
    summaries = []
    threshold_used = 0.5

    for seed, jp in zip(seeds, json_paths):
        p = Path(jp)
        if not p.exists():
            print(f"  [WARN] Introuvable: {p}")
            all_rows.append([])
            summaries.append({"total": 0, "correct": 0, "FP": 0, "FN": 0, "accuracy": 0})
            continue

        scores, threshold_used, model_name, protocol = load_scores(p)
        rows = build_rows(scores, threshold_used)
        s    = summarize(rows)
        all_rows.append(rows)
        summaries.append(s)

        print(f"  Seed {seed}: correct={s['correct']} | FP={s['FP']} | "
              f"FN={s['FN']} | accuracy={s['accuracy']:.4f}")

        # Sauvegarder CSVs par seed
        seed_dir = out_dir / f"seed{seed}"
        save_csv(rows, seed_dir / "all_predictions.csv")
        save_csv(rows, seed_dir / "only_errors.csv", only_errors=True)

        # Afficher les erreurs
        errors = [r for r in rows if r["error_type"] != "correct"]
        if errors:
            print(f"    Erreurs seed {seed}:")
            for e in errors:
                print(f"      {e['error_type']}: vidéo {e['video_id']} "
                      f"(label={'réel' if e['label']==0 else 'attaque'}, "
                      f"score={e['score']:.4f})")

    # Sauvegarder résumé global
    global_summary = {
        "model": "deep_behav_no_pts consecutive",
        "threshold": threshold_used,
        "seeds": seeds,
        "per_seed": [
            {f"seed_{s}": sm} for s, sm in zip(seeds, summaries)
        ],
        "mean_errors": round(
            sum(sm["FP"] + sm["FN"] for sm in summaries) / len(summaries), 2
        ),
    }
    with (out_dir / "global_summary.json").open("w") as f:
        json.dump(global_summary, f, indent=2)
    print(f"\n[OK] Résumé global: {out_dir}/global_summary.json")

    # Générer les figures
    print("\nGénération des figures...")
    fig_fp_fn_per_seed(summaries, seeds,
                        out_dir / "fig1_fp_fn_per_seed.png")
    fig_hard_videos(all_rows, seeds, threshold_used,
                    out_dir / "fig2_hard_videos.png")
    fig_score_distribution(all_rows, seeds, threshold_used,
                            out_dir / "fig3_score_distribution.png")
    fig_score_means(all_rows, seeds,
                    out_dir / "fig4_score_means.png")

    print(f"\n✅ Analyse complète sauvegardée dans : {out_dir}")
    print("   fig1_fp_fn_per_seed.png    — FP/FN par seed")
    print("   fig2_hard_videos.png       — vidéos difficiles avec scores")
    print("   fig3_score_distribution.png — histogramme scores")
    print("   fig4_score_means.png       — score moyen réel vs attaque")


if __name__ == "__main__":
    main()