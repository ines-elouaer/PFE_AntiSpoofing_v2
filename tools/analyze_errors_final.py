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
  5. [NOUVEAU] Seuil optimal par seed (minimise ACER)
  6. [NOUVEAU] Inspection des vidéos difficiles récurrentes
  7. [CORRECTION] fig5 utilise threshold_compare_summary.json pour
     montrer la vraie comparaison fixed05 vs valopt avec th extrêmes

Usage depuis E:\PFE_AntiSpoofing_v2 :

    python tools/analyze_errors_final.py

Ou avec chemins personnalisés :
    python tools/analyze_errors_final.py ^
        --seed42 reports/eval/step2/deep_behav_no_pts/valopt/deep_behav_no_pts_consecutive_seed42/test_scores.json ^
        --seed43 reports/eval/step2/deep_behav_no_pts/valopt/deep_behav_no_pts_consecutive_seed43/test_scores.json ^
        --seed44 reports/eval/step2/deep_behav_no_pts/valopt/deep_behav_no_pts_consecutive_seed44/test_scores.json ^
        --threshold_compare reports/eval/step1/deep_behav_no_pts/deep_behav_no_pts_threshold_compare_summary.json ^
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


# ─── Palettes ────────────────────────────────────────────────────────

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

C = DARK


# ─── Figures ─────────────────────────────────────────────────────────

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

    avg_err = sum(s["FP"] + s["FN"] for s in summaries) / len(summaries)
    ax.text(0.5, 0.92,
            f"Moyenne : {avg_err:.1f} erreur(s) / {summaries[0]['total']} vidéos",
            transform=ax.transAxes, ha="center", color=C["text"],
            fontsize=9, style="italic")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  [OK] {out_path}")


def fig_hard_videos(all_rows_by_seed, seeds, threshold, out_path: Path):
    """Barplot des vidéos mal classées avec leurs scores exacts."""
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
    """Score moyen réel vs attaque par seed."""
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


# ─── [NOUVEAU] Seuil optimal ─────────────────────────────────────────

def find_optimal_threshold(rows):
    """
    Cherche le seuil qui minimise l'ACER sur les données fournies.
    Retourne (meilleur_seuil, meilleur_acer, apcer, bpcer)

    ACER  = (APCER + BPCER) / 2
    APCER = FP / total réels    (Attack Presentation Classification Error Rate)
    BPCER = FN / total attaques (Bona-fide Presentation Classification Error Rate)

    IMPORTANT : utiliser sur val set en production, pas sur test set.

    CORRECTION v3 :
    - Plage restreinte à [0.20, 0.80] pour éviter les seuils extrêmes
      qui classifient tout dans une seule classe (faux ACER=0)
    - Vérification que le seuil optimal préserve les deux classes
    """
    y_true  = np.array([r["label"] for r in rows])
    y_score = np.array([r["score"] for r in rows])

    best_th    = 0.5
    best_acer  = float("inf")
    best_apcer = 0.0
    best_bpcer = 0.0

    real_mask   = y_true == 0
    attack_mask = y_true == 1

    if real_mask.sum() == 0 or attack_mask.sum() == 0:
        print("  [WARN] Impossible de calculer ACER — une classe est absente.")
        return 0.5, 1.0, 1.0, 1.0

    # CORRECTION : plage 0.20→0.80 uniquement
    # Les seuils extrêmes (<0.20 ou >0.80) sur un petit test set (24 vidéos)
    # produisent ACER=0 en classifiant tout dans une classe — c'est du
    # surapprentissage sur le seuil, pas un vrai gain de performance.
    for th in np.arange(0.20, 0.81, 0.01):
        y_pred = (y_score >= th).astype(int)

        # CORRECTION : vérifier que les deux classes restent représentées
        # après application du seuil (au moins 1 prédiction de chaque classe)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue  # seuil dégénéré — tout dans une classe

        apcer = np.sum(y_pred[real_mask] == 1)   / real_mask.sum()
        bpcer = np.sum(y_pred[attack_mask] == 0) / attack_mask.sum()
        acer  = (apcer + bpcer) / 2.0

        if acer < best_acer:
            best_acer  = acer
            best_th    = th
            best_apcer = apcer
            best_bpcer = bpcer

    return (
        round(float(best_th),    2),
        round(float(best_acer),  4),
        round(float(best_apcer), 4),
        round(float(best_bpcer), 4),
    )


def load_threshold_compare(json_path: Path):
    """
    Charge les vraies données de comparaison fixed05 vs valopt
    depuis threshold_compare_summary.json.

    CORRECTION : La fig5 précédente comparait consecutive th=0.5 vs
    consecutive th=0.5 (identiques) → trivial et sans intérêt.
    La vraie comparaison est fixed05 vs valopt sur le même modèle
    Deep+Behav, où valopt utilise des seuils extrêmes (0.23, 0.11)
    qui dégradent les performances sur le test set.
    """
    if not json_path.exists():
        print(f"  [WARN] threshold_compare introuvable : {json_path}")
        return None

    data = json.loads(json_path.read_text(encoding="utf-8"))

    seeds = [42, 43, 44]

    # Extraire les runs fixed05
    fixed_runs = {
        r["seed"]: r for r in data["runs"]
        if r["threshold_protocol"] == "fixed05"
    }
    # Extraire les runs valopt
    valopt_runs = {
        r["seed"]: r for r in data["runs"]
        if r["threshold_protocol"] == "valopt"
    }

    results = []
    for s in seeds:
        f = fixed_runs.get(s, {})
        v = valopt_runs.get(s, {})
        results.append({
            "seed":        s,
            "th_fixed":    f.get("threshold_used_th", 0.5),
            "acer_fixed":  round(f.get("ACER", 0), 4),
            "th_valopt":   v.get("threshold_used_th", 0.5),
            "acer_valopt": round(v.get("ACER", 0), 4),
        })

    return results


def fig_threshold_comparison_real(compare_data: list, out_path: Path):
    """
    CORRECTION v5 : Utilise les vraies données de threshold_compare_summary.json
    pour montrer que valopt avec des seuils extrêmes (0.23, 0.11) dégrade
    les performances sur le test set, justifiant le choix de fixed05.

    Données réelles :
    Seed 42 : fixed=0.5 ACER=11.11%  vs  valopt=0.50 ACER=11.11% (identiques)
    Seed 43 : fixed=0.5 ACER=16.67%  vs  valopt=0.23 ACER=25.00% ← dégradation
    Seed 44 : fixed=0.5 ACER=16.67%  vs  valopt=0.11 ACER=16.67% (identiques)
    """
    seeds      = [d["seed"]        for d in compare_data]
    acer_fixed = [d["acer_fixed"]  for d in compare_data]
    acer_val   = [d["acer_valopt"] for d in compare_data]
    th_val     = [d["th_valopt"]   for d in compare_data]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["ax"])

    x = np.arange(len(seeds))
    w = 0.35

    b1 = ax.bar(x - w/2, acer_fixed, w,
                color=C["green"], alpha=0.85,
                label="fixed05 — th=0.5 fixe (stable)")
    b2 = ax.bar(x + w/2, acer_val,   w,
                color=C["red"], alpha=0.85,
                label="valopt  — th optimisé sur val set (instable)")

    # Valeurs sur les barres
    for bars, vals in [(b1, acer_fixed), (b2, acer_val)]:
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.004,
                f"{v*100:.2f}%",
                ha="center", color=C["white"],
                fontsize=9, fontweight="bold"
            )

    # Annotations seuil valopt + dégradation
    for i, (th, v_val, v_fix) in enumerate(zip(th_val, acer_val, acer_fixed)):
        bar_x  = x[i] + w/2
        bar_top = v_val
        degradation = v_val - v_fix

        if abs(degradation) > 0.001:
            # Dégradation réelle → afficher avec flèche rouge
            direction = "↑" if degradation > 0 else "↓"
            color_ann = C["yellow"] if degradation > 0 else C["green"]
            offset = max(0.025, max(acer_val) * 0.18)
            ax.annotate(
                f"th={th:.2f}\n+{degradation*100:.1f}% {direction}",
                xy=(bar_x, bar_top),
                xytext=(bar_x, bar_top + offset),
                ha="center", color=color_ann,
                fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color_ann, lw=1.2),
            )
        else:
            # Identiques → afficher le seuil sans alarme
            ax.text(
                bar_x, bar_top + 0.012,
                f"th={th:.2f}\n≡ fixe",
                ha="center", color=C["green"],
                fontsize=8, style="italic"
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds],
                       color=C["text"], fontsize=11)
    ax.set_ylabel("ACER ↓  (meilleur = bas)", color=C["text"], fontsize=11)
    ax.set_ylim(0, max(max(acer_fixed), max(acer_val)) + 0.12)
    ax.set_title(
        "Choix du Threshold — fixed05 vs valopt\n"
        "Pourquoi th=0.5 fixe est plus robuste que le threshold optimisé sur val",
        color=C["white"], fontsize=12,
    )
    ax.tick_params(colors=C["text"])
    for sp in ax.spines.values(): sp.set_color(C["grid"])
    ax.legend(
        framealpha=0.3, facecolor=C["legend"],
        edgecolor=C["grid"], labelcolor=C["text"], fontsize=9,
    )

    # Stats résumé en bas
    mean_fixed = float(np.mean(acer_fixed))
    std_fixed  = float(np.std(acer_fixed))
    mean_val   = float(np.mean(acer_val))
    std_val    = float(np.std(acer_val))

    ax.text(
        0.5, -0.13,
        f"fixed05 : ACER moyen = {mean_fixed*100:.2f}% ± {std_fixed*100:.1f}%   "
        f"|   valopt : ACER moyen = {mean_val*100:.2f}% ± {std_val*100:.1f}%   "
        f"→  fixed05 est {std_val/std_fixed:.1f}× plus stable",
        transform=ax.transAxes, ha="center",
        color=C["text"], fontsize=8.5, style="italic"
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  [OK] {out_path}")


# ─── [NOUVEAU] Inspection vidéos difficiles ──────────────────────────

def inspect_hard_video(video_id, all_rows_by_seed, seeds):
    """
    Affiche les stats complètes d'une vidéo sur tous les seeds.
    Permet d'identifier si l'erreur est systématique (problème de données)
    ou aléatoire (instabilité du modèle).
    """
    print(f"\n{'='*55}")
    print(f"  INSPECTION VIDÉO : {video_id}")
    print(f"{'='*55}")

    found      = False
    n_errors   = 0
    scores_all = []

    for seed, rows in zip(seeds, all_rows_by_seed):
        for r in rows:
            if r["video_id"] == video_id:
                found = True
                scores_all.append(r["score"])
                is_error = r["error_type"] != "correct"
                if is_error:
                    n_errors += 1
                status = "❌ ERREUR" if is_error else "✅ correct"
                print(
                    f"  Seed {seed} | "
                    f"label={'réel    ' if r['label']==0 else 'attaque '} | "
                    f"score={r['score']:.4f} | "
                    f"pred={'attaque' if r['prediction']==1 else 'réel   '} | "
                    f"{status}  [{r['error_type']}]"
                )

    if not found:
        print(f"  ⚠️  Vidéo '{video_id}' absente du set de test.")
    else:
        print(f"\n  Résumé : {n_errors}/{len(seeds)} seeds → erreur")
        print(f"  Scores : min={min(scores_all):.4f} | "
              f"max={max(scores_all):.4f} | "
              f"mean={np.mean(scores_all):.4f}")

        # Diagnostic automatique
        if n_errors == len(seeds):
            print("  🔴 Diagnostic : erreur SYSTÉMATIQUE sur tous les seeds")
            print("     → Vidéo structurellement ambiguë (données ou label)")
        elif n_errors >= 2:
            print("  🟡 Diagnostic : erreur FRÉQUENTE — cas difficile confirmé")
            print("     → Vérifier manuellement les frames de cette vidéo")
        else:
            print("  🟢 Diagnostic : erreur isolée — probablement instabilité seed")

    print(f"{'='*55}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse d'erreurs — Deep+Behav Consecutive"
    )
    parser.add_argument("--seed42",
        default=r"reports\eval\step2\deep_behav_no_pts\valopt\deep_behav_no_pts_consecutive_seed42\test_scores.json")
    parser.add_argument("--seed43",
        default=r"reports\eval\step2\deep_behav_no_pts\valopt\deep_behav_no_pts_consecutive_seed43\test_scores.json")
    parser.add_argument("--seed44",
        default=r"reports\eval\step2\deep_behav_no_pts\valopt\deep_behav_no_pts_consecutive_seed44\test_scores.json")
    parser.add_argument("--threshold_compare",
        default=r"reports\eval\step1\deep_behav_no_pts\deep_behav_no_pts_threshold_compare_summary.json",
        help="JSON de comparaison fixed05 vs valopt pour fig5")
    parser.add_argument("--out_dir",
        default=r"reports\errors\step2_consecutive_final")
    parser.add_argument("--light", action="store_true",
        help="Mode clair (fond blanc) pour mémoire et rapport")
    args = parser.parse_args()

    global C
    if args.light:
        C = LIGHT
        print("Mode : clair (fond blanc)")
    else:
        print("Mode : sombre (fond noir)")

    seeds      = [42, 43, 44]
    json_paths = [args.seed42, args.seed43, args.seed44]
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nChargement des données...")
    all_rows      = []
    summaries     = []
    threshold_used = 0.5

    # Listes pour la figure de comparaison de seuils
    results_th05 = []
    results_opt  = []

    for seed, jp in zip(seeds, json_paths):
        p = Path(jp)
        if not p.exists():
            print(f"  [WARN] Introuvable: {p}")
            all_rows.append([])
            summaries.append({"total": 0, "correct": 0, "FP": 0, "FN": 0, "accuracy": 0})
            results_th05.append({"acer_th05": 1.0})
            results_opt.append({"acer_opt": 1.0, "th_opt": 0.5})
            continue

        scores, threshold_used, model_name, protocol = load_scores(p)
        rows = build_rows(scores, threshold_used)
        s    = summarize(rows)
        all_rows.append(rows)
        summaries.append(s)

        print(f"\n  Seed {seed}: correct={s['correct']} | FP={s['FP']} | "
              f"FN={s['FN']} | accuracy={s['accuracy']:.4f}")

        # ── [NOUVEAU] Seuil optimal ───────────────────────────────────
        opt_th, opt_acer, opt_apcer, opt_bpcer = find_optimal_threshold(rows)

        # ACER à th=0.5
        real_mask   = np.array([r["label"] for r in rows]) == 0
        attack_mask = np.array([r["label"] for r in rows]) == 1
        y_pred_05   = np.array([r["prediction"] for r in rows])
        apcer_05 = np.sum(y_pred_05[real_mask] == 1)   / real_mask.sum()
        bpcer_05 = np.sum(y_pred_05[attack_mask] == 0) / attack_mask.sum()
        acer_05  = round(float((apcer_05 + bpcer_05) / 2), 4)

        print(f"    th=0.50  → ACER={acer_05:.4f}  "
              f"(APCER={apcer_05:.4f} | BPCER={bpcer_05:.4f})")
        print(f"    th={opt_th:.2f}   → ACER={opt_acer:.4f}  "
              f"(APCER={opt_apcer:.4f} | BPCER={opt_bpcer:.4f})  ← optimal")

        gain = round((acer_05 - opt_acer) * 100, 2)
        if gain > 0:
            print(f"    ✅ Gain ACER : -{gain}% en utilisant th={opt_th:.2f}")
        else:
            print(f"    ℹ️  th=0.5 déjà optimal pour ce seed")

        results_th05.append({"acer_th05": acer_05})
        results_opt.append({"acer_opt": opt_acer, "th_opt": opt_th})

        # Recalcule les erreurs avec le seuil optimal
        rows_opt = build_rows(scores, opt_th)
        s_opt    = summarize(rows_opt)
        print(f"    Avec th={opt_th:.2f} → FP={s_opt['FP']} | "
              f"FN={s_opt['FN']} | accuracy={s_opt['accuracy']:.4f}")
        # ─────────────────────────────────────────────────────────────

        # Sauvegarder CSVs par seed
        seed_dir = out_dir / f"seed{seed}"
        save_csv(rows, seed_dir / "all_predictions.csv")
        save_csv(rows, seed_dir / "only_errors.csv", only_errors=True)

        # Sauvegarder aussi les prédictions avec seuil optimal
        rows_opt_export = build_rows(scores, opt_th)
        save_csv(rows_opt_export,
                 seed_dir / f"predictions_th{str(opt_th).replace('.','')}.csv")

        # Afficher les erreurs à th=0.5
        errors = [r for r in rows if r["error_type"] != "correct"]
        if errors:
            print(f"    Erreurs seed {seed} (th=0.5):")
            for e in errors:
                print(f"      {e['error_type']}: vidéo {e['video_id']} "
                      f"(label={'réel' if e['label']==0 else 'attaque'}, "
                      f"score={e['score']:.4f})")

    # ── [NOUVEAU] Inspection des vidéos récurrentes ───────────────────
    print("\n" + "="*55)
    print("  INSPECTION DES VIDÉOS DIFFICILES RÉCURRENTES")
    print("="*55)
    hard_videos = ["10_1", "10_2", "13_6"]
    for vid in hard_videos:
        inspect_hard_video(vid, all_rows, seeds)

    # ── Sauvegarder résumé global ─────────────────────────────────────
    global_summary = {
        "model":     "deep_behav_no_pts consecutive",
        "threshold": threshold_used,
        "seeds":     seeds,
        "per_seed": [
            {
                f"seed_{s}": {
                    **sm,
                    "acer_th05": results_th05[i]["acer_th05"],
                    "acer_opt":  results_opt[i]["acer_opt"],
                    "th_opt":    results_opt[i]["th_opt"],
                }
            }
            for i, (s, sm) in enumerate(zip(seeds, summaries))
        ],
        "mean_errors": round(
            sum(sm["FP"] + sm["FN"] for sm in summaries) / len(summaries), 2
        ),
    }
    with (out_dir / "global_summary.json").open("w") as f:
        json.dump(global_summary, f, indent=2)
    print(f"\n[OK] Résumé global: {out_dir}/global_summary.json")

    # ── Générer les figures ───────────────────────────────────────────
    print("\nGénération des figures...")
    fig_fp_fn_per_seed(summaries, seeds,
                       out_dir / "fig1_fp_fn_per_seed.png")
    fig_hard_videos(all_rows, seeds, threshold_used,
                    out_dir / "fig2_hard_videos.png")
    fig_score_distribution(all_rows, seeds, threshold_used,
                           out_dir / "fig3_score_distribution.png")
    fig_score_means(all_rows, seeds,
                    out_dir / "fig4_score_means.png")

    # [CORRECTION v5] Figure comparaison seuils — utilise les vraies données
    # threshold_compare_summary.json (fixed05 vs valopt avec th extrêmes)
    compare_data = load_threshold_compare(Path(args.threshold_compare))
    if compare_data:
        fig_threshold_comparison_real(
            compare_data,
            out_dir / "fig5_threshold_comparison.png"
        )
        print("   fig5 : vraies données fixed05 vs valopt (th=0.23, th=0.11)")
    else:
        print("   [SKIP] fig5 — threshold_compare_summary.json introuvable")
        print(f"   Chemin attendu : {args.threshold_compare}")

    print(f"\n✅ Analyse complète sauvegardée dans : {out_dir}")
    print("   fig1_fp_fn_per_seed.png        — FP/FN par seed")
    print("   fig2_hard_videos.png           — vidéos difficiles avec scores")
    print("   fig3_score_distribution.png    — histogramme scores")
    print("   fig4_score_means.png           — score moyen réel vs attaque")
    print("   fig5_threshold_comparison.png  — [CORRIGÉ] fixed05 vs valopt réel")


if __name__ == "__main__":
    main()