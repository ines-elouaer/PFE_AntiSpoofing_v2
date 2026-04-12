"""
fix_confusion_matrix.py
========================
Vérifie et corrige les confusion_matrix.csv manquants.

Le vrai problème n'est pas que la clé est None —
c'est que le fichier CSV pointé n'existe pas physiquement
(chemins Windows/Linux incorrects après déplacement de dossiers).

Ce script :
  1. Parcourt tous les test_scores.json
  2. Vérifie si le fichier confusion_matrix.csv existe réellement
  3. Si absent : le recalcule depuis test_scores et le sauvegarde
  4. Met à jour le chemin dans le JSON

Usage :
    python tools/fix_confusion_matrix.py --root reports/eval
"""

import os
import csv
import json
import argparse


def confusion_from_scores(test_scores, th):
    tn = fp = fn = tp = 0
    for d in test_scores.values():
        y    = int(d["label"])
        pred = 1 if float(d["score"]) >= th else 0
        if   y == 0 and pred == 0: tn += 1
        elif y == 0 and pred == 1: fp += 1
        elif y == 1 and pred == 0: fn += 1
        else:                       tp += 1
    return tn, fp, fn, tp


def save_cm_csv(path, tn, fp, fn, tp):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["",            "Pred_Real", "Pred_Attack"])
        w.writerow(["True_Real",   tn,          fp])
        w.writerow(["True_Attack", fn,          tp])


def get_threshold(data):
    """Récupère le seuil utilisé depuis le JSON."""
    th_info = data.get("threshold_used", {})
    th = th_info.get("th", None)
    if th is None:
        th = 0.5
    return float(th)


def check_cm_exists(data, dirpath):
    """
    Retourne True si le fichier confusion_matrix.csv existe réellement.
    Gère les chemins Windows/Linux/relatifs incorrects.
    """
    # Chercher dans les deux emplacements possibles
    artifacts = data.get("artifacts", {})
    cm_path   = artifacts.get("confusion_matrix_csv", None)

    # 1. Chemin exact dans le JSON
    if cm_path and os.path.exists(cm_path):
        return True

    # 2. confusion_matrix.csv dans le même dossier que test_scores.json
    local_cm = os.path.join(dirpath, "confusion_matrix.csv")
    if os.path.exists(local_cm):
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"reports\eval",
                        help="Racine où chercher les test_scores.json")
    parser.add_argument("--dry_run", action="store_true",
                        help="Afficher seulement sans corriger")
    args = parser.parse_args()

    fixed   = 0
    already = 0
    errors  = 0

    for dirpath, _, filenames in os.walk(args.root):
        for fname in filenames:
            if fname != "test_scores.json":
                continue

            json_path = os.path.join(dirpath, fname)

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[ERR] Lecture impossible : {json_path} — {e}")
                errors += 1
                continue

            # Vérifier si CM existe réellement
            if check_cm_exists(data, dirpath):
                already += 1
                continue

            # Recalculer depuis test_scores
            test_scores = data.get("test_scores", {})
            if not test_scores:
                print(f"[SKIP] Pas de test_scores : {json_path}")
                errors += 1
                continue

            th = get_threshold(data)
            tn, fp, fn, tp = confusion_from_scores(test_scores, th)

            cm_out = os.path.join(dirpath, "confusion_matrix.csv")

            if not args.dry_run:
                save_cm_csv(cm_out, tn, fp, fn, tp)

                # Mettre à jour le JSON
                if "artifacts" not in data:
                    data["artifacts"] = {}
                data["artifacts"]["confusion_matrix_csv"] = cm_out

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            print(f"[FIX] {dirpath}")
            print(f"      th={th} | TN={tn} FP={fp} FN={fn} TP={tp}")
            fixed += 1

    print(f"\n{'='*50}")
    print(f"Total scannés  : {fixed + already + errors}")
    print(f"Déjà OK        : {already}")
    print(f"Corrigés       : {fixed}")
    print(f"Erreurs        : {errors}")
    if args.dry_run:
        print("(dry_run=True — aucune modification effectuée)")


if __name__ == "__main__":
    main()