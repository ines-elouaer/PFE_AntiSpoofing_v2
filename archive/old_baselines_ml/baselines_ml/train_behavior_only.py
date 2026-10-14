"""
Behavior-Only Baseline
=======================
Entraîne LogReg + MLP sur le vecteur comportemental dim=9.
Produit : F1, ACER, AUC + feature importance classée.

Usage depuis la racine du projet :
    python -m src.baselines_ml.train_behavior_only

Ou avec chemins explicites :
    python -m src.baselines_ml.train_behavior_only ^
        --train_csv data/processed/casia/behav/train_behav_norm.csv ^
        --val_csv   data/processed/casia/behav/val_behav_norm.csv ^
        --test_csv  data/processed/casia/behav/test_behav_norm.csv ^
        --out_dir   reports/baselines/behavior_only ^
        --seeds 42 43 44
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)

FEAT_COLS = [
    "ear_mean", "ear_std", "ear_min", "ear_max", "blink_count",
    "motion_mean", "motion_std", "motion_max", "skipped_rate",
]


def compute_pad_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_attack = tp + fn
    n_bona   = tn + fp
    apcer = fn / n_attack if n_attack > 0 else 0.0
    bpcer = fp / n_bona   if n_bona   > 0 else 0.0
    acer  = 0.5 * (apcer + bpcer)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return {
        "ACC":   float(accuracy_score(y_true, y_pred)),
        "F1":    float(f1_score(y_true, y_pred, zero_division=0)),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER":  float(acer),
        "AUC":   auc,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def feature_importance(model):
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    elif hasattr(model, "coefs_"):
        imp = np.abs(model.coefs_[0]).mean(axis=1)
    else:
        return {c: 0.0 for c in FEAT_COLS}
    total = imp.sum() if imp.sum() > 0 else 1.0
    return {FEAT_COLS[i]: float(imp[i] / total) for i in range(len(FEAT_COLS))}


def run_one_seed(X_train, y_train, X_test, y_test, seed):
    lr = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    prob_lr = lr.predict_proba(X_test)[:, 1]

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    mlp.fit(X_train, y_train)
    pred_mlp = mlp.predict(X_test)
    prob_mlp = mlp.predict_proba(X_test)[:, 1]

    return {
        "logreg": {**compute_pad_metrics(y_test, pred_lr, prob_lr),
                   "feature_importance": feature_importance(lr)},
        "mlp":    {**compute_pad_metrics(y_test, pred_mlp, prob_mlp),
                   "feature_importance": feature_importance(mlp)},
    }


def aggregate(all_results, seeds):
    model_names  = list(all_results[0].keys())
    metric_names = ["ACC", "F1", "AUC", "APCER", "BPCER", "ACER"]
    summary = {}
    for model in model_names:
        summary[model] = {}
        for m in metric_names:
            vals = [all_results[s][model][m] for s in range(len(seeds))]
            summary[model][m] = {
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "values": [round(v, 6) for v in vals],
            }
        fi_keys = list(all_results[0][model]["feature_importance"].keys())
        summary[model]["feature_importance"] = {
            feat: float(np.mean([all_results[s][model]["feature_importance"].get(feat, 0.0)
                                 for s in range(len(seeds))]))
            for feat in fi_keys
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Behavior-only PAD baseline")
    parser.add_argument("--train_csv",
        default=r"data\processed\casia\behav\train_behav_norm.csv")
    parser.add_argument("--val_csv",
        default=r"data\processed\casia\behav\val_behav_norm.csv")
    parser.add_argument("--test_csv",
        default=r"data\processed\casia\behav\test_behav_norm.csv")
    parser.add_argument("--out_dir",
        default=r"reports\baselines\behavior_only")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)
    test_df  = pd.read_csv(args.test_csv)

    trainval = pd.concat([train_df, val_df], ignore_index=True)
    X_tv = trainval[FEAT_COLS].values.astype(np.float32)
    y_tv = trainval["label"].values.astype(int)
    X_te = test_df[FEAT_COLS].values.astype(np.float32)
    y_te = test_df["label"].values.astype(int)

    print(f"Train+Val : {len(y_tv)} videos")
    print(f"Test      : {len(y_te)} videos  "
          f"(Real={int((y_te==0).sum())} | Attack={int((y_te==1).sum())})")

    all_results = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        res = run_one_seed(X_tv, y_tv, X_te, y_te, seed)
        all_results.append(res)
        for model, m in res.items():
            print(f"  [{model:7s}] F1={m['F1']:.4f} | ACER={m['ACER']:.4f} | "
                  f"APCER={m['APCER']:.4f} | BPCER={m['BPCER']:.4f} | AUC={m['AUC']:.4f}")

    summary = aggregate(all_results, args.seeds)

    print("\n========== RÉSUMÉ (mean +/- std) ==========")
    for model, ms in summary.items():
        print(f"\n  [{model}]")
        for metric in ["ACC", "F1", "ACER", "APCER", "BPCER", "AUC"]:
            print(f"    {metric:6s}: {ms[metric]['mean']:.4f} +/- {ms[metric]['std']:.4f}")
        print(f"    Feature importance (classée) :")
        fi = ms["feature_importance"]
        for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
            bar = "#" * int(imp * 30)
            print(f"      {feat:15s} {bar:30s} {imp:.4f}")

    out = {
        "seeds": args.seeds,
        "feat_cols": FEAT_COLS,
        "n_test": int(len(y_te)),
        "n_trainval": int(len(y_tv)),
        "summary": summary,
    }
    out_path = os.path.join(args.out_dir, "behavior_only_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Sauvegarde : {out_path}")

    print("\n========== COMPARAISON ABLATION ==========")
    print(f"  {'Modele':40s} {'F1':>7} {'ACER':>7} {'AUC':>7}")
    print(f"  {'-'*56}")
    deep_ref = [
        ("CNN seul",                     1.0000, 0.0000, 1.0000),
        ("CNN+LSTM (no PTS)",             0.9449, 0.1296, 0.9660),
        ("Deep+Behav (no PTS) valopt",    0.9554, 0.1204, 0.9877),
        ("Deep+Behav consecutive [best]", 0.9639, 0.0926, 0.9877),
    ]
    for name, f1, acer, auc in deep_ref:
        print(f"  {name:40s} {f1:7.4f} {acer:7.4f} {auc:7.4f}")
    print(f"  {'-'*56}")
    for model, ms in summary.items():
        name = f"Behavior-only ({model})"
        print(f"  {name:40s} {ms['F1']['mean']:7.4f} "
              f"{ms['ACER']['mean']:7.4f} {ms['AUC']['mean']:7.4f}")


if __name__ == "__main__":
    main()
