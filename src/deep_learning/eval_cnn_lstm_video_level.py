import os
import csv
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.deep_learning.datasets_sequence import CASIASequenceDataset
from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD
from src.deep_learning.metrics_pad import compute_apcer_bpcer_acer


@torch.no_grad()
def predict_video_scores(model, loader, device) -> Dict[str, Dict]:
    model.eval()
    softmax = nn.Softmax(dim=1)
    scores = {}

    for batch in tqdm(loader, desc="Predict videos"):
        if len(batch) == 3:
            x, y, vid = batch
            behav = None
        else:
            x, y, vid, behav = batch

        x = x.to(device)
        behav = None if behav is None else behav.to(device)

        logits = model(x, behav)
        probs = softmax(logits).cpu()

        for i in range(x.size(0)):
            v = vid[i]
            scores[v] = {
                "label": int(y[i].item()),
                "score": float(probs[i, 1].item())  # spoof / attack score
            }
    return scores


def confusion_at_threshold(video_scores: Dict[str, Dict], th: float) -> Tuple[int, int, int, int]:
    tn = fp = fn = tp = 0
    for d in video_scores.values():
        y = int(d["label"])
        pred = 1 if float(d["score"]) >= th else 0

        if y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1
        elif y == 1 and pred == 1:
            tp += 1

    return tn, fp, fn, tp


def f1_at_threshold(video_scores: Dict[str, Dict], th: float) -> float:
    tn, fp, fn, tp = confusion_at_threshold(video_scores, th)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def acc_at_threshold(video_scores: Dict[str, Dict], th: float) -> float:
    tn, fp, fn, tp = confusion_at_threshold(video_scores, th)
    total = tn + fp + fn + tp
    return (tn + tp) / total if total else 0.0


def find_best_threshold_on_val(
    video_scores: Dict[str, Dict],
    step: float = 0.01,
    prefer_closest_to: float = 0.5,
) -> Dict:
    candidates = []

    t = 0.0
    while t <= 1.000001:
        apcer, bpcer, acer = compute_apcer_bpcer_acer(video_scores, threshold=t)
        f1 = f1_at_threshold(video_scores, t)
        acc = acc_at_threshold(video_scores, t)

        candidates.append(
            {
                "th": round(t, 4),
                "acer": float(acer),
                "apcer": float(apcer),
                "bpcer": float(bpcer),
                "f1": float(f1),
                "acc": float(acc),
            }
        )
        t += step

    candidates = sorted(
        candidates,
        key=lambda d: (
            d["acer"],                     # min ACER
            -d["f1"],                     # max F1
            -d["acc"],                    # max ACC
            abs(d["th"] - prefer_closest_to),  # closest to 0.5
        ),
    )

    best = dict(candidates[0])
    same_main = [
        d for d in candidates
        if d["acer"] == best["acer"] and d["f1"] == best["f1"] and d["acc"] == best["acc"]
    ]
    best["n_tied_candidates"] = len(same_main)
    return best


def try_auc(video_scores: Dict[str, Dict]) -> Tuple[float, bool]:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return 0.0, False

    y_true = [d["label"] for d in video_scores.values()]
    y_score = [d["score"] for d in video_scores.values()]
    return float(roc_auc_score(y_true, y_score)), True


def score_means(video_scores: Dict[str, Dict]) -> Tuple[float, float]:
    attacks = [d["score"] for d in video_scores.values() if int(d["label"]) == 1]
    reals = [d["score"] for d in video_scores.values() if int(d["label"]) == 0]
    a_mean = sum(attacks) / len(attacks) if attacks else 0.0
    r_mean = sum(reals) / len(reals) if reals else 0.0
    return a_mean, r_mean


def infer_model_name_from_config(cfg: dict) -> str:
    use_behav = bool(cfg.get("use_behav", False))
    use_pts = bool(cfg.get("use_pts", False))

    if use_behav and use_pts:
        return "deep_behav_pts"
    if use_behav and not use_pts:
        return "deep_behav_no_pts"
    if use_pts and not use_behav:
        return "pts_cnn_lstm"
    return "cnn_lstm"


def save_confusion_csv(path: str, tn: int, fp: int, fn: int, tp: int):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "pred_real", "pred_attack"])
        w.writerow(["real", tn, fp])
        w.writerow(["attack", fn, tp])


def save_score_distribution_csv(path: str, video_scores: Dict[str, Dict]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label", "score"])
        for vid, d in sorted(video_scores.items()):
            w.writerow([vid, int(d["label"]), float(d["score"])])


def save_roc_points_csv(path: str, video_scores: Dict[str, Dict]) -> bool:
    try:
        from sklearn.metrics import roc_curve
    except Exception:
        return False

    y_true = [d["label"] for d in video_scores.values()]
    y_score = [d["score"] for d in video_scores.values()]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        for a, b, c in zip(fpr, tpr, thresholds):
            w.writerow([float(a), float(b), float(c)])
    return True


def banking_decision(score: float, accept_threshold: float, reject_threshold: float) -> str:
    """
    Banking logic:
      - low spoof score => ACCEPT
      - intermediate => RETRY
      - high spoof score => REJECT
    """
    if score < accept_threshold:
        return "ACCEPT"
    if score < reject_threshold:
        return "RETRY"
    return "REJECT"


def calibrate_banking_thresholds(val_scores: Dict[str, Dict],
                                  security_mode: str = "strict") -> Dict:
    """
    Calibre automatiquement les seuils accept/reject sur la validation.

    strict   : FN = 0 garanti (aucune attaque acceptée) — recommandé banking
    balanced : minimise ACER (équilibre sécurité / acceptation)

    Retourne dict avec accept_threshold, reject_threshold, fn_val, reasoning.
    """
    if security_mode == "strict":
        best_accept = 0.01
        for th in [i / 100 for i in range(1, 60)]:
            fn = sum(1 for d in val_scores.values()
                     if int(d["label"]) == 1 and float(d["score"]) < th)
            if fn == 0:
                best_accept = th
            else:
                break  # dès qu'un FN apparaît on s'arrête

        th_reject    = min(0.60, max(best_accept + 0.20, 0.50))
        fn_final     = sum(1 for d in val_scores.values()
                           if int(d["label"]) == 1 and float(d["score"]) < best_accept)
        fp_accept    = sum(1 for d in val_scores.values()
                           if int(d["label"]) == 0 and float(d["score"]) < best_accept)
        total_real   = sum(1 for d in val_scores.values() if int(d["label"]) == 0)

        return {
            "accept_threshold": round(best_accept, 2),
            "reject_threshold": round(th_reject,   2),
            "fn_val":           fn_final,
            "fp_accept_val":    fp_accept,
            "accept_rate_real": round(fp_accept / total_real, 4) if total_real else 0.0,
            "mode":             "strict (FN=0 garanti)",
            "reasoning": (
                f"Seuil accept={best_accept:.2f} garanti FN=0 sur VAL. "
                f"Seuil reject={th_reject:.2f} définit la zone RETRY."
            ),
        }

    else:  # balanced
        best_acer, best_th = 1.0, (0.30, 0.60)
        for th_a in [i / 100 for i in range(5, 45, 5)]:
            for th_r in [j / 100 for j in range(50, 80, 5)]:
                if th_r <= th_a:
                    continue
                _, _, acer = compute_apcer_bpcer_acer(val_scores, threshold=th_a)
                if acer < best_acer:
                    best_acer = acer
                    best_th   = (th_a, th_r)

        fn_final   = sum(1 for d in val_scores.values()
                         if int(d["label"]) == 1 and float(d["score"]) < best_th[0])
        fp_accept  = sum(1 for d in val_scores.values()
                         if int(d["label"]) == 0 and float(d["score"]) < best_th[0])
        total_real = sum(1 for d in val_scores.values() if int(d["label"]) == 0)

        return {
            "accept_threshold": best_th[0],
            "reject_threshold": best_th[1],
            "fn_val":           fn_final,
            "fp_accept_val":    fp_accept,
            "accept_rate_real": round(fp_accept / total_real, 4) if total_real else 0.0,
            "mode":             f"balanced (ACER_val={best_acer:.4f})",
            "reasoning": (
                f"Seuils optimisés sur ACER validation. "
                f"accept={best_th[0]:.2f}, reject={best_th[1]:.2f}."
            ),
        }


def analyze_banking_errors(decisions: Dict[str, Dict]) -> Dict:
    """
    Analyse les erreurs du point de vue sécurité bancaire.
      false_accept  : attaque acceptée   (CRITIQUE)
      false_reject  : réel rejeté        (gênant)
      missed_retry  : attaque en RETRY   (risque modéré)
    """
    out = {"false_accept": [], "false_reject": [], "missed_retry": [],
           "correct_accept": [], "correct_reject": []}
    for vid, d in decisions.items():
        label, dec, score = int(d["label"]), d["decision"], float(d["score"])
        if   label == 1 and dec == "ACCEPT":
            out["false_accept"].append({"video_id": vid, "score": score, "severity": "CRITIQUE"})
        elif label == 0 and dec == "REJECT":
            out["false_reject"].append({"video_id": vid, "score": score, "severity": "MODEREE"})
        elif label == 1 and dec == "RETRY":
            out["missed_retry"].append({"video_id": vid, "score": score, "severity": "FAIBLE"})
        elif label == 0 and dec == "ACCEPT":
            out["correct_accept"].append({"video_id": vid, "score": score})
        elif label == 1 and dec == "REJECT":
            out["correct_reject"].append({"video_id": vid, "score": score})
    return out


def build_banking_decisions(
    video_scores: Dict[str, Dict],
    accept_threshold: float,
    reject_threshold: float,
) -> Dict[str, Dict]:
    out = {}
    for vid, d in video_scores.items():
        score = float(d["score"])
        out[vid] = {
            "label": int(d["label"]),
            "score": score,
            "decision": banking_decision(score, accept_threshold, reject_threshold),
        }
    return out


def summarize_banking_decisions(decisions: Dict[str, Dict]) -> Dict[str, int]:
    summary = {"ACCEPT": 0, "RETRY": 0, "REJECT": 0}
    for d in decisions.values():
        summary[d["decision"]] += 1
    return summary


def save_banking_decisions_csv(path: str, decisions: Dict[str, Dict]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label", "score", "decision"])
        for vid, d in sorted(decisions.items()):
            w.writerow([vid, int(d["label"]), float(d["score"]), d["decision"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="Folder containing best_model.pth")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save evaluation outputs")

    parser.add_argument("--val_csv", default=r"data\processed\CASIA\splits_subject\val.csv")
    parser.add_argument("--test_csv", default=r"data\processed\CASIA\splits_subject\test.csv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--behav_val_csv", default=r"data\processed\CASIA\behav\val_behav.csv")
    parser.add_argument("--behav_test_csv", default=r"data\processed\CASIA\behav\test_behav.csv")

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=["cnn_lstm", "pts_cnn_lstm", "deep_behav", "deep_behav_no_pts", "deep_behav_pts"],
        help="Canonical model name. If omitted, inferred from checkpoint config."
    )

    parser.add_argument(
        "--threshold_protocol",
        type=str,
        default="fixed05",
        choices=["valopt", "fixed05", "banking"],
        help=(
            "Threshold protocol: "
            "valopt = threshold selected on VAL; "
            "fixed05 = fixed threshold 0.5 for standardized comparison; "
            "banking = calibrated thresholds for app decision logic."
        )
    )

    parser.add_argument(
        "--accept_threshold",
        type=float,
        default=None,
        help="Seuil accept manuel. Si None avec --auto_calibrate => calibration auto sur VAL."
    )
    parser.add_argument(
        "--reject_threshold",
        type=float,
        default=None,
        help="Seuil reject manuel. Si None avec --auto_calibrate => calibration auto sur VAL."
    )
    parser.add_argument(
        "--auto_calibrate",
        action="store_true",
        default=False,
        help="Calibre automatiquement les seuils banking sur VAL (recommandé)."
    )
    parser.add_argument(
        "--security_mode",
        type=str,
        default="strict",
        choices=["strict", "balanced"],
        help=(
            "strict   = FN=0 garanti (aucune attaque acceptée) — recommandé\n"
            "balanced = minimise ACER (équilibre sécurité/acceptation)"
        )
    )

    parser.add_argument(
        "--val_sample_mode",
        type=str,
        default="uniform",
        choices=["uniform", "random_clip", "consecutive", "center_consecutive"],
    )
    parser.add_argument(
        "--test_sample_mode",
        type=str,
        default="uniform",
        choices=["uniform", "random_clip", "consecutive", "center_consecutive"],
    )

    args = parser.parse_args()

    # Résoudre les seuils banking (auto ou manuels)
    banking_calib_info = None
    if args.threshold_protocol == "banking":
        if args.auto_calibrate or (args.accept_threshold is None or args.reject_threshold is None):
            # Calibration auto — nécessite les prédictions VAL en premier
            # On marque pour calibrer après predict_video_scores(val)
            _banking_auto_calibrate = True
            _banking_accept = None
            _banking_reject = None
        else:
            _banking_auto_calibrate = False
            _banking_accept = args.accept_threshold
            _banking_reject = args.reject_threshold
            if not (0.0 <= _banking_accept < _banking_reject <= 1.0):
                raise ValueError(
                    "Seuils invalides : 0 <= accept_threshold < reject_threshold <= 1"
                )
    else:
        _banking_auto_calibrate = False
        _banking_accept = args.accept_threshold if args.accept_threshold else 0.30
        _banking_reject = args.reject_threshold if args.reject_threshold else 0.60

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(args.exp_dir, "best_model.pth")

    out_dir = args.out_dir if args.out_dir is not None else args.exp_dir
    os.makedirs(out_dir, exist_ok=True)

    results_txt_path = os.path.join(out_dir, "results.txt")
    scores_json_path = os.path.join(out_dir, "test_scores.json")
    confusion_csv_path = os.path.join(out_dir, "confusion_matrix.csv")
    score_dist_csv_path = os.path.join(out_dir, "score_distribution.csv")
    roc_points_csv_path = os.path.join(out_dir, "roc_points.csv")
    banking_csv_path = os.path.join(out_dir, "banking_decisions.csv")

    ckpt = torch.load(model_path, map_location=device)

# --------------------------------------------------
# 1) récupérer config si elle existe
# --------------------------------------------------
    if isinstance(ckpt, dict) and "config" in ckpt:
      cfg = ckpt.get("config", {})
    else:
      cfg = {}

    use_behav = bool(cfg.get("use_behav", True))
    behav_dim = int(cfg.get("behav_dim", 9))
    behav_hidden = int(cfg.get("behav_hidden", 16))

    model = CNN_LSTM_PAD(
      hidden=cfg.get("hidden", 256),
      num_layers=cfg.get("num_layers", 1),
      bidir=cfg.get("bidir", False),
      temporal_pool=cfg.get("temporal_pool", "median"),
      pretrained_backbone=True,
      use_behav=use_behav,
      behav_dim=behav_dim,
      behav_hidden=behav_hidden,
      ).to(device)

# --------------------------------------------------
# 2) récupérer state_dict selon le format checkpoint
# --------------------------------------------------
    if isinstance(ckpt, dict) and "model_state" in ckpt:
      state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
       state_dict = ckpt["model_state_dict"]
    else:
    # checkpoint sauvegardé directement comme state_dict
      state_dict = ckpt

# gérer DataParallel éventuel
    if isinstance(state_dict, dict) and all(k.startswith("module.") for k in state_dict.keys()):
      state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    T = cfg.get("T", 16)
    img_size = cfg.get("img_size", 224)
    seed = cfg.get("seed", 42)

    model_name = args.model_name if args.model_name is not None else infer_model_name_from_config(cfg)
    protocol_name = f"{model_name}_{args.threshold_protocol}"

    val_ds = CASIASequenceDataset(
        args.val_csv,
        T=T,
        img_size=img_size,
        aug_mode="none",
        sample_mode=args.val_sample_mode,
        seed=seed,
        behav_csv=(args.behav_val_csv if use_behav else None),
    )

    test_ds = CASIASequenceDataset(
        args.test_csv,
        T=T,
        img_size=img_size,
        aug_mode="none",
        sample_mode=args.test_sample_mode,
        seed=seed,
        behav_csv=(args.behav_test_csv if use_behav else None),
    )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    val_scores = predict_video_scores(model, val_loader, device)
    best_val   = find_best_threshold_on_val(val_scores, step=0.01)

    # Calibration auto des seuils banking sur VAL
    if args.threshold_protocol == "banking" and _banking_auto_calibrate:
        banking_calib_info = calibrate_banking_thresholds(val_scores, args.security_mode)
        _banking_accept    = banking_calib_info["accept_threshold"]
        _banking_reject    = banking_calib_info["reject_threshold"]
        print(f"[AUTO-CALIBRATION banking/{args.security_mode}]")
        print(f"  accept_threshold = {_banking_accept:.2f}")
        print(f"  reject_threshold = {_banking_reject:.2f}")
        print(f"  FN sur VAL       = {banking_calib_info['fn_val']}")
        print(f"  {banking_calib_info['reasoning']}")

    if args.threshold_protocol == "fixed05":
        th_used: Optional[float] = 0.5
        th_source = "fixed05"
    elif args.threshold_protocol == "valopt":
        th_used = float(best_val["th"])
        th_source = "valopt"
    else:
        th_used = None
        th_source = "banking"

    test_scores = predict_video_scores(model, test_loader, device)

    if args.threshold_protocol in ["fixed05", "valopt"]:
        apcer, bpcer, acer = compute_apcer_bpcer_acer(test_scores, threshold=th_used)
        f1 = f1_at_threshold(test_scores, th_used)
        acc = acc_at_threshold(test_scores, th_used)
        tn, fp, fn, tp = confusion_at_threshold(test_scores, th_used)
    else:
        apcer = bpcer = acer = None
        f1 = acc = None
        tn = fp = fn = tp = None

    auc, ok_auc = try_auc(test_scores)
    a_mean, r_mean = score_means(test_scores)

    save_score_distribution_csv(score_dist_csv_path, test_scores)
    roc_ok = save_roc_points_csv(roc_points_csv_path, test_scores)

    if args.threshold_protocol in ["fixed05", "valopt"]:
        save_confusion_csv(confusion_csv_path, tn, fp, fn, tp)

    banking_decisions = None
    banking_summary   = None
    banking_errors    = None
    if args.threshold_protocol == "banking":
        banking_decisions = build_banking_decisions(
            test_scores,
            accept_threshold=_banking_accept,
            reject_threshold=_banking_reject,
        )
        banking_summary = summarize_banking_decisions(banking_decisions)
        banking_errors  = analyze_banking_errors(banking_decisions)
        save_banking_decisions_csv(banking_csv_path, banking_decisions)

    sweep_txt = None
    if args.threshold_protocol in ["fixed05", "valopt"]:
        sweep_lines = ["th\tACC\tAPCER\tBPCER\tACER\tF1"]
        for k in range(10, 100, 10):
            th = k / 100
            ap, bp, ac = compute_apcer_bpcer_acer(test_scores, threshold=th)
            sweep_lines.append(
                f"{th:.1f}\t{acc_at_threshold(test_scores, th):.4f}\t{ap:.4f}\t{bp:.4f}\t{ac:.4f}\t{f1_at_threshold(test_scores, th):.4f}"
            )
        sweep_txt = "\n".join(sweep_lines)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = []
    report.append("========================================")
    report.append("CASIA CNN+LSTM Evaluation (Video-Level)")
    report.append("========================================")
    report.append(f"Date: {now}")
    report.append(f"Device: {device}")
    report.append(f"Model name: {model_name}")
    report.append(f"Protocol name: {protocol_name}")
    report.append(f"Threshold protocol: {args.threshold_protocol}")
    report.append(f"Fusion behav: {use_behav} (behav_dim={behav_dim})")
    report.append(f"Model: MobileNetV3-Large + LSTM (T={T}, pool={cfg.get('temporal_pool', 'mean')})")
    report.append(f"VAL sample_mode: {args.val_sample_mode}")
    report.append(f"TEST sample_mode: {args.test_sample_mode}")
    report.append(f"Threshold (best on VAL): {best_val['th']:.4f}")

    if th_used is not None:
        report.append(f"Threshold (USED on TEST): {th_used:.4f}")
    else:
        report.append("Threshold (USED on TEST): N/A (banking decision mode)")

    report.append(f"Threshold source: {th_source}")
    report.append("")

    if args.threshold_protocol in ["fixed05", "valopt"]:
        report.append(f"Video Accuracy: {acc:.4f}")
        report.append(f"F1: {f1:.4f}")
        report.append(f"APCER: {apcer:.4f}")
        report.append(f"BPCER: {bpcer:.4f}")
        report.append(f"ACER: {acer:.4f}")
        report.append(f"ROC-AUC: {auc:.4f}" if ok_auc else "ROC-AUC: N/A (sklearn not installed)")
        report.append("")
        report.append("--- Confusion Matrix ---")
        report.append(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    else:
        report.append(f"ROC-AUC: {auc:.4f}" if ok_auc else "ROC-AUC: N/A (sklearn not installed)")
        report.append("")
        report.append("--- Calibration des Seuils Banking ---")
        if banking_calib_info:
            report.append(f"Mode          : {banking_calib_info['mode']}")
            report.append(f"Raisonnement  : {banking_calib_info['reasoning']}")
            report.append(f"FN sur VAL    : {banking_calib_info['fn_val']} "
                          f"({'OK' if banking_calib_info['fn_val']==0 else 'ATTENTION'})")
        else:
            report.append(f"Mode          : manuel")
        report.append("")
        report.append("--- Banking Decision Policy ---")
        report.append(f"ACCEPT if score < {_banking_accept:.4f}")
        report.append(f"RETRY  if {_banking_accept:.4f} <= score < {_banking_reject:.4f}")
        report.append(f"REJECT if score >= {_banking_reject:.4f}")
        report.append("")
        report.append("--- Banking Decision Summary ---")
        report.append(
            f"ACCEPT={banking_summary['ACCEPT']}  "
            f"RETRY={banking_summary['RETRY']}  "
            f"REJECT={banking_summary['REJECT']}"
        )
        report.append("")
        report.append("--- Analyse Erreurs Bancaires ---")
        fa = banking_errors.get("false_accept", [])
        fr = banking_errors.get("false_reject", [])
        mr = banking_errors.get("missed_retry", [])
        report.append(f"[CRITIQUE] False Accept (attaque->ACCEPT) : {len(fa)}")
        for e in fa:
            report.append(f"    FAILLE: vidéo {e['video_id']} score={e['score']:.4f}")
        if not fa:
            report.append("    OK — aucune attaque acceptée")
        report.append(f"[MODERE]   False Reject (réel->REJECT)    : {len(fr)}")
        for e in fr:
            report.append(f"    !  vidéo {e['video_id']} score={e['score']:.4f}")
        report.append(f"[FAIBLE]   Missed RETRY (attaque->RETRY)  : {len(mr)}")
        for e in mr:
            report.append(f"    ~  vidéo {e['video_id']} score={e['score']:.4f}")
        report.append("")
        report.append(f"Sécurité : {'OK — aucune attaque acceptée' if len(fa)==0 else 'ECHEC — attaques acceptées!'}")

    if sweep_txt is not None:
        report.append("")
        report.append("--- Threshold Sweep ---")
        report.append(sweep_txt)

    report.append("")
    report.append("--- Score Separation ---")
    report.append(f"Attack mean: {a_mean:.4f}")
    report.append(f"Real mean: {r_mean:.4f}")
    report.append("")
    report.append("--- Exported Artifacts ---")

    if args.threshold_protocol in ["fixed05", "valopt"]:
        report.append(f"confusion_matrix.csv: {confusion_csv_path}")
    report.append(f"score_distribution.csv: {score_dist_csv_path}")
    report.append(f"roc_points.csv: {roc_points_csv_path if roc_ok else 'not generated (sklearn missing)'}")
    if args.threshold_protocol == "banking":
        report.append(f"banking_decisions.csv: {banking_csv_path}")

    report_txt = "\n".join(report)
    print(report_txt)

    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    with open(scores_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "protocol_name": protocol_name,
                "threshold_protocol": args.threshold_protocol,
                "threshold_val": best_val,
                "threshold_used": {
                    "th": th_used,
                    "source": th_source
                },
                "banking_policy": (
                    {
                        "accept_threshold":  _banking_accept,
                        "reject_threshold":  _banking_reject,
                        "security_mode":     args.security_mode,
                        "auto_calibrated":   _banking_auto_calibrate,
                        "calibration_info":  banking_calib_info,
                        "decision_summary":  banking_summary,
                        "error_analysis":    banking_errors,
                    }
                    if args.threshold_protocol == "banking" else None
                ),
                "metrics_test": (
                    {
                        "ACC": acc,
                        "F1": f1,
                        "APCER": apcer,
                        "BPCER": bpcer,
                        "ACER": acer,
                        "AUC": auc if ok_auc else None,
                        "TN": tn,
                        "FP": fp,
                        "FN": fn,
                        "TP": tp
                    }
                    if args.threshold_protocol in ["fixed05", "valopt"] else
                    {
                        "AUC": auc if ok_auc else None
                    }
                ),
                "score_stats": {
                    "attack_mean": a_mean,
                    "real_mean": r_mean
                },
                "sampling": {
                    "val_sample_mode": args.val_sample_mode,
                    "test_sample_mode": args.test_sample_mode,
                    "T": T
                },
                "artifacts": {
                    "confusion_matrix_csv": (
                        confusion_csv_path if args.threshold_protocol in ["fixed05", "valopt"] else None
                    ),
                    "score_distribution_csv": score_dist_csv_path,
                    "roc_points_csv": roc_points_csv_path if roc_ok else None,
                    "banking_decisions_csv": banking_csv_path if args.threshold_protocol == "banking" else None
                },
                "test_scores": test_scores,
                "banking_decisions": banking_decisions,
                "config": cfg,
                "checkpoint_path": model_path,
                "output_dir": out_dir,
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {results_txt_path}")
    print(f"Saved: {scores_json_path}")
    if args.threshold_protocol in ["fixed05", "valopt"]:
        print(f"Saved: {confusion_csv_path}")
    print(f"Saved: {score_dist_csv_path}")
    if roc_ok:
        print(f"Saved: {roc_points_csv_path}")
    if args.threshold_protocol == "banking":
        print(f"Saved: {banking_csv_path}")
    

if __name__ == "__main__":
    main()