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
        default=0.30,
        help="For banking mode: score below this => ACCEPT"
    )
    parser.add_argument(
        "--reject_threshold",
        type=float,
        default=0.60,
        help="For banking mode: score above or equal this => REJECT; otherwise RETRY"
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

    if args.threshold_protocol == "banking":
        if not (0.0 <= args.accept_threshold < args.reject_threshold <= 1.0):
            raise ValueError(
                "In banking mode, thresholds must satisfy: "
                "0 <= accept_threshold < reject_threshold <= 1"
            )

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
    cfg = ckpt.get("config", {})

    use_behav = bool(cfg.get("use_behav", False))
    behav_dim = int(cfg.get("behav_dim", 9))
    behav_hidden = int(cfg.get("behav_hidden", 32))

    model = CNN_LSTM_PAD(
        hidden=cfg.get("hidden", 256),
        num_layers=cfg.get("num_layers", 1),
        bidir=cfg.get("bidir", False),
        temporal_pool=cfg.get("temporal_pool", "mean"),
        pretrained_backbone=True,
        use_behav=use_behav,
        behav_dim=behav_dim,
        behav_hidden=behav_hidden,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])

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
    best_val = find_best_threshold_on_val(val_scores, step=0.01)

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
    banking_summary = None
    if args.threshold_protocol == "banking":
        banking_decisions = build_banking_decisions(
            test_scores,
            accept_threshold=args.accept_threshold,
            reject_threshold=args.reject_threshold,
        )
        banking_summary = summarize_banking_decisions(banking_decisions)
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
        report.append("--- Banking Decision Policy ---")
        report.append(f"ACCEPT if score < {args.accept_threshold:.4f}")
        report.append(f"RETRY  if {args.accept_threshold:.4f} <= score < {args.reject_threshold:.4f}")
        report.append(f"REJECT if score >= {args.reject_threshold:.4f}")
        report.append("")
        report.append("--- Banking Decision Summary ---")
        report.append(
            f"ACCEPT={banking_summary['ACCEPT']}  "
            f"RETRY={banking_summary['RETRY']}  "
            f"REJECT={banking_summary['REJECT']}"
        )

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
                        "accept_threshold": args.accept_threshold,
                        "reject_threshold": args.reject_threshold,
                        "decision_summary": banking_summary,
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