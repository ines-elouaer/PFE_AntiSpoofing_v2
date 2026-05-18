import os
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.deep_learning.datasets_frame import CASIAFrameDataset
from src.deep_learning.metrics_pad import (
    aggregate_video_scores,
    compute_apcer_bpcer_acer,
)
from src.deep_learning.train_cnn import build_model


def confusion_at_threshold(video_scores: Dict[str, Dict], th: float) -> Tuple[int, int, int, int]:
    tn = fp = fn = tp = 0
    for d in video_scores.values():
        y = int(d["label"])
        s = float(d["score"])
        pred = 1 if s >= th else 0

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


def _collect_threshold_candidates(video_scores: Dict[str, Dict], step: float = 0.01) -> List[Dict]:
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
    return candidates


def find_best_threshold_on_val(video_scores: Dict[str, Dict], step: float = 0.01) -> Dict:
    candidates = _collect_threshold_candidates(video_scores, step=step)

    # 1) keep thresholds with minimum ACER
    min_acer = min(c["acer"] for c in candidates)
    best = [c for c in candidates if c["acer"] == min_acer]

    # 2) among them, keep maximum F1
    max_f1 = max(c["f1"] for c in best)
    best = [c for c in best if c["f1"] == max_f1]

    # 3) among them, keep maximum ACC
    max_acc = max(c["acc"] for c in best)
    best = [c for c in best if c["acc"] == max_acc]

    # 4) among remaining equally-good thresholds, choose the one closest to 0.5
    best.sort(key=lambda c: abs(c["th"] - 0.5))
    chosen = best[0].copy()

    # keep some debugging info
    chosen["n_tied_candidates"] = len(best)
    chosen["selection_rule"] = "min_ACER -> max_F1 -> max_ACC -> closest_to_0.5"
    return chosen


@torch.no_grad()
def predict_frame_preds(model, loader, device, desc="Predict"):
    model.eval()
    softmax = nn.Softmax(dim=1)
    frame_preds = []

    for x, y, vid in tqdm(loader, desc=desc):
        x = x.to(device)
        logits = model(x)
        probs = softmax(logits).detach().cpu()

        for i in range(x.size(0)):
            frame_preds.append(
                {
                    "video_id": vid[i],
                    "label": int(y[i].item()),
                    "score_attack": float(probs[i, 1].item()),
                }
            )
    return frame_preds


def try_auc(video_scores: Dict[str, Dict]) -> Tuple[float, bool]:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return 0.0, False

    y_true = [d["label"] for d in video_scores.values()]
    y_score = [d["score"] for d in video_scores.values()]
    return float(roc_auc_score(y_true, y_score)), True


def score_means(video_scores: Dict[str, Dict]) -> Tuple[float, float]:
    attacks = [d["score"] for d in video_scores.values() if d["label"] == 1]
    reals = [d["score"] for d in video_scores.values() if d["label"] == 0]
    a_mean = sum(attacks) / len(attacks) if attacks else 0.0
    r_mean = sum(reals) / len(reals) if reals else 0.0
    return a_mean, r_mean


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="Folder containing best_model.pth")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save evaluation outputs")
    parser.add_argument("--val_csv", default=r"data\processed\CASIA\splits_subject\val.csv")
    parser.add_argument("--test_csv", default=r"data\processed\CASIA\splits_subject\test.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--model_name",
        type=str,
        default="cnn",
        choices=["cnn"],
        help="Canonical model name."
    )

    parser.add_argument(
        "--threshold_protocol",
        type=str,
        default="valopt",
        choices=["valopt", "fixed05"],
        help="Threshold protocol: valopt = select on VAL; fixed05 = use 0.5 on TEST."
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(args.exp_dir, "best_model.pth")

    out_dir = args.out_dir if args.out_dir is not None else args.exp_dir
    os.makedirs(out_dir, exist_ok=True)

    results_txt_path = os.path.join(out_dir, "results.txt")
    scores_json_path = os.path.join(out_dir, "test_scores.json")

    val_ds = CASIAFrameDataset(args.val_csv, img_size=224, aug_mode="none")
    test_ds = CASIAFrameDataset(args.test_csv, img_size=224, aug_mode="none")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model().to(device)

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        cfg = ckpt.get("config", {})
    else:
        state = ckpt
        cfg = {}

    model.load_state_dict(state)

    protocol_name = f"{args.model_name}_{args.threshold_protocol}"

    # 1) VAL predictions and threshold calibration
    val_frame_preds = predict_frame_preds(model, val_loader, device, desc="Predict VAL frames")
    val_video_scores = aggregate_video_scores(val_frame_preds)
    best_val = find_best_threshold_on_val(val_video_scores, step=0.01)

    if args.threshold_protocol == "fixed05":
        th_used = 0.5
        th_source = "fixed05"
    else:
        th_used = float(best_val["th"])
        th_source = "valopt"

    # 2) TEST evaluation
    test_frame_preds = predict_frame_preds(model, test_loader, device, desc="Predict TEST frames")
    test_video_scores = aggregate_video_scores(test_frame_preds)

    apcer, bpcer, acer = compute_apcer_bpcer_acer(test_video_scores, threshold=th_used)
    acc = acc_at_threshold(test_video_scores, th_used)
    f1 = f1_at_threshold(test_video_scores, th_used)
    tn, fp, fn, tp = confusion_at_threshold(test_video_scores, th_used)
    auc, ok_auc = try_auc(test_video_scores)
    a_mean, r_mean = score_means(test_video_scores)

    sweep_lines = ["th\tACC\tAPCER\tBPCER\tACER\tF1"]
    for k in range(10, 100, 10):
        th = k / 100
        ap, bp, ac = compute_apcer_bpcer_acer(test_video_scores, threshold=th)
        sweep_lines.append(
            f"{th:.1f}\t{acc_at_threshold(test_video_scores, th):.4f}\t{ap:.4f}\t{bp:.4f}\t{ac:.4f}\t{f1_at_threshold(test_video_scores, th):.4f}"
        )
    sweep_txt = "\n".join(sweep_lines)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = []
    report.append("========================================")
    report.append("CASIA CNN Evaluation (Video-Level)")
    report.append("========================================")
    report.append(f"Date: {now}")
    report.append(f"Device: {device}")
    report.append(f"Model name: {args.model_name}")
    report.append(f"Protocol name: {protocol_name}")
    report.append(f"Threshold protocol: {args.threshold_protocol}")
    report.append("Model: MobileNetV3-Large")
    report.append(f"Checkpoint: {model_path}")
    report.append(f"Threshold (best on VAL): {best_val['th']:.4f}")
    report.append(f"Threshold (USED on TEST): {th_used:.4f}")
    report.append(f"Threshold source: {th_source}")
    report.append(f"Tied optimal thresholds on VAL: {best_val.get('n_tied_candidates', 1)}")
    report.append(f"Threshold selection rule: {best_val.get('selection_rule', 'legacy')}")
    report.append("")
    report.append(f"Video Accuracy: {acc:.4f}")
    report.append(f"F1: {f1:.4f}")
    report.append(f"APCER: {apcer:.4f}")
    report.append(f"BPCER: {bpcer:.4f}")
    report.append(f"ACER: {acer:.4f}")
    report.append(f"ROC-AUC: {auc:.4f}" if ok_auc else "ROC-AUC: N/A (sklearn not installed)")
    report.append("")
    report.append("--- Confusion Matrix ---")
    report.append(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    report.append("")
    report.append("--- Threshold Sweep ---")
    report.append(sweep_txt)
    report.append("")
    report.append("--- Score Separation ---")
    report.append(f"Attack mean: {a_mean:.4f}")
    report.append(f"Real mean: {r_mean:.4f}")

    report_txt = "\n".join(report)
    print(report_txt)

    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    with open(scores_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "protocol_name": protocol_name,
                "threshold_protocol": args.threshold_protocol,
                "threshold_val": best_val,
                "threshold_used": {
                    "th": th_used,
                    "source": th_source
                },
                "metrics_test": {
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
                },
                "score_stats": {
                    "attack_mean": a_mean,
                    "real_mean": r_mean
                },
                "test_scores": test_video_scores,
                "config": cfg,
                "checkpoint_path": model_path,
                "output_dir": out_dir,
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {results_txt_path}")
    print(f"Saved: {scores_json_path}")


if __name__ == "__main__":
    main()