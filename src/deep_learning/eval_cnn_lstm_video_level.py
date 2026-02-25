import os
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.deep_learning.datasets_sequence import CASIASequenceDataset
from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD
from src.deep_learning.metrics_pad import compute_apcer_bpcer_acer


@torch.no_grad()
def predict_video_scores(model, loader, device) -> Dict[str, Dict]:
    """Return: {vid: {"label": int, "score": float}} where score = P(attack)."""
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
                "score": float(probs[i, 1].item())  # P(attack)
            }
    return scores


def confusion_at_threshold(video_scores: Dict[str, Dict], th: float) -> Tuple[int, int, int, int]:
    tn = fp = fn = tp = 0
    for d in video_scores.values():
        y = d["label"]
        pred = 1 if d["score"] >= th else 0
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


def find_best_threshold_on_val(video_scores: Dict[str, Dict], step: float = 0.01) -> Dict:
   
    best = {"th": 0.5, "acer": 1.0, "apcer": 1.0, "bpcer": 1.0, "f1": 0.0, "acc": 0.0}
    t = 0.0
    while t <= 1.000001:
        apcer, bpcer, acer = compute_apcer_bpcer_acer(video_scores, threshold=t)
        f1 = f1_at_threshold(video_scores, t)
        acc = acc_at_threshold(video_scores, t)
        if (acer < best["acer"]) or (acer == best["acer"] and f1 > best["f1"]):
            best = {
                "th": round(t, 4),
                "acer": acer,
                "apcer": apcer,
                "bpcer": bpcer,
                "f1": f1,
                "acc": acc,
            }
        t += step
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
    attacks = [d["score"] for d in video_scores.values() if d["label"] == 1]
    reals = [d["score"] for d in video_scores.values() if d["label"] == 0]
    a_mean = sum(attacks) / len(attacks) if attacks else 0.0
    r_mean = sum(reals) / len(reals) if reals else 0.0
    return a_mean, r_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="Folder containing best_model.pth")
    parser.add_argument("--val_csv", default=r"data\processed\CASIA\splits_subject\val.csv")
    parser.add_argument("--test_csv", default=r"data\processed\CASIA\splits_subject\test.csv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--behav_val_csv", default=r"data\processed\CASIA\behav\val_behav.csv")
    parser.add_argument("--behav_test_csv", default=r"data\processed\CASIA\behav\test_behav.csv")

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, use this fixed threshold on TEST (e.g. 0.5). If None, threshold is selected from VAL (min ACER)."
    )

    args = parser.parse_args()

    exp_dir = args.exp_dir
    model_path = os.path.join(exp_dir, "best_model.pth")
    results_txt_path = os.path.join(exp_dir, "results.txt")
    scores_json_path = os.path.join(exp_dir, "test_scores.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    val_ds = CASIASequenceDataset(
        args.val_csv,
        T=T,
        img_size=img_size,
        aug_mode="none",
        sample_mode="uniform",
        seed=seed,
        behav_csv=(args.behav_val_csv if use_behav else None),
    )
    test_ds = CASIASequenceDataset(
        args.test_csv,
        T=T,
        img_size=img_size,
        aug_mode="none",
        sample_mode="uniform",
        seed=seed,
        behav_csv=(args.behav_test_csv if use_behav else None),
    )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    val_scores = predict_video_scores(model, val_loader, device)
    best_val = find_best_threshold_on_val(val_scores, step=0.01)


    if args.threshold is not None:
        th_used = float(args.threshold)
        th_source = f"FIXED ({th_used:.2f})"
    else:
        th_used = float(best_val["th"])
        th_source = f"VAL ({th_used:.2f})"

    test_scores = predict_video_scores(model, test_loader, device)

    apcer, bpcer, acer = compute_apcer_bpcer_acer(test_scores, threshold=th_used)
    f1 = f1_at_threshold(test_scores, th_used)
    acc = acc_at_threshold(test_scores, th_used)
    tn, fp, fn, tp = confusion_at_threshold(test_scores, th_used)
    auc, ok_auc = try_auc(test_scores)
    a_mean, r_mean = score_means(test_scores)

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
    report.append(f"Fusion behav: {use_behav} (behav_dim={behav_dim})")
    report.append(f"Model: MobileNetV3-Large + LSTM (T={T}, pool={cfg.get('temporal_pool','mean')})")
    report.append(f"Threshold (from VAL): {best_val['th']:.2f}")
    report.append(f"Threshold (USED on TEST): {th_used:.2f}  [{th_source}]")
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
                "threshold_val": best_val,
                "threshold_used": {"th": th_used, "source": th_source},
                "test_scores": test_scores,
                "config": cfg,
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {results_txt_path}")
    print(f"Saved: {scores_json_path}")


if __name__ == "__main__":
    main()