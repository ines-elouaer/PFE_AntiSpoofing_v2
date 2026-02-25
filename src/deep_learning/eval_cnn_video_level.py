import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from tqdm import tqdm

from src.deep_learning.datasets_frame import CASIAFrameDataset
from src.deep_learning.metrics_pad import aggregate_video_scores, compute_apcer_bpcer_acer


def build_model():
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def confusion_at_threshold(video_scores, th: float):
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
        else:
            tp += 1
    return tn, fp, fn, tp


def f1_at_threshold(video_scores, th: float) -> float:
    tn, fp, fn, tp = confusion_at_threshold(video_scores, th)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def acc_at_threshold(video_scores, th: float) -> float:
    tn, fp, fn, tp = confusion_at_threshold(video_scores, th)
    total = tn + fp + fn + tp
    return (tn + tp) / total if total else 0.0


def find_best_threshold_on_val(video_scores, step: float = 0.01):
    best = {"th": 0.5, "acer": 1.0, "apcer": 1.0, "bpcer": 1.0, "f1": 0.0, "acc": 0.0}
    t = 0.0
    while t <= 1.000001:
        apcer, bpcer, acer = compute_apcer_bpcer_acer(video_scores, threshold=t)
        f1 = f1_at_threshold(video_scores, t)
        acc = acc_at_threshold(video_scores, t)

        # critère principal: ACER min ; tie-break: F1 max
        if (acer < best["acer"]) or (acer == best["acer"] and f1 > best["f1"]):
            best = {
                "th": round(t, 4),
                "acer": float(acer),
                "apcer": float(apcer),
                "bpcer": float(bpcer),
                "f1": float(f1),
                "acc": float(acc),
            }
        t += step
    return best


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
            frame_preds.append({
                "video_id": vid[i],
                "label": int(y[i].item()),
                "score_attack": float(probs[i, 1].item())
            })
    return frame_preds


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to best_model.pth")
    parser.add_argument("--exp_dir", default=None, help="Where to write results (default: folder of ckpt)")
    parser.add_argument("--val_csv", default=r"data\processed\CASIA\splits_subject\val.csv")
    parser.add_argument("--test_csv", default=r"data\processed\CASIA\splits_subject\test.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model_path = args.ckpt
    exp_dir = args.exp_dir if args.exp_dir else os.path.dirname(model_path)
    os.makedirs(exp_dir, exist_ok=True)

    out_results_txt = os.path.join(exp_dir, "results.txt")
    out_scores_json = os.path.join(exp_dir, "test_scores.json")

    # data
    val_ds = CASIAFrameDataset(args.val_csv, img_size=224, aug_mode="none")
    test_ds = CASIAFrameDataset(args.test_csv, img_size=224, aug_mode="none")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    model = build_model().to(device)

    # load checkpoint (support 2 formats)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        # au cas où le checkpoint est directement state_dict
        state = ckpt
    model.load_state_dict(state)

    # 1) VAL -> best threshold
    val_frame_preds = predict_frame_preds(model, val_loader, device, desc="Predict VAL frames")
    val_video_scores = aggregate_video_scores(val_frame_preds)
    best = find_best_threshold_on_val(val_video_scores, step=0.01)

    # 2) TEST -> metrics at best threshold
    test_frame_preds = predict_frame_preds(model, test_loader, device, desc="Predict TEST frames")
    test_video_scores = aggregate_video_scores(test_frame_preds)

    th = best["th"]
    apcer, bpcer, acer = compute_apcer_bpcer_acer(test_video_scores, threshold=th)
    acc = acc_at_threshold(test_video_scores, th)
    f1 = f1_at_threshold(test_video_scores, th)
    tn, fp, fn, tp = confusion_at_threshold(test_video_scores, th)

    # Save results.txt
    with open(out_results_txt, "w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write("CASIA CNN Evaluation (Video-Level)\n")
        f.write("Protocol: threshold chosen on VAL, applied on TEST\n")
        f.write("========================================\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write("Model: MobileNetV3-Large\n")
        f.write(f"Checkpoint: {model_path}\n\n")

        f.write("---- Best threshold on VAL ----\n")
        f.write(json.dumps(best, indent=2) + "\n\n")

        f.write("---- TEST metrics (using VAL threshold) ----\n")
        f.write(f"Threshold: {th}\n")
        f.write(f"ACC:  {acc:.4f}\n")
        f.write(f"F1:   {f1:.4f}\n")
        f.write(f"APCER:{apcer:.4f}\n")
        f.write(f"BPCER:{bpcer:.4f}\n")
        f.write(f"ACER: {acer:.4f}\n\n")

        f.write("Confusion Matrix (TEST)\n")
        f.write(f"TN={tn} FP={fp} FN={fn} TP={tp}\n")

   
    with open(out_scores_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold_val": best,
                "test_scores": test_video_scores
            },
            f,
            indent=2
        )

    print("✅ Evaluation complete.")
    print("Saved:", out_results_txt)
    print("Saved:", out_scores_json)


if __name__ == "__main__":
    main()