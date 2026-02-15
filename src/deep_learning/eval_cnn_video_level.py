import os
import csv
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


def compute_threshold_sweep(video_scores):
    thresholds = [i / 10 for i in range(1, 10)]  # 0.1..0.9
    results = []

    for th in thresholds:
        apcer, bpcer, acer = compute_apcer_bpcer_acer(video_scores, threshold=th)

        correct = 0
        total = 0
        for _, d in video_scores.items():
            y_true = d["label"]
            y_pred = 1 if d["score"] >= th else 0
            correct += int(y_pred == y_true)
            total += 1

        acc = correct / total if total else 0.0

        results.append({
            "threshold": th,
            "ACC": acc,
            "APCER": apcer,
            "BPCER": bpcer,
            "ACER": acer
        })

    return results


def compute_video_accuracy(video_scores, threshold: float):
    correct = 0
    total = 0
    for _, d in video_scores.items():
        y_true = d["label"]
        y_pred = 1 if d["score"] >= threshold else 0
        correct += int(y_pred == y_true)
        total += 1
    return correct / total if total else 0.0


@torch.no_grad()
def main():
    test_csv = r"data\processed\CASIA\splits_subject\test.csv"
    exp_dir = r"experiments\exp3_casia_mnv3large_strong"

    model_path = os.path.join(exp_dir, "best_model.pth")
    out_results_txt = os.path.join(exp_dir, "results.txt")

    threshold = 0.5
    batch_size = 32
    num_workers = 0

    os.makedirs(exp_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = CASIAFrameDataset(test_csv, img_size=224, aug_mode="none")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model().to(device)

    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    softmax = nn.Softmax(dim=1)

    frame_preds = []
    for x, y, vid in tqdm(loader, desc="Predict test frames"):
        x = x.to(device)
        logits = model(x)
        probs = softmax(logits).detach().cpu()

        for i in range(x.size(0)):
            frame_preds.append({
                "video_id": vid[i],
                "label": int(y[i].item()),
                "score_attack": float(probs[i, 1].item())
            })

  
    video_scores = aggregate_video_scores(frame_preds)

   
    apcer, bpcer, acer = compute_apcer_bpcer_acer(video_scores, threshold=threshold)
    video_acc = compute_video_accuracy(video_scores, threshold=threshold)

    sweep_results = compute_threshold_sweep(video_scores)

   
    attack_scores = [d["score"] for d in video_scores.values() if d["label"] == 1]
    real_scores = [d["score"] for d in video_scores.values() if d["label"] == 0]
    attack_mean = sum(attack_scores) / len(attack_scores) if attack_scores else 0.0
    real_mean = sum(real_scores) / len(real_scores) if real_scores else 0.0

    
    with open(out_results_txt, "w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write("CASIA CNN Evaluation (Video-Level)\n")
        f.write("========================================\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write("Model: MobileNetV3-Large\n")
        f.write(f"Threshold: {threshold}\n\n")

        f.write(f"Video Accuracy: {video_acc:.4f}\n")
        f.write(f"APCER: {apcer:.4f}\n")
        f.write(f"BPCER: {bpcer:.4f}\n")
        f.write(f"ACER: {acer:.4f}\n\n")

        f.write("--- Threshold Sweep ---\n")
        f.write("th\tACC\tAPCER\tBPCER\tACER\n")
        for row in sweep_results:
            f.write(
                f"{row['threshold']:.1f}\t"
                f"{row['ACC']:.4f}\t"
                f"{row['APCER']:.4f}\t"
                f"{row['BPCER']:.4f}\t"
                f"{row['ACER']:.4f}\n"
            )

        f.write("\n--- Score Separation ---\n")
        f.write(f"Attack mean: {attack_mean:.4f}\n")
        f.write(f"Real mean: {real_mean:.4f}\n")

    print("✅ Evaluation complete.")
    print("Saved:", out_results_txt)


if __name__ == "__main__":
    main()
