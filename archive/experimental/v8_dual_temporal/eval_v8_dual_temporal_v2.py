from pathlib import Path
import sys
import argparse
import json
import pickle
import torch
from torch.utils.data import DataLoader
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.v8.v8_dual_temporal_lib import (
    DualTemporalDataset,
    DualTemporalPAD,
    BEHAV_SEQ_COLS,
    compute_metrics,
)

NUM_WORKERS = 0


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, loader, device, threshold=0.5):
    model.eval()

    rows = []
    y_all = []
    pred_all = []
    score_all = []

    for x, behavior_seq, y, vids in loader:
        x = x.to(device)
        behavior_seq = behavior_seq.to(device)

        logits = model(x, behavior_seq)
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1]
        pred = (scores >= threshold).long()

        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()
        score_np = scores.cpu().numpy()

        for vid, yy, pp, ss in zip(vids, y_np, pred_np, score_np):
            rows.append({
                "video_id": str(vid),
                "label": int(yy),
                "pred_label": int(pp),
                "score_spoof": float(ss),
            })

        y_all.extend(y_np.tolist())
        pred_all.extend(pred_np.tolist())
        score_all.extend(score_np.tolist())

    metrics = compute_metrics(y_all, pred_all, score_all)
    metrics["threshold"] = float(threshold)

    return pd.DataFrame(rows), metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--behavior_seq_csv", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--threshold", default="auto")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    ckpt = torch.load(args.checkpoint, map_location=device)

    if "config" in ckpt:
        config = ckpt["config"]
    else:
        config = {}

    video_lstm_hidden = int(config.get("video_lstm_hidden", 256))
    behavior_lstm_hidden = int(config.get("behavior_lstm_hidden", 32))
    dropout = float(config.get("dropout", 0.6))

    if args.threshold == "auto":
        threshold = float(ckpt.get("best_threshold_from_val", 0.5))
    else:
        threshold = float(args.threshold)

    print("Threshold:", threshold)

    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)

    ds = DualTemporalDataset(
        args.frames_csv,
        args.behavior_seq_csv,
        T=args.T,
        img_size=args.img_size,
        scaler=scaler,
        fit_scaler=False,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = DualTemporalPAD(
        behavior_dim=len(BEHAV_SEQ_COLS),
        video_lstm_hidden=video_lstm_hidden,
        behavior_lstm_hidden=behavior_lstm_hidden,
        dropout=dropout,
        pretrained_backbone=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    pred_df, metrics = predict(model, loader, device, threshold=threshold)

    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = {
        "model": str(args.checkpoint),
        "threshold": threshold,
        "metrics": metrics,
        "outputs": {
            "predictions": str(out_dir / "predictions.csv"),
            "metrics": str(out_dir / "metrics.json"),
        },
    }

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
