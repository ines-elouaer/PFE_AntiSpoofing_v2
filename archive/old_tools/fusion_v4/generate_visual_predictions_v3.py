from pathlib import Path
import argparse
import inspect
import sys
import json

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD
from src.deep_learning.datasets_frame import build_transforms


BEHAV_COLS_15 = [
    "ear_mean",
    "ear_std",
    "ear_min",
    "ear_max",
    "blink_count",
    "motion_mean",
    "motion_std",
    "motion_max",
    "skipped_rate",
    "rppg_dominant_freq",
    "rppg_hr_estimate",
    "rppg_snr",
    "rppg_signal_std",
    "rppg_skipped_rate",
    "rppg_valid",
]


def load_checkpoint_state(path):
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict):
        for key in ["model_state", "model_state_dict", "state_dict"]:
            if key in ckpt:
                print(f"[INFO] Checkpoint loaded from key: {key}")
                return ckpt[key]

    print("[INFO] Checkpoint loaded directly as state_dict")
    return ckpt


def build_model(args):
    sig = inspect.signature(CNN_LSTM_PAD.__init__)
    allowed = set(sig.parameters.keys())

    kwargs = {
        "hidden": args.hidden,
        "num_layers": args.num_layers,
        "bidir": bool(args.bidir),
        "lstm_dropout": args.lstm_dropout,
        "head_dropout": args.head_dropout,
        "pretrained_backbone": False,
        "temporal_pool": args.temporal_pool,
        "use_behav": bool(args.use_behav),
        "behav_dim": args.behav_dim,
        "behav_hidden": args.behav_hidden,
    }

    if "use_gated_fusion" in allowed:
        kwargs["use_gated_fusion"] = bool(args.use_gated_fusion)

    kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    return CNN_LSTM_PAD(**kwargs)


def center_consecutive_sample(rows, T):
    rows = rows.sort_values("frame_idx").reset_index(drop=True)
    n = len(rows)

    if n == 0:
        raise RuntimeError("Video without frames.")

    if n >= T:
        start = max(0, (n - T) // 2)
        return rows.iloc[start:start + T].copy()

    out = rows.copy()
    while len(out) < T:
        out = pd.concat([out, rows.iloc[[-1]]], ignore_index=True)

    return out


def load_sequence(rows, T, img_size, tf):
    sampled = center_consecutive_sample(rows, T)

    frames = []
    for _, r in sampled.iterrows():
        p = Path(r["path"])
        if not p.exists():
            raise FileNotFoundError(p)

        img = Image.open(p).convert("RGB")
        frames.append(tf(img))

    x = torch.stack(frames, dim=0).unsqueeze(0)
    return x


def load_behav_vector(behav_df, video_id, behav_dim):
    row = behav_df[behav_df["video_id"].astype(str) == str(video_id)]

    if len(row) == 0:
        return torch.zeros(1, behav_dim, dtype=torch.float32)

    vals = []
    for col in BEHAV_COLS_15[:behav_dim]:
        if col in row.columns:
            vals.append(float(row.iloc[0][col]))
        else:
            vals.append(0.0)

    return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)


def compute_metrics(df):
    y = df["label"].astype(int)
    p = df["pred_label"].astype(int)

    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())

    real_count = tn + fp
    spoof_count = fn + tp

    apcer = fn / spoof_count if spoof_count else 0.0
    bpcer = fp / real_count if real_count else 0.0
    acer = (apcer + bpcer) / 2.0
    acc = (tn + tp) / len(df) if len(df) else 0.0

    return {
        "total": int(len(df)),
        "real_count": int(real_count),
        "spoof_count": int(spoof_count),
        "tn_real": tn,
        "fp_real_as_spoof": fp,
        "fn_spoof_as_real": fn,
        "tp_spoof": tp,
        "accuracy": float(acc),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER": float(acer),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--behav_csv", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidir", type=int, default=0)
    parser.add_argument("--lstm_dropout", type=float, default=0.2)
    parser.add_argument("--head_dropout", type=float, default=0.5)
    parser.add_argument("--temporal_pool", default="mean")

    parser.add_argument("--use_behav", type=int, default=1)
    parser.add_argument("--behav_dim", type=int, default=15)
    parser.add_argument("--behav_hidden", type=int, default=16)
    parser.add_argument("--use_gated_fusion", type=int, default=1)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("========== GENERATE VISUAL PREDICTIONS V3 ==========")
    print("Device:", device)
    print("Model :", args.model)
    print("Frames:", args.frames_csv)
    print("Behav :", args.behav_csv)

    frames = pd.read_csv(args.frames_csv)
    behav = pd.read_csv(args.behav_csv)

    frames["video_id"] = frames["video_id"].astype(str)
    behav["video_id"] = behav["video_id"].astype(str)

    required = {"video_id", "label", "frame_idx", "path"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"frames_csv missing columns: {missing}")

    video_ids = sorted(frames["video_id"].unique())

    labels = frames.groupby("video_id")["label"].first().to_dict()

    model = build_model(args)
    state = load_checkpoint_state(args.model)

    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)

    print("Missing keys   :", len(missing_keys))
    print("Unexpected keys:", len(unexpected_keys))

    if len(missing_keys) > 0:
        print("[WARN] Missing example:", missing_keys[:10])
    if len(unexpected_keys) > 0:
        print("[WARN] Unexpected example:", unexpected_keys[:10])

    model.to(device)
    model.eval()

    tf = build_transforms(args.img_size, aug_mode="none")

    rows_out = []

    with torch.no_grad():
        for i, vid in enumerate(video_ids, start=1):
            rows = frames[frames["video_id"] == vid].copy()

            x = load_sequence(rows, args.T, args.img_size, tf).to(device)
            b = load_behav_vector(behav, vid, args.behav_dim).to(device)

            logits = model(x, b)
            probs = torch.softmax(logits, dim=1)[0]
            score_spoof = float(probs[1].detach().cpu().item())
            pred_label = 1 if score_spoof >= 0.5 else 0

            rows_out.append({
                "video_id": vid,
                "label": int(labels[vid]),
                "score_spoof": score_spoof,
                "pred_label": pred_label,
            })

            if i % 20 == 0:
                print(f"[INFO] processed {i}/{len(video_ids)} videos")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame(rows_out)
    pred_path = out_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    metrics = compute_metrics(pred_df)
    metrics_path = out_dir / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved predictions:", pred_path)
    print("Saved metrics    :", metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
