from pathlib import Path
import argparse
import inspect
import sys

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD
from src.deep_learning.datasets_frame import build_transforms


BEHAV_COLS = [
    "ear_mean",
    "ear_std",
    "ear_min",
    "ear_max",
    "blink_count",
    "motion_mean",
    "motion_std",
    "motion_max",
    "skipped_rate",
]

RPPG_COLS = [
    "rppg_dominant_freq",
    "rppg_hr_estimate",
    "rppg_snr",
    "rppg_signal_std",
    "rppg_skipped_rate",
    "rppg_valid",
]

ALL_COLS = BEHAV_COLS + RPPG_COLS


def load_checkpoint_state(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model_state", "model_state_dict", "state_dict"]:
            if key in ckpt:
                return ckpt[key]
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
        "use_behav": True,
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

    if n >= T:
        start = max(0, (n - T) // 2)
        return rows.iloc[start:start + T].copy()

    out = rows.copy()
    while len(out) < T:
        out = pd.concat([out, rows.iloc[[-1]]], ignore_index=True)
    return out


def load_sequence(frames_df, video_id, T, img_size):
    rows = frames_df[frames_df["video_id"].astype(str) == str(video_id)].copy()

    if len(rows) == 0:
        raise RuntimeError(f"No frames for video_id={video_id}")

    sampled = center_consecutive_sample(rows, T)
    tf = build_transforms(img_size, aug_mode="none")

    tensors = []
    for _, r in sampled.iterrows():
        img = Image.open(r["path"]).convert("RGB")
        tensors.append(tf(img))

    return torch.stack(tensors, dim=0).unsqueeze(0)


def load_feature_vector(behav_df, video_id, mode):
    row = behav_df[behav_df["video_id"].astype(str) == str(video_id)]

    if len(row) == 0:
        return torch.zeros(1, len(ALL_COLS), dtype=torch.float32)

    vals = []
    for col in ALL_COLS:
        vals.append(float(row.iloc[0][col]) if col in row.columns else 0.0)

    vec = torch.tensor(vals, dtype=torch.float32)

    if mode == "normal":
        pass

    elif mode == "zero_behavior":
        vec[0:len(BEHAV_COLS)] = 0.0

    elif mode == "zero_rppg":
        vec[len(BEHAV_COLS):] = 0.0

    elif mode == "zero_all":
        vec[:] = 0.0

    else:
        raise ValueError(f"Invalid ablation mode: {mode}")

    return vec.unsqueeze(0)


def predict(model, x, behav):
    with torch.no_grad():
        logits = model(x, behav)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred = 1 if prob >= 0.5 else 0
    return prob, pred


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions", required=True)
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

    parser.add_argument("--behav_dim", type=int, default=15)
    parser.add_argument("--behav_hidden", type=int, default=16)
    parser.add_argument("--use_gated_fusion", type=int, default=1)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pred_df = pd.read_csv(args.predictions)
    frames_df = pd.read_csv(args.frames_csv)
    behav_df = pd.read_csv(args.behav_csv)

    model = build_model(args)
    state = load_checkpoint_state(args.model)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Device:", device)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model.to(device)
    model.eval()

    modes = ["normal", "zero_behavior", "zero_rppg", "zero_all"]
    rows_out = []

    for _, row in pred_df.iterrows():
        video_id = row["video_id"]
        y = int(row["label"])

        x = load_sequence(frames_df, video_id, args.T, args.img_size).to(device)

        out_row = {
            "video_id": video_id,
            "label": y,
            "original_pred_label": int(row["pred_label"]),
            "original_score_spoof": float(row["score_spoof"]),
        }

        for mode in modes:
            behav = load_feature_vector(behav_df, video_id, mode).to(device)
            score, pred = predict(model, x, behav)

            out_row[f"{mode}_score_spoof"] = score
            out_row[f"{mode}_pred_label"] = pred

        rows_out.append(out_row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame(rows_out)

    # Deltas
    out["delta_zero_behavior"] = out["zero_behavior_score_spoof"] - out["normal_score_spoof"]
    out["delta_zero_rppg"] = out["zero_rppg_score_spoof"] - out["normal_score_spoof"]
    out["delta_zero_all"] = out["zero_all_score_spoof"] - out["normal_score_spoof"]

    out_path = out_dir / "ablation_behavior_rppg_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    summary_rows = []

    for mode in modes:
        pred_col = f"{mode}_pred_label"

        tn = int(((out["label"] == 0) & (out[pred_col] == 0)).sum())
        fp = int(((out["label"] == 0) & (out[pred_col] == 1)).sum())
        fn = int(((out["label"] == 1) & (out[pred_col] == 0)).sum())
        tp = int(((out["label"] == 1) & (out[pred_col] == 1)).sum())

        real_count = tn + fp
        spoof_count = fn + tp

        apcer = fn / spoof_count if spoof_count else 0.0
        bpcer = fp / real_count if real_count else 0.0
        acer = (apcer + bpcer) / 2.0

        summary_rows.append({
            "mode": mode,
            "tn_real": tn,
            "fp_real_as_spoof": fp,
            "fn_spoof_as_real": fn,
            "tp_spoof": tp,
            "APCER": apcer,
            "BPCER": bpcer,
            "ACER": acer,
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "ablation_behavior_rppg_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print("========== ABLATION SUMMARY ==========")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
