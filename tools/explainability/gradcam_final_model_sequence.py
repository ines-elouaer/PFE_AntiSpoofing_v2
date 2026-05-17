from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inspect
import cv2
import numpy as np
import pandas as pd
import torch
# Disable cuDNN for Grad-CAM through LSTM/RNN backward.
# This keeps model.eval() but avoids: cudnn RNN backward can only be called in training mode.
torch.backends.cudnn.enabled = False
from PIL import Image

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
                print(f"[INFO] Checkpoint state loaded from key: {key}")
                return ckpt[key]

    print("[INFO] Checkpoint loaded directly as state_dict")
    return ckpt


def build_model(args):
    """
    Build CNN_LSTM_PAD with flexible arguments.
    This avoids errors if your current CNN_LSTM_PAD class does or does not support gated fusion.
    """
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
    model = CNN_LSTM_PAD(**kwargs)
    return model


def center_consecutive_sample(rows, T):
    rows = rows.sort_values("frame_idx").reset_index(drop=True)
    n = len(rows)

    if n == 0:
        raise RuntimeError("No frames for selected video.")

    if n >= T:
        start = max(0, (n - T) // 2)
        return rows.iloc[start:start + T].copy()

    out = rows.copy()
    while len(out) < T:
        out = pd.concat([out, rows.iloc[[-1]]], ignore_index=True)

    return out


def load_sequence(frames_csv, video_id, T, img_size):
    frames = pd.read_csv(frames_csv)
    frames["video_id"] = frames["video_id"].astype(str)

    rows = frames[frames["video_id"] == str(video_id)].copy()

    if len(rows) == 0:
        raise RuntimeError(f"video_id not found in frames_csv: {video_id}")

    sampled = center_consecutive_sample(rows, T)
    tf = build_transforms(img_size, aug_mode="none")

    tensors = []
    original_images = []
    original_paths = []

    for _, r in sampled.iterrows():
        p = Path(r["path"])

        if not p.exists():
            raise FileNotFoundError(p)

        img = Image.open(p).convert("RGB")
        tensors.append(tf(img))

        bgr = cv2.imread(str(p))
        if bgr is None:
            raise FileNotFoundError(p)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (img_size, img_size))
        original_images.append(rgb)
        original_paths.append(str(p))

    x = torch.stack(tensors, dim=0).unsqueeze(0)  # [1, T, C, H, W]

    return x, original_images, original_paths, sampled


def load_behav(behav_csv, video_id, behav_dim):
    if behav_csv is None:
        return torch.zeros(1, behav_dim, dtype=torch.float32)

    behav = pd.read_csv(behav_csv)
    behav["video_id"] = behav["video_id"].astype(str)

    row = behav[behav["video_id"] == str(video_id)]

    if len(row) == 0:
        print(f"[WARN] No behavior found for {video_id}. Using zeros.")
        return torch.zeros(1, behav_dim, dtype=torch.float32)

    values = []

    for col in BEHAV_COLS_15[:behav_dim]:
        if col in row.columns:
            values.append(float(row.iloc[0][col]))
        else:
            values.append(0.0)

    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


class FinalModelGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self.forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, x, behav, target_class, frame_position):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x, behav)
        probs = torch.softmax(logits, dim=1)
        prob = probs[0, target_class].item()

        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        acts = self.activations
        grads = self.gradients

        if acts.dim() != 4:
            raise RuntimeError(f"Unexpected activation shape: {acts.shape}")

        # Model processes B*T frames. Since B=1, frame_position corresponds to index.
        idx = frame_position

        act = acts[idx]      # [C, H, W]
        grad = grads[idx]    # [C, H, W]

        weights = grad.mean(dim=(1, 2))  # [C]
        cam = (weights[:, None, None] * act).sum(dim=0)
        cam = torch.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        cam_np = cam_np - cam_np.min()

        if cam_np.max() > 1e-8:
            cam_np = cam_np / cam_np.max()

        return cam_np, logits.detach().cpu().numpy(), prob


def overlay_cam(rgb_img, cam):
    rgb_float = rgb_img.astype(np.float32) / 255.0

    cam_resized = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = 0.55 * rgb_float + 0.45 * heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--behav_csv", required=True)
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--target_class", type=int, default=1, help="0=REAL, 1=SPOOF")
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

    parser.add_argument("--frame_position", default="center", help="center or numeric index 0..T-1")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("========== FINAL MODEL GRAD-CAM ==========")
    print("Device       :", device)
    print("Model        :", args.model)
    print("Video        :", args.video_id)
    print("Target class :", args.target_class)

    model = build_model(args)
    state = load_checkpoint_state(args.model)

    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Missing keys   :", len(missing))
    print("Unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print("[WARN] Missing keys example:", missing[:10])

    if len(unexpected) > 0:
        print("[WARN] Unexpected keys example:", unexpected[:10])

    model.to(device)
    model.eval()

    x, original_images, original_paths, sampled = load_sequence(
        args.frames_csv,
        args.video_id,
        args.T,
        args.img_size,
    )

    behav = load_behav(args.behav_csv, args.video_id, args.behav_dim)

    x = x.to(device)
    behav = behav.to(device)

    if args.frame_position == "center":
        frame_idx_local = args.T // 2
    else:
        frame_idx_local = int(args.frame_position)

    frame_idx_local = max(0, min(args.T - 1, frame_idx_local))

    target_layer = model.backbone.features[-1]

    gradcam = FinalModelGradCAM(model, target_layer)

    cam, logits, prob = gradcam(
        x=x,
        behav=behav,
        target_class=args.target_class,
        frame_position=frame_idx_local,
    )

    gradcam.remove_hooks()

    overlay = overlay_cam(original_images[frame_idx_local], cam)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), overlay_bgr)

    print("Logits:", logits)
    print(f"Probability target class {args.target_class}: {prob:.6f}")
    print("Frame used:", original_paths[frame_idx_local])
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
