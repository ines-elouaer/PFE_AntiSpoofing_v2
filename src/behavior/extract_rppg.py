"""
extract_rppg.py
================

Extraction professionnelle des features rPPG par vidéo.

Objectif :
- Extraire un signal physiologique approximatif depuis les variations RGB du visage.
- Ajouter des features rPPG au vecteur comportemental.
- Produire un CSV par split : train_rppg.csv, val_rppg.csv, test_rppg.csv.

Features générées :
- rppg_dominant_freq
- rppg_hr_estimate
- rppg_snr
- rppg_signal_std
- rppg_skipped_rate
- rppg_valid

Usage :
python -m src.behavior.extract_rppg ^
  --split_csv data/mixed_casia_axon_local_msu/mixed_train_frames.csv ^
  --out_csv data/mixed_casia_axon_local_msu_rppg/mixed_train_rppg.csv ^
  --fps 25 ^
  --model models/face_landmarker.task
"""

from __future__ import annotations

import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


LEFT_CHEEK = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]


def read_rows(csv_path: str) -> List[Dict]:
    rows = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            row["frame_idx"] = int(row["frame_idx"])
            rows.append(row)
    return rows


def group_by_video(rows: List[Dict]) -> Dict[str, List[Dict]]:
    by_vid = defaultdict(list)
    for row in rows:
        by_vid[row["video_id"]].append(row)
    return by_vid


def get_cheek_roi(landmarks, w: int, h: int, margin: int = 5):
    indices = LEFT_CHEEK + RIGHT_CHEEK

    xs = [landmarks[i].x * w for i in indices if i < len(landmarks)]
    ys = [landmarks[i].y * h for i in indices if i < len(landmarks)]

    if not xs or not ys:
        return None

    x1 = max(0, int(min(xs)) - margin)
    y1 = max(0, int(min(ys)) - margin)
    x2 = min(w, int(max(xs)) + margin)
    y2 = min(h, int(max(ys)) + margin)

    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    return x1, y1, x2, y2


def extract_rgb_signal_from_frames(frame_rows, landmarker):
    """
    Retourne signal_r, signal_g, signal_b, n_skipped.
    """
    import cv2

    signal_r = []
    signal_g = []
    signal_b = []
    n_skipped = 0

    sorted_rows = sorted(frame_rows, key=lambda r: int(r["frame_idx"]))

    for row in sorted_rows:
        img_path = row["path"]
        frame = cv2.imread(img_path)

        if frame is None:
            n_skipped += 1
            continue

        landmarks = landmarker.detect_landmarks(frame)

        if landmarks is None:
            n_skipped += 1
            continue

        h, w = frame.shape[:2]
        roi = get_cheek_roi(landmarks, w, h)

        if roi is None:
            n_skipped += 1
            continue

        x1, y1, x2, y2 = roi
        patch = frame[y1:y2, x1:x2]

        if patch.size == 0:
            n_skipped += 1
            continue

        # OpenCV lit en BGR.
        signal_b.append(float(patch[:, :, 0].mean()))
        signal_g.append(float(patch[:, :, 1].mean()))
        signal_r.append(float(patch[:, :, 2].mean()))

    return (
        np.array(signal_r, dtype=np.float32),
        np.array(signal_g, dtype=np.float32),
        np.array(signal_b, dtype=np.float32),
        n_skipped,
    )


def bandpass_filter(signal: np.ndarray, fps: float, low_hz: float = 0.75, high_hz: float = 4.0):
    """
    Filtrage FFT simple.
    Bande 0.75–4 Hz ≈ 45–240 bpm.
    """
    if len(signal) < 8:
        return signal.astype(np.float32)

    n = len(signal)
    centered = signal - signal.mean()

    fft = np.fft.rfft(centered)
    freq = np.fft.rfftfreq(n, d=1.0 / fps)

    mask = (freq >= low_hz) & (freq <= high_hz)
    fft_filtered = fft * mask

    return np.fft.irfft(fft_filtered, n=n).astype(np.float32)


def chrom_rppg(signal_r: np.ndarray, signal_g: np.ndarray, signal_b: np.ndarray):
    """
    Méthode CHROM simplifiée.
    """
    if len(signal_r) < 4:
        return np.zeros(max(len(signal_r), 1), dtype=np.float32)

    mean_r = signal_r.mean() if abs(signal_r.mean()) > 1e-6 else 1.0
    mean_g = signal_g.mean() if abs(signal_g.mean()) > 1e-6 else 1.0
    mean_b = signal_b.mean() if abs(signal_b.mean()) > 1e-6 else 1.0

    r_n = signal_r / mean_r
    g_n = signal_g / mean_g
    b_n = signal_b / mean_b

    xs = 3.0 * r_n - 2.0 * g_n
    ys = 1.5 * r_n + g_n - 1.5 * b_n

    std_xs = xs.std() if xs.std() > 1e-6 else 1.0
    std_ys = ys.std() if ys.std() > 1e-6 else 1.0
    alpha = std_xs / std_ys

    return (xs - alpha * ys).astype(np.float32)


def extract_rppg_features(
    signal_r: np.ndarray,
    signal_g: np.ndarray,
    signal_b: np.ndarray,
    fps: float,
    n_frames: int,
    n_skipped: int,
):
    skipped_rate = float(n_skipped / max(n_frames, 1))

    if len(signal_r) < 8:
        return {
            "rppg_dominant_freq": 0.0,
            "rppg_hr_estimate": 0.0,
            "rppg_snr": 0.0,
            "rppg_signal_std": 0.0,
            "rppg_skipped_rate": round(skipped_rate, 6),
            "rppg_valid": 0.0,
        }

    rppg_raw = chrom_rppg(signal_r, signal_g, signal_b)
    rppg_filt = bandpass_filter(rppg_raw, fps=fps)

    n = len(rppg_filt)
    fft = np.fft.rfft(rppg_filt - rppg_filt.mean())
    freq = np.fft.rfftfreq(n, d=1.0 / fps)
    psd = np.abs(fft) ** 2

    mask_band = (freq >= 0.75) & (freq <= 4.0)

    if mask_band.sum() == 0:
        dominant_freq = 0.0
        hr_estimate = 0.0
        snr = 0.0
    else:
        psd_band = psd.copy()
        psd_band[~mask_band] = 0.0

        idx_peak = int(np.argmax(psd_band))
        dominant_freq = float(freq[idx_peak])
        hr_estimate = dominant_freq * 60.0

        power_peak = float(psd[idx_peak])
        power_band = float(psd[mask_band].sum())

        snr = 10.0 * np.log10(power_peak / (power_band - power_peak + 1e-8))

    signal_std = float(rppg_filt.std()) if len(rppg_filt) > 0 else 0.0

    # Critère simple de validité.
    # Tu peux l’ajuster après analyse.
    valid = 1.0
    if skipped_rate > 0.40:
        valid = 0.0
    if hr_estimate < 45.0 or hr_estimate > 240.0:
        valid = 0.0
    if signal_std < 1e-6:
        valid = 0.0

    return {
        "rppg_dominant_freq": round(float(dominant_freq), 6),
        "rppg_hr_estimate": round(float(hr_estimate), 4),
        "rppg_snr": round(float(snr), 4),
        "rppg_signal_std": round(float(signal_std), 6),
        "rppg_skipped_rate": round(skipped_rate, 6),
        "rppg_valid": round(valid, 1),
    }


def precompute_rppg(split_csv: str, out_csv: str, fps: float, model_path: str):
    from src.behavior.mp_landmarks import FaceLandmarkerHelper

    rows = read_rows(split_csv)
    by_vid = group_by_video(rows)

    landmarker = FaceLandmarkerHelper(model_path=model_path)

    out_rows = []
    total = len(by_vid)

    for i, (vid, frs) in enumerate(by_vid.items(), start=1):
        frs_sorted = sorted(frs, key=lambda r: int(r["frame_idx"]))
        label = int(frs_sorted[0]["label"])

        sig_r, sig_g, sig_b, n_skipped = extract_rgb_signal_from_frames(
            frs_sorted,
            landmarker,
        )

        feats = extract_rppg_features(
            sig_r,
            sig_g,
            sig_b,
            fps=fps,
            n_frames=len(frs_sorted),
            n_skipped=n_skipped,
        )

        out_rows.append({
            "video_id": vid,
            "label": label,
            "n_frames": len(frs_sorted),
            "n_skipped": n_skipped,
            **feats,
        })

        if i % 10 == 0 or i == total:
            print(
                f"[{i}/{total}] {vid} | "
                f"HR={feats['rppg_hr_estimate']:.1f} | "
                f"SNR={feats['rppg_snr']:.2f} | "
                f"skip={feats['rppg_skipped_rate']:.2f} | "
                f"valid={feats['rppg_valid']}"
            )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    if not out_rows:
        raise RuntimeError("Aucune vidéo traitée.")

    cols = list(out_rows[0].keys())
    with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n[OK] Saved: {out_csv} ({len(out_rows)} videos)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--model", default="models/face_landmarker.task")
    args = parser.parse_args()

    print("========== rPPG EXTRACTION ==========")
    print("split_csv:", args.split_csv)
    print("out_csv  :", args.out_csv)
    print("fps      :", args.fps)
    print("model    :", args.model)

    precompute_rppg(
        split_csv=args.split_csv,
        out_csv=args.out_csv,
        fps=args.fps,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()