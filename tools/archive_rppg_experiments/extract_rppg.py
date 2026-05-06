"""
rPPG — Remote PhotoPlethysmoGraphy
====================================
Extrait le signal de pouls à distance depuis les frames vidéo.

Principe :
  - La peau change légèrement de couleur à chaque battement de cœur
  - On extrait la moyenne RGB sur la zone joue (ROI) frame par frame
  - On filtre le signal entre 0.75 Hz et 4 Hz (45–240 bpm)
  - On extrait 4 features : hr_estimate, snr, dominant_freq, signal_std

Ces 4 features s'ajoutent au vecteur comportemental : dim=9 → dim=13

Usage (precompute pour tout le dataset) :
    python -m src.behavior.extract_rppg ^
        --split_csv  data/processed/casia/splits_subject/train.csv ^
        --out_csv    data/processed/casia/behav/train_rppg.csv ^
        --fps        25

Puis normaliser avec normalize_behav.py (étendre FEAT_COLS pour inclure rPPG).
"""

import os
import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# ─────────────────────── landmarks joue (MediaPipe) ──────────────────

# Indices MediaPipe Face Mesh pour les joues gauche et droite
# Zone joue gauche  : landmark 234 (bas joue) + 93, 132, 58, 172
# Zone joue droite  : landmark 454 + 323, 361, 288, 397
LEFT_CHEEK  = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
FOREHEAD    = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323]


def get_cheek_roi(lm, w, h, side="both"):
    """
    Retourne le bounding box (x1,y1,x2,y2) de la zone joue
    à partir des landmarks MediaPipe.
    """
    if side == "left":
        indices = LEFT_CHEEK
    elif side == "right":
        indices = RIGHT_CHEEK
    else:
        indices = LEFT_CHEEK + RIGHT_CHEEK

    xs = [lm[i].x * w for i in indices if i < len(lm)]
    ys = [lm[i].y * h for i in indices if i < len(lm)]

    if not xs:
        return None

    margin = 5
    x1 = max(0, int(min(xs)) - margin)
    y1 = max(0, int(min(ys)) - margin)
    x2 = min(w, int(max(xs)) + margin)
    y2 = min(h, int(max(ys)) + margin)

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    return x1, y1, x2, y2


def extract_rgb_signal_from_frames(frame_rows, landmarker, fps=25.0):
    """
    Extrait le signal RGB moyen sur la zone joue pour chaque frame.

    Args:
        frame_rows : liste de dicts avec clé "path" (depuis le CSV split)
        landmarker : FaceLandmarkerHelper de mp_landmarks.py
        fps        : frames par seconde (pour le filtrage fréquentiel)

    Returns:
        signal_r, signal_g, signal_b : arrays numpy, une valeur par frame
        n_skipped : nombre de frames sans détection
    """
    import cv2

    signal_r = []
    signal_g = []
    signal_b = []
    n_skipped = 0

    sorted_rows = sorted(frame_rows, key=lambda r: int(r["frame_idx"]))

    for r in sorted_rows:
        img_path = r["path"]
        frame = cv2.imread(img_path)
        if frame is None:
            n_skipped += 1
            continue

        lm = landmarker.detect_landmarks(frame)
        if lm is None:
            n_skipped += 1
            continue

        h, w = frame.shape[:2]
        roi = get_cheek_roi(lm, w, h, side="both")

        if roi is None:
            n_skipped += 1
            continue

        x1, y1, x2, y2 = roi
        patch = frame[y1:y2, x1:x2]   # BGR

        if patch.size == 0:
            n_skipped += 1
            continue

        # Moyenne BGR sur le patch (cv2 = BGR donc [:,:,2]=R, [:,:,1]=G, [:,:,0]=B)
        signal_b.append(float(patch[:, :, 0].mean()))
        signal_g.append(float(patch[:, :, 1].mean()))
        signal_r.append(float(patch[:, :, 2].mean()))

    return (
        np.array(signal_r, dtype=np.float32),
        np.array(signal_g, dtype=np.float32),
        np.array(signal_b, dtype=np.float32),
        n_skipped,
    )


# ─────────────────────── traitement signal ───────────────────────────

def bandpass_filter(signal, fps, low_hz=0.75, high_hz=4.0):
    """
    Filtre passe-bande simple par FFT.
    Garde uniquement les fréquences entre low_hz et high_hz (45–240 bpm).
    """
    if len(signal) < 8:
        return signal

    n    = len(signal)
    fft  = np.fft.rfft(signal - signal.mean())
    freq = np.fft.rfftfreq(n, d=1.0 / fps)

    # Mettre à zéro les fréquences hors bande
    mask = (freq >= low_hz) & (freq <= high_hz)
    fft_filtered = fft * mask

    return np.fft.irfft(fft_filtered, n=n).astype(np.float32)


def chrom_rppg(signal_r, signal_g, signal_b):
    """
    Méthode CHROM (De Haan & Jeanne, 2013) — simple et robuste.
    Xs = 3R - 2G
    Ys = 1.5R + G - 1.5B
    signal_rppg = Xs - alpha * Ys  où alpha = std(Xs)/std(Ys)
    """
    if len(signal_r) < 4:
        return np.zeros(max(len(signal_r), 1), dtype=np.float32)

    # Normalisation par la moyenne (évite les variations d'illumination)
    mean_r = signal_r.mean() if signal_r.mean() != 0 else 1.0
    mean_g = signal_g.mean() if signal_g.mean() != 0 else 1.0
    mean_b = signal_b.mean() if signal_b.mean() != 0 else 1.0

    r_n = signal_r / mean_r
    g_n = signal_g / mean_g
    b_n = signal_b / mean_b

    Xs = 3 * r_n - 2 * g_n
    Ys = 1.5 * r_n + g_n - 1.5 * b_n

    std_xs = Xs.std() if Xs.std() > 1e-6 else 1.0
    std_ys = Ys.std() if Ys.std() > 1e-6 else 1.0
    alpha  = std_xs / std_ys

    return (Xs - alpha * Ys).astype(np.float32)


def extract_rppg_features(signal_r, signal_g, signal_b, fps=25.0):
    """
    Calcule les 4 features rPPG depuis les signaux RGB bruts.

    Returns dict avec :
        rppg_dominant_freq  : fréquence dominante en Hz (≈ HR/60)
        rppg_hr_estimate    : estimation HR en bpm
        rppg_snr            : rapport signal/bruit (en dB)
        rppg_signal_std     : écart-type du signal filtré (énergie)
    """
    # Cas dégénéré
    if len(signal_r) < 8:
        return {
            "rppg_dominant_freq": 0.0,
            "rppg_hr_estimate":   0.0,
            "rppg_snr":           0.0,
            "rppg_signal_std":    0.0,
        }

    # 1. Construire le signal rPPG via CHROM
    rppg_raw = chrom_rppg(signal_r, signal_g, signal_b)

    # 2. Filtrage passe-bande 0.75–4 Hz
    rppg_filt = bandpass_filter(rppg_raw, fps, low_hz=0.75, high_hz=4.0)

    # 3. Spectre de puissance
    n    = len(rppg_filt)
    fft  = np.fft.rfft(rppg_filt - rppg_filt.mean())
    freq = np.fft.rfftfreq(n, d=1.0 / fps)
    psd  = np.abs(fft) ** 2

    # 4. Fréquence dominante dans la bande [0.75, 4.0] Hz
    mask_band = (freq >= 0.75) & (freq <= 4.0)
    if mask_band.sum() == 0:
        dominant_freq = 0.0
        hr_estimate   = 0.0
        snr           = 0.0
    else:
        psd_band     = psd.copy()
        psd_band[~mask_band] = 0.0
        idx_peak     = np.argmax(psd_band)
        dominant_freq = float(freq[idx_peak])
        hr_estimate   = dominant_freq * 60.0    # Hz → bpm

        # SNR = puissance pic / puissance totale dans la bande
        power_peak  = float(psd[idx_peak])
        power_band  = float(psd[mask_band].sum())
        snr = 10 * np.log10(power_peak / (power_band - power_peak + 1e-8))

    signal_std = float(rppg_filt.std()) if len(rppg_filt) > 0 else 0.0

    return {
        "rppg_dominant_freq": round(dominant_freq, 6),
        "rppg_hr_estimate":   round(hr_estimate, 4),
        "rppg_snr":           round(float(snr), 4),
        "rppg_signal_std":    round(signal_std, 6),
    }


# ─────────────────────── precompute (script principal) ───────────────

def read_rows(csv_path):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["frame_idx"] = int(row["frame_idx"])
            row["label"]     = int(row["label"])
            rows.append(row)
    return rows


def precompute_rppg(split_csv, out_csv, fps=25.0,
                    model_path="models/face_landmarker.task"):
    """
    Lit le CSV de split, calcule les features rPPG pour chaque vidéo,
    et sauvegarde dans out_csv.
    """
    from src.behavior.mp_landmarks import FaceLandmarkerHelper

    rows     = read_rows(split_csv)
    by_vid   = defaultdict(list)
    for r in rows:
        by_vid[r["video_id"]].append(r)

    landmarker = FaceLandmarkerHelper(model_path=model_path)

    out_rows = []
    total    = len(by_vid)
    for i, (vid, frs) in enumerate(by_vid.items()):
        vid_label = int(frs[0]["label"])
        frs_sorted = sorted(frs, key=lambda x: x["frame_idx"])

        sig_r, sig_g, sig_b, n_skip = extract_rgb_signal_from_frames(
            frs_sorted, landmarker, fps=fps
        )

        feats = extract_rppg_features(sig_r, sig_g, sig_b, fps=fps)

        out_rows.append({
            "video_id":  vid,
            "label":     vid_label,
            "n_frames":  len(frs_sorted),
            "n_skipped": n_skip,
            **feats,
        })

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {vid} | "
                  f"HR~{feats['rppg_hr_estimate']:.1f} bpm | "
                  f"SNR={feats['rppg_snr']:.2f} dB | "
                  f"skipped={n_skip}")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    if not out_rows:
        print("[WARN] Aucune vidéo traitée.")
        return

    cols = list(out_rows[0].keys())
    with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n[OK] Sauvegarde : {out_csv} ({len(out_rows)} videos)")


def main():
    parser = argparse.ArgumentParser(description="Precompute rPPG features")
    parser.add_argument("--split_csv",
        default=r"data\processed\casia\splits_subject\train.csv")
    parser.add_argument("--out_csv",
        default=r"data\processed\casia\behav\train_rppg.csv")
    parser.add_argument("--fps",    type=float, default=25.0)
    parser.add_argument("--model",
        default="models/face_landmarker.task")
    args = parser.parse_args()

    print(f"Extraction rPPG : {args.split_csv}")
    print(f"FPS             : {args.fps}")
    precompute_rppg(
        split_csv  = args.split_csv,
        out_csv    = args.out_csv,
        fps        = args.fps,
        model_path = args.model,
    )

    # Lancer pour les 3 splits :
    print("\n[INFO] Pour traiter les 3 splits :")
    print("  python -m src.behavior.extract_rppg "
          "--split_csv data/processed/casia/splits_subject/train.csv "
          "--out_csv   data/processed/casia/behav/train_rppg.csv")
    print("  python -m src.behavior.extract_rppg "
          "--split_csv data/processed/casia/splits_subject/val.csv "
          "--out_csv   data/processed/casia/behav/val_rppg.csv")
    print("  python -m src.behavior.extract_rppg "
          "--split_csv data/processed/casia/splits_subject/test.csv "
          "--out_csv   data/processed/casia/behav/test_rppg.csv")


if __name__ == "__main__":
    main()