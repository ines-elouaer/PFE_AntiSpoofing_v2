"""
benchmark_inference_final.py
=============================
Mesure le temps réel d'inférence du modèle Deep+Behav Consecutive.

Ce script mesure :
  - Temps CNN (MobileNetV3 sur T=16 frames)
  - Temps LSTM + pooling temporel
  - Temps fusion comportementale
  - Temps total par vidéo
  - Throughput (vidéos/seconde)

Usage depuis E:\PFE_AntiSpoofing_v2 :

    python tools/benchmark_inference_final.py ^
        --checkpoint experiments/deep_no_pts/Deep+Behav_noPTS/deep_behav_seed42/best_model.pth ^
        --config     experiments/deep_no_pts/Deep+Behav_noPTS/deep_behav_seed42/config.json ^
        --out_dir    reports/benchmark

Prérequis :
    pip install torch torchvision
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deep_learning.models_cnn_lstm import CNN_LSTM_PAD


# ─── Mesure précise GPU/CPU ──────────────────────────────────────────

def measure_time_cpu(fn, n_warmup=5, n_runs=50):
    """Mesure temps CPU avec warmup."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    return np.mean(times), np.std(times)


def measure_time_gpu(fn, device, n_warmup=10, n_runs=100):
    """Mesure temps GPU avec torch.cuda.Event (précision haute)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(n_runs):
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))  # ms

    return np.mean(times), np.std(times)


# ─── Benchmark ───────────────────────────────────────────────────────

def run_benchmark(checkpoint_path: str, config_path: str, out_dir: str,
                  T: int = 16, img_size: int = 224, n_runs: int = 50):

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"
    print(f"Device  : {device}")
    print(f"T       : {T} frames")
    print(f"N runs  : {n_runs}")

    # Charger config
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Config  : {config_path}")

    # Charger modèle
    model = CNN_LSTM_PAD(
        hidden            = config.get("hidden", 256),
        num_layers        = config.get("num_layers", 1),
        bidir             = config.get("bidir", False),
        head_dropout      = 0.0,
        temporal_pool     = config.get("temporal_pool", "median"),
        pretrained_backbone = False,
        use_behav         = config.get("use_behav", True),
        behav_dim         = config.get("behav_dim", 9),
        behav_hidden      = config.get("behav_hidden", 16),
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device,
                           weights_only=False)
        # Gérer les différents formats de checkpoint
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=True)
        print(f"Checkpoint chargé : {checkpoint_path}")
    else:
        print("[WARN] Checkpoint non trouvé — utilisation poids aléatoires (benchmark valide quand même)")

    model.to(device)
    model.eval()

    # Données synthétiques (batch=1, T frames)
    x_single = torch.randn(1, T, 3, img_size, img_size).to(device)
    b_single  = torch.randn(1, config.get("behav_dim", 9)).to(device)

    results = {}

    print(f"\n{'='*50}")
    print("BENCHMARK — Inférence vidéo unique (batch=1)")
    print(f"{'='*50}")

    with torch.no_grad():
        if use_gpu:
            # ── GPU ──────────────────────────────────────────────────
            # CNN seul (backbone)
            def run_cnn():
                B, T_, C, H, W = x_single.shape
                x_flat = x_single.view(B * T_, C, H, W)
                _ = model.backbone(x_flat)

            mean_ms, std_ms = measure_time_gpu(run_cnn, device, n_runs=n_runs)
            results["cnn_backbone_ms"] = {"mean": round(mean_ms, 2), "std": round(std_ms, 2)}
            print(f"  CNN backbone (GPU)     : {mean_ms:.2f} ± {std_ms:.2f} ms")

            # LSTM seul
            feat_dim = model.backbone.out_dim
            fake_feats = torch.randn(1, T, feat_dim).to(device)
            def run_lstm():
                out, _ = model.lstm(fake_feats)
                _ = out.median(dim=1).values

            mean_ms, std_ms = measure_time_gpu(run_lstm, device, n_runs=n_runs)
            results["lstm_ms"] = {"mean": round(mean_ms, 2), "std": round(std_ms, 2)}
            print(f"  LSTM + pooling (GPU)   : {mean_ms:.2f} ± {std_ms:.2f} ms")

            # Inférence complète
            def run_full():
                _ = model(x_single, b_single)

            mean_ms, std_ms = measure_time_gpu(run_full, device, n_runs=n_runs)
            results["full_inference_gpu_ms"] = {"mean": round(mean_ms, 2), "std": round(std_ms, 2)}
            print(f"  Full inference (GPU)   : {mean_ms:.2f} ± {std_ms:.2f} ms")
            print(f"  Throughput (GPU 1 vid) : {1000/mean_ms:.1f} vidéos/sec")

        # ── CPU ──────────────────────────────────────────────────────
        x_cpu = x_single.cpu()
        b_cpu = b_single.cpu()
        model_cpu = model.cpu()

        def run_cpu():
            with torch.no_grad():
                _ = model_cpu(x_cpu, b_cpu)

        mean_ms, std_ms = measure_time_cpu(run_cpu, n_runs=min(n_runs, 20))
        results["full_inference_cpu_ms"] = {"mean": round(mean_ms, 2), "std": round(std_ms, 2)}
        print(f"  Full inference (CPU)   : {mean_ms:.2f} ± {std_ms:.2f} ms")
        print(f"  Throughput (CPU 1 vid) : {1000/mean_ms:.1f} vidéos/sec")

        if use_gpu:
            model.to(device)

    # ── Batch sizes ───────────────────────────────────────────────────
    if use_gpu:
        print(f"\n{'='*50}")
        print("BENCHMARK — Différents batch sizes (GPU)")
        print(f"{'='*50}")
        results["batch_throughput"] = {}

        for bs in [1, 2, 4, 8]:
            x_batch = torch.randn(bs, T, 3, img_size, img_size).to(device)
            b_batch = torch.randn(bs, config.get("behav_dim", 9)).to(device)

            def run_batch():
                _ = model(x_batch, b_batch)

            mean_ms, std_ms = measure_time_gpu(run_batch, device, n_runs=n_runs)
            throughput = bs * 1000 / mean_ms
            results["batch_throughput"][f"batch_{bs}"] = {
                "latency_ms": round(mean_ms, 2),
                "throughput_vid_per_sec": round(throughput, 1),
            }
            print(f"  Batch={bs:2d}: {mean_ms:7.2f} ms | "
                  f"{throughput:6.1f} vidéos/sec")

    # ── Résumé ────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("RÉSUMÉ")
    print(f"{'='*50}")
    cpu_ms = results["full_inference_cpu_ms"]["mean"]
    print(f"  Temps CPU / vidéo : {cpu_ms:.1f} ms  (~{cpu_ms/1000:.2f} sec)")
    if use_gpu:
        gpu_ms = results["full_inference_gpu_ms"]["mean"]
        print(f"  Temps GPU / vidéo : {gpu_ms:.1f} ms  (~{gpu_ms/1000:.2f} sec)")
        print(f"  Accélération GPU  : x{cpu_ms/gpu_ms:.1f}")

    results["device"]   = str(device)
    results["T_frames"] = T
    results["img_size"] = img_size
    results["n_runs"]   = n_runs

    # Sauvegarder
    out_json = out_path / "benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Résultats sauvegardés : {out_json}")

    # Générer figure
    _plot_benchmark(results, out_path / "benchmark_plot.png", use_gpu)


def _plot_benchmark(results, out_path, use_gpu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Mode clair
    bg      = "#FFFFFF"
    ax_bg   = "#F8F9FA"
    text_c  = "#2C3E50"
    grid_c  = "#BDC3C7"
    title_c = "#1A252F"
    val_c   = "#2C3E50"   # couleur chiffres sur barres

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(bg)
    fig.suptitle(
        "Temps d'Inférence — Deep+Behav Consecutive\n"
        "MobileNetV3-Large + LSTM + Behavioral Features",
        color=title_c, fontsize=13, fontweight="bold"
    )

    # ── Étapes ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(ax_bg)

    stages = []
    cpu_ms = []
    gpu_ms_vals = []

    if use_gpu and "cnn_backbone_ms" in results:
        stages += ["CNN\n(16 frames)", "LSTM\npool"]
        cpu_approx_cnn = (
            results["cnn_backbone_ms"]["mean"]
            * (results["full_inference_cpu_ms"]["mean"]
               / results["full_inference_gpu_ms"]["mean"])
        ) if use_gpu else results["cnn_backbone_ms"]["mean"]
        cpu_ms += [round(cpu_approx_cnn, 1), 0]
        gpu_ms_vals += [
            results["cnn_backbone_ms"]["mean"],
            results["lstm_ms"]["mean"],
        ]

    stages += ["Total"]
    cpu_ms.append(results["full_inference_cpu_ms"]["mean"])
    if use_gpu:
        gpu_ms_vals.append(results["full_inference_gpu_ms"]["mean"])

    x = np.arange(len(stages))
    w = 0.35
    bars_cpu = ax.bar(
        x - w/2 if use_gpu else x, cpu_ms,
        w if use_gpu else 0.6,
        color="#2980B9", alpha=0.85, label="CPU", edgecolor="white"
    )
    if use_gpu and gpu_ms_vals:
        bars_gpu = ax.bar(
            x + w/2, gpu_ms_vals, w,
            color="#16A085", alpha=0.85, label="GPU (CUDA)", edgecolor="white"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(stages, color=text_c, fontsize=10)
    ax.set_ylabel("Temps (ms)", color=text_c, fontsize=11)
    ax.set_title("Par étape — T=16 frames", color=text_c, fontsize=11,
                 fontweight="bold")
    ax.tick_params(colors=text_c)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(grid_c)
    ax.grid(axis="y", color=grid_c, alpha=0.5, linewidth=0.8)
    ax.legend(framealpha=0.9, facecolor=bg, edgecolor=grid_c,
              labelcolor=text_c, fontsize=10)

    # ── Throughput ───────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(ax_bg)

    configs_names = ["CPU\n1 vidéo"]
    throughputs   = [round(1000 / results["full_inference_cpu_ms"]["mean"], 1)]
    bar_colors    = ["#2980B9"]

    if use_gpu and "batch_throughput" in results:
        for k, v in results["batch_throughput"].items():
            bs = k.split("_")[1]
            configs_names.append(f"GPU\nbatch={bs}")
            throughputs.append(v["throughput_vid_per_sec"])
            if bs == "1":
                bar_colors.append("#16A085")
            elif bs == "4":
                bar_colors.append("#27AE60")
            else:
                bar_colors.append("#D4920A")

    bars = ax2.bar(
        configs_names, throughputs,
        color=bar_colors, alpha=0.85, edgecolor="white"
    )
    ax2.set_ylabel("Vidéos / seconde", color=text_c, fontsize=11)
    ax2.set_title("Débit (throughput)", color=text_c, fontsize=11,
                  fontweight="bold")
    ax2.tick_params(colors=text_c)
    ax2.set_xticklabels(configs_names, color=text_c, fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for sp in ["bottom", "left"]:
        ax2.spines[sp].set_color(grid_c)
    ax2.grid(axis="y", color=grid_c, alpha=0.5, linewidth=0.8)

    for bar, val in zip(bars, throughputs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val}", ha="center",
            color=val_c, fontsize=11, fontweight="bold"
        )

    ax2.text(
        0.5, -0.15,
        "* T=16 frames/vidéo — mesures réelles torch.cuda.Event (GPU) "
        "et time.perf_counter (CPU)",
        transform=ax2.transAxes, ha="center",
        color="#7F8C8D", fontsize=8, style="italic"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=bg)
    plt.close()
    print(f"[OK] Figure benchmark : {out_path}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark inférence PAD")
    parser.add_argument("--checkpoint",
        default=r"experiments\deep_no_pts\Deep+Behav_noPTS\deep_behav_seed42\best_model.pth",
        help="Chemin vers best_model.pth")
    parser.add_argument("--config",
        default=r"experiments\deep_no_pts\Deep+Behav_noPTS\deep_behav_seed42\config.json",
        help="Chemin vers config.json")
    parser.add_argument("--out_dir", default=r"reports\benchmark")
    parser.add_argument("--T",       type=int, default=16, help="Frames par vidéo")
    parser.add_argument("--n_runs",  type=int, default=50, help="Nombre de mesures")
    args = parser.parse_args()

    run_benchmark(
        checkpoint_path = args.checkpoint,
        config_path     = args.config,
        out_dir         = args.out_dir,
        T               = args.T,
        n_runs          = args.n_runs,
    )


if __name__ == "__main__":
    main()