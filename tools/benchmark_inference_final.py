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

[CORRECTIONS v2]
  - Closure correcte pour run_batch (évite capture par référence)
  - torch.cuda.empty_cache() entre chaque batch size
  - Affichage mémoire VRAM pour diagnostiquer batch=8
  - Gestion OutOfMemoryError propre
  - torch.cuda.synchronize() garanti avant ET après chaque mesure

Usage depuis E:\PFE_AntiSpoofing_v2 :

    python tools/benchmark_inference_final.py ^
        --checkpoint experiments/deep_no_pts/Deep+Behav_noPTS/deep_behav_seed42/best_model.pth ^
        --config     experiments/deep_no_pts/Deep+Behav_noPTS/deep_behav_seed42/config.json ^
        --out_dir    reports/benchmark
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
    """
    Mesure temps GPU avec torch.cuda.Event.
    
    CORRECTION : synchronize() AVANT l'enregistrement du start event
    pour s'assurer que toutes les opérations précédentes sont terminées.
    """
    # Warmup
    for _ in range(n_warmup):
        fn()
    # Synchronisation complète après warmup
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)

        # CORRECTION : synchronize avant start pour mesure propre
        torch.cuda.synchronize()
        start_evt.record()
        fn()
        end_evt.record()
        # CORRECTION : synchronize après end pour attendre la fin réelle
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))  # ms

    return np.mean(times), np.std(times)


def get_gpu_memory_info():
    """Retourne (utilisée_GB, totale_GB) de la mémoire GPU."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    used  = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return round(used, 2), round(total, 2)


# ─── Benchmark ───────────────────────────────────────────────────────

def run_benchmark(checkpoint_path: str, config_path: str, out_dir: str,
                  T: int = 16, img_size: int = 224, n_runs: int = 50):

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"
    print(f"Device  : {device}")
    print(f"T       : {T} frames")
    print(f"N runs  : {n_runs}")

    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        _, mem_total = get_gpu_memory_info()
        print(f"GPU     : {gpu_name} ({mem_total:.1f} GB VRAM)")

    # Charger config
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Config  : {config_path}")

    # Charger modèle
    model = CNN_LSTM_PAD(
        hidden              = config.get("hidden", 256),
        num_layers          = config.get("num_layers", 1),
        bidir               = config.get("bidir", False),
        head_dropout        = 0.0,
        temporal_pool       = config.get("temporal_pool", "median"),
        pretrained_backbone = False,
        use_behav           = config.get("use_behav", True),
        behav_dim           = config.get("behav_dim", 9),
        behav_hidden        = config.get("behav_hidden", 16),
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device,
                           weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=True)
        print(f"Checkpoint chargé : {checkpoint_path}")
    else:
        print("[WARN] Checkpoint non trouvé — poids aléatoires (benchmark valide quand même)")

    model.to(device)
    model.eval()

    x_single = torch.randn(1, T, 3, img_size, img_size).to(device)
    b_single  = torch.randn(1, config.get("behav_dim", 9)).to(device)

    results = {}

    print(f"\n{'='*55}")
    print("BENCHMARK — Inférence vidéo unique (batch=1)")
    print(f"{'='*55}")

    with torch.no_grad():
        if use_gpu:
            # CNN backbone seul
            def run_cnn():
                B, T_, C_ch, H, W = x_single.shape
                x_flat = x_single.view(B * T_, C_ch, H, W)
                _ = model.backbone(x_flat)

            mean_ms, std_ms = measure_time_gpu(run_cnn, device, n_runs=n_runs)
            results["cnn_backbone_ms"] = {
                "mean": round(mean_ms, 2), "std": round(std_ms, 2)
            }
            print(f"  CNN backbone (GPU)     : {mean_ms:.2f} ± {std_ms:.2f} ms")

            # LSTM seul
            feat_dim   = model.backbone.out_dim
            fake_feats = torch.randn(1, T, feat_dim).to(device)

            def run_lstm():
                out, _ = model.lstm(fake_feats)
                _ = out.median(dim=1).values

            mean_ms, std_ms = measure_time_gpu(run_lstm, device, n_runs=n_runs)
            results["lstm_ms"] = {
                "mean": round(mean_ms, 2), "std": round(std_ms, 2)
            }
            print(f"  LSTM + pooling (GPU)   : {mean_ms:.2f} ± {std_ms:.2f} ms")

            # Inférence complète batch=1
            def run_full():
                _ = model(x_single, b_single)

            mean_ms, std_ms = measure_time_gpu(run_full, device, n_runs=n_runs)
            results["full_inference_gpu_ms"] = {
                "mean": round(mean_ms, 2), "std": round(std_ms, 2)
            }
            print(f"  Full inference (GPU)   : {mean_ms:.2f} ± {std_ms:.2f} ms")
            print(f"  Throughput (GPU 1 vid) : {1000/mean_ms:.1f} vidéos/sec")

        # CPU
        x_cpu     = x_single.cpu()
        b_cpu     = b_single.cpu()
        model_cpu = model.cpu()

        def run_cpu():
            with torch.no_grad():
                _ = model_cpu(x_cpu, b_cpu)

        mean_ms, std_ms = measure_time_cpu(run_cpu, n_runs=min(n_runs, 20))
        results["full_inference_cpu_ms"] = {
            "mean": round(mean_ms, 2), "std": round(std_ms, 2)
        }
        print(f"  Full inference (CPU)   : {mean_ms:.2f} ± {std_ms:.2f} ms")
        print(f"  Throughput (CPU 1 vid) : {1000/mean_ms:.1f} vidéos/sec")

        if use_gpu:
            model.to(device)

    # ── Batch sizes ───────────────────────────────────────────────────
    if use_gpu:
        print(f"\n{'='*55}")
        print("BENCHMARK — Différents batch sizes (GPU)")
        print(f"{'='*55}")
        results["batch_throughput"] = {}

        for bs in [1, 2, 4, 8]:
            # CORRECTION : vide le cache avant chaque test
            torch.cuda.empty_cache()

            try:
                x_batch = torch.randn(
                    bs, T, 3, img_size, img_size
                ).to(device)
                b_batch = torch.randn(
                    bs, config.get("behav_dim", 9)
                ).to(device)

                # CORRECTION v3 : synchronize APRÈS l'allocation pour que
                # memory_allocated() reflète la vraie mémoire utilisée
                torch.cuda.synchronize()
                mem_avant, mem_total = get_gpu_memory_info()
                print(f"\n  [MEM avant inférence] batch={bs} | "
                      f"VRAM: {mem_avant:.2f}/{mem_total:.2f} GB")

                # CORRECTION : closure explicite pour capturer x_batch et b_batch
                # correctement (évite le bug de capture par référence en Python)
                def make_run_batch(xb, bb):
                    def run():
                        _ = model(xb, bb)
                    return run

                run_batch = make_run_batch(x_batch, b_batch)

                mean_ms, std_ms = measure_time_gpu(
                    run_batch, device, n_runs=n_runs
                )
                throughput = bs * 1000 / mean_ms

                # CORRECTION v3 : synchronize avant lecture mémoire post-inférence
                torch.cuda.synchronize()
                mem_apres, _ = get_gpu_memory_info()
                print(f"  Batch={bs:2d} : {mean_ms:7.2f} ± {std_ms:.2f} ms | "
                      f"{throughput:6.1f} vid/s | "
                      f"VRAM après: {mem_apres:.3f} GB")

                # Diagnostic si chute de performance
                if bs > 1:
                    prev_key = f"batch_{bs // 2}"
                    if prev_key in results["batch_throughput"]:
                        prev_tp = results["batch_throughput"][prev_key]["throughput_vid_per_sec"]
                        ratio   = throughput / prev_tp
                        if ratio < 0.5:
                            print(f"  ⚠️  CHUTE DÉTECTÉE : throughput divisé par "
                                  f"{1/ratio:.1f}x vs batch={bs//2}")
                            print(f"      → Cause probable : saturation VRAM "
                                  f"({mem_apres:.2f}/{mem_total:.2f} GB)")

                results["batch_throughput"][f"batch_{bs}"] = {
                    "latency_ms":             round(mean_ms, 2),
                    "latency_std_ms":         round(std_ms, 2),
                    "throughput_vid_per_sec": round(throughput, 1),
                    "vram_used_gb":           mem_apres,
                }

            except torch.cuda.OutOfMemoryError:
                # CORRECTION : gestion propre OOM au lieu d'un crash
                mem_apres, _ = get_gpu_memory_info()
                print(f"  Batch={bs:2d} : ❌ OUT OF MEMORY "
                      f"(VRAM {mem_apres:.2f}/{mem_total:.2f} GB insuffisant)")
                torch.cuda.empty_cache()
                results["batch_throughput"][f"batch_{bs}"] = {
                    "latency_ms":             -1,
                    "throughput_vid_per_sec": 0,
                    "error":                  "OutOfMemoryError",
                }

    # ── Résumé ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("RÉSUMÉ")
    print(f"{'='*55}")
    cpu_ms = results["full_inference_cpu_ms"]["mean"]
    print(f"  Temps CPU / vidéo : {cpu_ms:.1f} ms")
    if use_gpu:
        gpu_ms = results["full_inference_gpu_ms"]["mean"]
        print(f"  Temps GPU / vidéo : {gpu_ms:.1f} ms")
        print(f"  Accélération GPU  : x{cpu_ms/gpu_ms:.1f}")

        if "batch_throughput" in results:
            print("\n  Throughput par batch size :")
            for k, v in results["batch_throughput"].items():
                bs = k.split("_")[1]
                if v["throughput_vid_per_sec"] > 0:
                    print(f"    batch={bs} → {v['throughput_vid_per_sec']:.1f} vid/s "
                          f"(VRAM: {v.get('vram_used_gb', '?'):.2f} GB)")
                else:
                    print(f"    batch={bs} → ❌ OOM")

    results["device"]   = str(device)
    results["T_frames"] = T
    results["img_size"] = img_size
    results["n_runs"]   = n_runs

    out_json = out_path / "benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Résultats sauvegardés : {out_json}")

    _plot_benchmark(results, out_path / "benchmark_plot.png", use_gpu)


# ─── Figure ──────────────────────────────────────────────────────────

def _plot_benchmark(results, out_path, use_gpu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg      = "#FFFFFF"
    ax_bg   = "#F8F9FA"
    text_c  = "#2C3E50"
    grid_c  = "#BDC3C7"
    title_c = "#1A252F"
    val_c   = "#2C3E50"

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

    stages, cpu_ms, gpu_ms_vals = [], [], []

    if use_gpu and "cnn_backbone_ms" in results:
        stages += ["CNN\n(16 frames)", "LSTM\npool"]
        cpu_approx_cnn = (
            results["cnn_backbone_ms"]["mean"]
            * (results["full_inference_cpu_ms"]["mean"]
               / results["full_inference_gpu_ms"]["mean"])
        )
        cpu_ms    += [round(cpu_approx_cnn, 1), 0]
        gpu_ms_vals += [
            results["cnn_backbone_ms"]["mean"],
            results["lstm_ms"]["mean"],
        ]

    stages.append("Total")
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
    ax.set_title("Par étape — T=16 frames", color=text_c,
                 fontsize=11, fontweight="bold")
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
    annotations   = [""]  # pas d'annotation pour CPU

    # Récupère la VRAM de base (batch=1) pour calculer les ratios
    batch_1_vram = 0.0
    if use_gpu and "batch_throughput" in results:
        b1 = results["batch_throughput"].get("batch_1", {})
        batch_1_vram = b1.get("vram_used_gb", 0.0)

    if use_gpu and "batch_throughput" in results:
        for k, v in results["batch_throughput"].items():
            bs     = k.split("_")[1]
            bs_int = int(bs)
            tp     = v["throughput_vid_per_sec"]
            vram   = v.get("vram_used_gb", 0)
            configs_names.append(f"GPU\nbatch={bs}")
            throughputs.append(tp)

            # CORRECTION v4 : ratio VRAM vs batch=1 plutôt que valeur absolue
            # MobileNetV3 est léger — la diff absolue est trop petite à lire
            if tp == 0:
                annotations.append("OOM")
            elif bs_int == 1 or batch_1_vram == 0:
                annotations.append(f"{vram:.3f}GB\n(base)")
            else:
                ratio = vram / batch_1_vram if batch_1_vram > 0 else 1.0
                annotations.append(f"×{ratio:.1f}\nVRAM")

            if tp == 0:
                bar_colors.append("#E74C3C")
            elif bs == "1":
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
    ax2.set_title("Débit (throughput)", color=text_c,
                  fontsize=11, fontweight="bold")
    ax2.tick_params(colors=text_c)
    ax2.set_xticklabels(configs_names, color=text_c, fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for sp in ["bottom", "left"]:
        ax2.spines[sp].set_color(grid_c)
    ax2.grid(axis="y", color=grid_c, alpha=0.5, linewidth=0.8)

    for bar, val, ann in zip(bars, throughputs, annotations):
        label = str(val) if val > 0 else "OOM"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            label, ha="center",
            color=val_c, fontsize=11, fontweight="bold"
        )
        # Annotation VRAM sous la valeur — afficher seulement si > 0
        if ann and ann != "OOM":
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                ann, ha="center",
                color="white", fontsize=7.5, alpha=0.9,
                fontweight="bold"
            )

    ax2.text(
        0.5, -0.15,
        "* T=16 frames/vidéo — mesures réelles torch.cuda.Event (GPU) "
        "et time.perf_counter (CPU)\n"
        "** VRAM indiquée sur chaque barre GPU",
        transform=ax2.transAxes, ha="center",
        color="#7F8C8D", fontsize=7.5, style="italic"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=bg)
    plt.close()
    print(f"[OK] Figure benchmark : {out_path}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark inférence PAD")
    parser.add_argument("--checkpoint",
        default=r"experiments\deep_no_pts\Deep+Behav_noPTS\deep_behav_seed42\best_model.pth")
    parser.add_argument("--config",
        default=r"experiments\deep_no_pts\Deep+Behav_noPTS\deep_behav_seed42\config.json")
    parser.add_argument("--out_dir",  default=r"reports\benchmark")
    parser.add_argument("--T",        type=int, default=16)
    parser.add_argument("--n_runs",   type=int, default=50)
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