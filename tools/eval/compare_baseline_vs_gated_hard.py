from pathlib import Path
import pandas as pd


def load_predictions(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "pred_label" not in df.columns:
        df["pred_label"] = (df["score_spoof"] >= 0.5).astype(int)

    return df


def summarize(df, model_name):
    real_real = int(((df["label"] == 0) & (df["pred_label"] == 0)).sum())
    real_spoof = int(((df["label"] == 0) & (df["pred_label"] == 1)).sum())
    spoof_real = int(((df["label"] == 1) & (df["pred_label"] == 0)).sum())
    spoof_spoof = int(((df["label"] == 1) & (df["pred_label"] == 1)).sum())

    real_total = real_real + real_spoof
    spoof_total = spoof_real + spoof_spoof

    bpcer = real_spoof / real_total if real_total else 0.0
    apcer = spoof_real / spoof_total if spoof_total else 0.0
    acer = (bpcer + apcer) / 2

    return {
        "model": model_name,
        "REAL->REAL": real_real,
        "REAL->SPOOF": real_spoof,
        "SPOOF->REAL": spoof_real,
        "SPOOF->SPOOF": spoof_spoof,
        "BPCER": round(bpcer, 4),
        "APCER": round(apcer, 4),
        "ACER": round(acer, 4),
        "mean_score_spoof": round(float(df["score_spoof"].mean()), 4),
    }


def main():
    baseline_path = "reports/mixed_casia_axon_local_msu_final_eval/mixed_test/predictions.csv"
    new_path = "reports/mixed_casia_axon_local_msu_gated_hard_balanced_eval/mixed_test/predictions.csv"

    baseline = load_predictions(baseline_path)
    new = load_predictions(new_path)

    rows = [
        summarize(baseline, "baseline_concat"),
        summarize(new, "gated_hard_balanced"),
    ]

    out = pd.DataFrame(rows)

    out_dir = Path("reports/comparison_baseline_vs_gated_hard")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "comparison_metrics.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8")

    print(out.to_string(index=False))
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()