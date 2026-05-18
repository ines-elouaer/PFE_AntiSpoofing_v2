from pathlib import Path
import argparse
import json
import pandas as pd


FEAT_COLS = [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--stats_json", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    out = df.copy()

    for col in FEAT_COLS:
        if col not in out.columns:
            out[col] = 0.0

        if col not in stats:
            raise KeyError(f"Colonne {col} absente du fichier stats: {args.stats_json}")

        mean = float(stats[col]["mean"])
        std = float(stats[col]["std"])

        if std < 1e-8:
            std = 1.0

        out[col] = (out[col].fillna(0.0) - mean) / std

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")

    print("Input :", args.input_csv)
    print("Stats :", args.stats_json)
    print("Saved :", args.out_csv)


if __name__ == "__main__":
    main()