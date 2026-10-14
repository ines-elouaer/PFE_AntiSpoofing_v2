from pathlib import Path
import argparse
import json
import pandas as pd


BASE_COLS = [
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

FEAT_COLS = BASE_COLS + RPPG_COLS


def fit_stats(train_df):
    stats = {}
    for col in FEAT_COLS:
        if col not in train_df.columns:
            train_df[col] = 0.0

        mean = float(train_df[col].mean())
        std = float(train_df[col].std())

        if std < 1e-8:
            std = 1.0

        stats[col] = {"mean": mean, "std": std}

    return stats


def apply_norm(df, stats):
    out = df.copy()

    for col in FEAT_COLS:
        if col not in out.columns:
            out[col] = 0.0

        mean = stats[col]["mean"]
        std = stats[col]["std"]
        out[col] = (out[col].fillna(0.0) - mean) / std

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    val = pd.read_csv(args.val_csv)
    test = pd.read_csv(args.test_csv)

    stats = fit_stats(train)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_norm = apply_norm(train, stats)
    val_norm = apply_norm(val, stats)
    test_norm = apply_norm(test, stats)

    train_out = out_dir / "mixed_train_behav_rppg_norm.csv"
    val_out = out_dir / "mixed_val_behav_rppg_norm.csv"
    test_out = out_dir / "mixed_test_behav_rppg_norm.csv"
    stats_out = out_dir / "behav_rppg_norm_stats.json"

    train_norm.to_csv(train_out, index=False, encoding="utf-8")
    val_norm.to_csv(val_out, index=False, encoding="utf-8")
    test_norm.to_csv(test_out, index=False, encoding="utf-8")

    with stats_out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Saved:")
    print(train_out)
    print(val_out)
    print(test_out)
    print(stats_out)


if __name__ == "__main__":
    main()