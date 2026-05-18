from pathlib import Path
import argparse
import pandas as pd


RPPG_COLS = [
    "rppg_dominant_freq",
    "rppg_hr_estimate",
    "rppg_snr",
    "rppg_signal_std",
    "rppg_skipped_rate",
    "rppg_valid",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behav_csv", required=True)
    parser.add_argument("--rppg_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    behav = pd.read_csv(args.behav_csv)
    rppg = pd.read_csv(args.rppg_csv)

    behav["video_id"] = behav["video_id"].astype(str)
    rppg["video_id"] = rppg["video_id"].astype(str)

    keep = ["video_id"] + [c for c in RPPG_COLS if c in rppg.columns]
    rppg = rppg[keep].copy()

    merged = behav.merge(rppg, on="video_id", how="left")

    for c in RPPG_COLS:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = merged[c].fillna(0.0)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False, encoding="utf-8")

    print("Behav rows :", len(behav))
    print("rPPG rows  :", len(rppg))
    print("Merged rows:", len(merged))
    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()