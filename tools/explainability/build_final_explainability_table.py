from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradcam_summary", required=True)
    parser.add_argument("--behavior_case_csv", required=True)
    parser.add_argument("--ablation_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    grad = pd.read_csv(args.gradcam_summary)
    beh = pd.read_csv(args.behavior_case_csv)
    abl = pd.read_csv(args.ablation_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # On garde quelques colonnes simples de l'ablation
    cols_ablation = [
        "video_id",
        "normal_score_spoof",
        "zero_behavior_score_spoof",
        "zero_rppg_score_spoof",
        "zero_all_score_spoof",
        "delta_zero_behavior",
        "delta_zero_rppg",
        "delta_zero_all",
    ]

    abl_small = abl[[c for c in cols_ablation if c in abl.columns]].copy()

    # On garde quelques features importantes
    important_features = [
        "video_id",
        "case",
        "blink_count",
        "ear_std",
        "motion_mean",
        "motion_std",
        "motion_max",
        "rppg_snr",
        "rppg_signal_std",
        "rppg_valid",
    ]

    beh_small = beh[[c for c in important_features if c in beh.columns]].copy()

    out = grad.merge(beh_small, on="video_id", how="left", suffixes=("", "_behav"))
    out = out.merge(abl_small, on="video_id", how="left")

    out_path = out_dir / "final_explainability_table.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    print("Saved:", out_path)
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
