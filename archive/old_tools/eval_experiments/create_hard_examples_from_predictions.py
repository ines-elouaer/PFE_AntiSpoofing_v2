from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
        help="Chemin vers predictions.csv généré sur le train."
    )

    parser.add_argument(
        "--out_dir",
        default="data/hard_examples",
        help="Dossier de sortie."
    )

    parser.add_argument(
        "--weight",
        type=float,
        default=3.0,
        help="Poids des hard examples."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Seuil utilisé si pred_label n'existe pas."
    )

    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.csv introuvable: {pred_path}")

    df = pd.read_csv(pred_path)

    required = {"video_id", "label", "score_spoof"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    if "pred_label" not in df.columns:
        df["pred_label"] = (df["score_spoof"] >= args.threshold).astype(int)

    hard_real = df[(df["label"] == 0) & (df["pred_label"] == 1)].copy()
    hard_spoof = df[(df["label"] == 1) & (df["pred_label"] == 0)].copy()

    hard_real_out = hard_real[["video_id"]].drop_duplicates().copy()
    hard_real_out["hard_type"] = "REAL_AS_SPOOF"
    hard_real_out["sample_weight"] = args.weight

    hard_spoof_out = hard_spoof[["video_id"]].drop_duplicates().copy()
    hard_spoof_out["hard_type"] = "SPOOF_AS_REAL"
    hard_spoof_out["sample_weight"] = args.weight

    hard_real_path = out_dir / "hard_real_samples.csv"
    hard_spoof_path = out_dir / "hard_spoof_samples.csv"

    hard_real_out.to_csv(hard_real_path, index=False, encoding="utf-8")
    hard_spoof_out.to_csv(hard_spoof_path, index=False, encoding="utf-8")

    print("========== HARD EXAMPLES FROM TRAIN ==========")
    print(f"Predictions file : {pred_path}")
    print(f"Total videos     : {len(df)}")
    print(f"REAL -> SPOOF    : {len(hard_real_out)}")
    print(f"SPOOF -> REAL    : {len(hard_spoof_out)}")

    print("\\nSaved:")
    print(f"  {hard_real_path}")
    print(f"  {hard_spoof_path}")

    if len(hard_real_out) > 0:
        print("\\nHard REAL examples:")
        print(hard_real_out.to_string(index=False))

    if len(hard_spoof_out) > 0:
        print("\\nHard SPOOF examples:")
        print(hard_spoof_out.to_string(index=False))


if __name__ == "__main__":
    main()
