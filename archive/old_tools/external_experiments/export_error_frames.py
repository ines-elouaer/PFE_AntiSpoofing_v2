from pathlib import Path
import argparse
import shutil
import pandas as pd


def select_center_frames(frames_df, video_id, n_frames=8):
    """
    Sélectionne quelques frames représentatives au centre d'une vidéo.
    """
    rows = frames_df[frames_df["video_id"].astype(str) == str(video_id)].copy()

    if len(rows) == 0:
        return rows

    rows = rows.sort_values("frame_idx").reset_index(drop=True)

    if len(rows) <= n_frames:
        return rows

    center = len(rows) // 2
    half = n_frames // 2

    start = max(0, center - half)
    end = min(len(rows), start + n_frames)

    return rows.iloc[start:end].copy()


def export_case_frames(case_df, frames_df, out_dir, case_name, n_frames=8):
    case_out = out_dir / case_name
    case_out.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for _, pred in case_df.iterrows():
        video_id = str(pred["video_id"])
        score = float(pred["score_spoof"])

        safe_video_id = video_id.replace("\\", "_").replace("/", "_").replace(":", "_")
        video_out = case_out / safe_video_id
        video_out.mkdir(parents=True, exist_ok=True)

        selected = select_center_frames(frames_df, video_id, n_frames=n_frames)

        if len(selected) == 0:
            print(f"[WARN] No frames found for {video_id}")
            continue

        for i, fr in enumerate(selected.itertuples(index=False), start=1):
            src = Path(fr.path)

            if not src.exists():
                print(f"[WARN] Missing frame: {src}")
                continue

            dst_name = f"{i:02d}_frame_{int(fr.frame_idx):05d}_score_{score:.4f}.jpg"
            dst = video_out / dst_name

            shutil.copy2(src, dst)

            summary_rows.append({
                "case": case_name,
                "video_id": video_id,
                "score_spoof": score,
                "frame_idx": int(fr.frame_idx),
                "src_frame": str(src),
                "exported_frame": str(dst),
            })

        print(f"[OK] {case_name} | {video_id} | score={score:.4f} | frames={len(selected)}")

    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--frames_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_frames", type=int, default=8)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    frames = pd.read_csv(args.frames_csv)

    required_pred_cols = {"video_id", "label", "pred_label", "score_spoof"}
    missing_pred = required_pred_cols - set(predictions.columns)
    if missing_pred:
        raise ValueError(f"Colonnes manquantes dans predictions.csv: {missing_pred}")

    required_frame_cols = {"video_id", "path", "frame_idx"}
    missing_frames = required_frame_cols - set(frames.columns)
    if missing_frames:
        raise ValueError(f"Colonnes manquantes dans frames_csv: {missing_frames}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # REAL -> SPOOF
    real_as_spoof = predictions[
        (predictions["label"] == 0) & (predictions["pred_label"] == 1)
    ].copy()

    # SPOOF -> REAL
    spoof_as_real = predictions[
        (predictions["label"] == 1) & (predictions["pred_label"] == 0)
    ].copy()

    real_as_spoof = real_as_spoof.sort_values("score_spoof", ascending=False)
    spoof_as_real = spoof_as_real.sort_values("score_spoof", ascending=True)

    print("========== ERROR CASES ==========")
    print("REAL -> SPOOF:", len(real_as_spoof))
    print("SPOOF -> REAL:", len(spoof_as_real))

    all_rows = []

    all_rows += export_case_frames(
        real_as_spoof,
        frames,
        out_dir,
        "real_as_spoof",
        n_frames=args.n_frames,
    )

    all_rows += export_case_frames(
        spoof_as_real,
        frames,
        out_dir,
        "spoof_as_real",
        n_frames=args.n_frames,
    )

    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "exported_error_frames_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    # Sauvegarder aussi les CSV des erreurs vidéo
    real_as_spoof.to_csv(out_dir / "real_as_spoof_videos.csv", index=False, encoding="utf-8")
    spoof_as_real.to_csv(out_dir / "spoof_as_real_videos.csv", index=False, encoding="utf-8")

    print("\nSaved:")
    print(summary_path)
    print(out_dir / "real_as_spoof_videos.csv")
    print(out_dir / "spoof_as_real_videos.csv")


if __name__ == "__main__":
    main()