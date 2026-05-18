from pathlib import Path
import argparse
import shutil
import pandas as pd


def resolve_path(p: str, root: Path) -> Path:
    p = Path(str(p))
    if p.is_absolute():
        return p
    return (root / p).resolve()


def safe_name(x) -> str:
    return str(x).replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def get_case_name(label: int, pred_label: int) -> str:
    """
    label = 0 : REAL
    label = 1 : SPOOF

    pred_label = 0 : prédit REAL
    pred_label = 1 : prédit SPOOF
    """
    if label == 0 and pred_label == 1:
        return "real_as_spoof"      # faux rejet / BPCER
    if label == 0 and pred_label == 0:
        return "real_as_real"       # vrai utilisateur accepté
    if label == 1 and pred_label == 0:
        return "spoof_as_real"      # attaque acceptée / APCER
    if label == 1 and pred_label == 1:
        return "spoof_as_spoof"     # attaque rejetée correctement

    return "unknown_case"


def build_case_report(
    case_df: pd.DataFrame,
    frames: pd.DataFrame,
    video_meta: pd.DataFrame,
    out_dir: Path,
    root: Path,
    case_name: str,
    copy_frames: bool = True,
):
    case_out_dir = out_dir / case_name
    case_out_dir.mkdir(parents=True, exist_ok=True)

    frames_out_dir = case_out_dir / "frames_examples"
    if copy_frames:
        frames_out_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = [
        "video_id",
        "original_video_id",
        "label_name",
        "source_dataset",
        "domain",
        "device_id",
        "subject_id",
        "condition",
        "attack_type",
    ]

    for c in meta_cols:
        if c not in video_meta.columns:
            video_meta[c] = "unknown"

    case_df = case_df.merge(video_meta[meta_cols], on="video_id", how="left")

    if "score_spoof" in case_df.columns:
        case_df = case_df.sort_values("score_spoof", ascending=False)

    report_rows = []

    for _, row in case_df.iterrows():
        video_id = row["video_id"]
        score = float(row["score_spoof"])

        video_frames = frames[frames["video_id"] == video_id].sort_values("frame_idx").copy()

        if len(video_frames) > 0:
            idxs = sorted(set([
                0,
                len(video_frames) // 2,
                len(video_frames) - 1,
            ]))
        else:
            idxs = []

        copied_frame_paths = []
        clean_video_id = safe_name(video_id)

        if copy_frames:
            for i in idxs:
                frame_row = video_frames.iloc[i]
                src = resolve_path(frame_row["path"], root)

                if not src.exists():
                    continue

                dst_name = (
                    f"{clean_video_id}"
                    f"_score_{score:.4f}"
                    f"_frame_{int(frame_row['frame_idx']):03d}"
                    f"{src.suffix}"
                )

                dst = frames_out_dir / dst_name
                shutil.copy2(src, dst)
                copied_frame_paths.append(str(dst))

        label = int(row["label"])
        pred_label = int(row["pred_label"])

        true_class = "REAL" if label == 0 else "SPOOF"
        predicted_as = "REAL" if pred_label == 0 else "SPOOF"

        report_rows.append({
            "video_id": video_id,
            "original_video_id": row.get("original_video_id", "unknown"),
            "true_class": true_class,
            "predicted_as": predicted_as,
            "label": label,
            "pred_label": pred_label,
            "score_spoof": score,
            "source_dataset": row.get("source_dataset", "unknown"),
            "domain": row.get("domain", "unknown"),
            "device_id": row.get("device_id", "unknown"),
            "subject_id": row.get("subject_id", "unknown"),
            "condition": row.get("condition", "unknown"),
            "attack_type": row.get("attack_type", "unknown"),
            "num_frames": int(len(video_frames)),
            "example_frames_copied": " | ".join(copied_frame_paths),
        })

    report_df = pd.DataFrame(report_rows)

    out_csv = case_out_dir / f"{case_name}.csv"
    report_df.to_csv(out_csv, index=False, encoding="utf-8")

    return out_csv, report_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
        help="Chemin vers predictions.csv généré par l'évaluation.",
    )

    parser.add_argument(
        "--frames_csv",
        required=True,
        help="Chemin vers le CSV des frames utilisé pour l'évaluation.",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Dossier de sortie pour les inspections.",
    )

    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.50,
        help="Seuil utilisé si pred_label n'existe pas.",
    )

    parser.add_argument(
        "--no_copy_frames",
        action="store_true",
        help="Ne pas copier les images exemples.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    pred_path = resolve_path(args.predictions, root)
    frames_path = resolve_path(args.frames_csv, root)
    out_dir = resolve_path(args.out_dir, root)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.csv introuvable: {pred_path}")

    if not frames_path.exists():
        raise FileNotFoundError(f"frames_csv introuvable: {frames_path}")

    pred = pd.read_csv(pred_path)
    frames = pd.read_csv(frames_path)

    required_pred_cols = {"video_id", "label", "score_spoof"}
    missing_pred = required_pred_cols - set(pred.columns)
    if missing_pred:
        raise ValueError(f"Colonnes manquantes dans predictions.csv: {missing_pred}")

    required_frame_cols = {"video_id", "path", "frame_idx"}
    missing_frames = required_frame_cols - set(frames.columns)
    if missing_frames:
        raise ValueError(f"Colonnes manquantes dans frames_csv: {missing_frames}")

    # Si pred_label n'existe pas, on le reconstruit depuis score_spoof
    if "pred_label" not in pred.columns:
        pred["pred_label"] = (pred["score_spoof"] >= args.score_threshold).astype(int)

    # Métadonnées vidéo à partir du frames_csv
    video_meta = (
        frames.sort_values(["video_id", "frame_idx"])
        .groupby("video_id")
        .first()
        .reset_index()
    )

    # Déterminer le cas de chaque prédiction
    pred["case"] = pred.apply(
        lambda r: get_case_name(int(r["label"]), int(r["pred_label"])),
        axis=1,
    )

    case_names = [
        "real_as_spoof",
        "real_as_real",
        "spoof_as_real",
        "spoof_as_spoof",
    ]

    summary_rows = []

    for case_name in case_names:
        case_df = pred[pred["case"] == case_name].copy()

        out_csv, report_df = build_case_report(
            case_df=case_df,
            frames=frames,
            video_meta=video_meta,
            out_dir=out_dir,
            root=root,
            case_name=case_name,
            copy_frames=not args.no_copy_frames,
        )

        summary_rows.append({
            "case": case_name,
            "count": len(report_df),
            "csv": str(out_csv),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "inspection_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    total = len(pred)
    real_total = int((pred["label"] == 0).sum())
    spoof_total = int((pred["label"] == 1).sum())

    real_as_spoof = int(((pred["label"] == 0) & (pred["pred_label"] == 1)).sum())
    real_as_real = int(((pred["label"] == 0) & (pred["pred_label"] == 0)).sum())
    spoof_as_real = int(((pred["label"] == 1) & (pred["pred_label"] == 0)).sum())
    spoof_as_spoof = int(((pred["label"] == 1) & (pred["pred_label"] == 1)).sum())

    bpcer = real_as_spoof / real_total if real_total > 0 else 0.0
    apcer = spoof_as_real / spoof_total if spoof_total > 0 else 0.0
    acer = (apcer + bpcer) / 2

    print("\n========== INSPECTION DES PRÉDICTIONS ==========")
    print(f"Total vidéos      : {total}")
    print(f"Total REAL        : {real_total}")
    print(f"Total SPOOF       : {spoof_total}")

    print("\n========== MATRICE D'INSPECTION ==========")
    print(f"REAL  -> REAL     : {real_as_real}")
    print(f"REAL  -> SPOOF    : {real_as_spoof}")
    print(f"SPOOF -> REAL     : {spoof_as_real}")
    print(f"SPOOF -> SPOOF    : {spoof_as_spoof}")

    print("\n========== MÉTRIQUES PAD ==========")
    print(f"BPCER = REAL -> SPOOF / REAL total  = {bpcer:.4f} ({bpcer * 100:.2f}%)")
    print(f"APCER = SPOOF -> REAL / SPOOF total = {apcer:.4f} ({apcer * 100:.2f}%)")
    print(f"ACER  = moyenne(APCER, BPCER)       = {acer:.4f} ({acer * 100:.2f}%)")

    print("\n========== FICHIERS GÉNÉRÉS ==========")
    print(f"Résumé : {summary_csv}")

    for _, row in summary_df.iterrows():
        print(f"{row['case']} : {row['count']} vidéos -> {row['csv']}")


if __name__ == "__main__":
    main()