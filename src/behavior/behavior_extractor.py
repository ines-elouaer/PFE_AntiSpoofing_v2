from pathlib import Path
import argparse
import json
import pickle
import sys

import pandas as pd


# ==========================================================
# Project imports
# ==========================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.behavior.landmarks import FaceLandmarkerHelper, video_features_from_signals
from src.behavior.frame_behavior_extractor import extract_ear_and_motion_from_frames


# ==========================================================
# Constants
# ==========================================================

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
]


DEFAULT_FACE_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
DEFAULT_SCALER = PROJECT_ROOT / "data" / "processed" / "casia" / "behav" / "behav_scaler.pkl"


# ==========================================================
# Helpers
# ==========================================================

def resolve_path(path_value: str, project_root: Path) -> str:
    """
    Résout un chemin frame.
    - Si le chemin est absolu : on le garde.
    - Si le chemin est relatif : on le transforme en chemin absolu depuis project_root.
    """
    p = Path(str(path_value))

    if p.is_absolute():
        return str(p)

    return str((project_root / p).resolve())


def load_scaler(scaler_path: Path):
    """
    Charge le StandardScaler CASIA.
    On essaie d'abord pickle, puis joblib si nécessaire.
    """
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler introuvable: {scaler_path}")

    try:
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        import joblib
        return joblib.load(scaler_path)


def normalize_behavior(raw_df: pd.DataFrame, scaler, feature_cols):
    norm_df = raw_df.copy()

    for col in feature_cols:
        if col not in norm_df.columns:
            norm_df[col] = 0.0

    norm_df[feature_cols] = (
        norm_df[feature_cols]
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .astype(float)
    )

    X = norm_df[feature_cols].values
    X_norm = scaler.transform(X)

    for i, col in enumerate(feature_cols):
        norm_df[col] = X_norm[:, i]

    return norm_df


def safe_get(row, col, default="unknown"):
    if col in row and not pd.isna(row[col]):
        return row[col]
    return default


def build_metadata(meta: pd.Series):
    """
    Conserve les colonnes utiles si elles existent.
    """
    label = int(safe_get(meta, "label", 0))
    label_name = str(safe_get(meta, "label_name", "REAL" if label == 0 else "SPOOF"))

    out = {
        "video_id": str(safe_get(meta, "video_id", "unknown_video")),
        "label": label,
        "label_name": label_name,
        "subject_id": str(safe_get(meta, "subject_id", "unknown_subject")),
        "device_id": str(safe_get(meta, "device_id", "unknown_device")),
        "condition": str(safe_get(meta, "condition", "unknown")),
        "take": str(safe_get(meta, "take", "unknown")),
        "attack_type": str(safe_get(meta, "attack_type", "unknown")),
        "domain": str(safe_get(meta, "domain", "unknown")),
        "split": str(safe_get(meta, "split", "unknown")),
    }

    # Colonnes optionnelles utiles
    for extra_col in ["level", "source_split", "source_video_path", "source_video_dir"]:
        if extra_col in meta.index:
            out[extra_col] = str(safe_get(meta, extra_col, "unknown"))

    return out


def extract_behavior_from_manifest(
    frames_csv: Path,
    raw_out_csv: Path,
    norm_out_csv: Path | None,
    summary_out_json: Path,
    scaler_path: Path | None,
    face_model_path: Path,
    every_n: int = 1,
    max_frames: int = 600,
):
    if not frames_csv.exists():
        raise FileNotFoundError(f"Frames CSV introuvable: {frames_csv}")

    if not face_model_path.exists():
        raise FileNotFoundError(f"FaceLandmarker introuvable: {face_model_path}")

    df = pd.read_csv(frames_csv)

    required_cols = {"path", "label", "video_id", "frame_idx"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Colonnes manquantes dans {frames_csv}: {missing}")

    df["frame_idx"] = df["frame_idx"].astype(int)

    print("========== CONFIG BEHAVIOR EXTRACTION ==========")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Frames CSV   : {frames_csv}")
    print(f"Raw out CSV  : {raw_out_csv}")
    print(f"Norm out CSV : {norm_out_csv if norm_out_csv else 'NON DEMANDE'}")
    print(f"Scaler       : {scaler_path if scaler_path else 'NON UTILISE'}")
    print(f"Face model   : {face_model_path}")
    print(f"Rows         : {len(df)}")
    print(f"Videos       : {df['video_id'].nunique()}")

    landmarker = FaceLandmarkerHelper(model_path=str(face_model_path))

    out_rows = []

    grouped = df.sort_values(["video_id", "frame_idx"]).groupby("video_id")
    total_videos = len(grouped)

    for idx, (video_id, group) in enumerate(grouped, start=1):
        group = group.sort_values("frame_idx").copy()

        # cv2.imread a besoin d'un vrai chemin lisible.
        group["path"] = group["path"].apply(lambda p: resolve_path(p, PROJECT_ROOT))

        frame_rows = group.to_dict("records")

        ears, motions, skipped, used = extract_ear_and_motion_from_frames(
            frame_rows=frame_rows,
            landmarker=landmarker,
            every_n=every_n,
            max_frames=max_frames,
        )

        feats = video_features_from_signals(ears, motions)
        skipped_rate = float(skipped) / float(max(used, 1))

        meta = build_metadata(group.iloc[0])
        meta["video_id"] = str(video_id)

        row = {
            **meta,
            **feats,
            "skipped_rate": skipped_rate,
            "used_frames_for_behavior": int(used),
            "skipped_frames_for_behavior": int(skipped),
        }

        out_rows.append(row)

        if idx % 25 == 0:
            print(f"[INFO] Progression: {idx}/{total_videos} vidéos traitées")

    raw_df = pd.DataFrame(out_rows)

    for col in FEAT_COLS:
        if col not in raw_df.columns:
            raw_df[col] = 0.0

    raw_df[FEAT_COLS] = (
        raw_df[FEAT_COLS]
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .astype(float)
    )

    # Ordre propre des colonnes
    base_cols = [
        "video_id",
        "label",
        "label_name",
        "subject_id",
        "device_id",
        "condition",
        "take",
        "attack_type",
        "domain",
        "split",
    ]

    optional_cols = [
        c for c in ["level", "source_split", "source_video_path", "source_video_dir"]
        if c in raw_df.columns
    ]

    diagnostic_cols = [
        "used_frames_for_behavior",
        "skipped_frames_for_behavior",
    ]

    final_cols = base_cols + optional_cols + FEAT_COLS + diagnostic_cols
    raw_df = raw_df[final_cols]

    raw_out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_out_csv, index=False, encoding="utf-8")

    normalized = False
    scaler_feature_cols = FEAT_COLS

    if scaler_path is not None and norm_out_csv is not None:
        scaler = load_scaler(scaler_path)

        if hasattr(scaler, "feature_names_in_"):
            scaler_feature_cols = list(scaler.feature_names_in_)

        missing_scaler_cols = [c for c in scaler_feature_cols if c not in raw_df.columns]

        if missing_scaler_cols:
            raise RuntimeError(
                "Colonnes attendues par le scaler absentes: "
                + str(missing_scaler_cols)
            )

        norm_df = normalize_behavior(
            raw_df=raw_df,
            scaler=scaler,
            feature_cols=scaler_feature_cols,
        )

        norm_out_csv.parent.mkdir(parents=True, exist_ok=True)
        norm_df.to_csv(norm_out_csv, index=False, encoding="utf-8")
        normalized = True

        # Sauvegarde des splits normalisés si la colonne split existe
        if "split" in norm_df.columns:
            split_dir = norm_out_csv.parent / "splits"
            split_dir.mkdir(parents=True, exist_ok=True)

            prefix = norm_out_csv.stem.replace("_behav_norm", "")

            for split in ["train", "val", "test"]:
                split_df = norm_df[norm_df["split"] == split].copy()
                split_df.to_csv(
                    split_dir / f"{prefix}_behav_{split}.csv",
                    index=False,
                    encoding="utf-8",
                )

    # Sauvegarde des splits bruts
    if "split" in raw_df.columns:
        split_dir = raw_out_csv.parent / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)

        prefix = raw_out_csv.stem.replace("_behav_raw", "")

        for split in ["train", "val", "test"]:
            split_df = raw_df[raw_df["split"] == split].copy()
            split_df.to_csv(
                split_dir / f"{prefix}_behav_raw_{split}.csv",
                index=False,
                encoding="utf-8",
            )

    summary = {
        "frames_csv": str(frames_csv),
        "raw_out_csv": str(raw_out_csv),
        "norm_out_csv": str(norm_out_csv) if norm_out_csv else None,
        "summary_out_json": str(summary_out_json),
        "scaler_path": str(scaler_path) if scaler_path else None,
        "normalized": normalized,
        "num_videos": int(raw_df["video_id"].nunique()),
        "num_rows": int(len(raw_df)),
        "feature_cols": FEAT_COLS,
        "scaler_feature_cols": scaler_feature_cols,
        "label_counts": raw_df["label_name"].value_counts().to_dict()
        if "label_name" in raw_df.columns else {},
        "domain_counts": raw_df["domain"].value_counts().to_dict()
        if "domain" in raw_df.columns else {},
        "split_counts": raw_df["split"].value_counts().to_dict()
        if "split" in raw_df.columns else {},
        "device_counts": raw_df["device_id"].value_counts().to_dict()
        if "device_id" in raw_df.columns else {},
        "mean_skipped_rate": float(raw_df["skipped_rate"].mean())
        if "skipped_rate" in raw_df.columns and len(raw_df) else 0.0,
        "max_skipped_rate": float(raw_df["skipped_rate"].max())
        if "skipped_rate" in raw_df.columns and len(raw_df) else 0.0,
    }

    summary_out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== SUMMARY BEHAVIOR ==========")
    print(f"Raw behavior CSV  : {raw_out_csv}")
    print(f"Norm behavior CSV : {norm_out_csv if normalized else 'NON GENERE'}")
    print(f"Summary JSON      : {summary_out_json}")
    print(f"Videos            : {summary['num_videos']}")
    print(f"Normalized        : {normalized}")
    print(f"Mean skipped_rate : {summary['mean_skipped_rate']:.4f}")
    print(f"Max skipped_rate  : {summary['max_skipped_rate']:.4f}")

    print("\n========== SPLIT COUNTS ==========")
    if "split" in raw_df.columns:
        print(raw_df["split"].value_counts())

    print("\n========== DEVICE COUNTS ==========")
    if "device_id" in raw_df.columns:
        print(raw_df["device_id"].value_counts())

    print("\n[OK] Extraction behavior terminée.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--frames_csv", required=True, type=str)
    parser.add_argument("--raw_out_csv", required=True, type=str)
    parser.add_argument("--norm_out_csv", default="", type=str)
    parser.add_argument("--summary_out_json", required=True, type=str)

    parser.add_argument(
        "--scaler_path",
        default=str(DEFAULT_SCALER),
        type=str,
        help="Chemin vers le StandardScaler CASIA train.",
    )

    parser.add_argument(
        "--face_model_path",
        default=str(DEFAULT_FACE_MODEL),
        type=str,
        help="Chemin vers models/face_landmarker.task",
    )

    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--every_n", default=1, type=int)
    parser.add_argument("--max_frames", default=600, type=int)

    args = parser.parse_args()

    frames_csv = Path(args.frames_csv)
    raw_out_csv = Path(args.raw_out_csv)
    norm_out_csv = Path(args.norm_out_csv) if args.norm_out_csv and not args.no_normalize else None
    summary_out_json = Path(args.summary_out_json)

    scaler_path = None if args.no_normalize else Path(args.scaler_path)
    face_model_path = Path(args.face_model_path)

    if not frames_csv.is_absolute():
        frames_csv = (PROJECT_ROOT / frames_csv).resolve()

    if not raw_out_csv.is_absolute():
        raw_out_csv = (PROJECT_ROOT / raw_out_csv).resolve()

    if norm_out_csv is not None and not norm_out_csv.is_absolute():
        norm_out_csv = (PROJECT_ROOT / norm_out_csv).resolve()

    if not summary_out_json.is_absolute():
        summary_out_json = (PROJECT_ROOT / summary_out_json).resolve()

    if scaler_path is not None and not scaler_path.is_absolute():
        scaler_path = (PROJECT_ROOT / scaler_path).resolve()

    if not face_model_path.is_absolute():
        face_model_path = (PROJECT_ROOT / face_model_path).resolve()

    extract_behavior_from_manifest(
        frames_csv=frames_csv,
        raw_out_csv=raw_out_csv,
        norm_out_csv=norm_out_csv,
        summary_out_json=summary_out_json,
        scaler_path=scaler_path,
        face_model_path=face_model_path,
        every_n=args.every_n,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
