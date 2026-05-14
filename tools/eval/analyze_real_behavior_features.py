from pathlib import Path
import argparse
import pandas as pd


BEHAV_FEATURES = [
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


def resolve_path(path_str: str, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return root / path


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} introuvable : {path}")
    return pd.read_csv(path)


def find_video_id_col(df: pd.DataFrame) -> str:
    possible_cols = ["video_id", "original_video_id"]
    for col in possible_cols:
        if col in df.columns:
            return col
    raise ValueError("Aucune colonne video_id ou original_video_id trouvée.")


def add_group(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    df = df.copy()
    df["decision_group"] = group_name
    return df


def detect_behavior_issues(row: pd.Series) -> str:
    issues = []

    if "blink_count" in row and pd.notna(row["blink_count"]):
        if row["blink_count"] <= 0:
            issues.append("aucun clignement détecté")

    if "motion_mean" in row and pd.notna(row["motion_mean"]):
        if row["motion_mean"] < 0.10:
            issues.append("mouvement moyen faible")

    if "motion_std" in row and pd.notna(row["motion_std"]):
        if row["motion_std"] < 0.05:
            issues.append("variation de mouvement faible")

    if "skipped_rate" in row and pd.notna(row["skipped_rate"]):
        if row["skipped_rate"] > 0.30:
            issues.append("détection visage instable")

    if "ear_std" in row and pd.notna(row["ear_std"]):
        if row["ear_std"] < 0.01:
            issues.append("EAR très stable / peu de variation oculaire")

    if not issues:
        return "aucun problème comportemental évident"

    return " | ".join(issues)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--real_as_real",
        default="reports/mixed_casia_axon_local_msu_motion_aware_test_eval/inspection_cases/real_as_real/real_as_real.csv",
        help="CSV des vidéos REAL correctement classées REAL.",
    )

    parser.add_argument(
        "--real_as_spoof",
        default="reports/mixed_casia_axon_local_msu_motion_aware_test_eval/inspection_cases/real_as_spoof/real_as_spoof.csv",
        help="CSV des vidéos REAL classées SPOOF.",
    )

    parser.add_argument(
        "--behav_csv",
        default="data/mixed_casia_axon_local_msu/mixed_behav_norm.csv",
        help="CSV des features comportementales normalisées.",
    )

    parser.add_argument(
        "--out_dir",
        default="reports/behavior_analysis_real_errors",
        help="Dossier de sortie.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    real_as_real_path = resolve_path(args.real_as_real, root)
    real_as_spoof_path = resolve_path(args.real_as_spoof, root)
    behav_path = resolve_path(args.behav_csv, root)
    out_dir = resolve_path(args.out_dir, root)

    out_dir.mkdir(parents=True, exist_ok=True)

    real_as_real = load_csv(real_as_real_path, "real_as_real.csv")
    real_as_spoof = load_csv(real_as_spoof_path, "real_as_spoof.csv")
    behav = load_csv(behav_path, "mixed_behav_norm.csv")

    real_as_real = add_group(real_as_real, "REAL -> REAL")
    real_as_spoof = add_group(real_as_spoof, "REAL -> SPOOF")

    real_cases = pd.concat([real_as_real, real_as_spoof], ignore_index=True)

    if "video_id" not in real_cases.columns:
        raise ValueError("Les fichiers real_as_real / real_as_spoof doivent contenir video_id.")

    if "video_id" not in behav.columns:
        raise ValueError("Le fichier behavior doit contenir video_id.")

    existing_features = [f for f in BEHAV_FEATURES if f in behav.columns]

    if not existing_features:
        raise ValueError(
            "Aucune feature comportementale attendue trouvée. "
            f"Colonnes disponibles : {list(behav.columns)}"
        )

    merged = real_cases.merge(
        behav[["video_id"] + existing_features],
        on="video_id",
        how="left",
    )

    missing_behav = merged[existing_features].isna().all(axis=1).sum()
    if missing_behav > 0:
        print(f"[WARNING] {missing_behav} vidéos n'ont pas de features comportementales associées.")

    # Résumé statistique par groupe
    summary = (
        merged
        .groupby("decision_group")[existing_features]
        .agg(["mean", "std", "min", "max"])
    )

    # Tableau plus simple : moyennes uniquement
    mean_table = (
        merged
        .groupby("decision_group")[existing_features]
        .mean()
        .T
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    if "REAL -> REAL" in mean_table.columns and "REAL -> SPOOF" in mean_table.columns:
        mean_table["difference_spoof_minus_real"] = (
            mean_table["REAL -> SPOOF"] - mean_table["REAL -> REAL"]
        )

    # Analyse vidéo par vidéo pour les REAL -> SPOOF
    rejected = merged[merged["decision_group"] == "REAL -> SPOOF"].copy()

    if "score_spoof" in rejected.columns:
        rejected = rejected.sort_values("score_spoof", ascending=False)

    rejected["behavior_issue"] = rejected.apply(detect_behavior_issues, axis=1)

    detail_cols = [
        "video_id",
        "score_spoof",
        "source_dataset",
        "device_id",
        "subject_id",
        "condition",
        "attack_type",
        "decision_group",
    ]

    detail_cols = [c for c in detail_cols if c in rejected.columns]
    detail_cols = detail_cols + existing_features + ["behavior_issue"]

    rejected_detail = rejected[detail_cols].copy()

    # Sauvegarde
    merged_out = out_dir / "real_cases_with_behavior_features.csv"
    mean_out = out_dir / "comparison_real_real_vs_real_spoof_behavior_means.csv"
    rejected_out = out_dir / "real_rejected_as_spoof_behavior_diagnosis.csv"
    summary_out = out_dir / "behavior_summary_by_decision_group.csv"

    merged.to_csv(merged_out, index=False, encoding="utf-8")
    mean_table.to_csv(mean_out, index=False, encoding="utf-8")
    rejected_detail.to_csv(rejected_out, index=False, encoding="utf-8")
    summary.to_csv(summary_out, encoding="utf-8")

    print("\n========== FEATURES DISPONIBLES ==========")
    print(existing_features)

    print("\n========== COMPARAISON DES MOYENNES ==========")
    print(mean_table.to_string(index=False))

    print("\n========== VIDÉOS REAL REJETÉES AVEC DIAGNOSTIC ==========")
    print(rejected_detail.to_string(index=False))

    print("\n========== FICHIERS GÉNÉRÉS ==========")
    print(f"Cas REAL avec features : {merged_out}")
    print(f"Comparaison moyennes   : {mean_out}")
    print(f"Diagnostic faux rejets : {rejected_out}")
    print(f"Résumé complet         : {summary_out}")


if __name__ == "__main__":
    main()