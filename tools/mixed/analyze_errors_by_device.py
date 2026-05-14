from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRED_CSV = (
    PROJECT_ROOT
    / "reports"
    / "mixed_casia_axon_local_msu_final_eval"
    / "mixed_test"
    / "predictions.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "mixed_casia_axon_local_msu_final_eval"
    / "device_error_analysis"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


def decision_case(row):
    true_label = int(row["label"])
    pred_label = int(row["pred_label"])

    if true_label == 0 and pred_label == 0:
        return "REAL_AS_REAL"

    if true_label == 0 and pred_label == 1:
        return "REAL_AS_SPOOF"

    if true_label == 1 and pred_label == 0:
        return "SPOOF_AS_REAL"

    if true_label == 1 and pred_label == 1:
        return "SPOOF_AS_SPOOF"

    return "UNKNOWN"


def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Predictions introuvable: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV)

    required = {
        "video_id",
        "label",
        "pred_label",
        "score_spoof",
        "device_id",
        "source_dataset",
        "label_name",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans predictions.csv: {missing}")

    df["case"] = df.apply(decision_case, axis=1)

    df["true_label_name"] = df["label"].map({
        0: "REAL",
        1: "SPOOF",
    })

    df["pred_label_name"] = df["pred_label"].map({
        0: "REAL",
        1: "SPOOF",
    })

    # ======================================================
    # 1. Résumé global par appareil et type d'erreur
    # ======================================================

    summary = (
        df.groupby(["device_id", "case"])
        .size()
        .reset_index(name="count")
        .pivot(index="device_id", columns="case", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    for col in ["REAL_AS_REAL", "REAL_AS_SPOOF", "SPOOF_AS_REAL", "SPOOF_AS_SPOOF"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["total"] = (
        summary["REAL_AS_REAL"]
        + summary["REAL_AS_SPOOF"]
        + summary["SPOOF_AS_REAL"]
        + summary["SPOOF_AS_SPOOF"]
    )

    summary["real_total"] = summary["REAL_AS_REAL"] + summary["REAL_AS_SPOOF"]
    summary["spoof_total"] = summary["SPOOF_AS_REAL"] + summary["SPOOF_AS_SPOOF"]

    summary["BPCER_real_rejected"] = summary.apply(
        lambda r: r["REAL_AS_SPOOF"] / r["real_total"] if r["real_total"] > 0 else 0.0,
        axis=1,
    )

    summary["APCER_attack_accepted"] = summary.apply(
        lambda r: r["SPOOF_AS_REAL"] / r["spoof_total"] if r["spoof_total"] > 0 else 0.0,
        axis=1,
    )

    summary = summary[
        [
            "device_id",
            "total",
            "real_total",
            "spoof_total",
            "REAL_AS_REAL",
            "REAL_AS_SPOOF",
            "SPOOF_AS_REAL",
            "SPOOF_AS_SPOOF",
            "BPCER_real_rejected",
            "APCER_attack_accepted",
        ]
    ]

    summary_path = OUT_DIR / "summary_by_device_cases.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    # ======================================================
    # 2. Détails des REAL détectés comme SPOOF
    # ======================================================

    real_as_spoof = df[df["case"] == "REAL_AS_SPOOF"].copy()
    real_as_real = df[df["case"] == "REAL_AS_REAL"].copy()
    spoof_as_real = df[df["case"] == "SPOOF_AS_REAL"].copy()
    spoof_as_spoof = df[df["case"] == "SPOOF_AS_SPOOF"].copy()

    detail_cols = [
        "video_id",
        "original_video_id",
        "source_dataset",
        "device_id",
        "subject_id",
        "attack_type",
        "true_label_name",
        "pred_label_name",
        "score_spoof",
    ]

    detail_cols = [c for c in detail_cols if c in df.columns]

    real_as_spoof[detail_cols].sort_values(
        ["device_id", "score_spoof"],
        ascending=[True, False],
    ).to_csv(OUT_DIR / "real_detected_as_spoof.csv", index=False, encoding="utf-8")

    real_as_real[detail_cols].sort_values(
        ["device_id", "score_spoof"],
        ascending=[True, True],
    ).to_csv(OUT_DIR / "real_detected_as_real.csv", index=False, encoding="utf-8")

    spoof_as_real[detail_cols].sort_values(
        ["device_id", "score_spoof"],
        ascending=[True, True],
    ).to_csv(OUT_DIR / "spoof_detected_as_real.csv", index=False, encoding="utf-8")

    spoof_as_spoof[detail_cols].sort_values(
        ["device_id", "score_spoof"],
        ascending=[True, False],
    ).to_csv(OUT_DIR / "spoof_detected_as_spoof.csv", index=False, encoding="utf-8")

    # ======================================================
    # 3. Résumé par source_dataset + appareil
    # ======================================================

    source_device_summary = (
        df.groupby(["source_dataset", "device_id", "case"])
        .size()
        .reset_index(name="count")
        .pivot_table(
            index=["source_dataset", "device_id"],
            columns="case",
            values="count",
            fill_value=0,
        )
        .reset_index()
    )

    for col in ["REAL_AS_REAL", "REAL_AS_SPOOF", "SPOOF_AS_REAL", "SPOOF_AS_SPOOF"]:
        if col not in source_device_summary.columns:
            source_device_summary[col] = 0

    source_device_summary["total"] = (
        source_device_summary["REAL_AS_REAL"]
        + source_device_summary["REAL_AS_SPOOF"]
        + source_device_summary["SPOOF_AS_REAL"]
        + source_device_summary["SPOOF_AS_SPOOF"]
    )

    source_device_summary.to_csv(
        OUT_DIR / "summary_by_source_and_device.csv",
        index=False,
        encoding="utf-8",
    )

    # ======================================================
    # Console output
    # ======================================================

    print("\n========== SUMMARY BY DEVICE ==========")
    print(summary.to_string(index=False))

    print("\n========== REAL DETECTED AS SPOOF ==========")
    if len(real_as_spoof) == 0:
        print("Aucun REAL détecté comme SPOOF.")
    else:
        print(real_as_spoof[detail_cols].sort_values(
            ["device_id", "score_spoof"],
            ascending=[True, False],
        ).to_string(index=False))

    print("\n========== SPOOF DETECTED AS REAL ==========")
    if len(spoof_as_real) == 0:
        print("Aucun SPOOF détecté comme REAL.")
    else:
        print(spoof_as_real[detail_cols].sort_values(
            ["device_id", "score_spoof"],
            ascending=[True, True],
        ).to_string(index=False))

    print("\n========== SAVED FILES ==========")
    print(f"Summary by device       : {summary_path}")
    print(f"REAL as SPOOF details   : {OUT_DIR / 'real_detected_as_spoof.csv'}")
    print(f"REAL as REAL details    : {OUT_DIR / 'real_detected_as_real.csv'}")
    print(f"SPOOF as REAL details   : {OUT_DIR / 'spoof_detected_as_real.csv'}")
    print(f"SPOOF as SPOOF details  : {OUT_DIR / 'spoof_detected_as_spoof.csv'}")
    print(f"Source/device summary   : {OUT_DIR / 'summary_by_source_and_device.csv'}")

    print("\n[OK] Analyse par appareil terminée.")


if __name__ == "__main__":
    main()
    