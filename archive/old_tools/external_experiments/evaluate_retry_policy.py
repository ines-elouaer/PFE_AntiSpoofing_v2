from pathlib import Path
import argparse
import pandas as pd


def decide(score: float, low: float, high: float) -> str:
    """
    Politique métier :
    - score faible  : ACCEPT
    - score élevé   : REJECT
    - score moyen   : RETRY
    """
    if score < low:
        return "ACCEPT"
    if score > high:
        return "REJECT"
    return "RETRY"


def summarize_by_class(df: pd.DataFrame):
    rows = []

    for label_value, label_name in [(0, "REAL"), (1, "SPOOF")]:
        sub = df[df["label"] == label_value].copy()
        total = len(sub)

        if total == 0:
            continue

        accept = int((sub["business_decision"] == "ACCEPT").sum())
        retry = int((sub["business_decision"] == "RETRY").sum())
        reject = int((sub["business_decision"] == "REJECT").sum())

        rows.append({
            "class": label_name,
            "total": total,
            "ACCEPT": accept,
            "RETRY": retry,
            "REJECT": reject,
            "ACCEPT_%": round(accept / total * 100, 2),
            "RETRY_%": round(retry / total * 100, 2),
            "REJECT_%": round(reject / total * 100, 2),
        })

    return pd.DataFrame(rows)


def summarize_business_errors(df: pd.DataFrame):
    """
    Analyse métier :
    - REAL accepté     : bon
    - REAL retry       : acceptable, friction utilisateur
    - REAL rejeté      : mauvais faux rejet dur
    - SPOOF rejeté     : bon
    - SPOOF retry      : acceptable, pas accepté directement
    - SPOOF accepté    : dangereux
    """
    real = df[df["label"] == 0]
    spoof = df[df["label"] == 1]

    real_total = len(real)
    spoof_total = len(spoof)

    real_accept = int((real["business_decision"] == "ACCEPT").sum())
    real_retry = int((real["business_decision"] == "RETRY").sum())
    real_reject = int((real["business_decision"] == "REJECT").sum())

    spoof_accept = int((spoof["business_decision"] == "ACCEPT").sum())
    spoof_retry = int((spoof["business_decision"] == "RETRY").sum())
    spoof_reject = int((spoof["business_decision"] == "REJECT").sum())

    return pd.DataFrame([{
        "real_total": real_total,
        "spoof_total": spoof_total,

        "REAL_ACCEPT_good": real_accept,
        "REAL_RETRY_friction": real_retry,
        "REAL_REJECT_hard_false_reject": real_reject,

        "SPOOF_ACCEPT_danger": spoof_accept,
        "SPOOF_RETRY_safe_uncertain": spoof_retry,
        "SPOOF_REJECT_good": spoof_reject,

        "hard_false_reject_rate_%": round(real_reject / real_total * 100, 2) if real_total else 0.0,
        "dangerous_attack_accept_rate_%": round(spoof_accept / spoof_total * 100, 2) if spoof_total else 0.0,
    }])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--low", type=float, default=0.30)
    parser.add_argument("--high", type=float, default=0.85)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    df = pd.read_csv(pred_path)

    required = {"video_id", "label", "score_spoof"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans predictions.csv : {missing}")

    if args.low >= args.high:
        raise ValueError("--low doit être inférieur à --high")

    df["business_decision"] = df["score_spoof"].apply(
        lambda s: decide(float(s), args.low, args.high)
    )

    by_class = summarize_by_class(df)
    business = summarize_business_errors(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_out = out_dir / "predictions_with_retry_decision.csv"
    by_class_out = out_dir / "retry_policy_by_class.csv"
    business_out = out_dir / "retry_policy_business_summary.csv"

    df.to_csv(predictions_out, index=False, encoding="utf-8")
    by_class.to_csv(by_class_out, index=False, encoding="utf-8")
    business.to_csv(business_out, index=False, encoding="utf-8")

    print("========== RETRY POLICY ==========")
    print(f"Low threshold  : {args.low}")
    print(f"High threshold : {args.high}")

    print("\n========== BY CLASS ==========")
    print(by_class.to_string(index=False))

    print("\n========== BUSINESS SUMMARY ==========")
    print(business.to_string(index=False))

    print("\nSaved:")
    print(predictions_out)
    print(by_class_out)
    print(business_out)


if __name__ == "__main__":
    main()