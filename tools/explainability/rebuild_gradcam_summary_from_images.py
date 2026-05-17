from pathlib import Path
import pandas as pd
import re


def parse_file(path: Path):
    name = path.name

    # Exemple:
    # CASIA__10_2_score_0.0039_final_gradcam.jpg
    m = re.match(r"(.+)_score_([0-9.]+)_final_gradcam\.jpg", name)

    if not m:
        return None

    video_id = m.group(1)
    score = float(m.group(2))

    case = path.parent.name

    if case == "real_to_real":
        target_class = 0
    elif case == "spoof_to_spoof":
        target_class = 1
    elif case == "real_to_spoof":
        target_class = 1
    elif case == "spoof_to_real":
        target_class = 0
    else:
        target_class = None

    return {
        "case": case,
        "video_id": video_id,
        "score_spoof": score,
        "target_class": target_class,
        "gradcam_path": str(path),
    }


def main():
    root = Path("reports/gradcam_examples/final_model_mixed_cases")

    rows = []

    for img in root.rglob("*_final_gradcam.jpg"):
        item = parse_file(img)
        if item is not None:
            rows.append(item)

    out = pd.DataFrame(rows)

    out_path = root / "final_model_gradcam_summary.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    print("Images found:", len(rows))
    print("Saved:", out_path)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
