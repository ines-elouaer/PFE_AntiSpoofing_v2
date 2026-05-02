from pathlib import Path
import pandas as pd

from src.pad_system.router import PADRouter

ROOT = Path(r"E:\PFE_AntiSpoofing_v2")

router = PADRouter()

# =========================
# TEST IMAGE avec une frame CASIA existante
# =========================
frames_dir = ROOT / "data" / "processed" / "casia" / "all_img" / "color"

image_candidates = list(frames_dir.glob("*.jpg"))

if not image_candidates:
    print("[SKIP] Aucune image trouvée dans:", frames_dir)
else:
    image_path = image_candidates[0]
    print(f"\n[TEST IMAGE] image utilisée = {image_path}")

    result_img = router.analyze(str(image_path))

    print("\n=== TEST IMAGE ===")
    print(result_img)


# =========================
# TEST VIDEO depuis CSV CASIA
# =========================
test_csv = ROOT / "data" / "processed" / "CASIA" / "splits_subject" / "test.csv"

df = pd.read_csv(test_csv)

real_id = df[df["label"] == 0]["video_id"].drop_duplicates().iloc[0]
spoof_id = df[df["label"] == 1]["video_id"].drop_duplicates().iloc[0]

print(f"\n[TEST VIDEO REAL] video_id = {real_id}")
result_real = router.analyze(real_id)
print(result_real)

print(f"\n[TEST VIDEO SPOOF] video_id = {spoof_id}")
result_spoof = router.analyze(spoof_id)
print(result_spoof)