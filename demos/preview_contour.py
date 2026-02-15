import os
import glob
import cv2

from utils.face_utils_contour import FaceContourSegmenter

VALID_EXT = (".jpg", ".jpeg", ".png")

def save_side_by_side(img, face, out_path):
    H = 480
    left = cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H))
    right = cv2.resize(face, (H, H))
    combo = cv2.hconcat([left, right])
    cv2.imwrite(out_path, combo)

def list_images_recursive(folder):
    paths = glob.glob(os.path.join(folder, "**", "*"), recursive=True)
    paths = [p for p in paths if p.lower().endswith(VALID_EXT)]
    return sorted(paths)

def process_all_videos(frames_root, out_dir, n_per_video=3):
    
    os.makedirs(out_dir, exist_ok=True)
    seg = FaceContourSegmenter("models/face_landmarker.task")

    subfolders = sorted([
        os.path.join(frames_root, d)
        for d in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, d))
    ])

    total_saved = 0

    for sf in subfolders:
        video_name = os.path.basename(sf)  # real_01, fake_02...
        imgs = list_images_recursive(sf)[:n_per_video]

        if len(imgs) == 0:
            print(f"[SKIP] no images in: {sf}")
            continue

        for i, p in enumerate(imgs):
            img = cv2.imread(p)
            if img is None:
                print(f"[SKIP] unreadable: {p}")
                continue

            face, _ = seg.segment(img, out_size=(224, 224), pad=6, feather=4)
            if face is None:
                print(f"[SKIP] no face: {p}")
                continue

            out_path = os.path.join(out_dir, f"{video_name}_preview_{i:02d}.jpg")
            save_side_by_side(img, face, out_path)
            total_saved += 1

        print(f"[OK] previews for {video_name}: {len(imgs)} saved")

    return total_saved

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

  
    project_root = os.path.dirname(base)

    real_dir = os.path.join(project_root, "data", "frames", "real")
    fake_dir = os.path.join(project_root, "data", "frames", "fake")

    out_real = os.path.join(project_root, "results", "previews_contour", "real")
    out_fake = os.path.join(project_root, "results", "previews_contour", "fake")

    s1 = process_all_videos(real_dir, out_real, n_per_video=3)
    s2 = process_all_videos(fake_dir, out_fake, n_per_video=3)

    print(f"\n[OK] Saved previews REAL: {s1} -> {out_real}")
    print(f"[OK] Saved previews FAKE: {s2} -> {out_fake}")
