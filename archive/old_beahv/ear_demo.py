import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.mp_landmarks import FaceLandmarkerHelper, ear_from_landmarks, LEFT_EYE, RIGHT_EYE


def run_ear_demo(video_path, landmarker, every_n=1, max_frames=600):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    times, ears = [], []
    i = 0
    used = 0
    skipped = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % every_n == 0:
            lm = landmarker.detect_landmarks(frame)
            if lm is None:
                skipped += 1
            else:
                h, w = frame.shape[:2]
                ear_l = ear_from_landmarks(lm, LEFT_EYE, w, h)
                ear_r = ear_from_landmarks(lm, RIGHT_EYE, w, h)
                if ear_l is not None and ear_r is not None:
                    t = i / fps
                    times.append(t)
                    ears.append((ear_l + ear_r) / 2.0)

            used += 1
            if used >= max_frames:
                break

        i += 1

    cap.release()
    return np.array(times, dtype=np.float32), np.array(ears, dtype=np.float32), float(fps), int(skipped)


def save_plot(times, ears, out_path, title="EAR vs Time"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times, ears)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EAR")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def list_videos(folder):
    exts = (".mp4", ".avi", ".mov", ".mkv")
    paths = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in paths if p.lower().endswith(exts)]


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # dossiers vidéos
    real_dir = os.path.join(BASE_DIR, "data", "videos", "real")
    fake_dir = os.path.join(BASE_DIR, "data", "videos", "fake")

    # sortie
    out_root = os.path.join(BASE_DIR, "results", "ear_demo")
    out_real = os.path.join(out_root, "real")
    out_fake = os.path.join(out_root, "fake")
    os.makedirs(out_real, exist_ok=True)
    os.makedirs(out_fake, exist_ok=True)

    # modèle mediapipe
    model_path = os.path.join(BASE_DIR, "models", "face_landmarker.task")
    landmarker = FaceLandmarkerHelper(model_path)

    total_done = 0
    total_skip = 0

    for label, folder, out_dir in [
        ("REAL", real_dir, out_real),
        ("FAKE", fake_dir, out_fake),
    ]:
        vids = list_videos(folder)
        if not vids:
            print(f"[WARN] No videos found in: {folder}")
            continue

        for vp in vids:
            name = os.path.splitext(os.path.basename(vp))[0]
            try:
                times, ears, fps, skipped = run_ear_demo(vp, landmarker, every_n=1, max_frames=600)

                if len(ears) == 0:
                    print(f"[SKIP] {label} {os.path.basename(vp)} -> no EAR (no face?)")
                    total_skip += 1
                    continue

                out_plot = os.path.join(out_dir, f"{name}_ear_plot.png")
                out_csv = os.path.join(out_dir, f"{name}_ear_values.csv")

                save_plot(times, ears, out_plot, title=f"EAR vs Time ({os.path.basename(vp)})")
                np.savetxt(out_csv, np.column_stack([times, ears]), delimiter=",",
                           header="time_sec,ear", comments="")

                print(f"[OK] {label} -> {os.path.basename(vp)} | points={len(ears)} fps={fps:.2f} skipped={skipped}")
                total_done += 1

            except Exception as e:
                print(f"[ERROR] {label} {os.path.basename(vp)} -> {e}")
                total_skip += 1

    print("\n================ SUMMARY ================")
    print(f"[OK] Done: {total_done} | Skipped/Failed: {total_skip}")
    print(f"[INFO] Results in: {out_root}")
