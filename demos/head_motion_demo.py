import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.dnn_face import load_dnn, detect_face_dnn


def run_motion_demo(video_path, net, every_n=1, max_frames=900, conf_thresh=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    times, cx_list, cy_list, area_list, conf_list = [], [], [], [], []
    used = 0
    i = 0
    skipped = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % every_n == 0:
            bbox, conf = detect_face_dnn(net, frame, conf_thresh=conf_thresh)
            if bbox is None:
                skipped += 1
            else:
                (x, y, w, h) = bbox
                cx = x + w / 2.0
                cy = y + h / 2.0
                area = float(w * h)

                t = i / fps
                times.append(t)
                cx_list.append(cx)
                cy_list.append(cy)
                area_list.append(area)
                conf_list.append(conf)

            used += 1
            if used >= max_frames:
                break

        i += 1

    cap.release()

    return (
        np.array(times, dtype=np.float32),
        np.array(cx_list, dtype=np.float32),
        np.array(cy_list, dtype=np.float32),
        np.array(area_list, dtype=np.float32),
        np.array(conf_list, dtype=np.float32),
        float(fps),
        int(skipped),
    )


def save_plots(times, cx, cy, area, out_path, title="Head Motion (Face center + size)"):
    if len(times) == 0:
        raise RuntimeError("No detections to plot.")

    cx_n = (cx - cx.min()) / (cx.max() - cx.min() + 1e-6)
    cy_n = (cy - cy.min()) / (cy.max() - cy.min() + 1e-6)
    area_n = (area - area.min()) / (area.max() - area.min() + 1e-6)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(times, cx_n, label="cx (norm)")
    ax.plot(times, cy_n, label="cy (norm)")
    ax.plot(times, area_n, label="area (norm)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized value (0..1)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def list_videos(folder):
    exts = (".mp4", ".avi", ".mov", ".mkv")
    paths = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in paths if p.lower().endswith(exts)]


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    real_dir = os.path.join(BASE_DIR, "data", "videos", "real")
    fake_dir = os.path.join(BASE_DIR, "data", "videos", "fake")

    out_root = os.path.join(BASE_DIR, "results", "motion_demo")
    out_real = os.path.join(out_root, "real")
    out_fake = os.path.join(out_root, "fake")
    os.makedirs(out_real, exist_ok=True)
    os.makedirs(out_fake, exist_ok=True)

    net = load_dnn(os.path.join(BASE_DIR, "models"))

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
                times, cx, cy, area, confs, fps, skipped = run_motion_demo(
                    vp, net, every_n=1, max_frames=900, conf_thresh=0.5
                )

                if len(times) == 0:
                    print(f"[SKIP] {label} {os.path.basename(vp)} -> no face detected")
                    total_skip += 1
                    continue

                out_plot = os.path.join(out_dir, f"{name}_motion_plot.png")
                out_csv = os.path.join(out_dir, f"{name}_motion_values.csv")

                save_plots(times, cx, cy, area, out_plot, title=f"Motion vs Time ({os.path.basename(vp)})")

                data = np.column_stack([times, cx, cy, area, confs])
                np.savetxt(out_csv, data, delimiter=",",
                           header="time_sec,cx,cy,area,conf", comments="")

                print(f"[OK] {label} -> {os.path.basename(vp)} | points={len(times)} fps={fps:.2f} skipped={skipped}")
                total_done += 1

            except Exception as e:
                print(f"[ERROR] {label} {os.path.basename(vp)} -> {e}")
                total_skip += 1

    print("\n================ SUMMARY ================")
    print(f"[OK] Done: {total_done} | Skipped/Failed: {total_skip}")
    print(f"[INFO] Results in: {out_root}")
