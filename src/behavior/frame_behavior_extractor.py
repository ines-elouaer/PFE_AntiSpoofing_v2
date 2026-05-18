import cv2
import numpy as np

from .landmarks import (
    ear_from_landmarks, LEFT_EYE, RIGHT_EYE,
    NOSE_TIP,
)

def extract_ear_and_motion_from_frames(frame_rows, landmarker, every_n=1, max_frames=600):
    
    ears = []
    motions = []
    prev_xy = None
    skipped = 0
    used = 0

    for i, r in enumerate(frame_rows):
        if i % every_n != 0:
            continue

        img_path = r["path"]
        frame = cv2.imread(img_path)  
        if frame is None:
            skipped += 1
            used += 1
            continue

        lm = landmarker.detect_landmarks(frame)
        if lm is None:
            skipped += 1
            used += 1
            continue

        h, w = frame.shape[:2]

        ear_l = ear_from_landmarks(lm, LEFT_EYE, w, h)
        ear_r = ear_from_landmarks(lm, RIGHT_EYE, w, h)
        if ear_l is not None and ear_r is not None:
            ears.append((ear_l + ear_r) / 2.0)

        x = lm[NOSE_TIP].x * w
        y = lm[NOSE_TIP].y * h
        xy = np.array([x, y], dtype=np.float32)

        if prev_xy is not None:
            motions.append(float(np.linalg.norm(xy - prev_xy)))
        prev_xy = xy

        used += 1
        if used >= max_frames:
            break

    return np.array(ears, dtype=np.float32), np.array(motions, dtype=np.float32), skipped, used
